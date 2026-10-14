"""
Pane composition and H.264 encoding for review clips.

One :class:`ClipSession` owns one long-lived FFmpeg process fed raw RGB
frames, so the service never holds a whole clip in memory. Every pane is
letterboxed into its slot rather than cropped, so a review frame always shows
the whole camera image.
"""

from __future__ import annotations

import io
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageOps

from compositor.protocol import H264_CRF, H264_PRESET, H264_PROFILE, PANE_BACKGROUND, ClipSpec, Resolution, VideoProbe, pane_widths


def ffmpeg_version() -> str:
    """Report the FFmpeg build ``imageio-ffmpeg`` will run."""

    return str(imageio_ffmpeg.get_ffmpeg_version())


def compose_frame(panes: Sequence[bytes], *, resolution: Resolution, widths: Sequence[int]) -> bytes:
    """
    Compose one output frame from encoded pane images and return its raw ``rgb24`` bytes.

    :param panes: Encoded PNG or JPEG images, left to right.
    :param resolution: Output frame size.
    :param widths: Slot width for each pane; must sum to ``resolution.width``.
    :return: ``resolution.width * resolution.height * 3`` bytes.
    :raises ValueError: If the pane and slot counts differ or the slots do not fill the frame
    """

    if len(panes) != len(widths):
        raise ValueError(f"Received {len(panes)} panes for {len(widths)} slots")
    if sum(widths) != resolution.width:
        raise ValueError(f"Slot widths {tuple(widths)} do not sum to the frame width {resolution.width}")
    composite = Image.new("RGB", (resolution.width, resolution.height), PANE_BACKGROUND)
    left = 0
    for encoded, width in zip(panes, widths, strict=True):
        with Image.open(io.BytesIO(encoded)) as image:
            slot = ImageOps.pad(image.convert("RGB"), (width, resolution.height), color=PANE_BACKGROUND, method=Image.Resampling.LANCZOS)
        composite.paste(slot, (left, 0))
        left += width
    return composite.tobytes()


class ClipSession:
    """
    One open encode: an FFmpeg process fed composed frames in output order.

    :param spec: The validated clip request.
    :param video_path: Destination MP4.
    """

    def __init__(self, spec: ClipSpec, video_path: Path) -> None:
        self.spec = spec
        self.video_path = video_path
        self.widths = pane_widths(spec.resolution.width, spec.pane_count)
        self.frames_written = 0
        self.closed = False
        self._started = time.perf_counter()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(self._command(), stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _command(self) -> list[str]:
        return [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            str(self.spec.resolution),
            "-r",
            f"{self.spec.fps:g}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-preset",
            H264_PRESET,
            "-crf",
            str(H264_CRF),
            "-profile:v",
            H264_PROFILE,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.video_path),
        ]

    def write_frame(self, index: int, panes: Sequence[bytes]) -> None:
        """
        Compose and encode one frame.

        :raises ValueError: If the frame arrives out of order, after the last one, or with the wrong pane count
        :raises RuntimeError: If the session is closed or FFmpeg has exited
        """

        if self.closed:
            raise RuntimeError(f"Clip session for run {self.spec.run_id} is closed")
        if index != self.frames_written:
            raise ValueError(f"Frame index {index} arrived where {self.frames_written} was expected")
        if self.frames_written >= self.spec.frame_count:
            raise ValueError(f"Frame index {index} exceeds the declared frame count {self.spec.frame_count}")
        stdin = self._process.stdin
        if stdin is None or self._process.poll() is not None:
            raise RuntimeError(f"FFmpeg exited early with code {self._process.returncode}")
        stdin.write(compose_frame(panes, resolution=self.spec.resolution, widths=self.widths))
        self.frames_written += 1

    def finish(self) -> tuple[VideoProbe, float]:
        """
        Close the stream, wait for FFmpeg, and probe the encoded file.

        :raises RuntimeError: If frames are missing or FFmpeg failed
        """

        if self.frames_written != self.spec.frame_count:
            self.abort()
            raise RuntimeError(f"Clip received {self.frames_written} frames, expected {self.spec.frame_count}")
        stdin = self._process.stdin
        if stdin is not None:
            stdin.close()
        stderr = self._process.stderr.read() if self._process.stderr is not None else b""
        return_code = self._process.wait()
        self.closed = True
        if return_code != 0:
            raise RuntimeError(f"FFmpeg exited with code {return_code}: {stderr.decode(errors='replace')[-2000:]}")
        if not self.video_path.is_file() or self.video_path.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg did not create video '{self.video_path}'")
        return probe_video(self.video_path), time.perf_counter() - self._started

    def abort(self) -> None:
        """Kill FFmpeg and mark the session closed."""

        if self.closed:
            return
        self.closed = True
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()


def probe_video(path: Path) -> VideoProbe:
    """Read the encoded file back through FFmpeg and report what it actually contains."""

    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        metadata = next(reader)
    finally:
        reader.close()
    return VideoProbe(
        codec=str(metadata["codec"]),
        pixel_format=str(metadata["pix_fmt"]),
        resolution=Resolution(width=int(metadata["size"][0]), height=int(metadata["size"][1])),
        fps=float(metadata["fps"]),
        duration_s=float(metadata["duration"]),
    )


__all__ = ["ClipSession", "compose_frame", "ffmpeg_version", "probe_video"]

"""
The full-rate MP4 clip: frames spooled during the run, encoded by the ``compositor`` service after it.

A replay's length is set by its physical end, so its frame count is unknown until it ends, while the compositor
encodes a clip whose frame count is declared up front. The simulator therefore spools every rendered broadcast frame
as a JPEG in a run directory from a background thread (the step loop only hands over the array); after the replay it
opens one clip with the exact count and streams the frames to the service, which encodes them with FFmpeg and returns
its own read-back of the file.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from compositor.protocol import DEFAULT_PORT, SERVICE_NAME, ClipAccepted, ClipReceipt, ClipSpec, HealthReport, Resolution

JPEG_QUALITY: int = 90
SPOOL_QUEUE_FRAMES: int = 64


def compositor_url() -> str:
    """The encoder's endpoint, from the ``COMPOSITOR_URL`` the manifest gives the replay service."""

    return os.environ.get("COMPOSITOR_URL", f"http://{SERVICE_NAME}:{DEFAULT_PORT}")


def encode_jpeg(rgb: NDArray[np.uint8], quality: int = JPEG_QUALITY) -> bytes:
    """
    JPEG bytes of an RGB frame, by Rerun's encoder (the engine image carries no other).

    :raises RuntimeError: If the encoder returned no JPEG
    """

    import rerun as rr

    encoded = rr.Image(np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)).compress(jpeg_quality=quality)
    blob = getattr(encoded, "blob", None)
    if blob is None:
        raise RuntimeError("Rerun declined to JPEG-encode the frame")
    values = blob.as_arrow_array()[0].values.to_numpy(zero_copy_only=False)
    data = bytes(np.asarray(values, dtype=np.uint8).tobytes())
    if data[:2] != b"\xff\xd8":
        raise RuntimeError("the encoded frame is not a JPEG")
    return data


class FrameSpool:
    """
    Spool frames to numbered JPEG files on a background thread.

    :param directory: Run-scoped spool directory (created).
    :param quality: JPEG quality.
    """

    def __init__(self, directory: Path, *, quality: int = JPEG_QUALITY) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.quality = quality
        self.submitted = 0
        self.written = 0
        self.error: str | None = None
        self.size: tuple[int, int] | None = None
        self._queue: queue.Queue[tuple[int, NDArray[np.uint8]] | None] = queue.Queue(maxsize=SPOOL_QUEUE_FRAMES)
        self._thread = threading.Thread(target=self._pump, name="clip-spool", daemon=True)
        self._thread.start()

    def path(self, index: int) -> Path:
        """The spooled file of one frame."""

        return self.directory / f"frame_{index:06d}.jpg"

    def _pump(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            index, frame = item
            try:
                self.path(index).write_bytes(encode_jpeg(frame, self.quality))
                self.written += 1
            except Exception as exc:
                if self.error is None:
                    self.error = f"{type(exc).__name__}: {exc}"

    def submit(self, frame: NDArray[np.uint8]) -> None:
        """Queue one frame (in output order); blocks only when the encoder falls behind."""

        if self.size is None:
            self.size = (int(frame.shape[1]), int(frame.shape[0]))
        self._queue.put((self.submitted, np.ascontiguousarray(frame[:, :, :3])))
        self.submitted += 1

    def close(self) -> None:
        """
        Finish encoding.

        :raises RuntimeError: If any frame failed to encode or write
        """

        self._queue.put(None)
        self._thread.join()
        if self.error is not None:
            raise RuntimeError(f"clip spool failed: {self.error}")
        if self.written != self.submitted:
            raise RuntimeError(f"clip spool wrote {self.written} of {self.submitted} frames")


@dataclass(frozen=True)
class ClipResult:
    """
    The encoded clip.

    :ivar path: The downloaded MP4.
    :ivar receipt: The compositor's receipt (FFmpeg's read-back of the file).
    :ivar stream_s: Wall time to stream and encode, s.
    """

    path: Path
    receipt: ClipReceipt
    stream_s: float


class CompositorClient:
    """
    A connection to a healthy compositor; construction proves it is ready.

    :param endpoint: Base URL (default :func:`compositor_url`).
    :param timeout_s: Per-request timeout.
    """

    def __init__(self, endpoint: str | None = None, *, timeout_s: float = 300.0) -> None:
        import httpx

        self.endpoint = (endpoint or compositor_url()).rstrip("/")
        self._client = httpx.Client(base_url=self.endpoint, timeout=timeout_s)
        try:
            response = self._client.get("/health")
            response.raise_for_status()
            self.health = HealthReport.model_validate(response.json())
            if self.health.service != SERVICE_NAME or not self.health.ready:
                raise RuntimeError(f"compositor at {self.endpoint} is not ready: {self.health.detail}")
        except BaseException:
            self._client.close()
            raise

    def encode(
        self, spool: FrameSpool, *, run_id: str, scenario: str, clip: str, fps: float, destination: Path, cancel_check: Callable[[], None] | None = None
    ) -> ClipResult:
        """
        Encode a closed spool as one single-pane clip and download it.

        :raises RuntimeError: If the spool is empty or the service rejects a frame
        """

        if spool.size is None or spool.written == 0:
            raise RuntimeError("the clip has no frames")
        started = time.perf_counter()
        width, height = spool.size
        spec = ClipSpec(
            run_id=run_id, scenario=scenario, clip=clip, resolution=Resolution(width=width, height=height), fps=fps, pane_count=1, frame_count=spool.written
        )
        response = self._client.post("/clips", json=spec.model_dump(mode="json"))
        response.raise_for_status()
        accepted = ClipAccepted.model_validate(response.json())
        try:
            for index in range(spool.written):
                if cancel_check is not None and index % 100 == 0:
                    cancel_check()
                data = spool.path(index).read_bytes()
                frame = self._client.post(
                    f"/clips/{accepted.encode_id}/frames", data={"index": str(index)}, files=[("panes", (f"{index:06d}.jpg", data, "image/jpeg"))]
                )
                frame.raise_for_status()
            finished = self._client.post(f"/clips/{accepted.encode_id}/finish")
            finished.raise_for_status()
            receipt = ClipReceipt.model_validate(finished.json())
            destination.parent.mkdir(parents=True, exist_ok=True)
            with self._client.stream("GET", f"/clips/{accepted.encode_id}/video") as video, destination.open("wb") as handle:
                video.raise_for_status()
                for block in video.iter_bytes():
                    handle.write(block)
        finally:
            self._client.delete(f"/clips/{accepted.encode_id}")
        if destination.stat().st_size == 0:
            raise RuntimeError("the compositor returned an empty video")
        return ClipResult(path=destination, receipt=receipt, stream_s=time.perf_counter() - started)

    def close(self) -> None:
        """Close the connection."""

        self._client.close()


__all__ = ["ClipResult", "CompositorClient", "FrameSpool", "compositor_url", "encode_jpeg"]

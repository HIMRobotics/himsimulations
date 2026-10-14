"""
Camera video in the run's recording: every rendered frame at 24 fps, as an H.264 ``rr.VideoStream``.

One :class:`CameraStreamSink` per camera. The step loop calls :meth:`CameraStreamSink.push` with each rendered frame
and the simulation time it shows; the call copies the frame into a bounded queue and returns. A worker thread encodes
it (libx264 through PyAV, which both engine images install) and sends it to the recording in column batches, each
frame stamped with its own time, so the viewer plays the stream at the render rate instead of as a slideshow.

The stream is yuv420p High profile, BT.709 limited range signalled in the VUI, no B-frames, and an IDR with SPS/PPS on
the first frame and then every ``gop_seconds``. Rerun's web viewer decodes it with WebCodecs, its native viewer with
FFmpeg. Two 640x480 Orbbec views plus a 1280x720 broadcast view cost about 11-15 MiB per simulated minute.

With ``block_s`` > 0, a full queue makes :meth:`~CameraStreamSink.push` wait up to that long before it drops the
oldest queued frame, so a sink can promise every frame (the replay's telemetry test requires it). Drops and failures
are counted in :class:`StreamStats`, never raised into the run; :meth:`~CameraStreamSink.close` drains the queue.
Close every sink before the scenario body returns: the SDK finalizes the recording after the body.
"""

from __future__ import annotations

import contextlib
import io
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

SIM_TIMELINE = "sim_time"
WALL_TIMELINE = "wall_time"
"""The two timelines the Antioch SDK stamps on every write; frames carry both."""

_STOP = object()
_LOG = logging.getLogger(__name__)


class _ChannelLogger(Protocol):
    """The part of ``antioch.Logger`` used here: the channel prefix."""

    def _path(self, path: str) -> str: ...


def run_clocks() -> tuple[float | None, float]:
    """The active run's ``(sim_s, wall_s)``, the clocks ``antioch.Logger`` stamps; ``(None, 0.0)`` outside a run."""

    import antioch

    try:
        run = antioch.current_scenario_run()
    except Exception:  # StateError outside a run
        return None, 0.0
    return run.sim_s, run.wall_s


def log_at(path: str, archetype: Any, sim_time_s: float) -> bool:
    """
    Log one archetype at an explicit simulation time (Rerun keeps timeline positions per thread).

    :param path: Full entity path.
    :return: Whether a run was active to receive it.
    """

    from antioch.core.scenario import active_recording

    stream = active_recording()
    if stream is None:
        return False
    stream.set_time(SIM_TIMELINE, duration=float(sim_time_s))
    stream.log(path, archetype)
    return True


@dataclass
class StreamStats:
    """What one sink did; every field is cumulative."""

    entity_path: str
    width: int = 0
    height: int = 0
    pushed: int = 0
    dropped: int = 0
    """Frames lost: the oldest queued frame dropped on a full queue, or a frame rejected (not ``HxWx3/4`` uint8)."""
    encoded: int = 0
    logged: int = 0
    keyframes: int = 0
    encoded_bytes: int = 0
    encode_s: float = 0.0
    blocked_s: float = 0.0
    """Time :meth:`CameraStreamSink.push` waited on a full queue."""
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Plain values for results or JSON."""

        return asdict(self)


@dataclass(frozen=True)
class _Sample:
    data: bytes
    keyframe: bool
    sim_s: float
    wall_s: float


def output_size(width: int, height: int, max_width: int | None) -> tuple[int, int]:
    """The encoded size: downscaled to ``max_width`` keeping the aspect, rounded down to even (4:2:0 chroma)."""

    out_w, out_h = float(width), float(height)
    if max_width is not None and width > max_width:
        out_w, out_h = float(max_width), height * max_width / width
    return max(2, int(out_w + 1e-9) // 2 * 2), max(2, int(out_h + 1e-9) // 2 * 2)


class _H264Encoder:
    """libx264 producing one Annex B access unit per input frame."""

    def __init__(self, width: int, height: int, fps: float, *, crf: int, preset: str, gop: int, threads: int) -> None:
        import av

        self.width, self.height = width, height
        self._times: dict[int, tuple[float, float]] = {}
        self._next_pts = 0
        ctx: Any = av.CodecContext.create("libx264", "w")
        ctx.width, ctx.height = width, height
        ctx.pix_fmt = "yuv420p"
        rate = Fraction(fps).limit_denominator(1001)
        ctx.time_base = 1 / rate
        ctx.framerate = rate
        ctx.gop_size = gop
        ctx.max_b_frames = 0
        # BT.709, limited range, signalled in the SPS VUI so every decoder agrees.
        ctx.colorspace, ctx.color_primaries, ctx.color_trc, ctx.color_range = 1, 1, 1, 1
        ctx.thread_count = threads
        ctx.options = {
            "preset": preset,
            "tune": "zerolatency",  # no lookahead and no frame threads: one packet out per frame in
            "profile": "high",
            "crf": str(crf),
            "x264-params": f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes=0:repeat-headers=1",
        }
        ctx.open()
        self._ctx = ctx

    def encode(self, frame: NDArray[np.uint8], sim_s: float, wall_s: float) -> list[_Sample]:
        import av

        source = av.VideoFrame.from_ndarray(frame, format="rgba" if frame.shape[2] == 4 else "rgb24")
        yuv = source.reformat(
            width=self.width,
            height=self.height,
            format="yuv420p",
            src_colorspace="ITU709",
            dst_colorspace="ITU709",
            dst_color_range="MPEG",
            interpolation="BILINEAR",
            threads=1,
        )
        yuv.pts = self._next_pts
        self._times[self._next_pts] = (sim_s, wall_s)
        self._next_pts += 1
        return self._collect(self._ctx.encode(yuv))

    def flush(self) -> list[_Sample]:
        return self._collect(self._ctx.encode(None))

    def _collect(self, packets: Any) -> list[_Sample]:
        out: list[_Sample] = []
        for packet in packets:
            times = None if packet.pts is None else self._times.pop(int(packet.pts), None)
            if times is not None:
                out.append(_Sample(bytes(packet), bool(packet.is_keyframe), times[0], times[1]))
        return out


class CameraStreamSink:
    """
    Encode one camera's frames on a worker thread and log them as an H.264 stream.

    :param logger: The ``antioch.Logger`` whose prefix the channel joins, or ``None`` to use ``channel`` as the path.
    :param channel: Channel under the logger's prefix, e.g. ``cams/broadcast``.
    :param fps: Frame rate; sets the encoder's rate control and keyframe interval.
    :param crf: libx264 constant rate factor; lower is better and bigger.
    :param preset: libx264 preset.
    :param gop_seconds: Keyframe interval, s.
    :param threads: libx264 slice threads.
    :param max_width: Downscale wider frames to this width, keeping the aspect ratio.
    :param queue_frames: Frames buffered between the step loop and the encoder.
    :param block_s: How long :meth:`push` may wait on a full queue before dropping the oldest frame; 0 never waits.
    :param batch_frames: Samples per ``send_columns`` call; defaults to half a second of frames.
    :param max_latency_s: Wall seconds an encoded sample may wait for its batch.
    """

    def __init__(
        self,
        logger: _ChannelLogger | None,
        channel: str,
        fps: float,
        *,
        crf: int = 23,
        preset: str = "veryfast",
        gop_seconds: float = 2.0,
        threads: int = 2,
        max_width: int | None = None,
        queue_frames: int = 8,
        block_s: float = 0.0,
        batch_frames: int | None = None,
        max_latency_s: float = 1.0,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.path = logger._path(channel) if logger is not None else channel
        self.fps = float(fps)
        self._crf, self._preset, self._threads = int(crf), preset, int(threads)
        self._gop = max(1, round(gop_seconds * fps))
        self._max_width = max_width
        self._block_s = max(0.0, float(block_s))
        self._batch_frames = max(1, batch_frames if batch_frames is not None else round(fps / 2))
        self._max_latency_s = float(max_latency_s)
        self._stream: Any = None
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, queue_frames))
        self._closed = False
        self._encoder: _H264Encoder | None = None
        self._failed = False
        self.stats = StreamStats(entity_path=self.path)
        self._thread = threading.Thread(target=self._run, name=f"camera-stream:{self.path}", daemon=True)
        self._thread.start()

    # ---- step-loop side -------------------------------------------------------------------------------------------

    def push(self, frame_rgb: NDArray[np.uint8], sim_time: float | None = None) -> bool:
        """
        Queue one rendered frame.

        :param frame_rgb: ``HxWx3`` (RGB) or ``HxWx4`` (RGBA) uint8.
        :param sim_time: Simulation seconds the frame shows; ``None`` reads the active run's clock.
        :return: Whether the frame was queued.
        """

        if frame_rgb.ndim != 3 or frame_rgb.shape[2] not in (3, 4) or frame_rgb.dtype != np.uint8 or frame_rgb.size == 0:
            self.stats.dropped += 1
            return False
        return self._submit(np.array(frame_rgb, copy=True, order="C"), sim_time)

    def push_jpeg(self, data: bytes, sim_time: float | None = None) -> bool:
        """
        Queue one already-encoded JPEG (a recorded camera frame); the worker decodes it into the stream.

        :param data: JPEG bytes.
        :param sim_time: Simulation seconds the frame shows (its capture time); ``None`` reads the run's clock.
        :return: Whether the frame was queued.
        """

        if not data:
            self.stats.dropped += 1
            return False
        return self._submit(bytes(data), sim_time)

    def _submit(self, item: NDArray[np.uint8] | bytes, sim_time: float | None) -> bool:
        if self._closed:
            return False
        if self._stream is None:
            from antioch.core.scenario import active_recording

            self._stream = active_recording()
        run_sim_s, wall_s = run_clocks()
        sim_s = run_sim_s if sim_time is None else sim_time
        if sim_s is None:
            self.stats.dropped += 1
            return False
        entry = (item, float(sim_s), float(wall_s))
        self.stats.pushed += 1
        if self._block_s > 0.0:
            waited = time.perf_counter()
            try:
                self._queue.put(entry, timeout=self._block_s)
                return True
            except queue.Full:
                pass
            finally:
                self.stats.blocked_s += time.perf_counter() - waited
        while True:
            try:
                self._queue.put_nowait(entry)
                return True
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self._queue.get_nowait()
                    self.stats.dropped += 1

    def close(self, timeout_s: float = 120.0) -> StreamStats:
        """
        Encode and send every queued frame, then stop the worker. Idempotent, bounded by ``timeout_s``; never raises.

        :return: The sink's final statistics (check ``errors``).
        """

        if self._closed:
            return self.stats
        self._closed = True
        deadline = time.perf_counter() + timeout_s
        with contextlib.suppress(queue.Full):
            self._queue.put(_STOP, timeout=max(0.01, deadline - time.perf_counter()))
        self._thread.join(timeout=max(0.0, deadline - time.perf_counter()))
        if self._thread.is_alive():
            self.stats.errors.append(f"worker still running after {timeout_s:.1f} s")
        return self.stats

    # ---- worker side ----------------------------------------------------------------------------------------------

    def _run(self) -> None:
        pending: list[_Sample] = []
        first_pending_at = 0.0
        while True:
            timeout = None if not pending else max(0.0, first_pending_at + self._max_latency_s - time.perf_counter())
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                self._send(pending)
                pending = []
                continue
            if item is _STOP:
                break
            if self._failed:
                continue
            pixels, sim_s, wall_s = item
            started = time.perf_counter()
            try:
                if isinstance(pixels, bytes):
                    from PIL import Image

                    pixels = np.asarray(Image.open(io.BytesIO(pixels)).convert("RGB"))
                if self._encoder is None:
                    width, height = output_size(pixels.shape[1], pixels.shape[0], self._max_width)
                    self._encoder = _H264Encoder(width, height, self.fps, crf=self._crf, preset=self._preset, gop=self._gop, threads=self._threads)
                    self.stats.width, self.stats.height = width, height
                samples = self._encoder.encode(pixels, sim_s, wall_s)
            except Exception as error:  # one camera's failure must not take down the run
                self.stats.errors.append(f"encode: {type(error).__name__}: {error}")
                _LOG.warning("camera stream %s stopped encoding: %s", self.path, error)
                self._failed = True
                continue
            self.stats.encode_s += time.perf_counter() - started
            self.stats.encoded += 1
            if samples and not pending:
                first_pending_at = time.perf_counter()
            pending.extend(samples)
            if len(pending) >= self._batch_frames:
                self._send(pending)
                pending = []
        if self._encoder is not None and not self._failed:
            try:
                pending.extend(self._encoder.flush())
            except Exception as error:
                self.stats.errors.append(f"flush: {type(error).__name__}: {error}")
        self._send(pending)

    def _send(self, samples: list[_Sample]) -> None:
        if not samples:
            return
        if self._stream is None:
            self.stats.errors.append(f"no active recording; {len(samples)} samples not logged")
            return
        import rerun as rr

        blobs = [sample.data for sample in samples]
        try:
            if self.stats.logged == 0:
                rr.log(self.path, rr.VideoStream(codec=rr.VideoCodec.H264), static=True, recording=self._stream)
            indexes = [
                rr.TimeColumn(SIM_TIMELINE, duration=np.fromiter((s.sim_s for s in samples), np.float64, len(samples))),
                rr.TimeColumn(WALL_TIMELINE, duration=np.fromiter((s.wall_s for s in samples), np.float64, len(samples))),
            ]
            columns = rr.VideoStream.columns(sample=blobs, is_keyframe=[s.keyframe for s in samples])
            rr.send_columns(self.path, indexes=indexes, columns=columns, recording=self._stream)
        except Exception as error:  # telemetry is never the run's failure
            self.stats.errors.append(f"send: {type(error).__name__}: {error}")
            return
        self.stats.logged += len(samples)
        self.stats.keyframes += sum(1 for sample in samples if sample.keyframe)
        self.stats.encoded_bytes += sum(len(blob) for blob in blobs)


__all__ = ["SIM_TIMELINE", "WALL_TIMELINE", "CameraStreamSink", "StreamStats", "log_at", "output_size", "run_clocks"]

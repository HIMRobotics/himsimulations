"""
The review-clip encoder service.

The simulator opens a clip, streams its frames in output order as one to four
encoded panes each, and asks for a receipt; the encoder never buffers a whole
clip. Readiness is proven at startup by encoding and probing a two-frame,
two-pane file, so a session that reports healthy really can produce a
playable MP4.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from compositor.encoder import ClipSession, ffmpeg_version, probe_video
from compositor.protocol import DEFAULT_PORT, SERVICE_NAME, ClipAccepted, ClipReceipt, ClipSpec, HealthReport, Resolution

LOGGER = logging.getLogger(SERVICE_NAME)

_SESSIONS: dict[str, ClipSession] = {}
_HEALTH = {"ready": False, "detail": "warming up", "ffmpeg": "", "pillow": ""}
_WORK_ROOT = Path(os.environ.get("COMPOSITOR_WORK_DIR", tempfile.gettempdir())) / "compositor-clips"


def _encoded(size: tuple[int, int], color: tuple[int, int, int], image_format: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=image_format)
    return buffer.getvalue()


def warm_up() -> None:
    """Encode and probe a two-frame, two-pane clip so readiness means FFmpeg and Pillow work."""

    from PIL import __version__ as pillow_version

    _HEALTH["pillow"] = str(pillow_version)
    _HEALTH["ffmpeg"] = ffmpeg_version()
    _WORK_ROOT.mkdir(parents=True, exist_ok=True)
    warm = _WORK_ROOT / "warmup"
    shutil.rmtree(warm, ignore_errors=True)
    warm.mkdir(parents=True)
    spec = ClipSpec(run_id="warmup", scenario="warmup", clip="warmup", resolution=Resolution(width=96, height=48), fps=30.0, pane_count=2, frame_count=2)
    session = ClipSession(spec, warm / "warmup.mp4")
    panes = (_encoded((64, 48), (200, 20, 20), "JPEG"), _encoded((48, 64), (20, 20, 200), "PNG"))
    for index in range(spec.frame_count):
        session.write_frame(index, panes)
    probe, _seconds = session.finish()
    shutil.rmtree(warm, ignore_errors=True)
    _HEALTH["ready"] = True
    _HEALTH["detail"] = f"warm-up encoded {probe.resolution} {probe.codec} {probe.pixel_format}"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Prove the encoder works before the service reports healthy."""

    try:
        warm_up()
        LOGGER.info("compositor ready: %s", _HEALTH["detail"])
    except Exception as exc:  # readiness must report the reason, not crash the process
        _HEALTH["ready"] = False
        _HEALTH["detail"] = f"warm-up failed: {exc}"
        LOGGER.exception("compositor warm-up failed")
    yield
    for session in list(_SESSIONS.values()):
        session.abort()
    _SESSIONS.clear()
    shutil.rmtree(_WORK_ROOT, ignore_errors=True)


app = FastAPI(title="Table-tennis review-clip compositor", lifespan=lifespan)


@app.get("/health")
def health() -> HealthReport:
    """Report readiness proven by the startup encode."""

    return HealthReport(
        ready=bool(_HEALTH["ready"]),
        ffmpeg_version=str(_HEALTH["ffmpeg"]),
        pillow_version=str(_HEALTH["pillow"]),
        active_encodes=len(_SESSIONS),
        detail=str(_HEALTH["detail"]),
    )


@app.post("/clips", status_code=201)
def open_clip(spec: ClipSpec) -> ClipAccepted:
    """Open one clip encode and report the pane slots the service will use."""

    if not _HEALTH["ready"]:
        raise HTTPException(status_code=503, detail=str(_HEALTH["detail"]))
    encode_id = uuid.uuid4().hex
    session = ClipSession(spec, _WORK_ROOT / encode_id / f"{spec.clip}.mp4")
    _SESSIONS[encode_id] = session
    LOGGER.info("opened clip %s for run %s: %d frames of %d panes at %s", encode_id, spec.run_id, spec.frame_count, spec.pane_count, spec.resolution)
    return ClipAccepted(encode_id=encode_id, run_id=spec.run_id, pane_widths=session.widths)


def _session(encode_id: str) -> ClipSession:
    session = _SESSIONS.get(encode_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown clip session {encode_id}")
    return session


@app.post("/clips/{encode_id}/frames", status_code=204)
async def add_frame(encode_id: str, index: int = Form(...), panes: list[UploadFile] = File(...)) -> None:
    """Compose and encode one frame; frames must arrive in output order with the declared pane count."""

    session = _session(encode_id)
    if len(panes) != session.spec.pane_count:
        raise HTTPException(status_code=422, detail=f"Frame {index} carries {len(panes)} panes, the clip declared {session.spec.pane_count}")
    encoded = [await pane.read() for pane in panes]
    try:
        session.write_frame(index, encoded)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/clips/{encode_id}/finish")
def finish_clip(encode_id: str) -> ClipReceipt:
    """Close the stream and report FFmpeg's read-back of the encoded file."""

    session = _session(encode_id)
    try:
        probe, seconds = session.finish()
    except RuntimeError as exc:
        _SESSIONS.pop(encode_id, None)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ClipReceipt(
        encode_id=encode_id,
        run_id=session.spec.run_id,
        frames_written=session.frames_written,
        video_bytes=session.video_path.stat().st_size,
        probe=probe,
        ffmpeg_seconds=seconds,
    )


@app.get("/clips/{encode_id}/video")
def download_video(encode_id: str) -> FileResponse:
    """Return the encoded MP4 for the simulator to store as a run artifact."""

    session = _session(encode_id)
    if not session.closed or not session.video_path.is_file():
        raise HTTPException(status_code=409, detail=f"Clip {encode_id} is not finished")
    return FileResponse(session.video_path, media_type="video/mp4", filename=session.video_path.name)


@app.get("/clips/{encode_id}/probe")
def reprobe(encode_id: str) -> ClipReceipt:
    """Re-read the encoded file from disk; used to prove the stored artifact still decodes."""

    session = _session(encode_id)
    if not session.closed or not session.video_path.is_file():
        raise HTTPException(status_code=409, detail=f"Clip {encode_id} is not finished")
    return ClipReceipt(
        encode_id=encode_id,
        run_id=session.spec.run_id,
        frames_written=session.frames_written,
        video_bytes=session.video_path.stat().st_size,
        probe=probe_video(session.video_path),
        ffmpeg_seconds=0.0,
    )


@app.delete("/clips/{encode_id}", status_code=204)
def close_clip(encode_id: str) -> None:
    """Release one session and its working files."""

    session = _SESSIONS.pop(encode_id, None)
    if session is None:
        return
    session.abort()
    shutil.rmtree(session.video_path.parent, ignore_errors=True)


def main() -> None:
    """Serve the encoder."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("COMPOSITOR_PORT", DEFAULT_PORT)), log_level="info")


if __name__ == "__main__":
    main()

"""
The wire contract between the game harness and the ``compositor`` service.

This module is pure: pydantic models, the pane geometry both sides must agree
on, and nothing else. The simulator imports only this module from the service
package, so the encoder's FFmpeg and Pillow dependencies never enter an
engine image.

A review clip is a horizontal strip of one to four equal-width panes per
frame (for example the presentation camera, a virtual Orbbec view, and the
matching real Orbbec frame). Every pane is letterboxed into its slot, never
cropped, so the ball cannot be cut out of a review frame.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SERVICE_NAME: str = "compositor"
SCHEMA_VERSION: int = 1
DEFAULT_PORT: int = 8771
MAX_PANES: int = 4

PANE_BACKGROUND: tuple[int, int, int] = (16, 18, 22)
"""Letterbox colour behind a pane image that does not fill its slot."""

VIDEO_CODEC: str = "h264"
VIDEO_PIXEL_FORMAT_PREFIX: str = "yuv420p"
H264_PRESET: str = "medium"
H264_CRF: int = 18
H264_PROFILE: str = "high"


class Resolution(BaseModel):
    """
    A positive pixel resolution.

    :ivar width: Width in pixels.
    :ivar height: Height in pixels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @classmethod
    def parse(cls, value: str) -> Resolution:
        """
        Parse ``WIDTHxHEIGHT``.

        :raises ValueError: If the value is malformed
        """

        try:
            width_text, height_text = value.lower().split("x", maxsplit=1)
            return cls(width=int(width_text), height=int(height_text))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Resolution '{value}' must use WIDTHxHEIGHT") from exc

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


def pane_widths(width: int, pane_count: int) -> tuple[int, ...]:
    """
    Split a clip width into equal pane slots, giving any remainder to the leftmost panes.

    Both sides compute the split from the same numbers so a pane is never
    rescaled twice.

    :param width: Clip width in pixels.
    :param pane_count: Number of panes, one to :data:`MAX_PANES`.
    :return: Slot widths from left to right; they sum to ``width``.
    :raises ValueError: If the count is out of range or a slot would have no pixels
    """

    if not 1 <= pane_count <= MAX_PANES:
        raise ValueError(f"Pane count {pane_count} must be between 1 and {MAX_PANES}")
    base, remainder = divmod(width, pane_count)
    if base <= 0:
        raise ValueError(f"Width {width} leaves no pixels for {pane_count} panes")
    return tuple(base + (1 if index < remainder else 0) for index in range(pane_count))


class ClipSpec(BaseModel):
    """
    One review-clip encode request.

    :ivar schema_version: Wire schema version.
    :ivar run_id: The simulator's run identity, echoed in every reply.
    :ivar scenario: Scenario name that produced the frames.
    :ivar clip: Clip name inside the run, for example ``rally-03``.
    :ivar resolution: Clip output resolution.
    :ivar fps: Output frames per simulated second.
    :ivar pane_count: Panes per frame, laid out left to right.
    :ivar frame_count: Number of frames the simulator will send.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    clip: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    resolution: Resolution
    fps: float = Field(gt=0.0, le=120.0)
    pane_count: int = Field(ge=1, le=MAX_PANES)
    frame_count: int = Field(gt=0)

    @property
    def expected_duration_s(self) -> float:
        """Duration FFmpeg must report for a complete encode."""

        return self.frame_count / self.fps


class ClipAccepted(BaseModel):
    """
    The opened encode session.

    :ivar encode_id: Session identity used by the frame, finish, and video routes.
    :ivar run_id: The run identity from the request.
    :ivar pane_widths: Slot widths the service will letterbox each pane into.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    encode_id: str
    run_id: str
    pane_widths: tuple[int, ...]


class VideoProbe(BaseModel):
    """
    Read-back metadata for the encoded MP4, measured by FFmpeg on the file.

    :ivar codec: Decoder name reported for the video stream.
    :ivar pixel_format: Pixel format reported for the video stream.
    :ivar resolution: Decoded frame size.
    :ivar fps: Decoded frame rate.
    :ivar duration_s: Decoded duration in seconds.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    codec: str
    pixel_format: str
    resolution: Resolution
    fps: float = Field(gt=0.0)
    duration_s: float = Field(gt=0.0)


class ClipReceipt(BaseModel):
    """
    The finished encode with FFmpeg's read-back of the file.

    :ivar encode_id: Session identity.
    :ivar run_id: The run identity from the request.
    :ivar frames_written: Frames actually composed into the stream.
    :ivar video_bytes: Size of the encoded file.
    :ivar probe: FFmpeg's read-back of the encoded file.
    :ivar ffmpeg_seconds: Wall-clock seconds the encoder spent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    encode_id: str
    run_id: str
    frames_written: int = Field(ge=0)
    video_bytes: int = Field(ge=0)
    probe: VideoProbe
    ffmpeg_seconds: float = Field(ge=0.0)


class HealthReport(BaseModel):
    """
    Readiness of the encoder, proven by a warm-up encode at startup.

    :ivar ready: True only after the warm-up encode produced a playable file.
    :ivar service: Service name, checked by the health probe.
    :ivar schema_version: Wire schema version the service speaks.
    :ivar ffmpeg_version: FFmpeg build reported by ``imageio-ffmpeg``.
    :ivar pillow_version: Installed Pillow version.
    :ivar active_encodes: Currently open encode sessions.
    :ivar detail: Human-readable readiness detail or failure reason.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ready: bool
    service: str = SERVICE_NAME
    schema_version: int = SCHEMA_VERSION
    ffmpeg_version: str = ""
    pillow_version: str = ""
    active_encodes: int = 0
    detail: str = ""


__all__ = [
    "DEFAULT_PORT",
    "H264_CRF",
    "H264_PRESET",
    "H264_PROFILE",
    "MAX_PANES",
    "PANE_BACKGROUND",
    "SCHEMA_VERSION",
    "SERVICE_NAME",
    "VIDEO_CODEC",
    "VIDEO_PIXEL_FORMAT_PREFIX",
    "ClipAccepted",
    "ClipReceipt",
    "ClipSpec",
    "HealthReport",
    "Resolution",
    "VideoProbe",
    "pane_widths",
]

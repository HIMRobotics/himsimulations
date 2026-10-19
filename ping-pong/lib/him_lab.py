"""
The HIM lab's digital twin: the room, table and net, the calibrated lighting, the physics colliders and the cameras.

Two ways to put it on a stage:

- :func:`build_him_lab`: the replay's lab (Isaac Sim). The room, table and net load in the **table frame** (origin at
  the table-top centre, ``z = 0`` on the playing surface, +X toward the 8E camera's half, +Y away from the cameras),
  with invisible colliders authored from :mod:`lib.physics` and the calibrated ceiling lighting.
- :func:`author_him_twin`: a render-only room around a table at any world position (an Isaac Lab robot env), with
  the same lighting and the ``cine_wide`` camera; it refuses to author anything that carries physics.

The cameras: the two virtual overhead Orbbecs (8E and 8F) at their calibrated poses and lenses
(:func:`author_orbbec_camera`), the ``broadcast`` view and the ``cine_wide`` side view. Every prim this module authors
lives under :data:`STAGE_ROOT`, so :func:`clear_stage` resets the scene.

The lab assets are Blender reconstructions of HIM's lab with no physics schemas; contact geometry comes only from
:mod:`lib.physics`, the single source of physical truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lib.assets import ASSETS
from lib.physics.engine.colliders import ROOM_BOXES, TABLE_BOARD_THICKNESS_M
from lib.physics.params import BallTableParams, ContactMaterial
from lib.physics.reference.projection import CameraIntrinsics, PinholeCamera

if TYPE_CHECKING:
    from pxr import Usd

DATA = Path(__file__).resolve().parent / "data"

# ---- the stage and the lab assets -----------------------------------------------------------------------------------

STAGE_ROOT: str = "/World/TableTennis"
"""Every prim the project authors lives below this root, so clearing it resets the scene."""

ASSET_SURFACE_HEIGHT_M: float = 0.76
"""Playing-surface height above the floor in the room, table and net assets (ITTF); equals ``lib.physics``' table height."""

LAB_ROOT: str = f"{STAGE_ROOT}/Lab"
"""Parent of the room, table and net, lowered so the playing surface is at ``z = 0``."""

LAB_ASSETS: dict[str, tuple[str, str]] = {
    "lab_room": ("ping-pong/him-lab-room", "/tt_lab_room"),
    "table": ("ping-pong/table", "/tt_table"),
    "net": ("ping-pong/net", "/tt_net"),
    "paddle": ("ping-pong/paddle", "/tt_paddle_red"),
    "ball": ("ping-pong/ball", "/tt_ball"),
}
"""Each lab role's store asset and its default prim (:data:`lib.assets.ASSETS` pins the version)."""

LAB_PRIMS: dict[str, str] = {"lab_room": f"{LAB_ROOT}/room", "table": f"{LAB_ROOT}/table", "net": f"{LAB_ROOT}/net"}


def load_lab_asset(role: str, prim_path: str) -> str:
    """
    Reference one lab asset at ``prim_path``, at its pinned version.

    :param role: A key of :data:`LAB_ASSETS`.
    :return: ``name@version``.
    :raises RuntimeError: If the prim did not load
    """

    import antioch

    asset = ASSETS[LAB_ASSETS[role][0]]
    antioch.load_asset(asset.name, prim_path=prim_path, version=asset.version)
    if not antioch.stage().GetPrimAtPath(prim_path).IsValid():
        raise RuntimeError(f"asset {asset.name}@{asset.version} did not load at {prim_path}")
    return f"{asset.name}@{asset.version}"


def clear_stage(stage: Usd.Stage) -> None:
    """
    Remove :data:`STAGE_ROOT` from every layer so the next case starts clean.

    :raises RuntimeError: If the root survives removal
    """

    from pxr import Usd

    for layer in stage.GetLayerStack():
        with Usd.EditContext(stage, layer):
            if stage.GetPrimAtPath(STAGE_ROOT):
                stage.RemovePrim(STAGE_ROOT)
    if stage.GetPrimAtPath(STAGE_ROOT):
        raise RuntimeError(f"Stage root {STAGE_ROOT} survived clearing")


# ---- lighting -------------------------------------------------------------------------------------------------------

ToneMap = Literal["aces", "linear"]

TONEMAP_OPERATORS: dict[str, int] = {"aces": 6, "linear": 1}
"""RTX ``/rtx/post/tonemap/op`` value per operator (Isaac Sim 6.0.1); ACES is the shipped default."""

DEFAULT_LIGHT_PRIM_PATH: str = "/Environment/defaultLight"


class LabLighting(BaseModel):
    """
    One lighting of the lab: every ceiling LED panel's RectLight and the renderer's tone mapping.

    :ivar panel_intensity: Intensity of every panel (RTX units, ``normalize`` as authored).
    :ivar colour_temperature_k: Panel colour temperature, K; the panel colour is white so this alone tints it.
    :ivar tonemap: RTX tone-mapping operator.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    panel_intensity: float = Field(gt=0.0)
    colour_temperature_k: float = Field(ge=1500.0, le=12000.0)
    tonemap: ToneMap = "aces"


CALIBRATED_LIGHTING = LabLighting(panel_intensity=8960.0, colour_temperature_k=5500.0, tonemap="linear")
"""
The lab's light, calibrated against the recorded Orbbec frames: lighting sweeps (4500-7500 K, ACES and linear tone
mapping) rendered the virtual 8E and 8F views and compared table and floor colour (CIEDE2000, mean 13.0 here), frame
mean and glare position with the recordings. The linear tone mapper keeps the render proportional to radiance below
clipping, which the cameras' diagonal ISP model (:func:`isp_luts`) assumes.
"""


def apply_lab_lighting(stage: Usd.Stage, room_prim_path: str, lighting: LabLighting = CALIBRATED_LIGHTING) -> tuple[str, ...]:
    """
    Re-light every ceiling panel of the loaded room in RTX units and set the tone mapper; safe to call again.

    :param room_prim_path: The loaded room prim.
    :return: The re-lit RectLight prims.
    :raises RuntimeError: If the room holds no RectLight
    """

    import carb.settings
    from pxr import Gf, Usd, UsdLux

    panels = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(room_prim_path)) if prim.IsA(UsdLux.RectLight)]
    if not panels:
        raise RuntimeError(f"no RectLight below {room_prim_path}; the room asset carries one per ceiling panel")
    for prim in panels:
        light = UsdLux.RectLight(prim)
        light.GetIntensityAttr().Set(float(lighting.panel_intensity))
        light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        light.CreateEnableColorTemperatureAttr().Set(True)
        light.CreateColorTemperatureAttr().Set(float(lighting.colour_temperature_k))
    default_light = stage.GetPrimAtPath(DEFAULT_LIGHT_PRIM_PATH)
    if default_light and default_light.IsValid():
        default_light.SetActive(False)
    carb.settings.get_settings().set("/rtx/post/tonemap/op", TONEMAP_OPERATORS[lighting.tonemap])
    return tuple(str(prim.GetPath()) for prim in panels)


# ---- colliders ------------------------------------------------------------------------------------------------------
#
# The render meshes never collide. Contact geometry is authored from lib.physics as invisible primitives in the table
# frame, each binding one physics material whose restitution and friction are the lib.physics ball-pair values:
#
# - the table top: a box whose upper face is the playing surface z = 0;
# - the net: a thin box on x = 0, a steel post just outside each end and a clamp box each side;
# - the floor: a plane at z = -table.height;
# - the room: its walls, ceiling and solid furniture as boxes (the `wall` material);
# - each ball: a sphere of the ball's radius under its prim.
#
# Nothing here authors a physics scene, a solver or a rigid body: the engine adapter (lib.physics.engine) owns those,
# reading the same parameters. With the hybrid contacts the game ball is filtered from the table, net, floor and room
# colliders and meets them through the adapter's sweeps; retired balls keep the native colliders.

COLLIDER_ROOT: str = f"{STAGE_ROOT}/Colliders"
MATERIAL_ROOT: str = f"{STAGE_ROOT}/PhysicsMaterials"
ROOM_COLLIDER_ROOT: str = f"{COLLIDER_ROOT}/room"
NET_FIXTURE_COLLIDERS: tuple[str, ...] = tuple(f"{COLLIDER_ROOT}/net_{part}_{side}" for part in ("post", "clamp") for side in ("neg_y", "pos_y"))
"""The net posts' and clamps' native colliders (the hybrid net sweeps them too)."""
FLOOR_COLLIDER: str = f"{COLLIDER_ROOT}/floor"
ROOM_COLLIDERS: tuple[str, ...] = tuple(f"{ROOM_COLLIDER_ROOT}/{box.name}" for box in ROOM_BOXES)
"""The room's native colliders (the hybrid floor sweeps the floor and the room for the game ball)."""

TABLE_TOP_THICKNESS_M: float = TABLE_BOARD_THICKNESS_M
NET_POST_RADIUS_M: float = 0.0075
NET_POST_OUTSET_M: float = 0.0045
"""Post axis distance beyond the end of the net, as built in the net asset."""
NET_POST_BOTTOM_M: float = -0.03
NET_POST_TOP_M: float = 0.1585
"""Post span in the table frame, from the clamp to the cap above the net tape."""
NET_CLAMP_HALF_X_M: float = 0.016
NET_CLAMP_Y_M: tuple[float, float] = (0.7275, 0.927)
NET_CLAMP_Z_M: tuple[float, float] = (-0.089, 0.016)
"""Each net clamp's box in the table frame: it grips the table's side edge, 16 mm above the playing surface over the last
35 mm inside the side line, and reaches 89 mm below it."""

SurfaceName = Literal["table", "net", "floor", "rubber", "wall", "fixture", "post"]
ShapeKind = Literal["box", "cylinder", "plane", "sphere"]


class ColliderShape(BaseModel):
    """
    One authored collision primitive.

    :ivar prim_path: The ``UsdGeom`` primitive carrying ``CollisionAPI``.
    :ivar kind: Primitive type.
    :ivar centre: Centre in its parent's frame, m.
    :ivar size: Box edge lengths; cylinder ``(radius, radius, height)``; sphere ``(radius,) * 3``; plane zeros.
    :ivar surface: The physics material it binds; ``None`` for the ball, whose pair materials the adapter sets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prim_path: str
    kind: ShapeKind
    centre: tuple[float, float, float]
    size: tuple[float, float, float]
    surface: SurfaceName | None


class SurfaceMaterial(BaseModel):
    """
    One physics material and the ``lib.physics`` pair it carries.

    :ivar prim_path: The ``UsdShade.Material`` with ``PhysicsMaterialAPI``.
    :ivar restitution: Ball-pair coefficient of restitution.
    :ivar friction: Ball-pair Coulomb friction, used as both static and dynamic friction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prim_path: str
    restitution: float
    friction: float


class ColliderFacts(BaseModel):
    """
    What the collider functions authored.

    :ivar shapes: Every collision primitive.
    :ivar materials: Every physics material, by surface.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    shapes: tuple[ColliderShape, ...]
    materials: dict[str, SurfaceMaterial]


def surface_contacts(params: BallTableParams) -> dict[SurfaceName, ContactMaterial]:
    """The ``lib.physics`` ball-pair law each surface's material carries."""

    return {
        "table": params.table_contact,
        "net": params.net_contact,
        "floor": params.floor_contact,
        "rubber": params.rubber_contact,
        "fixture": params.fixture_contact,
        "post": params.post_contact,
    }


def static_collider_shapes(params: BallTableParams) -> tuple[ColliderShape, ...]:
    """
    The static contact geometry in the table frame, without touching USD.

    :return: Table top, net, net posts, net clamps, floor.
    """

    table, net = params.table, params.net
    half_span = params.net_half_span
    post_y = half_span + NET_POST_OUTSET_M
    post_height = NET_POST_TOP_M - NET_POST_BOTTOM_M
    post_z = 0.5 * (NET_POST_TOP_M + NET_POST_BOTTOM_M)
    return (
        ColliderShape(
            prim_path=f"{COLLIDER_ROOT}/table_top",
            kind="box",
            centre=(0.0, 0.0, -0.5 * TABLE_TOP_THICKNESS_M),
            size=(table.length, table.width, TABLE_TOP_THICKNESS_M),
            surface="table",
        ),
        ColliderShape(
            prim_path=f"{COLLIDER_ROOT}/net", kind="box", centre=(0.0, 0.0, 0.5 * net.height), size=(net.thickness, 2.0 * half_span, net.height), surface="net"
        ),
        ColliderShape(
            prim_path=f"{COLLIDER_ROOT}/net_post_neg_y",
            kind="cylinder",
            centre=(0.0, -post_y, post_z),
            size=(NET_POST_RADIUS_M, NET_POST_RADIUS_M, post_height),
            surface="post",
        ),
        ColliderShape(
            prim_path=f"{COLLIDER_ROOT}/net_post_pos_y",
            kind="cylinder",
            centre=(0.0, post_y, post_z),
            size=(NET_POST_RADIUS_M, NET_POST_RADIUS_M, post_height),
            surface="post",
        ),
        *(
            ColliderShape(
                prim_path=f"{COLLIDER_ROOT}/net_clamp_{name}",
                kind="box",
                centre=(0.0, sign * 0.5 * (NET_CLAMP_Y_M[0] + NET_CLAMP_Y_M[1]), 0.5 * (NET_CLAMP_Z_M[0] + NET_CLAMP_Z_M[1])),
                size=(2.0 * NET_CLAMP_HALF_X_M, NET_CLAMP_Y_M[1] - NET_CLAMP_Y_M[0], NET_CLAMP_Z_M[1] - NET_CLAMP_Z_M[0]),
                surface="fixture",
            )
            for name, sign in (("neg_y", -1.0), ("pos_y", 1.0))
        ),
        ColliderShape(prim_path=f"{COLLIDER_ROOT}/floor", kind="plane", centre=(0.0, 0.0, -table.height), size=(0.0, 0.0, 0.0), surface="floor"),
    )


def room_collider_shapes() -> tuple[ColliderShape, ...]:
    """The room's walls, ceiling and solid furniture as static boxes (the ``wall`` material), table frame."""

    return tuple(
        ColliderShape(prim_path=f"{ROOM_COLLIDER_ROOT}/{box.name}", kind="box", centre=box.centre, size=box.size, surface="wall") for box in ROOM_BOXES
    )


def ball_shape(ball_prim_path: str, params: BallTableParams) -> ColliderShape:
    """The ball sphere in the ball asset's frame (origin at the centre)."""

    radius = params.ball.radius
    return ColliderShape(prim_path=f"{ball_prim_path}/collider", kind="sphere", centre=(0.0, 0.0, 0.0), size=(radius, radius, radius), surface=None)


def _define_material(stage: Usd.Stage, path: str, contact: ContactMaterial) -> SurfaceMaterial:
    from pxr import UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    if api is None:
        raise RuntimeError(f"could not apply PhysicsMaterialAPI to {path}")
    api.CreateRestitutionAttr().Set(float(contact.restitution))
    api.CreateStaticFrictionAttr().Set(float(contact.friction))
    api.CreateDynamicFrictionAttr().Set(float(contact.friction))
    return SurfaceMaterial(prim_path=path, restitution=float(contact.restitution), friction=float(contact.friction))


def _author_shape(stage: Usd.Stage, shape: ColliderShape, material_path: str | None) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    if shape.kind == "box":
        geom = UsdGeom.Cube.Define(stage, shape.prim_path)
        geom.CreateSizeAttr(1.0)
        geom.AddTranslateOp().Set(Gf.Vec3d(*shape.centre))
        geom.AddScaleOp().Set(Gf.Vec3f(*shape.size))
    elif shape.kind == "cylinder":
        geom = UsdGeom.Cylinder.Define(stage, shape.prim_path)
        geom.CreateAxisAttr(UsdGeom.Tokens.z)
        geom.CreateRadiusAttr(shape.size[0])
        geom.CreateHeightAttr(shape.size[2])
        geom.AddTranslateOp().Set(Gf.Vec3d(*shape.centre))
    elif shape.kind == "sphere":
        geom = UsdGeom.Sphere.Define(stage, shape.prim_path)
        geom.CreateRadiusAttr(shape.size[0])
        geom.AddTranslateOp().Set(Gf.Vec3d(*shape.centre))
    else:
        geom = UsdGeom.Plane.Define(stage, shape.prim_path)
        geom.CreateAxisAttr(UsdGeom.Tokens.z)
        geom.CreateWidthAttr(40.0)
        geom.CreateLengthAttr(40.0)
        geom.AddTranslateOp().Set(Gf.Vec3d(*shape.centre))
    geom.CreatePurposeAttr(UsdGeom.Tokens.guide)
    geom.MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    if material_path is not None:
        binding = UsdShade.MaterialBindingAPI.Apply(geom.GetPrim())
        binding.Bind(UsdShade.Material(stage.GetPrimAtPath(material_path)), UsdShade.Tokens.weakerThanDescendants, "physics")


def author_lab_colliders(stage: Usd.Stage, params: BallTableParams) -> ColliderFacts:
    """
    Author the physics materials and the static colliders: table top, net, posts, clamps, floor, and the room.

    :param stage: The active stage, stopped, with :data:`STAGE_ROOT` defined.
    :param params: The physics the engine runs.
    :return: What was authored.
    """

    from pxr import UsdGeom

    materials = {surface: _define_material(stage, f"{MATERIAL_ROOT}/{surface}", contact) for surface, contact in surface_contacts(params).items()}
    UsdGeom.Scope.Define(stage, COLLIDER_ROOT)
    shapes = static_collider_shapes(params)
    for shape in shapes:
        _author_shape(stage, shape, None if shape.surface is None else materials[shape.surface].prim_path)
    materials["wall"] = _define_material(stage, f"{MATERIAL_ROOT}/wall", params.wall_contact)
    UsdGeom.Scope.Define(stage, ROOM_COLLIDER_ROOT)
    room = room_collider_shapes()
    for shape in room:
        _author_shape(stage, shape, materials["wall"].prim_path)
    return ColliderFacts(shapes=(*shapes, *room), materials=materials)


def author_ball_colliders(stage: Usd.Stage, params: BallTableParams, balls: tuple[str, ...]) -> tuple[ColliderShape, ...]:
    """Add a sphere collider under each loaded ball prim (ball asset frame)."""

    shapes = tuple(ball_shape(path, params) for path in balls)
    for shape in shapes:
        _author_shape(stage, shape, None)
    return shapes


# ---- cameras --------------------------------------------------------------------------------------------------------
#
# Each real overhead Orbbec (serials CP0E8530008E and CP0E8530008F) has one calibration per extrinsic epoch: OpenCV
# intrinsics and distortion, and a world-to-camera pose in the table frame (x_cam = R x_world + t). A USD camera looks
# down its local -Z with +Y up and the OpenCV camera down +Z with +Y down, so the prim's rotation is the OpenCV
# camera-to-world rotation followed by the mount diag(1, -1, -1). RTX's OmniLensDistortionOpenCvPinholeAPI renders
# fx, fy, cx, cy and the distortion exactly as OpenCV projects them. The principal point is shifted by half a pixel:
# OpenCV puts pixel centres on integer coordinates, a rendered pixel (i, j) covers [i, i + 1). Rendered ArUco stickers
# recover each pose with OpenCV's PnP to within 1 cm and 0.2 degrees.

CAMERA_ROOT: str = f"{STAGE_ROOT}/Cameras"
SERIALS: dict[str, str] = {"8E": "CP0E8530008E", "8F": "CP0E8530008F"}
"""Each overhead camera's short label and device serial."""
USD_FOCAL_LENGTH: float = 0.24
"""Raw USD focal length; only its ratio to the apertures sets the projection."""
CLIPPING_RANGE_M: tuple[float, float] = (0.05, 50.0)
USD_FROM_OPENCV: NDArray[np.float64] = np.diag([1.0, -1.0, -1.0])
PIXEL_CENTRE_OFFSET: float = 0.5
"""Rendered principal point minus the OpenCV one, px, on both axes."""
OPENCV_LENS_SCHEMA: str = "OmniLensDistortionOpenCvPinholeAPI"
OPENCV_LENS_PREFIX: str = "omni:lensdistortion:opencvPinhole"


class OrbbecCalibration(BaseModel):
    """
    One overhead camera's calibration for one extrinsic epoch.

    :ivar label: ``8E`` or ``8F``.
    :ivar serial: Device serial.
    :ivar epoch: Extrinsic epoch label (the camera moved between epochs).
    :ivar sessions: Recorded sessions (by camera key) with at least one segment in this epoch.
    :ivar intrinsics: Intrinsics and OpenCV distortion at the colour resolution.
    :ivar rotation: World-to-camera rotation ``R``.
    :ivar translation: World-to-camera translation ``t``, m.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    serial: str
    epoch: str
    sessions: tuple[str, ...]
    intrinsics: CameraIntrinsics
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    translation: tuple[float, float, float]

    @model_validator(mode="after")
    def _known_serial_and_proper_rotation(self) -> OrbbecCalibration:
        if SERIALS.get(self.label) != self.serial:
            raise ValueError(f"camera label {self.label!r} does not name serial {self.serial!r}; expected one of {SERIALS}")
        rot = np.asarray(self.rotation, dtype=float)
        if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-6) or np.linalg.det(rot) < 0.0:
            raise ValueError(f"camera {self.label} epoch {self.epoch}: rotation is not a proper rotation matrix")
        return self

    @property
    def camera(self) -> PinholeCamera:
        """The calibrated camera as the reference model's pinhole."""

        return PinholeCamera(self.intrinsics, np.asarray(self.rotation, dtype=float), np.asarray(self.translation, dtype=float), name=self.serial)


class OrbbecCalibrationSet(BaseModel):
    """
    Every calibrated camera epoch (``lib/data/orbbec_cameras.json``).

    :ivar schema_version: File schema version.
    :ivar frame: The world frame, in words.
    :ivar source: Provenance of the values.
    :ivar cameras: One entry per serial and epoch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    frame: str
    source: str
    cameras: tuple[OrbbecCalibration, ...] = Field(min_length=1)

    def select(self, label: str, *, session: str | None = None, epoch: str | None = None) -> OrbbecCalibration:
        """
        One camera's calibration, by session or epoch (``epoch`` wins).

        :raises KeyError: If no calibration, or more than one, matches
        """

        matches = [
            cal for cal in self.cameras if cal.label == label and (cal.epoch == epoch if epoch is not None else session is None or session in cal.sessions)
        ]
        if len(matches) != 1:
            available = sorted(f"{cal.label}/{cal.epoch}" for cal in self.cameras)
            raise KeyError(f"{len(matches)} calibrations match camera {label} (session={session}, epoch={epoch}); available: {available}")
        return matches[0]

    def latest(self, label: str) -> OrbbecCalibration:
        """A camera's last epoch."""

        return self.select(label, epoch=max(cal.epoch for cal in self.cameras if cal.label == label))


def load_calibration_set(path: Path | str = DATA / "orbbec_cameras.json") -> OrbbecCalibrationSet:
    """Read and validate the Orbbec calibrations."""

    return OrbbecCalibrationSet.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class UsdLens(BaseModel):
    """
    Raw ``UsdGeom.Camera`` optics for a calibration.

    :ivar focal_length: ``focalLength``.
    :ivar horizontal_aperture: ``horizontalAperture``.
    :ivar vertical_aperture: ``verticalAperture``.
    :ivar horizontal_aperture_offset: Moves the principal point along u.
    :ivar vertical_aperture_offset: Moves the principal point along v.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float
    horizontal_aperture_offset: float
    vertical_aperture_offset: float


def usd_lens(intrinsics: CameraIntrinsics, focal_length: float = USD_FOCAL_LENGTH) -> UsdLens:
    """
    USD camera optics that project like ``intrinsics`` without distortion.

    USD's window spans ``offset +- aperture / 2`` on the film, so in rendered (pixel-edge) coordinates
    ``fx = width * focal / horizontal_aperture``, ``cx = width / 2 - width * horizontal_offset / horizontal_aperture``
    and ``cy = height / 2 + height * vertical_offset / vertical_aperture``.
    """

    w, h = intrinsics.width, intrinsics.height
    cx, cy = intrinsics.cx + PIXEL_CENTRE_OFFSET, intrinsics.cy + PIXEL_CENTRE_OFFSET
    return UsdLens(
        focal_length=focal_length,
        horizontal_aperture=w * focal_length / intrinsics.fx,
        vertical_aperture=h * focal_length / intrinsics.fy,
        horizontal_aperture_offset=(w / 2.0 - cx) * focal_length / intrinsics.fx,
        vertical_aperture_offset=(cy - h / 2.0) * focal_length / intrinsics.fy,
    )


def usd_camera_matrix(rotation: ArrayLike, translation: ArrayLike) -> NDArray[np.float64]:
    """The USD camera prim's local-to-world matrix (row-vector layout) for an OpenCV pose."""

    rot = np.asarray(rotation, dtype=float).reshape(3, 3)
    centre = -rot.T @ np.asarray(translation, dtype=float).reshape(3)
    matrix = np.eye(4)
    matrix[:3, :3] = USD_FROM_OPENCV @ rot
    matrix[3, :3] = centre
    return matrix


def opencv_lens_attributes(intrinsics: CameraIntrinsics) -> dict[str, float | tuple[int, int] | str]:
    """The ``OmniLensDistortionOpenCvPinholeAPI`` attribute values, the principal point in rendered coordinates."""

    k = intrinsics
    values: dict[str, float | tuple[int, int] | str] = {"omni:lensdistortion:model": "opencvPinhole", f"{OPENCV_LENS_PREFIX}:imageSize": (k.width, k.height)}
    for name in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"):
        offset = PIXEL_CENTRE_OFFSET if name in ("cx", "cy") else 0.0
        values[f"{OPENCV_LENS_PREFIX}:{name}"] = float(getattr(k, name)) + offset
    for name in ("k4", "k5", "k6", "s1", "s2", "s3", "s4"):
        values[f"{OPENCV_LENS_PREFIX}:{name}"] = 0.0
    return values


class AuthoredCamera(BaseModel):
    """
    One camera prim this project authored.

    :ivar name: ``orbbec_8E``, ``orbbec_8F``, ``broadcast`` or ``cine_wide``.
    :ivar prim_path: The ``UsdGeom.Camera`` prim.
    :ivar resolution: Render size, width by height (px).
    :ivar calibrated: Whether it carries a measured calibration (the Orbbecs).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    prim_path: str
    resolution: tuple[int, int]
    calibrated: bool


def author_orbbec_camera(stage: Usd.Stage, calibration: OrbbecCalibration) -> AuthoredCamera:
    """
    Author one virtual Orbbec at its calibrated pose with the OpenCV lens (distortion rendered).

    :return: The camera, rendered at the calibration's resolution.
    """

    from pxr import Gf, Sdf, UsdGeom

    intrinsics = calibration.intrinsics
    name = f"orbbec_{calibration.label}"
    path = f"{CAMERA_ROOT}/{name}"
    optics = usd_lens(intrinsics)
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr(optics.focal_length)
    camera.CreateHorizontalApertureAttr(optics.horizontal_aperture)
    camera.CreateVerticalApertureAttr(optics.vertical_aperture)
    camera.CreateHorizontalApertureOffsetAttr(optics.horizontal_aperture_offset)
    camera.CreateVerticalApertureOffsetAttr(optics.vertical_aperture_offset)
    camera.CreateClippingRangeAttr(Gf.Vec2f(*CLIPPING_RANGE_M))
    camera.CreateFStopAttr(0.0)
    xformable = UsdGeom.Xformable(camera)
    xformable.ClearXformOpOrder()
    matrix = usd_camera_matrix(calibration.rotation, calibration.translation)
    xformable.AddTransformOp().Set(Gf.Matrix4d(*matrix.ravel().tolist()))
    prim = camera.GetPrim()
    prim.CreateAttribute("table_tennis:serial", Sdf.ValueTypeNames.String).Set(calibration.serial)
    prim.CreateAttribute("table_tennis:epoch", Sdf.ValueTypeNames.String).Set(calibration.epoch)
    prim.AddAppliedSchema(OPENCV_LENS_SCHEMA)
    for attribute, value in opencv_lens_attributes(intrinsics).items():
        if attribute.endswith(":imageSize"):
            width, height = value  # type: ignore[misc]
            prim.CreateAttribute(attribute, Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(int(width), int(height)))
        elif attribute.endswith(":model"):
            prim.CreateAttribute(attribute, Sdf.ValueTypeNames.Token).Set(str(value))
        else:
            prim.CreateAttribute(attribute, Sdf.ValueTypeNames.Float).Set(float(value))  # type: ignore[arg-type]
    return AuthoredCamera(name=name, prim_path=path, resolution=(intrinsics.width, intrinsics.height), calibrated=True)


def look_at_matrix(eye: tuple[float, float, float], target: tuple[float, float, float]) -> NDArray[np.float64]:
    """A camera's local-to-world matrix (USD row-vector layout; the camera looks down its -Z, +Y up)."""

    eye_m, target_m = np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64)
    backward = (eye_m - target_m) / np.linalg.norm(eye_m - target_m)
    right = np.cross((0.0, 0.0, 1.0), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    matrix = np.eye(4)
    matrix[0, :3], matrix[1, :3], matrix[2, :3], matrix[3, :3] = right, up, backward, eye_m
    return matrix


@dataclass(frozen=True)
class PresentationView:
    """
    A fixed presentation camera in the table frame (16:9 apertures, no distortion).

    :ivar name: Camera name.
    :ivar resolution: Render size, px.
    :ivar eye_m: Camera position, m.
    :ivar target_m: Point it looks at, m.
    :ivar focal_length: Raw USD focal length (with the 0.36 x 0.2025 apertures: the field of view).
    """

    name: str
    resolution: tuple[int, int]
    eye_m: tuple[float, float, float]
    target_m: tuple[float, float, float]
    focal_length: float


BROADCAST = PresentationView("broadcast", (1280, 720), eye_m=(-3.2, -3.6, 1.9), target_m=(0.2, 0.0, -0.1), focal_length=0.27)
"""The replay's cinematic view: high three-quarter from the cameras' side behind the 8F end, both players head to foot (67 degrees wide)."""

CINE_WIDE = PresentationView("cine_wide", (1920, 1080), eye_m=(0.0, -4.4, 1.7), target_m=(0.0, 0.3, 0.05), focal_length=0.30)
"""The television side view: level with the net, the whole table and both ready areas (out to x = +-2.2 m), 62 degrees wide."""


def author_presentation_camera(stage: Usd.Stage, view: PresentationView) -> AuthoredCamera:
    """Author a presentation camera (``broadcast`` or ``cine_wide``) under :data:`CAMERA_ROOT`."""

    from pxr import Gf, UsdGeom

    path = f"{CAMERA_ROOT}/{view.name}"
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(view.focal_length)
    camera.CreateHorizontalApertureAttr(0.36)
    camera.CreateVerticalApertureAttr(0.2025)
    camera.CreateClippingRangeAttr(Gf.Vec2f(*CLIPPING_RANGE_M))
    camera.CreateFStopAttr(0.0)
    xformable = UsdGeom.Xformable(camera)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(Gf.Matrix4d(*look_at_matrix(view.eye_m, view.target_m).ravel().tolist()))
    return AuthoredCamera(name=view.name, prim_path=path, resolution=view.resolution, calibrated=False)


VIEWPORT_CAMERA: str = "/OmniverseKit_Persp"
"""Kit's own perspective camera: the viewport's camera, which a run streamed to Mission Control shows and its viewer flies."""


def aim_viewport(view: PresentationView) -> None:
    """
    Stand the viewport's camera (:data:`VIEWPORT_CAMERA`) at ``view``'s pose, on a stage whose world is the table frame.

    A run streamed to Mission Control opens on this view. A viewer who flies around moves only the viewport's own
    camera, never a camera the run records; a headless run never renders the viewport.
    """

    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(eye=np.asarray(view.eye_m), target=np.asarray(view.target_m), camera_prim_path=VIEWPORT_CAMERA)


@dataclass(frozen=True)
class PinholeSpec:
    """
    A camera for telemetry and visibility tests: OpenCV intrinsics, distortion and pose in the table frame.

    :ivar name: Camera name.
    :ivar width: Image width, px.
    :ivar height: Image height, px.
    :ivar fx: Focal length, px.
    :ivar fy: Focal length, px.
    :ivar cx: Principal point, px (OpenCV pixel-centre convention).
    :ivar cy: Principal point, px.
    :ivar rotation: World-to-camera rotation ``R`` (OpenCV axes).
    :ivar translation: World-to-camera translation ``t``, m.
    :ivar calibrated: Whether it carries a measured calibration (the Orbbecs) rather than a presentation lens.
    :ivar distortion: OpenCV ``(k1, k2, p1, p2, k3)``.
    """

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    calibrated: bool
    distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    def project(self, point: NDArray[np.float64]) -> tuple[float, float] | None:
        """
        Pixel of a world point through the lens, in Rerun's convention (pixel centres at +0.5).

        :return: ``(u, v)``, or ``None`` behind the camera or off the image.
        """

        camera = self.rotation @ point + self.translation
        if camera[2] <= 1e-6:
            return None
        x, y = camera[0] / camera[2], camera[1] / camera[2]
        k1, k2, p1, p2, k3 = self.distortion
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        u = self.fx * xd + self.cx + 0.5
        v = self.fy * yd + self.cy + 0.5
        if not (0.0 <= u < self.width and 0.0 <= v < self.height):
            return None
        return float(u), float(v)


def orbbec_pinhole(calibration: OrbbecCalibration) -> PinholeSpec:
    """A calibrated Orbbec as a telemetry pinhole."""

    k = calibration.intrinsics
    return PinholeSpec(
        name=f"orbbec_{calibration.label}",
        width=k.width,
        height=k.height,
        fx=k.fx,
        fy=k.fy,
        cx=k.cx,
        cy=k.cy,
        rotation=np.asarray(calibration.rotation, dtype=np.float64),
        translation=np.asarray(calibration.translation, dtype=np.float64),
        calibrated=True,
        distortion=(k.k1, k.k2, k.p1, k.p2, k.k3),
    )


def presentation_pinhole(view: PresentationView) -> PinholeSpec:
    """A presentation camera as a telemetry pinhole (OpenCV axes, centred principal point, no distortion)."""

    matrix = look_at_matrix(view.eye_m, view.target_m)
    rotation = np.stack([matrix[0, :3], -matrix[1, :3], -matrix[2, :3]])
    rotation /= np.linalg.norm(rotation, axis=1, keepdims=True)
    width, height = view.resolution
    return PinholeSpec(
        name=view.name,
        width=width,
        height=height,
        fx=width * view.focal_length / 0.36,
        fy=height * view.focal_length / 0.2025,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        rotation=rotation,
        translation=-rotation @ matrix[3, :3],
        calibrated=False,
    )


def _srgb_to_linear(srgb: NDArray[np.float64]) -> NDArray[np.float64]:
    c = np.asarray(srgb, dtype=float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(linear: NDArray[np.float64]) -> NDArray[np.float64]:
    c = np.clip(np.asarray(linear, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255.0


def isp_luts(path: Path | str = DATA / "orbbec_isp.json") -> dict[str, NDArray[np.uint8]]:
    """
    Each virtual Orbbec's image signal processing as three 256-entry lookup tables (``orbbec_<label>``).

    The real cameras ran auto white balance and exposure, so the two recorded views differ in tint and level; RTX
    renders physical light with neither. Each virtual camera applies one fixed gain per channel in linear light
    (``lib/data/orbbec_isp.json``), fitted at the calibrated lighting against the recorded table and floor colours.
    """

    entries: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    levels = _srgb_to_linear(np.arange(256, dtype=float))
    return {
        f"orbbec_{label}": np.stack([np.round(_linear_to_srgb(levels * float(gain))).astype(np.uint8) for gain in entry["gains"]])
        for label, entry in entries.items()
    }


# ---- building the lab -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HimLab:
    """
    The replay's lab on the stage.

    :ivar assets: ``name@version`` per loaded role.
    :ivar colliders: The collision shapes and materials.
    :ivar panels: The re-lit ceiling panels.
    """

    assets: dict[str, str]
    colliders: ColliderFacts
    panels: tuple[str, ...]


def build_him_lab(stage: Usd.Stage, physics: BallTableParams, lighting: LabLighting = CALIBRATED_LIGHTING) -> HimLab:
    """
    Clear the stage and compose the lab in the table frame: room, table and net, the colliders, the lighting.

    :param stage: The active stage, stopped.
    :param physics: The ball physics the engine runs (the colliders' materials).
    :param lighting: The ceiling panels' light.
    :return: What was built.
    """

    from pxr import Gf, UsdGeom

    clear_stage(stage)
    UsdGeom.Xform.Define(stage, STAGE_ROOT)
    root = UsdGeom.Xform.Define(stage, LAB_ROOT)
    root.ClearXformOpOrder()
    root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -ASSET_SURFACE_HEIGHT_M))
    assets = {role: load_lab_asset(role, path) for role, path in LAB_PRIMS.items()}
    colliders = author_lab_colliders(stage, physics)
    panels = apply_lab_lighting(stage, LAB_PRIMS["lab_room"], lighting)
    return HimLab(assets=assets, colliders=colliders, panels=panels)


def author_him_cameras(stage: Usd.Stage, calibrations: dict[str, OrbbecCalibration]) -> dict[str, AuthoredCamera]:
    """The virtual Orbbecs at the given calibrations and the broadcast camera, keyed by camera name."""

    cameras = {f"orbbec_{label}": author_orbbec_camera(stage, calibration) for label, calibration in calibrations.items()}
    cameras["broadcast"] = author_presentation_camera(stage, BROADCAST)
    return cameras


TWIN_ROOT: str = "/World/HimTwin"
"""The render-only twin's room, outside any cloned env so it is never cloned."""

PHYSICS_PREFIXES: tuple[str, ...] = ("Physics", "Physx")
"""USD Physics and PhysX type and schema names: colliders, bodies, masses, materials, joints and scenes all start so."""


def physics_prim_count(stage: Usd.Stage) -> int:
    """How many prims of the whole stage carry physics (a physics type or applied schema), instance proxies included."""

    from pxr import Usd

    count = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        names = [str(prim.GetTypeName()), *(str(name) for name in prim.GetAppliedSchemas())]
        count += any(name.startswith(PHYSICS_PREFIXES) for name in names)
    return count


@dataclass(frozen=True)
class HimTwin:
    """
    The render-only twin around a robot's table.

    :ivar room_prim: The room's prim.
    :ivar cine_wide: The ``cine_wide`` camera (render it at its resolution).
    :ivar orbbec_8f: The calibrated Orbbec 8F at its latest epoch, or ``None``.
    :ivar panels: The re-lit ceiling panels.
    """

    room_prim: str
    cine_wide: AuthoredCamera
    orbbec_8f: AuthoredCamera | None
    panels: tuple[str, ...]


def author_him_twin(stage: Usd.Stage, table_origin_world: ArrayLike, *, orbbec_8f: bool = True) -> HimTwin:
    """
    Dress a table at ``table_origin_world`` in the HIM lab, render only: the room, the calibrated lighting, ``cine_wide``
    and (optionally) the Orbbec 8F view. Call on a stopped stage, after the robot scene is built.

    :param stage: The active stage.
    :param table_origin_world: The table frame's origin (the table-top centre) in the world, m.
    :param orbbec_8f: Also author the calibrated Orbbec 8F camera.
    :return: The twin.
    :raises RuntimeError: If the twin changed the stage's count of physics prims
    """

    import antioch
    from pxr import Gf, UsdGeom

    before = physics_prim_count(stage)
    origin = np.asarray(table_origin_world, dtype=np.float64).reshape(3)
    room_root = UsdGeom.Xform.Define(stage, TWIN_ROOT)
    room_root.ClearXformOpOrder()
    room_root.AddTranslateOp().Set(Gf.Vec3d(float(origin[0]), float(origin[1]), float(origin[2]) - ASSET_SURFACE_HEIGHT_M))
    asset = ASSETS[LAB_ASSETS["lab_room"][0]]
    room_prim = f"{TWIN_ROOT}/room"
    UsdGeom.Xform.Define(stage, room_prim).GetPrim().GetReferences().AddReference(str(antioch.fetch_asset(asset.name, version=asset.version)))
    panels = apply_lab_lighting(stage, room_prim)
    # The cameras are authored in the table frame: their root goes to the table.
    root = UsdGeom.Xform.Define(stage, STAGE_ROOT)
    root.ClearXformOpOrder()
    root.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in origin)))
    cine = author_presentation_camera(stage, CINE_WIDE)
    camera_8f = author_orbbec_camera(stage, load_calibration_set().latest("8F")) if orbbec_8f else None
    after = physics_prim_count(stage)
    if after != before:
        raise RuntimeError(f"the HIM twin changed the stage's physics prims from {before} to {after}; it must be render only")
    return HimTwin(room_prim=room_prim, cine_wide=cine, orbbec_8f=camera_8f, panels=panels)


__all__ = [
    "BROADCAST",
    "CALIBRATED_LIGHTING",
    "CAMERA_ROOT",
    "CINE_WIDE",
    "COLLIDER_ROOT",
    "FLOOR_COLLIDER",
    "LAB_PRIMS",
    "NET_FIXTURE_COLLIDERS",
    "ROOM_COLLIDERS",
    "SERIALS",
    "STAGE_ROOT",
    "VIEWPORT_CAMERA",
    "AuthoredCamera",
    "ColliderFacts",
    "HimLab",
    "HimTwin",
    "LabLighting",
    "OrbbecCalibration",
    "OrbbecCalibrationSet",
    "PinholeSpec",
    "PresentationView",
    "aim_viewport",
    "apply_lab_lighting",
    "author_ball_colliders",
    "author_him_cameras",
    "author_him_twin",
    "author_lab_colliders",
    "author_orbbec_camera",
    "author_presentation_camera",
    "build_him_lab",
    "clear_stage",
    "isp_luts",
    "load_calibration_set",
    "load_lab_asset",
    "orbbec_pinhole",
    "presentation_pinhole",
    "static_collider_shapes",
]

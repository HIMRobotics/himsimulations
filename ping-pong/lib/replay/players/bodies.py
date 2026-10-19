"""
The players' bodies: ActorCore characters with a live UsdSkel ``Pose`` prim the motion writes every render tick.

Each body is spawned with ``antioch.load_asset`` **after** ``world.reset()`` (a skinned body on the stage stalls the
reset), its physics APIs are stripped, and a ``SkelAnimation`` named ``Pose`` is bound as the skeleton's animation
source. No Animation Graph is involved, so headless and streamed runs behave the same. The prim carries only the floor
offset and the scale (centimetres to metres times the body's height match); the motion's root (``Hip``) local
transform carries position and orientation, so skeleton space has the table frame's axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from lib.assets import actorcore_body
from lib.him_lab import STAGE_ROOT
from lib.replay.players.skeleton import ROOT_JOINT, SKELETON_UNITS_PER_METRE, SkeletonRest, embedded_skeleton, skeleton_from_arrays

if TYPE_CHECKING:
    from pxr import Usd

PLAYERS_ROOT: str = f"{STAGE_ROOT}/Players"
POSE_PRIM: str = "Pose"
CLIPS_PRIM: str = "Animation"
"""The ActorCore body's built-in ``SkelAnimation`` clips (26 per body, 78 time-sampled attributes): never played, since
the replay binds its own ``Pose``, but Fabric re-reads their samples whenever the timeline moves, every PhysX step
(about half of the physics plugin's time, 0.09 ms per step). Deactivated at spawn."""
FLOOR_Z_M: float = -0.76
"""The floor in the table frame (``lib.physics`` table height)."""


@dataclass
class CharacterHandle:
    """
    A spawned body and what its motion needs.

    :ivar role: ``suited`` or ``opponent``.
    :ivar body: ``asset@version``.
    :ivar look: Looks variant applied.
    :ivar path: Root prim.
    :ivar skeleton_path: The ``Skeleton`` prim.
    :ivar pose_path: The bound ``Pose`` animation.
    :ivar prim_scale: Uniform prim scale (0.01 times the body scale).
    :ivar skeleton: Rest skeleton read from the body's USD, metres at body scale 1.
    :ivar embedded_max_difference_m: Largest rest-offset difference from the embedded copy (:mod:`lib.replay.players.skeleton`), m (spawning refuses more than 2 mm or 0.5 deg).
    :ivar spawn_s: Wall time to spawn, s.
    :ivar clips_deactivated: The body's built-in animation clips deactivated at spawn (:data:`CLIPS_PRIM`).
    :ivar writes: Pose writes so far.
    :ivar write_seconds: Wall time spent writing poses, s.
    """

    role: str
    body: str
    look: str | None
    path: str
    skeleton_path: str
    pose_path: str
    prim_scale: float
    skeleton: SkeletonRest
    embedded_max_difference_m: float
    spawn_s: float
    clips_deactivated: int = 0
    writes: int = 0
    write_seconds: float = 0.0
    _pose: Any = field(default=None, repr=False)
    _translations: NDArray[np.float32] | None = field(default=None, repr=False)
    _quat_from_numpy: bool = field(default=False, repr=False)

    def write_pose(
        self,
        local_rotation: NDArray[np.float64],
        root_translation: NDArray[np.float64],
        translated: NDArray[np.int64] | None = None,
        local_translation: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Write one pose to the ``Pose`` prim.

        :param local_rotation: Every joint's local rotation ``(J, 4)``, ``(w, x, y, z)``, in the skeleton's joint order.
        :param root_translation: The root's local translation, skeleton metres at body scale 1.
        :param translated: Joint indices whose local translation the track animates (UsdSkel ``translations``).
        :param local_translation: Their translations, ``(T, 3)``, skeleton metres at body scale 1.
        """

        import time

        from pxr import Gf, Vt

        started = time.perf_counter()
        assert self._translations is not None
        translations = self._translations.copy()
        translations[self.skeleton.index(ROOT_JOINT)] = np.asarray(root_translation, dtype=np.float32) * SKELETON_UNITS_PER_METRE
        if translated is not None and local_translation is not None and len(translated):
            translations[translated] = np.asarray(local_translation, dtype=np.float32) * SKELETON_UNITS_PER_METRE
        self._pose.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(translations))
        q = np.asarray(local_rotation, dtype=np.float32)
        if self._quat_from_numpy:
            values = Vt.QuatfArray.FromNumpy(np.ascontiguousarray(q[:, [1, 2, 3, 0]]))
        else:
            values = Vt.QuatfArray([Gf.Quatf(float(r[0]), Gf.Vec3f(float(r[1]), float(r[2]), float(r[3]))) for r in q])
        self._pose.GetRotationsAttr().Set(values)
        self.writes += 1
        self.write_seconds += time.perf_counter() - started


def _set_root_transform(prim: Any, translation: tuple[float, float, float], scale: float) -> None:
    """Reuse the asset's authored translate/orient/scale ops (their precision is fixed by the asset)."""

    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    existing = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
    xformable.ClearXformOpOrder()
    translate = existing.get("xformOp:translate") or xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    orient = existing.get("xformOp:orient") or xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    scaling = existing.get("xformOp:scale") or xformable.AddScaleOp()
    double = UsdGeom.XformOp.PrecisionDouble
    translate.Set(Gf.Vec3d(*translation) if translate.GetPrecision() == double else Gf.Vec3f(*translation))
    orient.Set(Gf.Quatd(1.0) if orient.GetPrecision() == double else Gf.Quatf(1.0))
    scaling.Set(Gf.Vec3d(scale) if scaling.GetPrecision() == double else Gf.Vec3f(scale))
    xformable.SetXformOpOrder([translate, orient, scaling])


def _strip_physics(root: Any) -> None:
    from pxr import PhysxSchema, Usd, UsdPhysics

    for prim in Usd.PrimRange(root):
        for schema in (
            UsdPhysics.RigidBodyAPI,
            PhysxSchema.PhysxRigidBodyAPI,
            UsdPhysics.ArticulationRootAPI,
            UsdPhysics.CollisionAPI,
            UsdPhysics.MeshCollisionAPI,
        ):
            if prim.HasAPI(schema):
                prim.RemoveAPI(schema)


def _skeleton_from_usd(body: str, skeleton_prim: Any) -> tuple[SkeletonRest, NDArray[np.float32], NDArray[np.float64]]:
    """The rest skeleton in metres, plus the rest translations in skeleton units and rest rotations as authored."""

    from pxr import UsdSkel

    skeleton = UsdSkel.Skeleton(skeleton_prim)
    paths = [str(path) for path in skeleton.GetJointsAttr().Get()]
    names = [path.split("/")[-1] for path in paths]
    index = {path: position for position, path in enumerate(paths)}
    parents = [index.get(path.rsplit("/", 1)[0], -1) if "/" in path else -1 for path in paths]
    rest = skeleton.GetRestTransformsAttr().Get()
    translations = np.zeros((len(paths), 3))
    rotations = np.zeros((len(paths), 4))
    for position, matrix in enumerate(rest):
        translation = matrix.ExtractTranslation()
        quaternion = matrix.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        translations[position] = (translation[0], translation[1], translation[2])
        rotations[position] = (quaternion.GetReal(), imaginary[0], imaginary[1], imaginary[2])
    rest_skeleton = skeleton_from_arrays(body, names, parents, translations / SKELETON_UNITS_PER_METRE, rotations)
    return rest_skeleton, translations.astype(np.float32), rotations


def spawn_character(stage: Usd.Stage, *, role: str, asset: str, version: str, look: str | None, body_scale: float) -> CharacterHandle:
    """
    Spawn one body with a bound ``Pose`` prim, standing in its rest pose at the origin.

    Call after ``world.reset()``.

    :param stage: The live stage.
    :param role: ``suited`` or ``opponent``.
    :param asset: Catalog asset name.
    :param version: Pinned version.
    :param look: Looks variant, or ``None``.
    :param body_scale: Height match on top of the centimetre units.
    :return: The handle.
    :raises RuntimeError: If the asset has no skeleton or its joints disagree with the embedded copy
    """

    import time

    import antioch
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt

    started = time.perf_counter()
    UsdGeom.Xform.Define(stage, PLAYERS_ROOT)
    path = f"{PLAYERS_ROOT}/{role}"
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)
    # The package names the ActorCore body it was built with; this project's store carries a copy of it.
    copy = actorcore_body(asset, version)
    antioch.load_asset(copy.name, prim_path=path, version=copy.version)
    application = antioch.application()
    if application is not None:
        application.update()
    root = stage.GetPrimAtPath(path)
    if look is not None:
        variant_set = root.GetVariantSets().GetVariantSet("looks")
        if look not in variant_set.GetVariantNames():
            raise RuntimeError(f"{asset}@{version} has no look {look!r}; available: {variant_set.GetVariantNames()}")
        variant_set.SetVariantSelection(look)
    prim_scale = body_scale / SKELETON_UNITS_PER_METRE
    _set_root_transform(root, (0.0, 0.0, FLOOR_Z_M), prim_scale)
    _strip_physics(root)
    skeleton_prim = next((prim for prim in Usd.PrimRange(root) if prim.IsA(UsdSkel.Skeleton)), None)
    if skeleton_prim is None:
        raise RuntimeError(f"{asset}@{version} has no Skeleton prim")
    body = f"{asset}@{version}"
    skeleton, translations_units, rotations = _skeleton_from_usd(body, skeleton_prim)
    embedded = embedded_skeleton(body)
    if embedded.joints != skeleton.joints:
        raise RuntimeError(f"{body}: the spawned skeleton's joints differ from the embedded copy")
    difference = float(np.max(np.abs(embedded.rest_translation - skeleton.rest_translation)))
    dots = np.abs(np.sum(embedded.rest_rotation * skeleton.rest_rotation, axis=1))
    rotation_difference = float(np.degrees(2.0 * np.arccos(np.clip(dots.min(), -1.0, 1.0))))
    if difference > 0.002 or rotation_difference > 0.5:
        raise RuntimeError(f"{body}: rest skeleton differs from the embedded copy by {difference * 1e3:.2f} mm / {rotation_difference:.2f} deg")
    pose = UsdSkel.Animation.Define(stage, Sdf.Path(path).AppendChild(POSE_PRIM))
    pose.CreateJointsAttr().Set(UsdSkel.Skeleton(skeleton_prim).GetJointsAttr().Get())
    pose.CreateTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(translations_units))
    pose.CreateScalesAttr().Set(Vt.Vec3hArray([Gf.Vec3h(1.0, 1.0, 1.0)] * len(skeleton.joints)))
    pose.CreateRotationsAttr().Set(Vt.QuatfArray([_quatf(row) for row in rotations]))
    UsdSkel.BindingAPI.Apply(skeleton_prim).CreateAnimationSourceRel().SetTargets([pose.GetPath()])
    clips = root.GetChild(CLIPS_PRIM)
    clips_deactivated = 0
    if clips.IsValid() and not clips.IsInstanceProxy() and not skeleton_prim.GetPath().HasPrefix(clips.GetPath()):
        clips_deactivated = sum(1 for prim in Usd.PrimRange(clips) if prim.IsA(UsdSkel.Animation))
        clips.SetActive(False)
    handle = CharacterHandle(
        role=role,
        body=body,
        look=look,
        path=path,
        skeleton_path=str(skeleton_prim.GetPath()),
        pose_path=str(pose.GetPath()),
        prim_scale=prim_scale,
        skeleton=skeleton,
        embedded_max_difference_m=difference,
        spawn_s=time.perf_counter() - started,
        clips_deactivated=clips_deactivated,
        _pose=pose,
        _translations=translations_units,
    )
    handle._quat_from_numpy = _quatf_from_numpy_works(pose, rotations)
    return handle


def _quatf(row: Any) -> Any:
    from pxr import Gf

    return Gf.Quatf(float(row[0]), Gf.Vec3f(float(row[1]), float(row[2]), float(row[3])))


def _quatf_from_numpy_works(pose: Any, rotations: NDArray[np.float64]) -> bool:
    """Whether ``Vt.QuatfArray.FromNumpy`` takes ``(x, y, z, w)`` rows on this USD build (checked by a round trip)."""

    from pxr import Vt

    try:
        probe = np.ascontiguousarray(rotations[:, [1, 2, 3, 0]].astype(np.float32))
        array = Vt.QuatfArray.FromNumpy(probe)
        first = array[min(1, len(array) - 1)]
        expected = rotations[min(1, len(rotations) - 1)]
        imaginary = first.GetImaginary()
        got = np.array([first.GetReal(), imaginary[0], imaginary[1], imaginary[2]])
        return bool(np.allclose(got, expected, atol=1e-5))
    except Exception:
        return False
    finally:
        del pose


def usdskel_joint_world(stage: Usd.Stage, handle: CharacterHandle, joint: str) -> NDArray[np.float64]:
    """
    One joint's world position as UsdSkel itself computes it (to check the motion's forward kinematics).

    :return: Table-frame position, m.
    """

    from pxr import Usd, UsdGeom, UsdSkel

    cache = UsdSkel.Cache()
    root = UsdSkel.Root(next(prim for prim in Usd.PrimRange(stage.GetPrimAtPath(handle.path)) if prim.IsA(UsdSkel.Root)))
    cache.Populate(root, Usd.TraverseInstanceProxies())
    query = cache.GetSkelQuery(UsdSkel.Skeleton(stage.GetPrimAtPath(handle.skeleton_path)))
    transforms = query.ComputeJointWorldTransforms(UsdGeom.XformCache(Usd.TimeCode.Default()))
    translation = transforms[handle.skeleton.index(joint)].ExtractTranslation()
    return np.array([translation[0], translation[1], translation[2]])


__all__ = ["CLIPS_PRIM", "FLOOR_Z_M", "PLAYERS_ROOT", "POSE_PRIM", "CharacterHandle", "spawn_character", "usdskel_joint_world"]

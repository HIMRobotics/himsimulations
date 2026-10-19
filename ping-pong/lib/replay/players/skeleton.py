"""
The ActorCore CC3 rest skeleton of each character body: joint names, parents and rest transforms.

Every ActorCore body shares the 101-joint CC3 skeleton, but each body orders
its joints differently and has its own bone lengths, so a skeleton is always
keyed by ``asset@version``. Rest transforms are local (relative to the parent
joint), in metres at body scale 1 (the catalog USD is authored in centimetres,
so one skeleton unit is 0.01 m), with quaternions ``(w, x, y, z)``.

:data:`EMBEDDED_SKELETONS_PATH` carries the rest skeletons of the bodies the replay
spawns, exported from the ActorCore USDZ. It lets the motion code and the tests run
without a simulator; in Isaac Sim the spawned body's own ``UsdSkel`` rest transforms
are read and checked against it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

EMBEDDED_SKELETONS_PATH: Path = Path(__file__).resolve().parents[1] / "data" / "actorcore_skeletons.json"

SKELETON_UNITS_PER_METRE: float = 100.0
"""ActorCore catalog bodies are authored in centimetres on a metres-per-unit-1 stage."""

ROOT_JOINT: str = "Hip"
"""The animated root: its local transform carries the body's position and orientation in skeleton space."""


@dataclass(frozen=True)
class SkeletonRest:
    """
    One body's rest skeleton.

    :ivar body: ``asset@version`` the skeleton belongs to.
    :ivar joints: Joint names in the body's own order (parents before children).
    :ivar parents: Parent index per joint, -1 for the skeleton root.
    :ivar rest_translation: Local rest translation per joint, metres at body scale 1, shape ``(J, 3)``.
    :ivar rest_rotation: Local rest rotation per joint, ``(w, x, y, z)``, shape ``(J, 4)``.
    """

    body: str
    joints: tuple[str, ...]
    parents: NDArray[np.int64]
    rest_translation: NDArray[np.float64]
    rest_rotation: NDArray[np.float64]
    _index: dict[str, int] = field(init=False, repr=False, compare=False)
    parents_list: tuple[int, ...] = field(init=False, repr=False, compare=False)
    """``parents`` as Python ints (the per-joint forward pass reads them without numpy)."""
    rest_translation_list: tuple[tuple[float, float, float], ...] = field(init=False, repr=False, compare=False)
    """``rest_translation`` as Python floats (exact: float64 to float)."""

    def __post_init__(self) -> None:
        count = len(self.joints)
        if self.parents.shape != (count,) or self.rest_translation.shape != (count, 3) or self.rest_rotation.shape != (count, 4):
            raise ValueError(f"skeleton {self.body}: array shapes do not match {count} joints")
        for index, parent in enumerate(self.parents):
            if parent >= index:
                raise ValueError(f"skeleton {self.body}: joint {self.joints[index]} is listed before its parent")
        if ROOT_JOINT not in self.joints:
            raise ValueError(f"skeleton {self.body} has no {ROOT_JOINT} joint")
        object.__setattr__(self, "_index", {name: index for index, name in enumerate(self.joints)})
        object.__setattr__(self, "parents_list", tuple(int(parent) for parent in self.parents))
        object.__setattr__(self, "rest_translation_list", tuple(tuple(row) for row in np.asarray(self.rest_translation, dtype=np.float64).tolist()))
        for array in (self.parents, self.rest_translation, self.rest_rotation):
            array.setflags(write=False)

    def with_offsets(self, joints: tuple[str, ...], offsets: NDArray[np.float64]) -> SkeletonRest:
        """
        The same skeleton with some joints' local offsets replaced (a track's lengthened bones).

        :param joints: Joints whose offsets change.
        :param offsets: Their new local translations, metres at body scale 1, ``(len(joints), 3)``.
        """

        translation = np.array(self.rest_translation, dtype=np.float64)
        for name, offset in zip(joints, np.asarray(offsets, dtype=np.float64), strict=True):
            translation[self.index(name)] = offset
        return SkeletonRest(
            body=self.body, joints=self.joints, parents=np.array(self.parents), rest_translation=translation, rest_rotation=np.array(self.rest_rotation)
        )

    def index(self, name: str) -> int:
        """
        The index of a joint.

        :raises KeyError: For a joint this body does not have
        """

        try:
            return self._index[name]
        except KeyError:
            raise KeyError(f"skeleton {self.body} has no joint {name!r}") from None

    def chain(self, name: str) -> tuple[int, ...]:
        """Joint indices from the skeleton root down to ``name``, inclusive."""

        out: list[int] = []
        index = self.index(name)
        while index >= 0:
            out.append(index)
            index = int(self.parents[index])
        return tuple(reversed(out))


def skeleton_from_arrays(body: str, joints: list[str] | tuple[str, ...], parents: list[int], translation_m: object, rotation_wxyz: object) -> SkeletonRest:
    """
    Build a :class:`SkeletonRest` from plain sequences.

    :param body: ``asset@version``.
    :param joints: Joint names.
    :param parents: Parent indices.
    :param translation_m: Local rest translations, metres at body scale 1.
    :param rotation_wxyz: Local rest rotations.
    :return: The skeleton.
    """

    rotation = np.asarray(rotation_wxyz, dtype=np.float64).reshape(-1, 4)
    rotation = rotation / np.linalg.norm(rotation, axis=1, keepdims=True)
    return SkeletonRest(
        body=body,
        joints=tuple(str(name) for name in joints),
        parents=np.asarray(parents, dtype=np.int64),
        rest_translation=np.asarray(translation_m, dtype=np.float64).reshape(-1, 3),
        rest_rotation=rotation,
    )


@cache
def embedded_skeleton(body: str) -> SkeletonRest:
    """
    The embedded rest skeleton of one catalog body.

    :param body: ``asset@version``, e.g. ``ActorCore_Party_M_0001@2.0.2``.
    :return: The skeleton.
    :raises KeyError: If no skeleton is embedded for that body
    """

    document = json.loads(EMBEDDED_SKELETONS_PATH.read_text(encoding="utf-8"))
    bodies = document["bodies"]
    if body not in bodies:
        raise KeyError(f"no embedded skeleton for {body!r}; available: {sorted(bodies)}")
    entry = bodies[body]
    return skeleton_from_arrays(body, entry["joints"], entry["parents"], entry["rest_translation_m"], entry["rest_rotation_wxyz"])


__all__ = ["EMBEDDED_SKELETONS_PATH", "ROOT_JOINT", "SKELETON_UNITS_PER_METRE", "SkeletonRest", "embedded_skeleton", "skeleton_from_arrays"]

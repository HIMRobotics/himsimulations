"""
Every asset this project loads, pinned by name, version and sha256, in HIM Robotics' Antioch asset store.

Runs fetch them by ``name@version`` (``antioch.fetch_asset`` / ``antioch.load_asset``); ``docs/assets.md`` describes
each. The Unitree G1 USD is not here: Isaac Lab loads it from NVIDIA's asset root at run time.

The session packages name their player bodies by the ActorCore catalog names they were built with
(``ActorCore_Party_M_0001@2.0.2``); :func:`actorcore_body` maps those names to this store's copies, so the packages
stay byte-identical.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Asset(BaseModel):
    """
    One pinned asset.

    :ivar name: Store name (``ping-pong/...``).
    :ivar version: Pinned version.
    :ivar file: The file the version holds.
    :ivar sha256: Its sha256 (``None`` until the asset is built and published).
    :ivar bytes: Its size (``None`` until built).
    :ivar source: Where it comes from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^ping-pong/[a-z0-9-]+$")
    version: str = Field(min_length=1)
    file: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    bytes: int | None = Field(default=None, gt=0)
    source: str = Field(min_length=1)

    @property
    def ref(self) -> str:
        """``asset:<name>@<version>``, the form run parameters take."""

        return f"asset:{self.name}@{self.version}"


_SESSION_SOURCE = "the 2026-08-15 capture packaged for replay: registered mocap tracks, serves, strokes, camera epochs and Orbbec frames"
_LAB_SOURCE = "HIM lab reconstruction from the 2026-08-15 capture, built procedurally in Blender (metres, Z up, no physics schemas)"

ASSETS: dict[str, Asset] = {
    asset.name: asset
    for asset in (
        # The HIM lab twin: room, table, net, paddle and ball (lib/him_lab.py).
        Asset(
            name="ping-pong/him-lab-room",
            version="v2",
            file="tt_lab_room.usdz",
            sha256="fbe93d9d6048c43ba6a2140157aef41465a1a5be541ae859ff6348d5ab6acc25",
            bytes=2906859,
            source=_LAB_SOURCE,
        ),
        Asset(
            name="ping-pong/table",
            version="v2",
            file="tt_table.usdz",
            sha256="8a640b84135683d38451ff0de8544fbefad0816fb8bb61bf5da935e882b2168a",
            bytes=153169,
            source=_LAB_SOURCE + "; ITTF table with the lab's ArUco stickers, its top fitted to the recorded frames",
        ),
        Asset(
            name="ping-pong/net",
            version="v2",
            file="tt_net.usdz",
            sha256="63e76e2b470841e5ec5dddedc0b8fdf7c2f1ae8b08f8fb06430122db124e8030",
            bytes=89871,
            source=_LAB_SOURCE,
        ),
        Asset(
            name="ping-pong/paddle",
            version="v2",
            file="tt_paddle_red.usdz",
            sha256="02475ccff549f082f6b47b2b90fb40aaedcb33761a2d4dd2fa80132765515d13",
            bytes=91167,
            source=_LAB_SOURCE + "; red inverted-rubber paddle",
        ),
        Asset(
            name="ping-pong/ball",
            version="v2",
            file="tt_ball.usdz",
            sha256="bc3d4b33d3c223599744835394c19d96b7a55654ebefd22de6145aeabb8e8dfd",
            bytes=36573,
            source=_LAB_SOURCE + "; 40 mm ball",
        ),
        # The replay's two player bodies (Reallusion ActorCore).
        Asset(
            name="ping-pong/actorcore-party-m-0001",
            version="2.0.2",
            file="ActorCore_Party_M_0001.usdz",
            sha256="8a65a00ec072dec83721bc195b45813ccf74be82e401ab459b3d2fb9bd72e2eb",
            bytes=35855064,
            source="Reallusion ActorCore Party-M-0001, the suited player (look A_BlackJacket)",
        ),
        Asset(
            name="ping-pong/actorcore-uniform-m-0001",
            version="1.0.1",
            file="ActorCore_Uniform_M_0001.usdz",
            sha256="f98be406919fa43f0fcee93b7efb09e5a5149a5b1216b8cc224c3e2066b4a4a8",
            bytes=42041475,
            source="Reallusion ActorCore Uniform-M-0001, the inferred opponent",
        ),
        # The seven recorded sessions (lib/replay/session.py).
        *(
            Asset(name=f"ping-pong/him-session-{k}", version="v8", file=file, sha256=sha, bytes=size, source=f"HIM session {k}: {_SESSION_SOURCE}")
            for k, file, sha, size in (
                (1, "session-1-214831-v8.zip", "98c3de568c26e8f6e08c09c49d7d8d17db1ff25e41585c56de72a77f5158b6c6", 500268739),
                (2, "session-2-215350-v8.zip", "af27329f064220b77de90611976bdc09d1e8991520fba15d733688a25201793f", 314334603),
                (3, "session-3-215750-v8.zip", "25047faa0bfef34b9f1256177e2c83cc9022d12dcb6806370e24d5a268158436", 500181705),
                (4, "session-4-220448_60fps-v8.zip", "d9494b8cf2dd8117b0979835b96254db3d2f2c53b22146c36517718438ae0d6f", 448133590),
                (5, "session-5-221035_60fps-v8.zip", "477ea77c00462a1766f96fa8bff97439aa10b4ee693874fa10cc37fc6ebda3d6", 454485017),
                (6, "session-6-221649_60fps-v8.zip", "8d30a95ee114b4f5d5a2d9d228dd6d2229fdaf2802184aa3efe9f474a0f4ce88", 465253124),
                (7, "session-7-222542_60fps-v8.zip", "4bb41b7fcfdc4e3158dc81ee419361410edc32f782b6688730091543f3fe013c", 460696576),
            )
        ),
        # The trained G1 and the opponent it plays (lib/policy, lib/opponent).
        Asset(
            name="ping-pong/g1-rally-policy",
            version="e0e0bb45-it1500",
            file="model_final.pt",
            sha256="93298cdbb4febcb2a0cc0bcf92d64e655f06c30fc1f77f0ea59736e3f19241ca",
            bytes=7108103,
            source="The trained Unitree G1 rally policy (PPO, iteration 1500), an actor-critic state dict",
        ),
        Asset(
            name="ping-pong/him-opponent-model",
            version="v2",
            file="him-opponent-model-v2.zip",
            sha256="a22b3b84eb12f602c2e18b9187e323d01cd11dfe2cea773d741ee01ccf574f10",
            bytes=6696224,
            source="The HIM opponent distilled from the recorded sessions: shot-intent mixture model and rules",
        ),
        # Retargeting (lib/retarget): the human strokes in, and the reference output for the G1.
        Asset(
            name="ping-pong/him-mocap-strokes",
            version="v1",
            file="him-mocap-strokes-v1.zip",
            sha256="f3eed08c3706925f9cb55a0c22cd6bdd61c49f3ee2fcdb884bdf6353173e8b92",
            bytes=33351893,
            source="HIM's mocap strokes on the table frame with paddle contacts (tools/build_mocap_strokes.py)",
        ),
        Asset(
            name="ping-pong/g1-retargeted-strokes",
            version="v2",
            file="g1-strokes-v2.zip",
            sha256="da4a42ea0cb7043f6c75714232c6fd214bd75f2bc55a7991b8f5dcdb67f4b683",
            bytes=77002986,
            source="HIM's strokes retargeted onto the G1 by lib/retarget, the worked example's reference output",
        ),
    )
}
"""Every asset, by store name."""


def ref(name: str) -> str:
    """
    The pinned reference of a store name.

    :param name: ``ping-pong/...``.
    :return: ``asset:<name>@<version>``.
    :raises KeyError: For an asset this project does not pin.
    """

    return ASSETS[name].ref


def session_asset(session: int) -> str:
    """The recorded session's package, ``asset:ping-pong/him-session-<k>@v8``."""

    return ref(f"ping-pong/him-session-{session}")


ACTORCORE_BODIES: dict[str, str] = {
    "ActorCore_Party_M_0001@2.0.2": "ping-pong/actorcore-party-m-0001",
    "ActorCore_Uniform_M_0001@1.0.1": "ping-pong/actorcore-uniform-m-0001",
}
"""The ActorCore body a session package names, to this store's copy."""


def actorcore_body(asset: str, version: str) -> Asset:
    """
    The store copy of a body a session package names.

    :param asset: The package's ActorCore name, e.g. ``ActorCore_Party_M_0001``.
    :param version: Its version, e.g. ``2.0.2``.
    :return: The pinned copy.
    :raises KeyError: For a body this project does not carry.
    """

    copy = ASSETS[ACTORCORE_BODIES[f"{asset}@{version}"]]
    if copy.version != version:
        raise KeyError(f"{asset}@{version}: the store copy is version {copy.version}")
    return copy


G1_RALLY_POLICY: str = ref("ping-pong/g1-rally-policy")
OPPONENT_MODEL: str = ref("ping-pong/him-opponent-model")
MOCAP_STROKES: str = ref("ping-pong/him-mocap-strokes")
G1_RETARGETED_STROKES: str = ref("ping-pong/g1-retargeted-strokes")

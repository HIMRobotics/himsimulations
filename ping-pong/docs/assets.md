# Assets

An *asset* is a large file a simulation needs, such as a 3D model, a recording or a trained network, kept outside git.
Antioch stores assets for each organization in its *asset store*. Every asset has a name and a version, written
`name@version`, such as `ping-pong/him-session-2@v8`. A version never changes once it's published, so a run that
names one always gets exactly the same bytes.

Every asset this project loads lives in HIM Robotics' asset store under the `ping-pong/` prefix. Each one is pinned by
version in [`lib/assets.py`](../lib/assets.py), with its size and its sha256: a fingerprint of the file's bytes, so
a changed file can't slip in unnoticed. Runs fetch assets by `name@version` with `antioch.fetch_asset`, so you never
download anything by hand to run a scenario.

To look at one yourself, signed in to the HIM Robotics organization (run from `ping-pong/`):

```bash
uv run antioch asset list -q ping-pong                          # every asset whose name or description says ping-pong
uv run antioch asset show ping-pong/him-session-2 -v v8         # one version: its size, digest and description
uv run antioch asset pull ping-pong/him-session-2 -v v8 -o /tmp/  # download it
```

Mind the sizes before pulling: each session recording is 314-500 MB (see [pinned contents](#pinned-contents)).

Three things aren't in the store:

- **The Unitree G1's 3D model.** Isaac Lab loads Isaac's own G1 USD from NVIDIA's asset library on the machine at run
  time. The G1's URDF (BSD-3, Unitree) is kept in [`lib/robots/urdf/`](../lib/robots/urdf/NOTICE).
- **The ball-physics priors**, `lib/physics/data/priors.json`. They're small, so they stay in git.
- **The Orbbec camera calibration and image-processing settings**, `lib/data/`. Also small, also in git.

## What each asset is

"Used by" names the scenario that loads it: the replay (`replay_him_session`), the policy (`play_ping_pong_policy`,
whose film also draws the room), the robot scene (the policy and the template, `load_humanoid_at_table`) and
retargeting (`retarget_him_strokes`).

| Asset | Used by | What it is |
|---|---|---|
| `ping-pong/him-lab-room@v2` | replay, robot film | The HIM lab, 10 x 8 x 3 m, rebuilt from the 2026-08-15 capture: floor, walls, the twelve ceiling LED panels (re-lit by `lib.him_lab`), props |
| `ping-pong/table@v2` | replay | ITTF table 2.74 x 1.525 m with the lab's eight ArUco stickers (printed markers the cameras locate the table by); its top's finish fitted to the recorded frames |
| `ping-pong/net@v2` | replay | ITTF net assembly: 15.25 cm high, posts and clamps |
| `ping-pong/paddle@v2` | replay, robot | Red inverted-rubber paddle, origin at the blade centre, +Z the forehand face |
| `ping-pong/ball@v2` | replay | 40 mm ball |
| `ping-pong/actorcore-party-m-0001@2.0.2` | replay | Reallusion ActorCore 3D body of the suited (motion-captured) player, look `A_BlackJacket` |
| `ping-pong/actorcore-uniform-m-0001@1.0.1` | replay | Reallusion ActorCore 3D body of the opponent |
| `ping-pong/him-session-{1..7}@v8` | replay | One recorded session each: both players' motion tracks, serves, strokes with their reconstructed paddle contacts, the cameras' calibration over time, and every recorded Orbbec frame |
| `ping-pong/g1-rally-policy@e0e0bb45-it1500` | policy | The trained Unitree G1 rally policy: the network's weights after 1500 iterations of PPO training (a PyTorch actor-critic state dict) |
| `ping-pong/him-opponent-model@v2` | policy | The HIM opponent learnt from the recorded sessions: a network that picks each shot's intent, and rules for reaching the ball |
| `ping-pong/him-mocap-strokes@v1` | retargeting | HIM's mocap strokes on the table frame with the paddle state at each contact (`tools/build_mocap_strokes.py`) |
| `ping-pong/g1-retargeted-strokes@v2` | retargeting | HIM's strokes retargeted onto the G1: the worked example's reference output |

The session packages name their player bodies by the ActorCore catalog names they were built with
(`ActorCore_Party_M_0001@2.0.2`); `lib.assets.actorcore_body` maps those names to this store's copies, so the
packages are byte-identical to the ones the evidence was recorded with.

## Pinned contents

Each asset's size in bytes and its sha256. This table mirrors `lib/assets.py`, and `tests/test_contract.py` checks
that every pinned name, version and sha256 appears here, so the two can't drift apart.

| Asset | Bytes | sha256 |
|---|---|---|
| `ping-pong/him-lab-room@v2` | 2906859 | `fbe93d9d6048c43ba6a2140157aef41465a1a5be541ae859ff6348d5ab6acc25` |
| `ping-pong/table@v2` | 153169 | `8a640b84135683d38451ff0de8544fbefad0816fb8bb61bf5da935e882b2168a` |
| `ping-pong/net@v2` | 89871 | `63e76e2b470841e5ec5dddedc0b8fdf7c2f1ae8b08f8fb06430122db124e8030` |
| `ping-pong/paddle@v2` | 91167 | `02475ccff549f082f6b47b2b90fb40aaedcb33761a2d4dd2fa80132765515d13` |
| `ping-pong/ball@v2` | 36573 | `bc3d4b33d3c223599744835394c19d96b7a55654ebefd22de6145aeabb8e8dfd` |
| `ping-pong/actorcore-party-m-0001@2.0.2` | 35855064 | `8a65a00ec072dec83721bc195b45813ccf74be82e401ab459b3d2fb9bd72e2eb` |
| `ping-pong/actorcore-uniform-m-0001@1.0.1` | 42041475 | `f98be406919fa43f0fcee93b7efb09e5a5149a5b1216b8cc224c3e2066b4a4a8` |
| `ping-pong/him-session-1@v8` | 500268739 | `98c3de568c26e8f6e08c09c49d7d8d17db1ff25e41585c56de72a77f5158b6c6` |
| `ping-pong/him-session-2@v8` | 314334603 | `af27329f064220b77de90611976bdc09d1e8991520fba15d733688a25201793f` |
| `ping-pong/him-session-3@v8` | 500181705 | `25047faa0bfef34b9f1256177e2c83cc9022d12dcb6806370e24d5a268158436` |
| `ping-pong/him-session-4@v8` | 448133590 | `d9494b8cf2dd8117b0979835b96254db3d2f2c53b22146c36517718438ae0d6f` |
| `ping-pong/him-session-5@v8` | 454485017 | `477ea77c00462a1766f96fa8bff97439aa10b4ee693874fa10cc37fc6ebda3d6` |
| `ping-pong/him-session-6@v8` | 465253124 | `8d30a95ee114b4f5d5a2d9d228dd6d2229fdaf2802184aa3efe9f474a0f4ce88` |
| `ping-pong/him-session-7@v8` | 460696576 | `4bb41b7fcfdc4e3158dc81ee419361410edc32f782b6688730091543f3fe013c` |
| `ping-pong/g1-rally-policy@e0e0bb45-it1500` | 7108103 | `93298cdbb4febcb2a0cc0bcf92d64e655f06c30fc1f77f0ea59736e3f19241ca` |
| `ping-pong/him-opponent-model@v2` | 6696224 | `a22b3b84eb12f602c2e18b9187e323d01cd11dfe2cea773d741ee01ccf574f10` |
| `ping-pong/him-mocap-strokes@v1` | 33351893 | `f3eed08c3706925f9cb55a0c22cd6bdd61c49f3ee2fcdb884bdf6353173e8b92` |
| `ping-pong/g1-retargeted-strokes@v2` | 77002986 | `da4a42ea0cb7043f6c75714232c6fd214bd75f2bc55a7991b8f5dcdb67f4b683` |

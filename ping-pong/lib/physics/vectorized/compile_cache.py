"""
A persistent cache for torch.compile's output (the Inductor FX-graph cache, with the Triton kernels it compiled), kept as
an Antioch asset so a new session does not recompile the ball physics' phases from scratch.

Each session starts with an empty cache directory (``TORCHINDUCTOR_CACHE_DIR``, on the session's own volume), so every
run pays the full compile: about 240 s for a robot-ball coupling's phases on an RTX PRO 6000. :func:`restore` unpacks a published cache into that directory before the phases compile, and
:func:`publish` packs it after a run that compiled.

Torch looks an entry up by the hash of the graph, its inputs' properties and the compiler's configuration, so a stale or
foreign entry is never used, only missed. A cache never changes a result. The asset's name carries the torch and CUDA
versions and the GPU's architecture, so one package serves one compiler and one device kind.
"""

from __future__ import annotations

import os
import re
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

ASSET_PREFIX: str = "ping-pong/compile-cache"
"""Assets are ``<prefix>-<torch>-<cuda>-<sm>`` (one per compiler and device kind), one version per publication."""

PARTS: tuple[str, ...] = ("fxgraph", "triton", "aotautograd")
"""The cache directory's parts worth keeping: the compiled graphs, their Triton kernels and the AOT autograd entries."""


def cache_dir() -> Path:
    """The Inductor cache directory torch uses here (``TORCHINDUCTOR_CACHE_DIR``, else torch's own default)."""

    configured = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if configured:
        return Path(configured)
    from torch._inductor.runtime.runtime_utils import cache_dir as inductor_cache_dir  # pyright: ignore[reportMissingImports, reportPrivateImportUsage]

    return Path(inductor_cache_dir())


def asset_name() -> str:
    """This compiler's and device's asset: ``<prefix>-torch<version>-cu<cuda>-sm<major><minor>``."""

    from lib.physics.vectorized._torch import torch

    torch_version = torch.__version__.split("+")[0]
    cuda = (torch.version.cuda or "cpu").replace(".", "")
    sm = "".join(str(v) for v in torch.cuda.get_device_capability()) if torch.cuda.is_available() else "none"
    return re.sub(r"[^a-z0-9/-]", "-", f"{ASSET_PREFIX}-torch{torch_version}-cu{cuda}-sm{sm}".lower())


def restore(name: str | None = None) -> dict[str, Any]:
    """
    Unpack the latest published cache for this compiler and device into :func:`cache_dir`. Entries already there are
    kept. Any failure (no asset yet, no network) is reported, never raised: the phases then compile as usual.

    :return: ``{"asset", "restored", "files", "bytes", "s"}``, or ``{"asset", "restored": False, "error"}``.
    """

    import antioch

    name = name or asset_name()
    began = time.perf_counter()
    try:
        source = Path(antioch.fetch_asset(name))
        target = cache_dir()
        target.mkdir(parents=True, exist_ok=True)
        archive = source if source.is_file() else next(source.rglob("*.tar"))
        with tarfile.open(archive) as tar:
            members = [member for member in tar.getmembers() if member.name.split("/", 1)[0] in PARTS]
            tar.extractall(target, members=members, filter="data")
        size = sum(member.size for member in members)
        files = sum(1 for member in members if member.isfile())
        return {"asset": name, "restored": True, "files": files, "bytes": size, "s": round(time.perf_counter() - began, 1)}
    except Exception as error:  # a missing cache is not a failure: the phases compile as usual
        return {"asset": name, "restored": False, "error": repr(error)[:300], "s": round(time.perf_counter() - began, 1)}


def publish(version: str, *, name: str | None = None, description: str = "") -> dict[str, Any]:
    """
    Pack :func:`cache_dir`'s :data:`PARTS` into one tar and save it as a new version of the asset. Any failure is
    reported, never raised.

    :param version: The new version (a run id, say).
    :return: ``{"asset", "version", "published", "bytes", "s"}``, or ``{..., "published": False, "error"}``.
    """

    import antioch

    name = name or asset_name()
    began = time.perf_counter()
    try:
        root = cache_dir()
        with tempfile.TemporaryDirectory() as scratch:
            archive = Path(scratch) / "compile-cache.tar"
            with tarfile.open(archive, "w") as tar:
                for part in PARTS:
                    if (root / part).exists():
                        tar.add(root / part, arcname=part)
            size = archive.stat().st_size
            antioch.save_asset(
                name,
                version=version,
                source=str(archive),
                description=description or f"torch.compile cache of the ball physics' phases ({name.rsplit('/', 1)[-1]})",
            )
        return {"asset": name, "version": version, "published": True, "bytes": size, "s": round(time.perf_counter() - began, 1)}
    except Exception as error:  # a failed publication only costs the next run its compile
        return {"asset": name, "version": version, "published": False, "error": repr(error)[:300], "s": round(time.perf_counter() - began, 1)}


__all__ = ["ASSET_PREFIX", "PARTS", "asset_name", "cache_dir", "publish", "restore"]

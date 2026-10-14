"""
torch, imported on first use.

The simulator's modules never import a kit library at module scope (the demo contract), and the demo's own environment
has no torch. ``torch`` here resolves each attribute from the real module the first time it is used and keeps it, so
importing :mod:`lib.physics.vectorized` costs nothing and needs no torch, and calls run at full speed.
"""

from __future__ import annotations

from typing import Any


class _LazyTorch:
    """The torch module, imported when an attribute is first read."""

    def __getattr__(self, name: str) -> Any:
        import torch as module  # pyright: ignore[reportMissingImports]

        value = getattr(module, name)
        setattr(self, name, value)
        return value


torch: Any = _LazyTorch()
"""Use as the torch module."""


def available() -> bool:
    """Whether torch can be imported here."""

    try:
        import torch as module  # pyright: ignore[reportMissingImports]  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = ["available", "torch"]

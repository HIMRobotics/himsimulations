"""
Render products: one Replicator render product and CPU ``rgb`` annotator per camera, read as uint8 frames.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from lib.him_lab import AuthoredCamera


class CameraRig:
    """
    One render product and CPU RGB annotator per authored camera.

    A product's annotator returns the frame rendered one step earlier, so a frame read after a step shows the state
    before it.

    :param cameras: Authored cameras keyed by name.
    """

    def __init__(self, cameras: Mapping[str, AuthoredCamera]) -> None:
        import omni.replicator.core as rep

        self.cameras = dict(cameras)
        self._products: dict[str, Any] = {}
        self._annotators: dict[str, Any] = {}
        for name, camera in self.cameras.items():
            product = rep.create.render_product(camera.prim_path, camera.resolution)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            annotator.attach(cast("str", product))
            self._products[name] = product
            self._annotators[name] = annotator

    def read(self) -> dict[str, NDArray[np.uint8] | None]:
        """Each camera's latest frame, ``HxWx3`` uint8, or ``None`` before its first full-size frame."""

        frames: dict[str, NDArray[np.uint8] | None] = {}
        for name, annotator in self._annotators.items():
            data = np.asarray(annotator.get_data())
            width, height = self.cameras[name].resolution
            frames[name] = np.ascontiguousarray(data[..., :3], dtype=np.uint8) if data.ndim == 3 and data.shape[:2] == (height, width) else None
        return frames

    def close(self) -> None:
        """Detach the annotators and destroy the render products."""

        for name, annotator in self._annotators.items():
            product = self._products[name]
            annotator.detach(product)
            product.destroy()
        self._annotators.clear()
        self._products.clear()


__all__ = ["CameraRig"]

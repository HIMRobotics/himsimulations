"""
Compiling and capturing pure batched functions: a pure function of fixed-shape tensors becomes one CUDA graph replay.

:func:`compiler` wraps ``torch.compile`` (size-generic kernels, a whole-graph trace) and falls back to eager mode once,
with a warning, if compilation fails (the Isaac Lab engine image ships no C compiler, which triton needs).
:class:`GraphedFunction` copies its inputs into fixed buffers, runs the function three times to compile and warm up,
captures it, and from then on replays it: no Python, no host sync, one launch.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Sequence
from typing import Any

from lib.physics.vectorized._torch import torch

Tensor = Any


def compiler(enabled: bool, dynamic: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """A decorator: ``torch.compile`` (when ``enabled``), eager from the first compile failure on."""

    if not enabled:
        return lambda fn: fn

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        compiled = torch.compile(fn, dynamic=dynamic, fullgraph=True)
        state = {"eager": False}

        def call(*args: Any) -> Any:
            if state["eager"]:
                return fn(*args)
            try:
                return compiled(*args)
            except Exception as error:
                from torch._dynamo.exc import BackendCompilerFailed  # pyright: ignore[reportMissingImports]
                from torch._inductor.exc import InductorError  # pyright: ignore[reportMissingImports]

                if not isinstance(error, (BackendCompilerFailed, InductorError)):
                    raise
                state["eager"] = True
                warnings.warn(f"torch.compile failed for {fn.__name__}; running it eagerly: {error}", RuntimeWarning, stacklevel=2)
                return fn(*args)

        return call

    return wrap


class GraphedFunction:
    """
    A pure function of fixed-shape tensors, replayed as one CUDA graph.

    :param fn: ``fn(*inputs) -> tensor or tuple of tensors``; it must not change its inputs.
    :param examples: Tensors with the inputs' shapes, dtypes and device (copied into the fixed buffers).
    :param graphs: Capture (default: on a CUDA device); without, ``fn`` simply runs.
    """

    def __init__(self, fn: Callable[..., Any], examples: Sequence[Tensor], *, graphs: bool | None = None) -> None:
        self.fn = fn
        self.inputs = [example.clone() for example in examples]
        cuda = bool(self.inputs) and self.inputs[0].is_cuda
        self.graphs = cuda if graphs is None else bool(graphs)
        self.graph: Any = None
        self.outputs: Any = None
        self.setup_s: tuple[float, float] | None = None
        """The first call's (compile and warm-up, capture) times, s."""

    def __call__(self, *inputs: Tensor) -> Any:
        for buffer, value in zip(self.inputs, inputs, strict=True):
            buffer.copy_(value)
        if not self.graphs:
            return self.fn(*self.inputs)
        if self.graph is None:
            began = time.perf_counter()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.fn(*self.inputs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            warmed = time.perf_counter()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.outputs = self.fn(*self.inputs)
            torch.cuda.synchronize()
            self.setup_s = (round(warmed - began, 3), round(time.perf_counter() - warmed, 3))
        self.graph.replay()
        return self.outputs


__all__ = ["GraphedFunction", "compiler"]

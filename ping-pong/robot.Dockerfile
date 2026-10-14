FROM antioch-engine/isaac-lab-3.0:0.5.338

ENV ANTIOCH_PROJECT_DIR=/workspace/project
WORKDIR /workspace/project
# gcc and libc6-dev: torch.compile (Triton) builds its kernel launcher with a C
# compiler, and the GPU ball physics (lib/physics/vectorized) compiles each
# phase of its step before capturing it as a CUDA graph. The engine image has
# no C compiler.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
# PyAV 19.0.1, as in the replay image: the robot's 24 fps film streams H.264.
RUN (command -v uv >/dev/null 2>&1 && uv pip install --system --no-cache "av==19.0.1") \
    || python -m pip install --no-cache-dir "av==19.0.1"
# Project files arrive through antioch.yaml's sync mapping before every
# command runs; they are not build inputs. Install dependencies above this
# line: they are build inputs, and a file this Dockerfile COPYs must have a
# `!path` line in .dockerignore to reach the build.
#
# Every other runtime dependency in pyproject.toml is pinned to the version
# this engine image already owns, so nothing else is installed.
# tests/test_contract.py keeps these install lines and this tag aligned with
# the SDK pin.

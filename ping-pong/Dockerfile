FROM antioch-engine/isaac-sim-6.0.1:0.5.338

ENV ANTIOCH_PROJECT_DIR=/workspace/project
WORKDIR /workspace/project
# PyAV 19.0.1: its manylinux wheel bundles FFmpeg with libx264 and h264_nvenc
# for the H.264 camera streams. The engine image ships no `av`.
RUN (command -v uv >/dev/null 2>&1 && uv pip install --system --no-cache "av==19.0.1") \
    || python -m pip install --no-cache-dir "av==19.0.1"
# Project files arrive through antioch.yaml's sync mapping before every
# command runs; they are not build inputs. Install dependencies above this
# line: they are build inputs, and a file this Dockerfile COPYs must have a
# `!path` line in .dockerignore to reach the build.
#
# Every other runtime dependency in pyproject.toml is pinned to the version
# this engine image already owns, so nothing else is installed (installing
# the lock would replace engine-owned packages). tests/test_contract.py keeps
# the pins, this install line and this tag aligned with the SDK pin.

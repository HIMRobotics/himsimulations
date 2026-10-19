"""
Container health probe: the encoder is healthy only after its warm-up encode.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

from compositor.protocol import DEFAULT_PORT, SERVICE_NAME


def main() -> int:
    """Return 0 when the service reports a ready encoder of the expected name."""

    port = int(os.environ.get("COMPOSITOR_PORT", DEFAULT_PORT))
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=8) as response:
            payload = json.loads(response.read().decode())
    except Exception as exc:
        print(f"compositor health request failed: {exc}", file=sys.stderr)
        return 1
    if payload.get("service") != SERVICE_NAME or not payload.get("ready"):
        print(f"compositor not ready: {payload}", file=sys.stderr)
        return 1
    print(f"compositor ready: ffmpeg {payload.get('ffmpeg_version')} pillow {payload.get('pillow_version')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

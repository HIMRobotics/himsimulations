"""
Check that every Antioch project in this repository is locked to HIM Robotics' organization.

Each project's ``antioch.yaml`` names its ``organization_id``; the Antioch CLI and SDK refuse every remote command
(runs, sessions, asset pushes) unless the signed-in organization matches it. This check fails if any project is
missing the lock or names another organization. It needs only the standard library::

    python tools/check_org_lock.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HIM_ORGANIZATION = "org_R94gKeCzTxrpN61i"
REPOSITORY = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "node_modules"}


def organization_ids(manifest: Path) -> list[str]:
    """Every top-level ``organization_id`` the manifest sets."""

    return re.findall(r"^organization_id:\s*['\"]?([A-Za-z0-9_]+)['\"]?\s*(?:#.*)?$", manifest.read_text(encoding="utf-8"), flags=re.MULTILINE)


def main() -> int:
    manifests = [path for path in sorted(REPOSITORY.rglob("antioch.yaml")) if not SKIP.intersection(path.relative_to(REPOSITORY).parts)]
    if not manifests:
        print("no antioch.yaml found")
        return 1
    failures = 0
    for manifest in manifests:
        found = organization_ids(manifest)
        relative = manifest.relative_to(REPOSITORY)
        if found == [HIM_ORGANIZATION]:
            print(f"ok    {relative}")
        else:
            failures += 1
            print(f"FAIL  {relative}: organization_id {found or 'missing'}, expected {HIM_ORGANIZATION} (HIM Robotics)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

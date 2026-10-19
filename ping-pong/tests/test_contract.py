"""
The project's shape: the organization lock, the scenarios and suites, the engine pins and the import rules.

These tests read files, parse Python with ``ast`` and run the CLI's local ``collect``; none starts a simulator, a
service or needs a GPU.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from lib.assets import ASSETS
from lib.config import PLAY_PRESETS, REPLAY_PRESETS, RETARGET_PRESETS, TEMPLATE_PRESETS

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent

HIM_ORGANIZATION = "org_R94gKeCzTxrpN61i"
SDK_VERSION = "0.5.338"

ENGINE_IMAGES = {"replay": ("Dockerfile", "isaac-sim-6.0.1"), "robot": ("robot.Dockerfile", "isaac-lab-3.0")}
"""Each engine service's Dockerfile and the engine image it must start FROM."""

ENGINE_OWNED_VERSIONS = {"click": "8.1.7", "numpy": "2.3.1", "rerun-sdk": "0.38.1", "scipy": "1.17.0"}
"""Runtime dependencies both engine images already own at these exact versions; any other runtime dependency needs
its own install line in the engine Dockerfiles, and this map changes with it."""

ENGINE_IMAGE_INSTALLS = {"av": "19.0.1"}
"""What each engine Dockerfile installs itself: PyAV, for the H.264 camera streams."""

HOST_ONLY_PINS = {"pydantic": "2.13.5"}
"""The CLI digests submissions with ``Field(exclude_if=...)``, which pydantic < 2.12 ignores; the engines keep 2.11.10."""

SCENARIOS = {
    "replay_him_session": ("replay", set(REPLAY_PRESETS)),
    "play_ping_pong_policy": ("robot", set(PLAY_PRESETS)),
    "load_humanoid_at_table": ("robot", set(TEMPLATE_PRESETS)),
    "retarget_him_strokes": ("robot", set(RETARGET_PRESETS)),
}
"""Every scenario, its service and profile, and its cases (one per preset)."""

SUITES = {
    "replay-smoke": {("replay_him_session", "session-2-first-point"), ("replay_him_session", "nc-dead-table")},
    "robot-smoke": {("play_ping_pong_policy", "g1-rally-smoke"), ("load_humanoid_at_table", "unitree-g1"), ("retarget_him_strokes", "unitree-g1-smoke")},
    "replay-evidence": {("replay_him_session", "session-2")},
    "robot-evidence": {("play_ping_pong_policy", "g1-rally")},
}

SIMULATOR_ROOTS = ("pxr", "omni", "carb", "isaacsim", "isaaclab", "isaaclab_assets", "warp", "torch", "rsl_rl")
"""Modules that exist only inside an engine image (or are heavy): never imported at module scope in simulation code."""

CUSTOMER_FORBIDDEN = (
    re.compile(r"org_(?!R94gKeCzTxrpN61i)[A-Za-z0-9]{16}"),
    re.compile(r"antioch[-]demos"),
    re.compile(r"linear\.app"),
    re.compile(r"\bENG-\d+\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@antioch\.com"),
    re.compile(r"console\.(preview\.)?antioch\.com"),
    re.compile(r"/home/[a-z]"),
    re.compile(r"\breports/[a-z0-9_-]+\.md"),
)
"""Text that must never ship to HIM: another organization's id, Antioch-internal links, trackers and reports, emails,
local paths."""


def _manifest() -> dict:
    return yaml.safe_load((PROJECT / "antioch.yaml").read_text(encoding="utf-8"))


def _collect(*args: str) -> Any:
    command = [str(Path(sys.executable).with_name("antioch")), *args, "--json"]
    out = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=240, check=True)
    return json.loads(out.stdout)


def test_the_project_is_locked_to_him_robotics() -> None:
    """Every remote command refuses unless the signed-in organization is HIM Robotics."""

    assert _manifest()["organization_id"] == HIM_ORGANIZATION
    out = subprocess.run([sys.executable, str(REPOSITORY / "tools" / "check_org_lock.py")], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stdout + out.stderr


def test_collect_finds_exactly_the_four_scenarios() -> None:
    runs = _collect("scenario", "collect", "--all-cases")["runs"]
    found: dict[str, set[str]] = {}
    for run in runs:
        service, _cases = SCENARIOS[run["scenario"]]
        assert run["service"] == service and run["profile"] == service, run
        assert run["capture"] is False, run
        found.setdefault(run["scenario"], set()).add(run["case"])
    assert found == {name: cases for name, (_service, cases) in SCENARIOS.items()}


def test_collect_finds_exactly_the_four_suites_each_on_one_engine() -> None:
    suites = {suite["name"]: suite for suite in _collect("suite", "collect")}
    assert set(suites) == set(SUITES)
    for name, suite in suites.items():
        assert {(run["scenario"], run["case"]) for run in suite["runs"]} == SUITES[name], name
        assert len({run["profile"] for run in suite["runs"]}) == 1, f"{name} mixes engines"


def test_each_engine_service_has_its_own_profile() -> None:
    services = _manifest()["services"]
    for name, (dockerfile, _engine) in ENGINE_IMAGES.items():
        assert services[name]["profiles"] == [name]
        assert services[name]["build"]["dockerfile"] == dockerfile
    assert services["compositor"]["profiles"] == ["replay"]


@pytest.mark.parametrize("service", sorted(ENGINE_IMAGES))
def test_engine_images_match_the_sdk_pin(service: str) -> None:
    dockerfile, engine = ENGINE_IMAGES[service]
    text = (PROJECT / dockerfile).read_text(encoding="utf-8")
    assert re.findall(r"^FROM\s+(\S+)", text, flags=re.MULTILINE) == [f"antioch-engine/{engine}:{SDK_VERSION}"]
    for name, version in ENGINE_IMAGE_INSTALLS.items():
        assert f'"{name}=={version}"' in text, f"{dockerfile} must install {name}=={version}"


def test_runtime_pins_equal_what_the_engines_own() -> None:
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pins: dict[str, str] = {}
    for spec in project["dependencies"]:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^]]*\])?==(\S+)", spec)
        assert match is not None, f"{spec} must pin one exact version"
        pins[match.group(1)] = match.group(2)
    assert pins.pop("antioch-sim") == SDK_VERSION
    assert pins == ENGINE_OWNED_VERSIONS | ENGINE_IMAGE_INSTALLS | HOST_ONLY_PINS


def _simulation_sources() -> list[Path]:
    return [PROJECT / "main.py", PROJECT / "scenarios.py", *sorted((PROJECT / "lib").rglob("*.py"))]


def _module_scope_imports(path: Path) -> list[str]:
    """Modules imported at module scope, outside ``if TYPE_CHECKING:`` blocks."""

    found: list[str] = []

    def visit(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.Import):
                found.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                found.append(node.module or "")
            elif isinstance(node, ast.If):
                test = node.test
                if not (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"):
                    visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Try):
                visit(node.body)
                for handler in node.handlers:
                    visit(handler.body)

    visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body)
    return found


def test_simulator_modules_are_imported_lazily() -> None:
    """Every module imports without Isaac, so ``collect``, ``--help`` and the unit tests run anywhere."""

    offenders = [
        f"{path.relative_to(PROJECT)}: {module}"
        for path in _simulation_sources()
        for module in _module_scope_imports(path)
        if module.split(".")[0] in SIMULATOR_ROOTS
    ]
    assert not offenders, offenders


def test_every_asset_is_in_the_ping_pong_store_and_presets_name_only_pinned_assets() -> None:
    refs = {asset.ref for asset in ASSETS.values()}
    assert all(name.startswith("ping-pong/") for name in ASSETS)
    for params in REPLAY_PRESETS.values():
        assert params.package in refs
    for params in PLAY_PRESETS.values():
        assert params.policy in refs
    for params in RETARGET_PRESETS.values():
        assert f"asset:{params.source}@{params.source_version}" in refs


def _shipped_text_files() -> list[Path]:
    skip = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "runs", "outputs"}
    files = [path for path in PROJECT.rglob("*") if path.is_file() and not skip.intersection(path.relative_to(PROJECT).parts)]
    files += [REPOSITORY / "README.md", *(REPOSITORY / "tools").glob("*.py"), *(REPOSITORY / ".github").rglob("*.yml")]
    return [path for path in files if path.suffix in {".py", ".md", ".yaml", ".yml", ".toml", ".json", ".ipynb", ".txt", ""} and path.name != "uv.lock"]


def test_nothing_internal_ships_to_the_customer() -> None:
    offenders = []
    for path in _shipped_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        offenders += [f"{path.relative_to(REPOSITORY)}: {match.group(0)}" for pattern in CUSTOMER_FORBIDDEN for match in pattern.finditer(text)]
    assert not offenders, offenders


def test_docs_list_every_pinned_asset() -> None:
    """``docs/assets.md`` names every asset and its pinned sha256, so the documentation never drifts from the pins."""

    text = (PROJECT / "docs" / "assets.md").read_text(encoding="utf-8")
    for asset in ASSETS.values():
        assert f"`{asset.name}@{asset.version}`" in text, asset.name
        if asset.sha256 is not None:
            assert asset.sha256 in text, f"{asset.name}: docs/assets.md lacks sha256 {asset.sha256}"

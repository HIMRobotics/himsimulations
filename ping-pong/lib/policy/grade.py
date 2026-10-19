"""
The trained policy's grade: what one pass of rallies measured (:func:`rally_metrics`), and the run's verdict
(:func:`grade`).

**The headline is legal returns per receive.** A receive is a ball the opponent sent that is bound for the robot's
half. A legal return is the driver's call: the robot hit it, and its first table contact landed on the opponent's half
(net touches allowed). Rallies cut by a reset (a fall) and rallies still in flight when the pass ends are left out.

**Rally length** is the legal shots of both players in a rally. It is bounded by the opponent as much as by the robot:
the distilled opponent returns about nine in ten of the robot's legal balls.

Intervals are Wilson 95 %, which are too narrow: receives within a rally are not independent.

The policy's own numbers are reported, never gated: a policy is judged on its evidence, and your own policy will read
differently. The run's checks are the harness's and the physics': the blind control collapses, PhysX keeps headroom,
both passes run their full span, and the film holds every frame.
"""

from __future__ import annotations

import math
from typing import Any

SOURCE_LEGAL_PER_RECEIVE: tuple[float, float] = (0.958, 0.969)
"""Reported beside the reading, never gated: the trained policy's legal returns per receive in its source grades against
this opponent, lowest to highest. 0.958 on the opponent's feeds as single receives under nominal physics; 0.962 and
0.963 in 30 s of rallies (two seeds) and 0.969 over a 120 s pass, under per-rally randomised physics."""

BLIND_LEGAL_MAX: float = 0.02
"""The blind control (the ball and its forecast hidden from the policy) must return at most this share legally: a
policy that plays from the ball collapses without it (the source's blind control read 0.000)."""


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval of a binomial proportion (``(nan, nan)`` with no trials)."""

    if trials <= 0:
        return (math.nan, math.nan)
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return (centre - half, centre + half)


def rally_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """
    One pass's metrics from :meth:`lib.policy.rally.CoupledRally.play`'s record: the robot's receives, hits and legal
    returns (with the rate's interval), the rally length in legal shots, why rallies ended, and falls per 10 s.
    """

    import torch

    from lib.policy.rally import OPPONENT, OVER_REASONS, ROBOT

    completed = record["completed"]
    falls_per_10s = 10.0 * record["falls"] / max(record["env_seconds"], 1e-9)
    if not completed:
        return {"rallies": 0, "receives": 0, "legal_per_receive": None, "falls_per_10s": falls_per_10s}
    cat = {key: torch.cat([part[key] for part in completed]).cpu() for key in completed[0]}
    whole = cat["over_reason"] != 7
    receives = int(cat["robot_receives"][whole].sum())
    hits = int(cat["hits"][whole, ROBOT].sum())
    legal = int(cat["legal"][whole, ROBOT].sum())
    length = (cat["legal"][whole, ROBOT] + cat["legal"][whole, OPPONENT]).double()
    reasons = cat["over_reason"][whole]
    return {
        "rallies": int(whole.sum()),
        "rallies_cut_by_a_fall": int((~whole).sum()),
        "receives": receives,
        "hits": hits,
        "legal": legal,
        "legal_per_receive": legal / receives if receives else None,
        "legal_per_receive_ci95": list(wilson(legal, receives)) if receives else None,
        "hit_rate": hits / receives if receives else None,
        "legal_given_hit": legal / hits if hits else None,
        "opponent_legal_on_legal_balls": int(cat["legal"][whole, OPPONENT].sum()) / legal if legal else None,
        "rally_length_legal_shots": {
            "mean": float(length.mean()) if length.numel() else None,
            "p50": float(length.median()) if length.numel() else None,
            "max": int(length.max()) if length.numel() else None,
        },
        "ended_by": {OVER_REASONS.get(int(code), str(int(code))): int((reasons == code).sum()) for code in torch.unique(reasons).tolist()},
        "falls_per_10s": falls_per_10s,
        "pass_s": record["seconds"],
    }


def grade(
    run: Any,
    played: dict[str, Any],
    blind: dict[str, Any],
    *,
    spans_s: tuple[float, float],
    step_s: float,
    physx: dict[str, Any],
    film: Any = None,
    film_account: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Both passes' metrics, recorded on the run, and its checks (each recorded with :meth:`antioch.ScenarioRun.check`):

    - **the blind control collapses** (:data:`BLIND_LEGAL_MAX`): the policy plays from the ball;
    - **PhysX kept headroom** in every GPU buffer, and reported no overflow;
    - **both passes ran their span** (``spans_s``, the policy's and the control's, to within a policy step ``step_s``);
    - **the film holds every 1/24 s** of the policy's pass.

    The policy's legal returns per receive are reported with their interval, beside :data:`SOURCE_LEGAL_PER_RECEIVE`.

    :return: Every metric of both passes and the verdict, ``passed``.
    """

    from lib.robots.table_scene import physx_ok

    policy, control = rally_metrics(played), rally_metrics(blind)
    ran = (played["seconds"], blind["seconds"])
    checks = [
        run.check(
            "blind control collapses",
            control["receives"] > 0 and (control["legal_per_receive"] or 0.0) <= BLIND_LEGAL_MAX,
            detail=f"{_fmt(control['legal_per_receive'])} legal per receive over {control['receives']} receives, at most {BLIND_LEGAL_MAX}",
        ),
        run.check(
            "PhysX buffers not full",
            physx_ok(physx),
            detail=f"full {physx['full']}, {physx['overflow_errors']} overflow errors, {physx['errors']} PhysX errors",
        ),
        run.check(
            "both passes ran their span",
            all(abs(done - asked) < 0.5 * step_s for done, asked in zip(ran, spans_s, strict=True)),
            detail=f"policy {ran[0]:.2f} of {spans_s[0]:g} s, blind control {ran[1]:.2f} of {spans_s[1]:g} s",
        ),
    ]
    if film is not None and film_account is not None:
        ok, detail = film.at_24_fps(film_account, played["seconds"])
        checks.append(run.check("film at 24 fps", ok, detail=detail))
    low, high = SOURCE_LEGAL_PER_RECEIVE
    reading = f"{_fmt(policy['legal_per_receive'])} legal per receive {_interval(policy)} over {policy['receives']} receives"
    summary = {
        "passed": all(checks),
        "policy": policy,
        "blind_control": control,
        "source_legal_per_receive": [low, high],
        "physx_buffer_use": {k: round(v["use"], 4) for k, v in physx["buffers"].items()},
    }
    run.add_results(
        {
            "legal_per_receive": reading + f" (the source's grades read {low:.3f}-{high:.3f})",
            "policy": policy,
            "blind_control": control,
            "physx_buffer_use": summary["physx_buffer_use"],
        }
    )
    return summary


def _interval(metrics: dict[str, Any]) -> str:
    ci = metrics.get("legal_per_receive_ci95")
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "[n/a]"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


__all__ = ["BLIND_LEGAL_MAX", "SOURCE_LEGAL_PER_RECEIVE", "grade", "rally_metrics", "wilson"]

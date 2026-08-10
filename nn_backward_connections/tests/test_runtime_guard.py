"""Tests for the run-level wall-clock guard (queue item P05).

Two kinds of test live here.

*Behavioural* — ``resolve`` picks the right deadline and ``summarise`` tells a clean run
from a truncated one. The subtle case is stage 3: the sift stops early both when the clock
runs out (truncation) and when it reaches a true fixed point (convergence), and calling the
second one "degraded" would flag every mouse run forever.

*Sizing pins* — the guard is only safe if the deadline sits strictly between "what the
champion actually takes" and "what the cap allows". Both margins are asserted against the
measured numbers, so a later edit to ``reserve_s`` or ``max_wall_clock_s_per_run`` that
would silently start truncating microns fails here instead of in a 57-minute GPU run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("yaml", reason="the runtime guard reads campaign.yaml")

from eval import runtime_guard  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# resolve()
# ──────────────────────────────────────────────────────────────────────────────
def test_explicit_time_limit_wins():
    """The guard fills a hole; it never overrides an instruction."""
    plan = runtime_guard.resolve("connectome", explicit=123.0)
    assert plan["time_limit_s"] == 123.0
    assert plan["source"] == "explicit"


def test_resolves_from_campaign_yaml():
    plan = runtime_guard.resolve("microns")
    assert plan["source"] == "campaign", "the guard must be enabled in campaign.yaml"
    assert plan["cap_s"] == 3600.0
    assert plan["time_limit_s"] == plan["cap_s"] - plan["reserve_s"]
    assert 0 < plan["time_limit_s"] < plan["cap_s"]


def test_unreadable_config_falls_back_to_unguarded_but_says_so(tmp_path):
    """Running unguarded is the pre-P05 status quo, so it is a safe fallback —
    but it must never look like a guarded run."""
    plan = runtime_guard.resolve("mouse", path=tmp_path / "nope.yaml")
    assert plan["time_limit_s"] is None
    assert plan["source"] == "unavailable"
    assert runtime_guard.summarise(plan, wall_clock_s=1.0)["armed"] is False


def test_guard_can_be_disabled(tmp_path):
    cfg = tmp_path / "campaign.yaml"
    cfg.write_text("runtime:\n  max_wall_clock_s_per_run: 3600\n  guard:\n    enabled: false\n")
    plan = runtime_guard.resolve("mouse", path=cfg)
    assert plan["source"] == "disabled"
    assert plan["time_limit_s"] is None


def test_per_dataset_reserve_override(tmp_path):
    cfg = tmp_path / "campaign.yaml"
    cfg.write_text("runtime:\n"
                   "  max_wall_clock_s_per_run: 1000\n"
                   "  guard:\n"
                   "    enabled: true\n"
                   "    reserve_s:\n"
                   "      microns: 200\n"
                   "      default: 50\n")
    assert runtime_guard.resolve("microns", path=cfg)["time_limit_s"] == 800.0
    assert runtime_guard.resolve("connectome", path=cfg)["time_limit_s"] == 950.0


# ──────────────────────────────────────────────────────────────────────────────
# summarise()
# ──────────────────────────────────────────────────────────────────────────────
def _plan(limit=3450.0, cap=3600.0, warn=3400.0):
    return dict(time_limit_s=limit, cap_s=cap, warn_s=warn, reserve_s=cap - limit,
                source="campaign")


def test_clean_run_is_not_degraded():
    attrs = dict(n_alt_cycles=77, alt_cycles_requested=77,
                 n_sift_sweeps=40, sift_sweeps_requested=40, sift_converged=False)
    g = runtime_guard.summarise(_plan(), wall_clock_s=1238.0, variant_wall_s=1217.0,
                                attrs=attrs)
    assert g["armed"] and not g["degraded"]
    assert not g["deadline_reached"] and g["stages_truncated"] == {}
    assert not g["over_cap"] and not g["over_warn"]


def test_stage4_truncation_is_detected_exactly():
    """alternate_scc_sift's loop breaks on the clock and nothing else, so a short
    cycle count is unambiguous evidence the guard bound."""
    attrs = dict(n_alt_cycles=3, alt_cycles_requested=5)
    g = runtime_guard.summarise(_plan(), wall_clock_s=3500.0, variant_wall_s=3450.0,
                                attrs=attrs)
    assert g["degraded"]
    assert g["stages_truncated"]["stage4_alternation"] == dict(done=3, requested=5)
    assert g["deadline_reached"]


def test_converged_sift_is_not_truncation():
    """The false positive that would otherwise fire on every mouse run: the sift stops
    at a fixed point long before max_sweeps, and that is success, not degradation."""
    attrs = dict(n_sift_sweeps=9, sift_sweeps_requested=40, sift_converged=True)
    g = runtime_guard.summarise(_plan(), wall_clock_s=6.0, variant_wall_s=5.0, attrs=attrs)
    assert g["stages_truncated"] == {} and not g["degraded"]


def test_unconverged_short_sift_is_truncation():
    attrs = dict(n_sift_sweeps=9, sift_sweeps_requested=40, sift_converged=False)
    g = runtime_guard.summarise(_plan(), wall_clock_s=6.0, variant_wall_s=5.0, attrs=attrs)
    assert g["stages_truncated"]["stage3_sift"] == dict(done=9, requested=40)
    assert g["degraded"]


def test_stage_without_a_convergence_flag_is_not_guessed_at():
    attrs = dict(n_sift_sweeps=9, sift_sweeps_requested=40)     # no sift_converged
    g = runtime_guard.summarise(_plan(), wall_clock_s=6.0, variant_wall_s=5.0, attrs=attrs)
    assert g["stages_truncated"] == {}


def test_over_cap_is_flagged_against_the_record_wall_clock():
    """The 3600 s rule governs the RECORD's wall clock, not the variant's internal one."""
    g = runtime_guard.summarise(_plan(), wall_clock_s=3700.0, variant_wall_s=3400.0)
    assert g["over_cap"] and g["over_warn"]


# ──────────────────────────────────────────────────────────────────────────────
# Sizing pins — the deadline must not bind on the champion, and must fit the cap
# ──────────────────────────────────────────────────────────────────────────────
def _overheads() -> dict:
    p = _ROOT / "experiments" / "outputs" / "proto_P05.json"
    if not p.exists():
        pytest.skip("experiments/outputs/proto_P05.json not present")
    return {r["dataset"]: r["unguarded_overhead_s"]
            for r in json.loads(p.read_text())["rows"]}


def _champion_walls() -> dict:
    sota = json.loads((_ROOT / "autoresearch" / "sota.json").read_text())
    return {ds: float(v["wall_clock_s_approx"])
            for ds, v in sota["datasets"].items() if v.get("wall_clock_s_approx")}


@pytest.mark.parametrize("ds", ["connectome", "microns", "mouse"])
def test_deadline_does_not_bind_on_the_champion(ds):
    """The guard must be an emergency brake, not a new sizing rule.

    The champion's variant-internal time is its recorded wall clock minus the prologue and
    epilogue that sit outside the deadline. That must stay strictly under the deadline, or
    every confirmed number in sota.json stops reproducing.
    """
    plan = runtime_guard.resolve(ds)
    internal = _champion_walls()[ds] - _overheads()[ds]
    assert internal < plan["time_limit_s"], (
        f"{ds}: champion needs {internal:.0f}s of variant-internal time but the guard "
        f"would cut it at {plan['time_limit_s']:.0f}s — the guard would BIND on the "
        f"champion itself and sota.json would stop reproducing")


@pytest.mark.parametrize("ds", ["connectome", "microns", "mouse"])
def test_worst_case_run_fits_under_the_cap(ds):
    """Deadline + the overhead it cannot see must leave room for the abort granularity.

    A stage only notices the deadline at its own boundary, so the run can overshoot by one
    stage-3 sweep plus one stage-4 cycle. 78 s covers both on the slowest dataset.
    """
    plan = runtime_guard.resolve(ds)
    granularity_s = 78.0
    worst = plan["time_limit_s"] + _overheads()[ds] + granularity_s
    assert worst <= plan["cap_s"], (
        f"{ds}: worst-case run {worst:.0f}s exceeds the {plan['cap_s']:.0f}s cap")

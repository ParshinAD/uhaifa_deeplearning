"""Run-level wall-clock guard (queue item P05) — arm the deadline the variants already honour.

The defect
----------
Every stage of the champion pipeline *already* accepts a wall-clock budget and stops
cleanly at a safe boundary: :func:`mfas.baseline.rocket.run_rocket` checks ``time_limit``
each epoch, :func:`mfas.refine.underrelax.sift_underrelaxed` checks ``time_budget_s`` each
sweep, and :func:`mfas.refine.scc_recursive.alternate_scc_sift` checks it each alternation
cycle and passes the remainder down to its inner sift. The machinery is complete.

It was simply **never armed**. ``autoresearch/sweep.sh`` invokes ``eval/run_variant.py``
without ``--time-limit``, so ``time_limit=None`` reaches ``variant.run(...)``, and every one
of those checks is a no-op (``H42.run`` even derives its stage budgets as
``None if time_limit is None else ...``). A run therefore has no upper bound on its wall
clock at all — it stops when the algorithm's fixed cycle counts are exhausted, whenever
that happens to be.

That was tolerable while the champion sat at 3314 s of a 3600 s cap. H42 spent the margin:
microns now runs 3398-3418 s on 5/5 confirm runs, i.e. **182 s (5.1%) under the cap**. A
loaded machine, a thermal throttle or a background job does not slow such a run down
gracefully — it pushes it past the cap, and the run is then inadmissible after burning
~57 minutes of GPU time.

What this module does
---------------------
Resolves the deadline the runner hands to ``variant.run(...)`` when the caller gave none::

    time_limit = runtime.max_wall_clock_s_per_run - runtime.guard.reserve_s

An explicit ``--time-limit`` always wins; the guard only fills a hole.

Why ``reserve_s``, and why it is a measurement
----------------------------------------------
The deadline the variant enforces and the wall clock the cap governs are **not the same
clock**. ``results/*.json`` ``wall_clock_s`` is measured by the runner around the whole
``variant.run(...)`` call; a variant's ``time_limit`` is consumed from the start of
``run_rocket`` — after ``greedy_fas_order`` (stage 0) has already run — and the runner then
re-scores and saves after the call returns. Both ends sit outside the guard.

``experiments/outputs/proto_P05.json`` measures them (2026-08-11, this machine)::

    dataset      prologue (greedy-FAS + init)   epilogue (re-score + save)   total outside
    mouse                    0.09 s                      0.001 s               0.10 s
    microns                 35.88 s                      0.156 s              36.04 s
    connectome              20.46 s                      0.067 s              20.53 s

plus the **abort granularity**: a stage only notices the deadline at its own boundary, so a
run may overshoot by at most one stage-3 sweep plus one stage-4 alternation cycle.

``reserve_s = 150`` covers all three terms with room to spare, giving a deadline of
**3450 s** and a worst case of ``3450 + 36 (overhead) + ~60 (one sift sweep) + ~18 (one alt
cycle) = ~3564 s`` — inside the 3600 s cap.

The determinism constraint (and the honest cost)
------------------------------------------------
H42 established that stage 4 must **not** be *sized* by wall-clock: a time-derived cycle
count depends on machine load, and that would destroy the bit-reproducibility that
``sota.json`` (std = 0) and the P02 one-seed screen policy both rest on. This guard does not
size anything. The cycle counts stay the fixed constants they are; the deadline is an
**abort**, and on an idle machine it never fires, so every confirmed number is reproduced
bit-for-bit (verified — see ``experiments/log.md`` P05).

What it cannot do is make that free. microns' variant-internal time is 3382 s against a
3450 s deadline: **68 s, or 2.0%, of slack**. A machine more than 2% slower than this one
*will* trip the guard on microns, and the run will then be a truncated one — a couple of
stage-4 cycles short, therefore slightly lower-scoring and **not** comparable to a clean
run. That is the trade this item is making deliberately: an unbounded cap overrun becomes a
bounded, *recorded* degradation. It is recorded precisely so no future cycle pools a
truncated run into a mean without noticing (see :func:`summarise` and ``audit.py``'s
``runtime_guard`` check). Widening that 2% is not a guard problem — it needs microns to get
cheaper, which is Phase 2 / queue item S01.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["resolve", "summarise", "DEFAULT_RESERVE_S", "campaign_path"]

# Fallback reserve if campaign.yaml carries no explicit value. See the module docstring:
# 36 s measured overhead + ~78 s worst-case abort granularity + margin.
DEFAULT_RESERVE_S = 150.0

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def campaign_path() -> Path:
    """Path to the campaign config that supplies the cap."""
    return _ROOT / "autoresearch" / "campaign.yaml"


def _load_runtime_cfg(path: Optional[Path] = None) -> Dict[str, Any]:
    """Return ``campaign.yaml``'s ``runtime`` block, or ``{}`` if unreadable.

    Deliberately tolerant. An unreadable config must not stop a run: running unguarded is
    exactly the status quo this item is replacing, so falling back to it is not a
    regression — but the fallback is recorded (``source='unavailable'``) rather than
    silently indistinguishable from a guarded run.
    """
    p = campaign_path() if path is None else Path(path)
    try:
        import yaml
        cfg = yaml.safe_load(p.read_text()) or {}
        return dict(cfg.get("runtime") or {})
    except Exception:
        return {}


def resolve(dataset: str, explicit: Optional[float] = None,
            path: Optional[Path] = None) -> Dict[str, Any]:
    """Decide the ``time_limit`` to hand ``variant.run(...)``.

    Parameters
    ----------
    dataset : str
        Dataset name. Only used to look up a per-dataset ``reserve_s`` override; the cap
        itself is global.
    explicit : float, optional
        A ``--time-limit`` supplied by the caller. Always wins — the guard fills a hole,
        it does not override an instruction.
    path : Path, optional
        Alternate campaign.yaml (tests).

    Returns
    -------
    dict with ``time_limit_s`` (what to pass to the variant, may be ``None``), ``source``
    (``explicit`` | ``campaign`` | ``disabled`` | ``unavailable``), ``cap_s``,
    ``warn_s`` and ``reserve_s``.
    """
    rt = _load_runtime_cfg(path)
    cap = rt.get("max_wall_clock_s_per_run")
    warn = rt.get("warn_wall_clock_s_per_run")
    guard = dict(rt.get("guard") or {})

    base = dict(cap_s=float(cap) if cap else None,
                warn_s=float(warn) if warn else None,
                reserve_s=None)

    if explicit is not None:
        return dict(base, time_limit_s=float(explicit), source="explicit")

    if not rt or cap is None:
        return dict(base, time_limit_s=None, source="unavailable")

    if not guard.get("enabled", False):
        return dict(base, time_limit_s=None, source="disabled")

    reserve_cfg = guard.get("reserve_s")
    if isinstance(reserve_cfg, dict):
        reserve = reserve_cfg.get(dataset, reserve_cfg.get("default", DEFAULT_RESERVE_S))
    elif reserve_cfg is None:
        reserve = DEFAULT_RESERVE_S
    else:
        reserve = reserve_cfg

    reserve = float(reserve)
    return dict(base, reserve_s=reserve,
                time_limit_s=max(0.0, float(cap) - reserve), source="campaign")


def summarise(plan: Dict[str, Any], wall_clock_s: float,
              variant_wall_s: Optional[float] = None,
              attrs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the ``runtime_guard`` block for a run record.

    The point of this block is that **a truncated run must never be silently pooled with
    clean ones**. Two independent signals are recorded, because neither alone is sound:

    ``deadline_reached``
        Generic and variant-agnostic: the variant's own clock reached the deadline. True
        whenever some stage aborted on time, whatever the variant is.
    ``stages_truncated``
        Specific and exact, where the variant reports it: a stage that ran fewer
        iterations than it was configured for (``*_requested`` vs the count actually
        done). This is the measurement; ``deadline_reached`` is the tripwire that still
        works for a variant which reports nothing.

    "Fewer iterations than configured" is only evidence of *truncation* for a stage whose
    sole early exit is the clock. ``alternate_scc_sift``'s loop breaks on nothing else, so
    its count is exact. ``sift_underrelaxed`` also stops at a true fixed point (no node
    wants to move), which is convergence, not truncation — so a stage with a
    ``<stage>_converged`` flag is only reported when that flag is explicitly ``False``,
    and a stage that reports no flag at all is not guessed at.

    ``over_cap`` is the failure this whole item exists to prevent, and is checked against
    the RECORD's wall clock — the one the 3600 s rule actually governs.
    """
    attrs = attrs or {}
    cap = plan.get("cap_s")
    limit = plan.get("time_limit_s")

    truncated = {}
    for done_key, req_key, conv_key, name in (
        # stage 4: the alternation loop's ONLY break is the time budget -> count is exact.
        ("n_alt_cycles", "alt_cycles_requested", None, "stage4_alternation"),
        # stage 3: also exits at a fixed point, so it needs the convergence flag.
        ("n_sift_sweeps", "sift_sweeps_requested", "sift_converged", "stage3_sift"),
    ):
        done, req = attrs.get(done_key), attrs.get(req_key)
        if done is None or req is None or int(done) >= int(req):
            continue
        if conv_key is not None and attrs.get(conv_key) is not False:
            continue        # converged, or the variant does not say — do not guess
        truncated[name] = dict(done=int(done), requested=int(req))

    deadline_reached = (
        limit is not None and variant_wall_s is not None and variant_wall_s >= float(limit))

    return dict(
        armed=plan.get("source") in ("campaign", "explicit"),
        source=plan.get("source"),
        time_limit_s=limit,
        reserve_s=plan.get("reserve_s"),
        cap_s=cap,
        warn_s=plan.get("warn_s"),
        wall_clock_s=float(wall_clock_s),
        variant_wall_s=None if variant_wall_s is None else float(variant_wall_s),
        # Everything below is what makes a degraded run visible to audit.py.
        deadline_reached=bool(deadline_reached),
        stages_truncated=truncated,
        degraded=bool(deadline_reached or truncated),
        over_warn=bool(plan.get("warn_s") and wall_clock_s > float(plan["warn_s"])),
        over_cap=bool(cap and wall_clock_s > float(cap)),
    )

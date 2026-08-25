"""H56 prototype (rung 2) - is relabelling multi-start a mechanism, or just R times the compute?

The hypothesis (queue item H56)
-------------------------------
Running the champion pipeline on R randomly relabelled copies of the graph and keeping the
best-scoring order by the oracle beats a single canonical-labelling run, because relabelling is
the only source of score dispersion this deterministic pipeline has (P02: seeds are inert).

That premise is TRUE and was measured in cycle 9: ``experiments/outputs/proto_P09_connectome.json``
puts the connectome champion H42 over 84.11418066-84.15550043 across five labellings, sigma
0.019124 pp. H56 proposes to harvest it with best-of-R.

Why this gate is arithmetic and not GPU hours
---------------------------------------------
Everything H56 needs in order to be decided already exists as logged measurement:

  * the labelling distribution of the champion       -> proto_P09_connectome.json (5 paired points)
  * the labelling distribution on mouse              -> proto_P09_mouse.json      (25 paired points)
  * the per-run wall clock of the champion           -> the same files + sota.json
  * what a SINGLE run buys when given 2x the stage-4 seconds, i.e. the compute-matched
    control CAMPAIGN.md demands of every multi-start idea
                                                     -> proto_H42.json, sweeps=2 arm, 143 cycles
                                                        of (cum_wall_s, best_pct) on connectome

So this prototype re-analyses measurements rather than producing new large-graph ones. The one
thing it does measure fresh is the mouse arm, because mouse runs in ~1 s and because the CURRENT
mouse champion (H52) is not the arm P09 varied when it reported a mouse sigma - P09's mouse sigma
0.015352 pp is the H42 COMPARATOR's, and H52's own std over those 25 labellings was 0.0. That is
the difference between "there is dispersion to harvest on mouse" and "there is none", so it is
worth ~1 minute to replicate it at an INDEPENDENT permutation-seed base.

The two designs
---------------
``random_only``        best of R labellings drawn at random. This is H56 as written.
``canonical_anchored`` arm 0 is the canonical labelling (what sota.json records) and the other
                       R-1 are random. Monotone by construction - it can never score below the
                       champion - and therefore the strongest form of the idea. If the strongest
                       form fails, the idea fails.

The comparator is the champion's CANONICAL score, because that is the number in sota.json and the
number a promotion has to beat. Delta against the labelling MEAN is reported too, since that is
the more flattering framing and the reader should see both.

Run
---
    PY=/c/ProgramData/anaconda3/envs/allen/python.exe
    PYTHONPATH=src $PY experiments/proto_H56_multistart.py --arm mouse --relabels 40
    PYTHONPATH=src $PY experiments/proto_H56_multistart.py --arm analysis

Writes ``experiments/outputs/proto_H56.json`` (analysis) and ``proto_H56_mouse.json`` (mouse arm).
Never writes to ``results/`` - these are not harness runs and must never be pooled into evidence.
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import ndtr

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mfas import io                                              # noqa: E402
from mfas.metrics import pct, score_from_positions               # noqa: E402
from mfas.utils.seeding import seed_everything, select_device    # noqa: E402
from eval.frozen_guard import verify_frozen_manifest             # noqa: E402
from eval import runtime_guard                                   # noqa: E402

# CONFIG - every constant this prototype uses, with its provenance.
CONFIG = {
    # Promotion bar the harvested gain must clear on connectome, campaign.yaml
    # datasets.connectome.min_promotion_delta_pp.
    "min_promotion_delta_pp": 0.012,
    # Same for microns, campaign.yaml datasets.microns.min_promotion_delta_pp.
    "min_promotion_delta_pp_microns": 0.002,
    # Hard per-run cap, campaign.yaml runtime.max_wall_clock_s_per_run.
    "max_wall_clock_s_per_run": 3600.0,
    # Per-run wall clock of the champion. Two independent figures are carried on purpose: the P09
    # study's own measured mean (the same kind of run a multi-start arm would be) and the slower
    # confirm-role figure recorded in sota.json.
    "champion_wall_s": {"connectome_proto_p09": None,      # filled from the P09 file
                        "connectome_sota": 1238.0,
                        "microns_sota": 3418.0},
    # P07: this machine is ~20% slower under ordinary desktop load (cycle 6, 1483.9 s vs
    # 1226.5-1238.3 s on the same connectome verification). Carried as a sensitivity factor.
    "p07_load_factor": 1.20,
    # Largest R the extrapolation searches before giving up.
    "r_max_search": 200,
    # Quadrature grid for the order statistics of the normal, in standard-normal units.
    "quad_lo": -12.0, "quad_hi": 12.0, "quad_n": 400001,
    # Mouse arm.
    "mouse_champion": "H52",
    "mouse_perm_seed_base": 717000,   # INDEPENDENT of P09's 909000, so this is a replication
    "mouse_seed": 42,
}

_P09_CONNECTOME = _ROOT / "experiments" / "outputs" / "proto_P09_connectome.json"
_P09_MOUSE = _ROOT / "experiments" / "outputs" / "proto_P09_mouse.json"
_H42_PROTO = _ROOT / "experiments" / "outputs" / "proto_H42.json"


def _grid():
    """Standard-normal quadrature grid shared by both order-statistic integrals."""
    return np.linspace(CONFIG["quad_lo"], CONFIG["quad_hi"], CONFIG["quad_n"])


def expected_max_of_r(r):
    """E[max of ``r`` iid standard normals], by quadrature.

    Deterministic (no Monte Carlo), so the number is reproducible by checkout. The density of the
    maximum of r iid standard normals is ``r * phi(x) * Phi(x)**(r-1)``.
    """
    if r <= 1:
        return 0.0
    x = _grid()
    phi = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    dens = r * phi * np.power(ndtr(x), r - 1)
    return float(np.trapz(x * dens, x))


def expected_max_anchored(mu, sigma, c, r):
    """E[max(c, X_1..X_{r-1})] for X ~ N(mu, sigma), by quadrature.

    This is the canonical-anchored design: one arm is the fixed canonical labelling scoring ``c``
    and the other r-1 arms are random labellings. Equals ``c + E[(M_{r-1} - c)^+]``.
    """
    if r <= 1:
        return float(c)
    k = r - 1
    z = _grid()
    x = mu + sigma * z
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    dens = k * phi * np.power(ndtr(z), k - 1) / sigma          # density of M_k in x-space
    gain = np.trapz(np.maximum(x - c, 0.0) * dens, x)
    return float(c + gain)


def required_sigma_for_bar(bar, r_feasible):
    """Smallest labelling sigma at which best-of-R could clear ``bar``, given R is capped by time.

    This is the quantitative form of meta-rule M3 ("a multi-start idea must first show it CREATES
    dispersion"). Existence is not enough: with E[max of R] = mu + sigma * a_R, best-of-R over
    RANDOM labellings gains ``sigma * a_R`` over the labelling mean, so clearing a promotion bar
    of ``bar`` pp needs ``sigma >= bar / a_R``. R is not free - it is floor(guard deadline /
    per-run wall) - so the requirement is set by the runtime budget, not by the idea.

    The random-only form is the design-neutral LOWER bound on what is required: whenever the
    canonical labelling scores above the labelling mean (it does here, z = +1.03), the anchored
    design has to clear an even higher bar because it is measured against the canonical score.
    """
    a_r = expected_max_of_r(r_feasible)
    return dict(r_feasible=int(r_feasible), a_r=float(a_r),
                required_sigma_pp=(float(bar / a_r) if a_r > 0 else None))


def smallest_r_clearing(target, mu, sigma, c, anchored):
    """Smallest R whose expected best-of-R clears ``target``, or (None, None) if never."""
    for r in range(1, CONFIG["r_max_search"] + 1):
        val = (expected_max_anchored(mu, sigma, c, r) if anchored
               else mu + sigma * expected_max_of_r(r))
        if val >= target:
            return r, val
    return None, None


def exact_best_of_r(values, canonical, anchored):
    """Mean best-of-R over EVERY subset of the measured labellings, for each feasible R.

    ``values`` are the non-identity labelling scores; ``canonical`` is the identity one. In the
    anchored design the canonical arm is always present and R-1 are chosen from ``values``, so R
    runs to ``len(values)+1``; otherwise R runs to ``len(values)``. This is exact for the sample -
    no resampling and no distributional assumption.
    """
    out = []
    pool = tuple(values)
    r_hi = len(pool) + 1 if anchored else len(pool)
    for r in range(1, r_hi + 1):
        k = r - 1 if anchored else r
        subsets = list(itertools.combinations(pool, k))
        if not subsets:
            continue
        bests = [max(s + ((canonical,) if anchored else ())) for s in subsets]
        out.append(dict(r=r, n_subsets=len(bests),
                        mean_best_pct=float(np.mean(bests)),
                        min_best_pct=float(np.min(bests)),
                        max_best_pct=float(np.max(bests)),
                        delta_vs_canonical_pp=float(np.mean(bests) - canonical)))
    return out


def matched_compute_control():
    """What ONE run buys when handed 2x the stage-4 seconds - measured, not modelled.

    CAMPAIGN.md requires multi-start ideas to be matched on compute rather than compared at a
    single labelling's wall clock. ``proto_H42.json``'s sweeps=2 connectome arm is a 1201 s run of
    the production refiner that logged ``(cum_wall_s, best_pct)`` at all 143 cycles it completed,
    so the gain from doubling the refinement budget is a lookup: ``best_pct`` at ``cum_wall_s`` ~
    W/2 against ``best_pct`` at W.
    """
    d = json.loads(_H42_PROTO.read_text())
    arm = next(a for a in d["arms"]
               if a["dataset"] == "connectome" and a["sift_sweeps"] == 2)
    log = arm["log"]
    walls = np.array([c["cum_wall_s"] for c in log])
    bests = np.array([c["best_pct"] for c in log])
    w_full = float(walls[-1])
    i_half = int(np.argmin(np.abs(walls - w_full / 2.0)))
    return dict(
        source="experiments/outputs/proto_H42.json (connectome, sift_sweeps=2)",
        wall_full_s=w_full, wall_half_s=float(walls[i_half]),
        cycles_full=int(log[-1]["cycle"]), cycles_half=int(log[i_half]["cycle"]),
        pct_at_half=float(bests[i_half]), pct_at_full=float(bests[-1]),
        gain_from_doubling_pp=float(bests[-1] - bests[i_half]),
    )


def sigma_uncertainty(sigma, n, c_minus_mu, bar, control_pp):
    """How much of this verdict survives the fact that sigma is estimated from ``n`` points.

    The kill rests on three grounds and they are NOT equally secure. The runtime ground is
    structural. The effect-size and matched-compute grounds both scale with sigma, which here is a
    5-point estimate, and the chi-squared interval for a sample std at n = 5 is wide.

    So the honest thing is to evaluate the anchored best-of-2 gain at the ENDS of that interval,
    holding the measured ``canonical - mean`` gap fixed (it is a separate estimate), and report
    whether the verdict flips. It does at the upper end, and this function is what says so.
    """
    from scipy.stats import chi2                                   # local: only this needs it
    df = n - 1
    lo = float(sigma * np.sqrt(df / chi2.ppf(0.975, df)))
    hi = float(sigma * np.sqrt(df / chi2.ppf(0.025, df)))
    out = []
    for tag, sg in (("ci_lower", lo), ("point_estimate", float(sigma)), ("ci_upper", hi)):
        z = c_minus_mu / sg
        phi = float(np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi))
        gain = float(sg * (phi - z * (1.0 - float(ndtr(z)))))
        out.append(dict(which=tag, sigma_pp=sg,
                        anchored_best_of_2_gain_pp=gain,
                        clears_bar=bool(gain > bar),
                        beats_matched_compute_control=bool(gain > control_pp)))
    return dict(n_points=int(n), sigma_pp=float(sigma),
                sigma_ci95_pp=[lo, hi], c_minus_mu_pp=float(c_minus_mu),
                bar_pp=float(bar), matched_compute_control_pp=float(control_pp),
                scenarios=out,
                verdict_note=("The effect-size and matched-compute grounds FLIP at the upper end "
                              "of sigma's own confidence interval. The runtime ground - microns "
                              "admits R = 1, so a screen requiring a delta on both primaries is "
                              "impossible - does not depend on sigma and does not flip."))


def relabel(g, r, base):
    """Isomorphic copy of ``g`` under relabelling ``r`` at permutation-seed ``base`` (r=0 -> id).

    Identical in construction to ``experiments/proto_P09_relabel.relabel`` except that the seed
    base is a parameter, which is what makes this an INDEPENDENT sample rather than a re-run of
    P09's. Agreement with P09 at base 909000 is asserted by ``_check_relabel_agrees``.
    """
    n = g.n_nodes
    if r == 0:
        return g, np.arange(n, dtype=np.int64)
    perm = np.random.default_rng(base + r).permutation(n).astype(np.int64)
    node_ids_new = np.empty_like(g.node_ids)
    node_ids_new[perm] = g.node_ids
    g2 = replace(g,
                 src=perm[np.asarray(g.src)].astype(g.src.dtype, copy=False),
                 tgt=perm[np.asarray(g.tgt)].astype(g.tgt.dtype, copy=False),
                 node_ids=node_ids_new)
    return g2, perm


def _check_relabel_agrees(g):
    """Assert this script's relabelling is the same map P09 used, so the samples are comparable."""
    p09 = importlib.import_module("experiments.proto_P09_relabel")
    for r in (1, 2):
        _, mine = relabel(g, r, p09._PERM_SEED_BASE)
        _, theirs = p09.relabel(g, r)
        assert np.array_equal(mine, theirs), "relabel() diverged from proto_P09_relabel.relabel"


def run_mouse_arm(n_relabels, device_str, out_path):
    """Score the CURRENT mouse champion on ``n_relabels`` fresh relabellings; report its sigma."""
    verify_frozen_manifest()
    exp_id = CONFIG["mouse_champion"]
    g0 = io.load_dataset("mouse")
    _check_relabel_agrees(g0)
    variant = importlib.import_module("mfas.experiments." + exp_id)
    plan = runtime_guard.resolve("mouse", None)
    rows = []
    for r in range(0, n_relabels + 1):
        g, _ = relabel(g0, r, CONFIG["mouse_perm_seed_base"])
        assert float(g.weight.sum()) == g0.total_weight, "relabelling changed total weight"
        seed_everything(CONFIG["mouse_seed"])
        torch_device, dev_name = select_device(device_str)
        t0 = time.time()
        res = variant.run(g, seed=CONFIG["mouse_seed"], device=torch_device,
                          time_limit=plan["time_limit_s"])
        wall = time.time() - t0
        score = score_from_positions(res.best_positions, np.asarray(g.src),
                                     np.asarray(g.tgt), g.weight)
        assert score == res.best_score, "proto_H56/mouse score mismatch"
        rows.append(dict(r=r, identity=(r == 0), pct=pct(score, g0.total_weight),
                         score=float(score), wall_clock_s=wall, device=dev_name))
        print("[H56] mouse r=%d pct=%.10f wall=%.2fs" % (r, rows[-1]["pct"], wall), flush=True)
    v = np.array([x["pct"] for x in rows], dtype=np.float64)
    rec = dict(item="H56", arm="mouse", dataset="mouse", champion=exp_id,
               perm_seed_base=CONFIG["mouse_perm_seed_base"], seed=CONFIG["mouse_seed"],
               n_points=int(v.size), rows=rows,
               summary=dict(pct_mean=float(v.mean()),
                            pct_std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                            pct_min=float(v.min()), pct_max=float(v.max()),
                            n_distinct=int(np.unique(v).size),
                            span_pp=float(v.max() - v.min())))
    out_path.write_text(json.dumps(rec, indent=2))
    print("[H56] wrote " + str(out_path))
    print(json.dumps(rec["summary"], indent=2))
    return rec


def run_analysis(out_path, mouse_rec=None):
    """Decide H56 from logged measurement: order statistics against the bar and against the clock."""
    p09 = json.loads(_P09_CONNECTOME.read_text())
    rows = p09["rows"]
    canonical = float(next(x for x in rows if x["identity"])["comparator"]["pct"])
    non_id = [float(x["comparator"]["pct"]) for x in rows if not x["identity"]]
    allv = np.array([canonical] + non_id, dtype=np.float64)
    mu, sigma = float(allv.mean()), float(allv.std(ddof=1))
    wall = float(np.mean([x["comparator"]["wall_clock_s"] for x in rows]))
    CONFIG["champion_wall_s"]["connectome_proto_p09"] = wall

    bar = CONFIG["min_promotion_delta_pp"]
    target = canonical + bar
    deadline = float(runtime_guard.resolve("connectome", None)["time_limit_s"])
    microns_deadline = float(runtime_guard.resolve("microns", None)["time_limit_s"])
    cap = CONFIG["max_wall_clock_s_per_run"]

    feasible_r = {
        "connectome_at_proto_p09_wall": int(deadline // wall),
        "connectome_at_sota_wall": int(deadline // CONFIG["champion_wall_s"]["connectome_sota"]),
        "connectome_at_sota_wall_under_p07_load":
            int(deadline // (CONFIG["champion_wall_s"]["connectome_sota"]
                             * CONFIG["p07_load_factor"])),
        "microns_at_sota_wall": int(microns_deadline
                                    // CONFIG["champion_wall_s"]["microns_sota"]),
    }

    r_rand, v_rand = smallest_r_clearing(target, mu, sigma, canonical, anchored=False)
    r_anch, v_anch = smallest_r_clearing(target, mu, sigma, canonical, anchored=True)

    theory = []
    for r in range(1, 9):
        rand_pct = float(mu + sigma * expected_max_of_r(r))
        anch_pct = expected_max_anchored(mu, sigma, canonical, r)
        theory.append(dict(
            r=r,
            random_only_pct=rand_pct,
            random_only_delta_vs_canonical_pp=float(rand_pct - canonical),
            canonical_anchored_pct=anch_pct,
            canonical_anchored_delta_vs_canonical_pp=float(anch_pct - canonical),
            total_wall_s=float(r * wall),
            fits_guard_deadline=bool(r * wall <= deadline),
            fits_hard_cap=bool(r * wall <= cap),
        ))

    control = matched_compute_control()
    sigma_unc = sigma_uncertainty(sigma, int(allv.size), float(canonical - mu), bar,
                                  control["gain_from_doubling_pp"])

    p09m = json.loads(_P09_MOUSE.read_text())
    mouse = dict(source="experiments/outputs/proto_P09_mouse.json",
                 n_points=p09m["summary"]["n_points"],
                 comparator=p09m["comparator"],
                 comparator_pct_std=p09m["summary"]["comparator_pct_std"],
                 champion=p09m["variant"],
                 champion_pct_std=p09m["summary"]["variant_pct_std"])
    if mouse_rec is not None:
        mouse["replication"] = dict(source="experiments/outputs/proto_H56_mouse.json",
                                    champion=mouse_rec["champion"],
                                    perm_seed_base=mouse_rec["perm_seed_base"],
                                    **mouse_rec["summary"])

    rec = dict(
        item="H56", gate="prototype", date="2026-08-25",
        machine="Windows 10 / RTX 4060 Laptop GPU (CUDA); this arm is CPU-only re-analysis",
        _doc=("Decides H56 from logged measurement rather than new large-graph runs. Sources: "
              "proto_P09_connectome.json (labelling distribution of the champion), "
              "proto_P09_mouse.json + proto_H56_mouse.json (mouse), "
              "proto_H42.json (compute-matched control)."),
        reproduce=["PYTHONPATH=src $PY experiments/proto_H56_multistart.py --arm mouse --relabels 40",
                   "PYTHONPATH=src $PY experiments/proto_H56_multistart.py --arm analysis"],
        config=CONFIG,
        connectome=dict(
            source="experiments/outputs/proto_P09_connectome.json",
            champion="H42", n_labellings=int(allv.size),
            canonical_pct=canonical, labelling_mean_pct=mu, labelling_std_pp=sigma,
            canonical_minus_mean_pp=float(canonical - mu),
            canonical_z=float((canonical - mu) / sigma),
            canonical_rank_from_top=int(1 + (allv > canonical).sum()),
            per_run_wall_s=wall,
            guard_deadline_s=deadline, microns_guard_deadline_s=microns_deadline,
            hard_cap_s=cap, feasible_r=feasible_r,
            exact_random_only=exact_best_of_r(non_id, canonical, anchored=False),
            exact_canonical_anchored=exact_best_of_r(non_id, canonical, anchored=True),
            theory_best_of_r=theory,
            promotion_bar_pp=bar, promotion_target_pct=target,
            smallest_r_clearing_bar=dict(random_only=r_rand, random_only_pct=v_rand,
                                         canonical_anchored=r_anch,
                                         canonical_anchored_pct=v_anch,
                                         total_wall_s_at_r=(float(r_rand * wall)
                                                            if r_rand else None),
                                         over_hard_cap_x=(float(r_rand * wall / cap)
                                                          if r_rand else None)),
            required_sigma=dict(
                measured_sigma_pp=sigma,
                **required_sigma_for_bar(bar, feasible_r["connectome_at_proto_p09_wall"]),
                shortfall_ratio=float(
                    sigma / required_sigma_for_bar(
                        bar, feasible_r["connectome_at_proto_p09_wall"])["required_sigma_pp"]),
                h01_revival_threshold_pp=0.02,
                h01_revival_threshold_source=(
                    "autoresearch/killed.json H01.revival_if - 'std >> 0.02 pp on connectome'"),
            ),
            sigma_uncertainty=sigma_unc,
        ),
        matched_compute_control=control,
        mouse=mouse,
    )
    out_path.write_text(json.dumps(rec, indent=2))
    print("[H56] wrote " + str(out_path))
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description="H56 prototype - relabelling multi-start")
    p.add_argument("--arm", default="analysis", choices=["analysis", "mouse"])
    p.add_argument("--relabels", type=int, default=40,
                   help="mouse arm: number of NON-identity relabellings (r=0 always runs)")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    outdir = _ROOT / "experiments" / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    if args.arm == "mouse":
        run_mouse_arm(args.relabels, args.device, outdir / "proto_H56_mouse.json")
    else:
        mp = outdir / "proto_H56_mouse.json"
        mrec = json.loads(mp.read_text()) if mp.exists() else None
        run_analysis(outdir / "proto_H56.json", mouse_rec=mrec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

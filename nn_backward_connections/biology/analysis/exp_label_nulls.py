"""Label-permutation nulls for three descriptive headline findings (COSYNE upgrade).

The champion MFAS orders (fly H64, 84.258%; MICrONS H42, 83.241%) are FIXED
throughout — no optimization runs anywhere in this script. Every test shuffles
NODE LABELS on the fixed order and re-computes a descriptive statistic, turning
three purely descriptive findings into permutation-tested ones:

T1  FLY transmitter-feedback enrichment.
    Statistic: per-transmitter backward weight share = (weight of out-edges of
    t-neurons with rank[src] > rank[tgt]) / (total out-weight of t-neurons).
    Null (a) GLOBAL: permute `top_nt` labels across all 136,648 neurons.
    Null (b) STRATIFIED: permute `top_nt` within each `super_class` stratum —
    this is the null that matters, because it controls for composition (the
    optic lobes are ACh-heavy AND early in the order).

T2  MICrONS E/I identity shuffle (the never-run test of MICRONS_FINDINGS §8).
    Statistics: (a) per-layer inhibitory-vs-excitatory median-rank lead
    (median rank of E in layer L minus median rank of I in layer L; positive =
    inhibitory earlier), plus the mean lead over tested layers; (b) E-output
    vs I-output backward weight share and their difference E−I.
    Null: shuffle E/I identity among the 63,074 E/I-labeled neurons (NA labels
    stay NA) — GLOBAL, and STRATIFIED within cortical-layer band.

T3  FLY cell-type contiguity across ALL types (previously only spot-checked).
    Statistic: per-type rank IQR normalized by n (= 136,648; the repo's
    existing IQR/n convention from layering/tools/exp_degraded_order.py),
    summarized as the median over types with >= 20 members. Types come from
    data/external/flywire_neuron_annotations.tsv.gz with the exact fallback of
    layering/tools/exp_mirror_swap.py::load_types (cell_type, else
    "<class> (class)", else "unannotated"; "unannotated" excluded).
    Null: permute type labels across neurons — implemented as a joint random
    permutation of the rank vector, which is exactly equivalent and gives the
    correct joint distribution for the median-over-types statistic.

Frozen-oracle compliance: the feedforward/backward classification is the
oracle's own rule (edge u->v is feedforward iff rank[u] < rank[tgt-node]),
applied as a direct rank comparison on the fixed champion rank; the champion
orders themselves are loaded parity-gated through layering.io_utils, which
re-scores them with src/mfas/metrics.py. Nothing is optimized and no scorer
code is touched.

Gates (asserted before any test runs; values read live from the artifacts):
  - fly per-NT backward shares == layering/outputs/fly_streams/report_streams.json
    (ACh 11.36, GABA 20.90, Glu 22.25, DA 20.11, 5-HT 28.83, OA 16.72);
    brain-wide backward share == 100 − champion ff_pct (15.742).
  - MICrONS E/I backward shares and per-layer median ranks ==
    layering/outputs/microns_ann/diagnostics.json (E 26.62% vs I 5.21%;
    leads 6.6k–12.9k ranks).
  - fly median IQR/n over types with >= 100 members == 0.049 (QA anchor,
    QA_ORDER_AND_ROBUSTNESS.md).

Empirical p convention: two-sided, add-one on both tails —
  p = min(1, 2 * min((1 + #{null <= obs}) / (N+1), (1 + #{null >= obs}) / (N+1))).
With N = 2000 the smallest reportable p is 2/2001 ≈ 0.001; with N = 1000 it is
2/1001 ≈ 0.002 (one-sided floors: 1/2001, 1/1001). z = (obs − null mean)/null sd.

Run (repo root, CPU only, ~4 min total):
    PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
    $PY biology/analysis/exp_label_nulls.py            # all three tests
    $PY biology/analysis/exp_label_nulls.py --tests t1 t3   # fly only
    $PY biology/analysis/exp_label_nulls.py --tests t2      # microns only

Outputs:
    biology/results/label_nulls_fly.json        (T1 + T3)
    biology/results/label_nulls_microns.json    (T2)
    biology/results/label_nulls_fly_nulls.npz   (raw null draws, fly)
    biology/results/label_nulls_microns_nulls.npz
    biology/outputs/label_nulls_t1_fly_nt.png
    biology/outputs/label_nulls_t2_microns_ei.png
    biology/outputs/label_nulls_t3_fly_contiguity.png
"""
from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# CONFIG — every constant of the study lives here (no magic numbers below)
# ---------------------------------------------------------------------------
CONFIG = dict(
    seed=42,                      # base seed; each (test, null) gets its own
                                  # child stream default_rng([seed, stream_id])
    # T1 — fly transmitter enrichment
    t1_n_perm=2000,
    t1_stream_global=10,          # rng stream ids (documented, pinned)
    t1_stream_stratified=11,
    # T2 — MICrONS E/I shuffle
    t2_n_perm=2000,
    t2_min_group_n=20,            # a layer is tested only if both E and I >= this
    t2_stream_global=20,
    t2_stream_stratified=21,
    # T3 — fly type contiguity
    t3_n_perm=1000,
    t3_min_members=20,            # per task spec; >=100 is the QA-anchor gate
    t3_gate_min_members=100,
    t3_gate_anchor=0.049,         # QA_ORDER_AND_ROBUSTNESS.md table (3 dp)
    t3_gate_tol=5e-4,
    t3_exclude_type="unannotated",
    t3_stream=30,
    # paths
    fly_ann_npz=REPO_ROOT / "layering/outputs/fly_ann/node_ann.npz",
    fly_ann_meta=REPO_ROOT / "layering/outputs/fly_ann/node_ann_meta.json",
    fly_streams_report=REPO_ROOT / "layering/outputs/fly_streams/report_streams.json",
    fly_types_tsv=REPO_ROOT / "data/external/flywire_neuron_annotations.tsv.gz",
    microns_ann_npz=REPO_ROOT / "layering/outputs/microns_ann/node_ann.npz",
    microns_diagnostics=REPO_ROOT / "layering/outputs/microns_ann/diagnostics.json",
    results_dir=REPO_ROOT / "biology/results",
    outputs_dir=REPO_ROOT / "biology/outputs",
    fig_dpi=120,
    gate_tol=1e-9,                # relative, for shares reproduced from artifacts
)

# Validated categorical palette (dataviz reference instance, light mode).
COL_STRAT = "#2a78d6"   # stratified null (slot 1, blue)
COL_GLOB = "#eb6834"    # global null     (slot 2, orange)
COL_OBS = "#0b0b0b"     # observed        (text ink, never a series hue)
COL_REF = "#52514e"     # secondary ink / reference marks
COL_SURF = "#fcfcfb"    # chart surface


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def emp_stats(null: np.ndarray, obs: float) -> dict:
    """Summary of an observed statistic against its permutation null.

    Returns null mean/sd, z = (obs − mean)/sd, the add-one two-sided empirical
    p, and both tail exceedance counts (n_le = #{null <= obs}, n_ge >= obs)
    so the p is fully auditable from the JSON alone.
    """
    null = np.asarray(null, dtype=np.float64)
    n = null.shape[0]
    mean = float(null.mean())
    sd = float(null.std(ddof=1))
    n_le = int((null <= obs).sum())
    n_ge = int((null >= obs).sum())
    p = min(1.0, 2.0 * min((1 + n_le) / (n + 1), (1 + n_ge) / (n + 1)))
    z = float((obs - mean) / sd) if sd > 0 else float("inf")
    return dict(observed=float(obs), null_mean=mean, null_sd=sd, z=z,
                p_two_sided=p, n_le=n_le, n_ge=n_ge, n_perm=n)


def strata_indices(strata: np.ndarray) -> list:
    """Index arrays of each stratum value (for within-stratum permutation)."""
    return [np.where(strata == s)[0] for s in np.unique(strata)]


def permute_labels(labels: np.ndarray, rng: np.random.Generator,
                   strata_idx: list | None = None,
                   pool: np.ndarray | None = None) -> np.ndarray:
    """Return a permuted copy of `labels`.

    strata_idx: if given, permute independently within each index array
    (stratified null); otherwise permute globally. pool: if given (index
    array), only these positions take part in the (global) permutation —
    labels outside stay fixed (used for E/I where NA identities stay NA).
    """
    out = labels.copy()
    if strata_idx is not None:
        for idx in strata_idx:
            out[idx] = labels[idx][rng.permutation(idx.shape[0])]
    elif pool is not None:
        out[pool] = labels[pool][rng.permutation(pool.shape[0])]
    else:
        out = labels[rng.permutation(labels.shape[0])]
    return out


def node_out_weights(src, tgt, weight, rank):
    """Per-node total out-weight and backward out-weight (oracle rule).

    Backward edge: rank[src] > rank[tgt] (the frozen oracle's complement of
    feedforward rank[src] < rank[tgt]; ranks are a permutation, no ties).
    Accumulated in int64 (never float32), per the oracle's determinism rule.
    """
    n = rank.shape[0]
    w = weight.astype(np.int64, copy=False)
    back = rank[src] > rank[tgt]
    ow = np.bincount(src, weights=w, minlength=n).astype(np.int64)
    bw = np.bincount(src[back], weights=w[back], minlength=n).astype(np.int64)
    return ow, bw


def provenance() -> dict:
    return dict(
        script="biology/analysis/exp_label_nulls.py",
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        interpreter=sys.executable,
        seed=CONFIG["seed"],
    )


def rel_close(a: float, b: float, tol: float) -> bool:
    return bool(abs(a - b) <= tol * max(1.0, abs(a), abs(b)))


# ---------------------------------------------------------------------------
# T1 — fly transmitter backward-share enrichment
# ---------------------------------------------------------------------------
def run_t1(g, rank, ann, meta) -> tuple[dict, dict]:
    """Permutation test of per-transmitter backward weight share (fly)."""
    t0 = time.time()
    nt = ann["top_nt"].astype(np.int64)
    nt_names = meta["top_nt_values"]
    n_nt = len(nt_names)
    ow, bw = node_out_weights(g.src, g.tgt, g.weight, rank)

    def shares(labels):
        num = np.bincount(labels, weights=bw, minlength=n_nt)
        den = np.bincount(labels, weights=ow, minlength=n_nt)
        return 100.0 * num / den

    obs = shares(nt)

    # --- gate: reproduce report_streams.json fb_share_within_nt exactly -----
    report = json.loads(CONFIG["fly_streams_report"].read_text(encoding="utf-8"))
    gate = {}
    for i, nm in enumerate(nt_names):
        expected = float(report["nt_direction"][nm]["fb_share_within_nt"])
        ok = rel_close(obs[i], expected, CONFIG["gate_tol"])
        gate[nm] = dict(expected=expected, got=float(obs[i]), ok=ok)
        assert ok, f"T1 gate FAILED for {nm}: {obs[i]!r} vs report {expected!r}"
    brainwide = 100.0 * bw.sum() / ow.sum()
    gate["_brain_wide_backward_pct"] = dict(
        expected=100.0 - float(report["champion_ff_pct"]), got=float(brainwide))
    assert rel_close(brainwide, 100.0 - report["champion_ff_pct"], CONFIG["gate_tol"])

    # --- nulls ---------------------------------------------------------------
    n_perm = CONFIG["t1_n_perm"]
    strata_idx = strata_indices(ann["super_class"])
    nulls = {}
    for null_name, stream, sidx in [
            ("global", CONFIG["t1_stream_global"], None),
            ("stratified_super_class", CONFIG["t1_stream_stratified"], strata_idx)]:
        rng = np.random.default_rng([CONFIG["seed"], stream])
        draws = np.empty((n_perm, n_nt), dtype=np.float64)
        for k in range(n_perm):
            draws[k] = shares(permute_labels(nt, rng, strata_idx=sidx))
        nulls[null_name] = draws

    counts = np.bincount(nt, minlength=n_nt)
    table = {}
    for null_name, draws in nulls.items():
        table[null_name] = {
            nm: dict(emp_stats(draws[:, i], obs[i]),
                     n_neurons=int(counts[i]), out_weight=int(
                         np.bincount(nt, weights=ow, minlength=n_nt)[i]))
            for i, nm in enumerate(nt_names)}

    result = dict(
        statistic="per-transmitter backward weight share (%), source-neuron NT",
        null_definitions=dict(
            global_="permute top_nt across all neurons",
            stratified_super_class="permute top_nt within each super_class"),
        n_perm=n_perm, seed=CONFIG["seed"],
        rng_streams=dict(global_=[CONFIG["seed"], CONFIG["t1_stream_global"]],
                         stratified=[CONFIG["seed"], CONFIG["t1_stream_stratified"]]),
        brain_wide_backward_pct=float(brainwide),
        gates=gate, tables=table, seconds=round(time.time() - t0, 1),
        notes=["top_nt is a PREDICTION (Eckstein et al. 2024), not ground truth",
               "Kenyon cells were overridden to acetylcholine in the fly_ann "
               "cache (documented choice, kept here)",
               "NA transmitter (103 neurons) reported for completeness only"])
    raw = {f"t1_{k}": v for k, v in nulls.items()}
    raw["t1_observed"] = obs
    return result, raw


# ---------------------------------------------------------------------------
# T2 — MICrONS E/I identity shuffle
# ---------------------------------------------------------------------------
def run_t2(g, rank, ann, diag) -> tuple[dict, dict]:
    """Permutation test of the early-inhibition finding (MICrONS)."""
    t0 = time.time()
    ei = ann["ei"].astype(np.int64)        # 0 NA / 1 exc / 2 inh
    layer = ann["layer"].astype(np.int64)  # 0 NA / 1 L1 ... 5 L6 / 6 WM
    layer_names = ["NA", "L1", "L2/3", "L4", "L5", "L6", "WM"]
    ow, bw = node_out_weights(g.src, g.tgt, g.weight, rank)
    pool = np.where(ei != 0)[0]            # only E/I-labeled neurons shuffle

    # layers tested: both groups >= min_group_n
    tested_layers = [L for L in range(1, 7)
                     if ((layer == L) & (ei == 1)).sum() >= CONFIG["t2_min_group_n"]
                     and ((layer == L) & (ei == 2)).sum() >= CONFIG["t2_min_group_n"]]
    excluded = {layer_names[L]: dict(
        n_exc=int(((layer == L) & (ei == 1)).sum()),
        n_inh=int(((layer == L) & (ei == 2)).sum()))
        for L in range(1, 7) if L not in tested_layers}
    layer_idx = {L: np.where(layer == L)[0] for L in tested_layers}

    def leads(labels):
        out = np.empty(len(tested_layers))
        for j, L in enumerate(tested_layers):
            li = layer_idx[L]
            lab = labels[li]
            r = rank[li]
            out[j] = np.median(r[lab == 1]) - np.median(r[lab == 2])
        return out

    def bw_shares(labels):
        num = np.bincount(labels, weights=bw, minlength=3)
        den = np.bincount(labels, weights=ow, minlength=3)
        s = 100.0 * num / den
        return np.array([s[1], s[2], s[1] - s[2]])  # E, I, E−I

    obs_leads = leads(ei)
    obs_shares = bw_shares(ei)

    # --- gates: reproduce diagnostics.json exactly ---------------------------
    gate = {}
    d2 = diag["d2_ei_direction"]
    for key, val in [("exc", obs_shares[0]), ("inh", obs_shares[1])]:
        expected = float(d2[key]["backward_share_pct"])
        ok = rel_close(val, expected, CONFIG["gate_tol"])
        gate[f"backward_share_{key}"] = dict(expected=expected, got=float(val), ok=ok)
        assert ok, f"T2 gate FAILED {key}: {val!r} vs {expected!r}"
    d1 = diag["d1_rank_by_layer"]
    for j, L in enumerate(tested_layers):
        nm = layer_names[L]
        exp_lead = float(d1["exc"][nm]["median_rank"]) - float(d1["inh"][nm]["median_rank"])
        ok = rel_close(obs_leads[j], exp_lead, CONFIG["gate_tol"])
        gate[f"lead_{nm}"] = dict(expected=exp_lead, got=float(obs_leads[j]), ok=ok)
        assert ok, f"T2 gate FAILED lead {nm}: {obs_leads[j]!r} vs {exp_lead!r}"

    # --- nulls ---------------------------------------------------------------
    n_perm = CONFIG["t2_n_perm"]
    strata_idx = [np.where((layer == L) & (ei != 0))[0] for L in range(1, 7)
                  if ((layer == L) & (ei != 0)).sum() > 1]
    null_leads, null_shares = {}, {}
    for null_name, stream, sidx in [
            ("global", CONFIG["t2_stream_global"], None),
            ("stratified_layer", CONFIG["t2_stream_stratified"], strata_idx)]:
        rng = np.random.default_rng([CONFIG["seed"], stream])
        dl = np.empty((n_perm, len(tested_layers)))
        ds = np.empty((n_perm, 3))
        for k in range(n_perm):
            lab = permute_labels(ei, rng, strata_idx=sidx,
                                 pool=None if sidx is not None else pool)
            dl[k] = leads(lab)
            ds[k] = bw_shares(lab)
        null_leads[null_name] = dl
        null_shares[null_name] = ds

    table = {}
    for null_name in null_leads:
        entry = {}
        for j, L in enumerate(tested_layers):
            nm = layer_names[L]
            entry[f"lead_{nm}"] = dict(
                emp_stats(null_leads[null_name][:, j], obs_leads[j]),
                n_exc=int(((layer == L) & (ei == 1)).sum()),
                n_inh=int(((layer == L) & (ei == 2)).sum()))
        entry["lead_mean_over_layers"] = emp_stats(
            null_leads[null_name].mean(axis=1), obs_leads.mean())
        for i, nm in enumerate(["backward_share_exc", "backward_share_inh",
                                "backward_share_diff_E_minus_I"]):
            entry[nm] = emp_stats(null_shares[null_name][:, i], obs_shares[i])
        table[null_name] = entry

    result = dict(
        statistics=dict(
            a="per-layer median-rank lead = median rank(E in L) − median rank(I in L); "
              "positive = inhibitory earlier; plus mean over tested layers",
            b="E-output vs I-output backward weight share (%) and difference E−I (pp)"),
        null_definitions=dict(
            global_="shuffle E/I identity among the E/I-labeled neurons "
                    "(NA stays NA; #E and #I preserved)",
            stratified_layer="same, but within each cortical-layer band"),
        n_perm=n_perm, seed=CONFIG["seed"],
        rng_streams=dict(global_=[CONFIG["seed"], CONFIG["t2_stream_global"]],
                         stratified=[CONFIG["seed"], CONFIG["t2_stream_stratified"]]),
        tested_layers=[layer_names[L] for L in tested_layers],
        excluded_layers=excluded,
        n_pool=int(pool.shape[0]), n_exc=int((ei == 1).sum()),
        n_inh=int((ei == 2).sum()), n_ei_na=int((ei == 0).sum()),
        gates=gate, tables=table, seconds=round(time.time() - t0, 1),
        notes=["E/I from allen_soma_ei_class_model_v1 (soma-based classifier)",
               "WM excluded from statistic (a): only 18 inhibitory members "
               "(< 20 floor); its observed lead is negative (−3927)",
               "truncation bias: inhibitory axons are locally complete while "
               "excitatory axons are heavily truncated — see study caveats"])
    raw = {f"t2_leads_{k}": v for k, v in null_leads.items()}
    raw |= {f"t2_shares_{k}": v for k, v in null_shares.items()}
    raw["t2_observed_leads"] = obs_leads
    raw["t2_observed_shares"] = obs_shares
    return result, raw


# ---------------------------------------------------------------------------
# T3 — fly cell-type contiguity
# ---------------------------------------------------------------------------
def load_types(node_ids: np.ndarray) -> np.ndarray:
    """Per-node type string, exactly as layering/tools/exp_mirror_swap.py.

    cell_type, falling back to '<cell_class> (class)' (or super_class), else
    'unannotated'.
    """
    ctype = {}
    with gzip.open(CONFIG["fly_types_tsv"], "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t = row["cell_type"].strip()
            if not t:
                c = row["cell_class"].strip() or row["super_class"].strip()
                t = f"{c} (class)" if c else ""
            ctype[int(row["root_id"])] = t or "unannotated"
    return np.array([ctype.get(int(i), "unannotated") for i in node_ids])


def run_t3(g, rank) -> tuple[dict, dict]:
    """Permutation test of cell-type rank-block tightness (fly)."""
    t0 = time.time()
    n = rank.shape[0]
    types = load_types(g.node_ids)
    uniq, inv, counts = np.unique(types, return_inverse=True, return_counts=True)

    def iqr_over_n(vals: np.ndarray, axis=None) -> np.ndarray:
        q1, q3 = np.percentile(vals, [25, 75], axis=axis)
        return (q3 - q1) / n

    # --- gate: QA anchor (>=100 members, exclude 'unannotated') --------------
    keep100 = [i for i in range(len(uniq))
               if counts[i] >= CONFIG["t3_gate_min_members"]
               and uniq[i] != CONFIG["t3_exclude_type"]]
    med100 = float(np.median([iqr_over_n(rank[inv == i]) for i in keep100]))
    gate = dict(median_iqr_over_n_min100=dict(
        expected=CONFIG["t3_gate_anchor"], got=med100,
        n_types=len(keep100), tol=CONFIG["t3_gate_tol"]))
    assert abs(med100 - CONFIG["t3_gate_anchor"]) < CONFIG["t3_gate_tol"], \
        f"T3 gate FAILED: median IQR/n (>=100) {med100!r} vs anchor 0.049"

    # --- observed statistic (task convention: >= 20 members) -----------------
    keep = [i for i in range(len(uniq))
            if counts[i] >= CONFIG["t3_min_members"]
            and uniq[i] != CONFIG["t3_exclude_type"]]
    member_idx = [np.where(inv == i)[0] for i in keep]
    type_names = [str(uniq[i]) for i in keep]
    type_sizes = np.array([counts[i] for i in keep])
    obs_iqr = np.array([iqr_over_n(rank[idx]) for idx in member_idx])
    obs_median = float(np.median(obs_iqr))

    # --- null: joint permutation of the rank vector --------------------------
    # (== permuting type labels across neurons; each type of size k receives a
    # uniform random k-subset of the ranks 0..n−1, jointly without replacement)
    n_perm = CONFIG["t3_n_perm"]
    rng = np.random.default_rng([CONFIG["seed"], CONFIG["t3_stream"]])
    # vectorize by size class: all types of one size share a (m, k) gather
    size_groups = defaultdict(list)
    for ti, idx in enumerate(member_idx):
        size_groups[idx.shape[0]].append(ti)
    gather = {k: (np.array(tis), np.stack([member_idx[t] for t in tis]))
              for k, tis in size_groups.items()}
    null_iqr = np.empty((n_perm, len(keep)), dtype=np.float32)
    for p in range(n_perm):
        pr = rng.permutation(n)
        for k, (tis, mat) in gather.items():
            null_iqr[p, tis] = iqr_over_n(pr[mat], axis=1)
    null_median = np.median(null_iqr.astype(np.float64), axis=1)

    # --- per-type comparison --------------------------------------------------
    pct1 = np.percentile(null_iqr.astype(np.float64), 1, axis=0)
    tighter_than_pct1 = obs_iqr < pct1
    below_all = obs_iqr < null_iqr.min(axis=0)
    p_one_sided = (1 + (null_iqr <= obs_iqr[None, :]).sum(axis=0)) / (n_perm + 1)

    per_type = [dict(type=type_names[t], n=int(type_sizes[t]),
                     iqr_over_n=float(obs_iqr[t]),
                     null_pct1=float(pct1[t]),
                     p_one_sided_tighter=float(p_one_sided[t]))
                for t in np.argsort(-type_sizes)]

    result = dict(
        statistic="median over cell types (>=20 members) of per-type rank "
                  "IQR / n (n = 136,648)",
        null_definition="permute type labels across neurons (joint random "
                        "permutation of the rank vector); each type of size k "
                        "gets a uniform random k-subset of ranks",
        n_perm=n_perm, seed=CONFIG["seed"],
        rng_stream=[CONFIG["seed"], CONFIG["t3_stream"]],
        n_types=len(keep), min_members=CONFIG["t3_min_members"],
        excluded_type=CONFIG["t3_exclude_type"],
        gates=gate,
        median_statistic=emp_stats(null_median, obs_median),
        fraction_tighter_than_null_pct1=float(tighter_than_pct1.mean()),
        n_tighter_than_null_pct1=int(tighter_than_pct1.sum()),
        fraction_below_all_null_draws=float(below_all.mean()),
        n_below_all_null_draws=int(below_all.sum()),
        per_type_p_floor=1.0 / (n_perm + 1),
        per_type=per_type,
        seconds=round(time.time() - t0, 1),
        notes=["types use the exp_mirror_swap load_types fallback: cell_type, "
               "else '<class> (class)', else 'unannotated' (excluded)",
               "gate uses the repo's earlier >=100-member convention "
               "(exp_degraded_order.py, anchor 0.049); the test itself uses "
               ">=20 per the study spec"])
    raw = dict(t3_null_median=null_median, t3_obs_iqr=obs_iqr,
               t3_null_pct1=pct1, t3_type_sizes=type_sizes)
    return result, raw


# ---------------------------------------------------------------------------
# Figures (matplotlib, Agg; dpi from CONFIG)
# ---------------------------------------------------------------------------
def _style_ax(ax):
    ax.set_facecolor(COL_SURF)
    ax.grid(True, color="#e6e5e1", linewidth=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COL_REF)
    ax.tick_params(colors=COL_REF, labelsize=8)


def fig_t1(res, raw, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["acetylcholine", "gaba", "glutamate", "dopamine", "serotonin",
             "octopamine"]
    labels = {"acetylcholine": "ACh", "gaba": "GABA", "glutamate": "Glu",
              "dopamine": "DA", "serotonin": "5-HT", "octopamine": "OA"}
    nt_names = list(res["tables"]["global"].keys())
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), facecolor=COL_SURF)
    for ax, nt in zip(axes.ravel(), order):
        i = nt_names.index(nt)
        gdraw = raw["t1_global"][:, i]
        sdraw = raw["t1_stratified_super_class"][:, i]
        obs = raw["t1_observed"][i]
        _style_ax(ax)
        ax.hist(gdraw, bins=40, color=COL_GLOB, alpha=0.45, zorder=2,
                label="null: global shuffle")
        ax.hist(sdraw, bins=40, color=COL_STRAT, alpha=0.6, zorder=3,
                label="null: within super_class")
        ax.axvline(obs, color=COL_OBS, linewidth=2, zorder=4, label="observed")
        st = res["tables"]["stratified_super_class"][nt]
        ax.set_title(f"{labels[nt]}  (n={st['n_neurons']:,})", fontsize=10,
                     color=COL_OBS)
        ax.text(0.02, 0.95, f"z={st['z']:+.1f}\np={st['p_two_sided']:.4g}",
                transform=ax.transAxes, va="top", fontsize=8, color=COL_REF)
        ax.set_xlabel("backward weight share (%)", fontsize=8, color=COL_REF)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
    for ax in axes[:, 0]:
        ax.set_ylabel("null draws", fontsize=8, color=COL_REF)
    fig.suptitle("T1 fly: per-transmitter backward share vs label-shuffle nulls "
                 f"({res['n_perm']} perms, seed {res['seed']}; z/p vs stratified null)",
                 fontsize=11, color=COL_OBS)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=CONFIG["fig_dpi"], facecolor=COL_SURF)
    plt.close(fig)


def fig_t2(res, raw, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = res["tested_layers"]
    fig = plt.figure(figsize=(12, 6.4), facecolor=COL_SURF)
    gs = fig.add_gridspec(2, max(len(layers), 2), hspace=0.55, wspace=0.35)
    for j, nm in enumerate(layers):
        ax = fig.add_subplot(gs[0, j])
        _style_ax(ax)
        ax.hist(raw["t2_leads_global"][:, j] / 1e3, bins=40, color=COL_GLOB,
                alpha=0.45, zorder=2)
        ax.hist(raw["t2_leads_stratified_layer"][:, j] / 1e3, bins=40,
                color=COL_STRAT, alpha=0.6, zorder=3)
        ax.axvline(raw["t2_observed_leads"][j] / 1e3, color=COL_OBS,
                   linewidth=2, zorder=4)
        st = res["tables"]["stratified_layer"][f"lead_{nm}"]
        ax.set_title(f"{nm}", fontsize=10, color=COL_OBS)
        ax.text(0.03, 0.95, f"z={st['z']:+.1f}\np={st['p_two_sided']:.4g}",
                transform=ax.transAxes, va="top", fontsize=7.5, color=COL_REF)
        ax.set_xlabel("I lead (k ranks)", fontsize=8, color=COL_REF)
        if j == 0:
            ax.set_ylabel("null draws", fontsize=8, color=COL_REF)

    axm = fig.add_subplot(gs[1, 0:2])
    _style_ax(axm)
    axm.hist(raw["t2_leads_global"].mean(axis=1) / 1e3, bins=40, color=COL_GLOB,
             alpha=0.45, zorder=2, label="null: global E/I shuffle")
    axm.hist(raw["t2_leads_stratified_layer"].mean(axis=1) / 1e3, bins=40,
             color=COL_STRAT, alpha=0.6, zorder=3, label="null: within layer")
    axm.axvline(raw["t2_observed_leads"].mean() / 1e3, color=COL_OBS,
                linewidth=2, zorder=4, label="observed")
    st = res["tables"]["stratified_layer"]["lead_mean_over_layers"]
    axm.set_title(f"mean I lead over layers  z={st['z']:+.1f}, "
                  f"p={st['p_two_sided']:.4g}", fontsize=10, color=COL_OBS)
    axm.set_xlabel("mean I lead (k ranks)", fontsize=8, color=COL_REF)
    axm.set_ylabel("null draws", fontsize=8, color=COL_REF)
    axm.legend(fontsize=7.5, frameon=False)

    axd = fig.add_subplot(gs[1, 2:4])
    _style_ax(axd)
    axd.hist(raw["t2_shares_global"][:, 2], bins=40, color=COL_GLOB,
             alpha=0.45, zorder=2)
    axd.hist(raw["t2_shares_stratified_layer"][:, 2], bins=40, color=COL_STRAT,
             alpha=0.6, zorder=3)
    axd.axvline(raw["t2_observed_shares"][2], color=COL_OBS, linewidth=2,
                zorder=4)
    st = res["tables"]["stratified_layer"]["backward_share_diff_E_minus_I"]
    axd.set_title(f"backward-share difference E−I  z={st['z']:+.1f}, "
                  f"p={st['p_two_sided']:.4g}", fontsize=10, color=COL_OBS)
    axd.set_xlabel("E backward share − I backward share (pp)", fontsize=8,
                   color=COL_REF)
    axd.set_ylabel("null draws", fontsize=8, color=COL_REF)

    fig.suptitle("T2 MICrONS: E/I identity shuffle "
                 f"({res['n_perm']} perms, seed {res['seed']}; z/p vs "
                 "within-layer null; observed = black line)",
                 fontsize=11, color=COL_OBS)
    fig.savefig(path, dpi=CONFIG["fig_dpi"], facecolor=COL_SURF,
                bbox_inches="tight")
    plt.close(fig)


def fig_t3(res, raw, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=COL_SURF)
    _style_ax(axl)
    axl.hist(raw["t3_null_median"], bins=40, color=COL_STRAT, alpha=0.7,
             zorder=2, label="null medians (label shuffle)")
    axl.axvline(res["median_statistic"]["observed"], color=COL_OBS,
                linewidth=2, zorder=4, label="observed")
    st = res["median_statistic"]
    axl.set_xlim(0, max(0.55, raw["t3_null_median"].max() * 1.05))
    axl.set_title(f"median IQR/n over {res['n_types']} types  "
                  f"z={st['z']:+.0f}, p={st['p_two_sided']:.4g}", fontsize=10,
                  color=COL_OBS)
    axl.set_xlabel("median per-type rank IQR / n", fontsize=8, color=COL_REF)
    axl.set_ylabel("null draws", fontsize=8, color=COL_REF)
    axl.legend(fontsize=7.5, frameon=False)

    _style_ax(axr)
    axr.scatter(raw["t3_type_sizes"], raw["t3_null_pct1"], s=8, color=COL_REF,
                alpha=0.5, zorder=2, label="per-type null 1st percentile",
                edgecolors="none")
    axr.scatter(raw["t3_type_sizes"], raw["t3_obs_iqr"], s=8, color=COL_STRAT,
                alpha=0.75, zorder=3, label="observed per-type IQR/n",
                edgecolors="none")
    axr.set_xscale("log")
    axr.set_title(f"{res['n_tighter_than_null_pct1']}/{res['n_types']} types "
                  f"tighter than their null 1st percentile "
                  f"({100 * res['fraction_tighter_than_null_pct1']:.1f}%)",
                  fontsize=10, color=COL_OBS)
    axr.set_xlabel("type size (members, log)", fontsize=8, color=COL_REF)
    axr.set_ylabel("rank IQR / n", fontsize=8, color=COL_REF)
    axr.legend(fontsize=7.5, loc="upper right", frameon=True,
               facecolor=COL_SURF, edgecolor="#e6e5e1", framealpha=0.95)

    fig.suptitle("T3 fly: cell-type rank-block tightness vs type-label "
                 f"permutation ({res['n_perm']} draws, seed {res['seed']})",
                 fontsize=11, color=COL_OBS)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=CONFIG["fig_dpi"], facecolor=COL_SURF)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def merged_write(json_path: Path, npz_path: Path, new_json: dict, new_raw: dict):
    """Write results, merging over any existing file (partial --tests runs
    must not clobber the other tests' sections)."""
    if json_path.exists():
        old = json.loads(json_path.read_text(encoding="utf-8"))
        old.update(new_json)
        new_json = old
    json_path.write_text(json.dumps(new_json, indent=1), encoding="utf-8")
    if npz_path.exists():
        with np.load(npz_path) as f:
            old_raw = {k: f[k] for k in f.files}
        old_raw.update(new_raw)
        new_raw = old_raw
    np.savez_compressed(npz_path, **new_raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tests", nargs="+", default=["t1", "t2", "t3"],
                    choices=["t1", "t2", "t3"])
    args = ap.parse_args()
    t0 = time.time()
    from layering import io_utils

    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["outputs_dir"].mkdir(parents=True, exist_ok=True)

    if {"t1", "t3"} & set(args.tests):
        g, order, rank, ff = io_utils.load_graph_and_champion_order("connectome")
        ann = np.load(CONFIG["fly_ann_npz"])
        meta = json.loads(CONFIG["fly_ann_meta"].read_text(encoding="utf-8"))
        assert np.array_equal(ann["rank"], rank), "fly annotation cache is stale"
        fly = dict(provenance=provenance(), dataset="connectome",
                   champion_ff_pct=float(ff), n=int(rank.shape[0]),
                   m=int(g.src.shape[0]))
        raw_fly = {}
        if "t1" in args.tests:
            fly["t1_nt_backward_share"], r = run_t1(g, rank, ann, meta)
            raw_fly |= r
            print(f"[t1] done in {fly['t1_nt_backward_share']['seconds']}s")
        if "t3" in args.tests:
            fly["t3_type_contiguity"], r = run_t3(g, rank)
            raw_fly |= r
            print(f"[t3] done in {fly['t3_type_contiguity']['seconds']}s")
        out = CONFIG["results_dir"] / "label_nulls_fly.json"
        merged_write(out, CONFIG["results_dir"] / "label_nulls_fly_nulls.npz",
                     fly, raw_fly)
        if "t1" in args.tests:
            fig_t1(fly["t1_nt_backward_share"], raw_fly,
                   CONFIG["outputs_dir"] / "label_nulls_t1_fly_nt.png")
        if "t3" in args.tests:
            fig_t3(fly["t3_type_contiguity"], raw_fly,
                   CONFIG["outputs_dir"] / "label_nulls_t3_fly_contiguity.png")
        print(f"[fly] wrote {out}")

    if "t2" in args.tests:
        g, order, rank, ff = io_utils.load_graph_and_champion_order("microns")
        ann = np.load(CONFIG["microns_ann_npz"])
        diag = json.loads(CONFIG["microns_diagnostics"].read_text(encoding="utf-8"))
        assert np.array_equal(ann["rank"], rank), "microns annotation cache is stale"
        mic = dict(provenance=provenance(), dataset="microns",
                   champion_ff_pct=float(ff), n=int(rank.shape[0]),
                   m=int(g.src.shape[0]))
        mic["t2_ei_shuffle"], raw_mic = run_t2(g, rank, ann, diag)
        print(f"[t2] done in {mic['t2_ei_shuffle']['seconds']}s")
        out = CONFIG["results_dir"] / "label_nulls_microns.json"
        merged_write(out, CONFIG["results_dir"] / "label_nulls_microns_nulls.npz",
                     mic, raw_mic)
        fig_t2(mic["t2_ei_shuffle"], raw_mic,
               CONFIG["outputs_dir"] / "label_nulls_t2_microns_ei.png")
        print(f"[microns] wrote {out}")

    print(f"total {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()

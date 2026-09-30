"""INDEPENDENT REVIEW of biology/studies/mirror_swap_retune.md (2026-09-30).

Re-derives every quoted number from the saved artifacts through the frozen
oracle (src/mfas/metrics.py, CPU) with reviewer-written code:

  1. Oracle re-scores of the three retuned rank vectors + permutation validity.
  2. Derived quantities (recovery pp / % of deficit, headline gaps, un-swap
     fractions, Spearman, mean |rank shift|) recomputed from artifacts only.
  3. Swap-start scores reconstructed from saved pairs + champion rank, checked
     bit-exactly against layering/outputs/qa/exp_mirror_swap.json; the full
     four-scheme pairing is ALSO rebuilt from annotations (reviewer replication
     of layering/tools/exp_mirror_swap.py's inline construction) and compared
     array-equal to the saved pair files.
  4. Red-team: binomial z for the 56.7% vs 41.5% stay-swapped contrast, and
     stay-fraction binned by within-pair rank displacement (pooled deciles) to
     test whether the mirror-vs-C1b contrast is a displacement artifact.

Writes biology/results/review/mirror_retune_review.json and prints a verdict
per claim. Read-only on all study artifacts.

Run:  $PY biology/analysis/review_mirror_retune.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent            # biology/analysis
REPO = HERE.parent.parent                          # repo root
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RESULTS = HERE.parent / "results"
OUT_DIR = RESULTS / "review"
OUT_JSON = OUT_DIR / "mirror_retune_review.json"
QA_JSON = REPO / "layering" / "outputs" / "qa" / "exp_mirror_swap.json"
STUDY_JSON = RESULTS / "mirror_retune.json"

# ---- claims quoted from biology/studies/mirror_retune / mirror_retune.json ----
CLAIMS = {
    "champion_pct": 84.25817950936937,
    "total_weight": 41_912_141,
    "retuned_champion_pct": 84.262121088016,
    "retuned_champion_int": 35_316_059,
    "retuned_mirror_pct": 84.32808288175973,
    "retuned_mirror_int": 35_343_705,
    "retuned_c1b_pct": 84.32280517475832,
    "retuned_c1b_int": 35_341_493,
    "start_mirror_pct": 82.04530520165983,
    "start_c1b_pct": 82.72064412075727,
    "gap_a_minus_b_pp": -0.06596179374372468,
    "gap_a_minus_c_pp": -0.06068408674231307,
    "gap_c_minus_b_pp": -0.005277707001411613,
    "mirror_recovered_pp": 2.2827776800998976,
    "mirror_recovered_frac": 1.0315894003318746,
    "c1b_recovered_pp": 1.6021610540010442,
    "c1b_recovered_frac": 1.0420319856489817,
    "mirror_frac_stayed": 0.5667664853633196,
    "mirror_frac_reverted": 0.4332335146366803,
    "c1b_frac_reverted": 0.5851586011555112,
    "c1b_frac_stayed": 0.4148413988444888,
    "rho_swap_vs_champ": 0.9118962794701951,
    "rho_ret_vs_champ": 0.9660137747739485,
    "rho_ret_vs_start": 0.9065808111064422,
    "shift_ret_vs_champ": 4695.775510801475,
    "shift_ret_vs_start": 6387.479611849423,
    "n_mirror_pairs": 65_452,
    "n_c1b_pairs": 62_137,
    # study line "Champion reference: 84.25817950936937 % (34,751,902 + gate...)"
    "study_champion_parenthetical_int": 34_751_902,
}

checks = []


def check(name, claimed, computed, tol=0.0):
    if isinstance(claimed, (int, np.integer)) and tol == 0.0:
        ok = int(claimed) == int(computed)
    else:
        ok = abs(float(claimed) - float(computed)) <= tol
    checks.append({"claim": name, "claimed": claimed,
                   "computed": (int(computed) if isinstance(computed, (int, np.integer))
                                else float(computed)),
                   "tol": tol, "ok": bool(ok)})
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}: claimed {claimed!r} vs computed {computed!r}")
    return ok


def is_permutation(r, n):
    r = np.asarray(r)
    return r.shape == (n,) and np.array_equal(np.sort(r), np.arange(n))


def my_swap(rank, pairs):
    """Reviewer swap: exchange the two members' ranks (simultaneous)."""
    r = rank.copy()
    a, b = pairs[:, 0], pairs[:, 1]
    ra, rb = rank[a].copy(), rank[b].copy()
    r[a] = rb
    r[b] = ra
    return r


def rebuild_all_pairings(rank, side, L, R, types, n, seed):
    """Reviewer replication of exp_mirror_swap.py main()'s pairing block
    (mirror, C1, C1b, C2 in that RNG order), written from that file directly."""
    pairs = []
    for t in np.unique(types):
        m = types == t
        li = np.where(m & (side == L))[0]
        ri = np.where(m & (side == R))[0]
        if len(li) < 1 or len(ri) < 1:
            continue
        li = li[np.argsort(rank[li], kind="stable")]
        ri = ri[np.argsort(rank[ri], kind="stable")]
        k = min(len(li), len(ri))
        pairs.append(np.column_stack([li[:k], ri[:k]]))
    pairs = np.concatenate(pairs, axis=0)
    n_pairs = len(pairs)

    rng = np.random.default_rng(seed)
    c1_pool = []
    for t in np.unique(types):
        for s_ in (L, R):
            idx = np.where((types == t) & (side == s_))[0]
            if len(idx) < 2:
                continue
            idx = rng.permutation(idx)
            k2 = (len(idx) // 2) * 2
            c1_pool.append(idx[:k2].reshape(-1, 2))
    c1_pool = np.concatenate(c1_pool, axis=0)
    take = rng.choice(len(c1_pool), size=min(n_pairs, len(c1_pool)), replace=False)
    c1_pairs = c1_pool[take]

    c1b = []
    for t in np.unique(types):
        for s_ in (L, R):
            idx = np.where((types == t) & (side == s_))[0]
            if len(idx) < 2:
                continue
            idx = rng.permutation(idx)
            half = len(idx) // 2
            a = idx[:half]
            b = idx[half:2 * half]
            a = a[np.argsort(rank[a], kind="stable")]
            b = b[np.argsort(rank[b], kind="stable")]
            c1b.append(np.column_stack([a, b]))
    c1b_pairs = np.concatenate(c1b, axis=0)

    perm = rng.permutation(n)
    c2_pairs = perm[: 2 * n_pairs].reshape(-1, 2)
    return {"mirror": pairs, "c1": c1_pairs, "c1b": c1b_pairs, "c2": c2_pairs}


def main():
    t0 = time.time()
    from layering import io_utils
    from src.mfas.metrics import score_from_order, pct
    from scipy.stats import spearmanr

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. oracle re-scores -----------------------------------
    print("== 1. Oracle re-scores (frozen oracle, CPU) ==")
    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order("connectome")
    src, tgt, w = g.src, g.tgt, g.weight
    Wtot = float(w.sum())
    n = g.n_nodes
    check("total_weight", CLAIMS["total_weight"], int(round(Wtot)))
    check("champion_pct (io_utils gate)", CLAIMS["champion_pct"], ff_pct, tol=1e-12)
    champ_int = int(round(score_from_order(rank, src, tgt, w)))
    print(f"  champion exact int score = {champ_int:,} "
          f"(study parenthetical says {CLAIMS['study_champion_parenthetical_int']:,})")
    champ_paren_ok = champ_int == CLAIMS["study_champion_parenthetical_int"]
    checks.append({"claim": "study parenthetical '(34,751,902 ...)' equals champion exact score",
                   "claimed": CLAIMS["study_champion_parenthetical_int"],
                   "computed": champ_int, "tol": 0, "ok": bool(champ_paren_ok)})

    ret = {}
    for name, pct_key, int_key in (
            ("champion", "retuned_champion_pct", "retuned_champion_int"),
            ("mirror", "retuned_mirror_pct", "retuned_mirror_int"),
            ("c1b", "retuned_c1b_pct", "retuned_c1b_int")):
        v = np.load(RESULTS / f"retuned_{name}_rank.npy")
        perm_ok = is_permutation(v, n)
        checks.append({"claim": f"retuned_{name}_rank.npy is a valid permutation of [0,{n})",
                       "claimed": True, "computed": bool(perm_ok), "tol": 0,
                       "ok": bool(perm_ok)})
        print(f"  [{'OK  ' if perm_ok else 'FAIL'}] retuned_{name} permutation valid")
        s = score_from_order(v, src, tgt, w)
        check(f"retuned_{name} exact int", CLAIMS[int_key], int(round(s)))
        check(f"retuned_{name} pct", CLAIMS[pct_key], pct(s, Wtot), tol=1e-12)
        ret[name] = {"rank": v, "int": int(round(s)), "pct": pct(s, Wtot)}

    # ---------------- 3. swap starts from saved pairs + QA json --------------
    print("\n== 3. Swap-start reconstruction from saved pairs ==")
    pairs_m = np.load(RESULTS / "mirror_pairs.npy")
    pairs_c = np.load(RESULTS / "c1b_pairs.npy")
    for nm, prs, nkey in (("mirror", pairs_m, "n_mirror_pairs"),
                          ("c1b", pairs_c, "n_c1b_pairs")):
        check(f"{nm}_pairs count", CLAIMS[nkey], len(prs))
        flat = prs.ravel()
        disjoint = len(np.unique(flat)) == flat.size
        inrange = flat.min() >= 0 and flat.max() < n
        checks.append({"claim": f"{nm}_pairs disjoint & in-range",
                       "claimed": True, "computed": bool(disjoint and inrange),
                       "tol": 0, "ok": bool(disjoint and inrange)})
        print(f"  [{'OK  ' if disjoint and inrange else 'FAIL'}] {nm}_pairs disjoint & in-range")

    start_m = my_swap(rank, pairs_m)
    start_c = my_swap(rank, pairs_c)
    for nm, st in (("mirror", start_m), ("c1b", start_c)):
        pok = is_permutation(st, n)
        checks.append({"claim": f"swapped start ({nm}) is a permutation", "claimed": True,
                       "computed": bool(pok), "tol": 0, "ok": bool(pok)})
    s_start_m = pct(score_from_order(start_m, src, tgt, w), Wtot)
    s_start_c = pct(score_from_order(start_c, src, tgt, w), Wtot)
    check("mirror start pct (from saved pairs)", CLAIMS["start_mirror_pct"], s_start_m, tol=1e-12)
    check("c1b start pct (from saved pairs)", CLAIMS["start_c1b_pct"], s_start_c, tol=1e-12)

    qa = json.loads(QA_JSON.read_text(encoding="utf-8"))
    check("mirror start == QA exp_mirror_swap.json (bit)",
          qa["mirror_swap"]["ff_pct"], s_start_m, tol=0.0)
    check("c1b start == QA exp_mirror_swap.json (bit)",
          qa["control_same_side_same_type_rank_matched"]["ff_pct"], s_start_c, tol=0.0)

    # full pairing rebuild from annotations (reviewer replication)
    print("\n  full four-scheme pairing rebuild from annotations ...")
    from layering.tools.exp_mirror_swap import CONFIG as SWAP_CONFIG, load_types
    d = np.load(SWAP_CONFIG["ann_npz"])
    meta = json.loads(Path(SWAP_CONFIG["ann_meta"]).read_text(encoding="utf-8"))
    assert np.array_equal(d["rank"], rank), "annotation cache is stale"
    side = d["side"]
    L = meta["side_values"].index("left")
    R = meta["side_values"].index("right")
    types = load_types(g.node_ids)
    reb = rebuild_all_pairings(rank, side, L, R, types, n, seed=42)
    checks.append({"claim": "saved mirror_pairs.npy == rebuilt-from-annotations pairing",
                   "claimed": True, "computed": bool(np.array_equal(reb["mirror"], pairs_m)),
                   "tol": 0, "ok": bool(np.array_equal(reb["mirror"], pairs_m))})
    checks.append({"claim": "saved c1b_pairs.npy == rebuilt-from-annotations pairing",
                   "claimed": True, "computed": bool(np.array_equal(reb["c1b"], pairs_c)),
                   "tol": 0, "ok": bool(np.array_equal(reb["c1b"], pairs_c))})
    print(f"  mirror pairs identical: {np.array_equal(reb['mirror'], pairs_m)}; "
          f"c1b identical: {np.array_equal(reb['c1b'], pairs_c)}")
    # the two schemes not saved by the study: score them against the QA json too
    s_c1 = pct(score_from_order(my_swap(rank, reb["c1"]), src, tgt, w), Wtot)
    s_c2 = pct(score_from_order(my_swap(rank, reb["c2"]), src, tgt, w), Wtot)
    check("C1 start == QA json (bit)",
          qa["control_same_side_same_type_random"]["ff_pct"], s_c1, tol=0.0)
    check("C2 start == QA json (bit)",
          qa["control_random_pairs"]["ff_pct"], s_c2, tol=0.0)

    # ---------------- 2. derived quantities ----------------------------------
    print("\n== 2. Derived quantities ==")
    ra, rb, rc = ret["champion"]["pct"], ret["mirror"]["pct"], ret["c1b"]["pct"]
    check("gap retuned(a)-retuned(b)", CLAIMS["gap_a_minus_b_pp"], ra - rb, tol=1e-12)
    check("gap retuned(a)-retuned(c)", CLAIMS["gap_a_minus_c_pp"], ra - rc, tol=1e-12)
    check("gap retuned(c)-retuned(b)", CLAIMS["gap_c_minus_b_pp"], rc - rb, tol=1e-12)
    check("mirror recovered pp", CLAIMS["mirror_recovered_pp"], rb - s_start_m, tol=1e-12)
    check("mirror recovered frac of deficit", CLAIMS["mirror_recovered_frac"],
          (rb - s_start_m) / (ff_pct - s_start_m), tol=1e-12)
    check("c1b recovered pp", CLAIMS["c1b_recovered_pp"], rc - s_start_c, tol=1e-12)
    check("c1b recovered frac of deficit", CLAIMS["c1b_recovered_frac"],
          (rc - s_start_c) / (ff_pct - s_start_c), tol=1e-12)

    # un-swap fractions: stayed == within-pair rank relation still the exchanged
    # one, i.e. sign(r_new[a]-r_new[b]) != sign(rank[a]-rank[b]) (champion's).
    def stay_frac(prs, r_new):
        a, b = prs[:, 0], prs[:, 1]
        orig = rank[a] < rank[b]
        new = r_new[a] < r_new[b]
        return float((new != orig).mean())

    st_m = stay_frac(pairs_m, ret["mirror"]["rank"])
    st_c = stay_frac(pairs_c, ret["c1b"]["rank"])
    check("mirror frac stayed swapped", CLAIMS["mirror_frac_stayed"], st_m, tol=1e-12)
    check("mirror frac reverted", CLAIMS["mirror_frac_reverted"], 1 - st_m, tol=1e-9)
    check("c1b frac stayed swapped", CLAIMS["c1b_frac_stayed"], st_c, tol=1e-12)
    check("c1b frac reverted", CLAIMS["c1b_frac_reverted"], 1 - st_c, tol=1e-9)

    rho1 = spearmanr(start_m, rank).statistic
    rho2 = spearmanr(ret["mirror"]["rank"], rank).statistic
    rho3 = spearmanr(ret["mirror"]["rank"], start_m).statistic
    check("Spearman swapped start vs champion", CLAIMS["rho_swap_vs_champ"], rho1, tol=1e-9)
    check("Spearman retuned mirror vs champion", CLAIMS["rho_ret_vs_champ"], rho2, tol=1e-9)
    check("Spearman retuned mirror vs swapped start", CLAIMS["rho_ret_vs_start"], rho3, tol=1e-9)
    check("mean |rank shift| retuned mirror vs champion", CLAIMS["shift_ret_vs_champ"],
          float(np.abs(ret["mirror"]["rank"] - rank).mean()), tol=1e-6)
    check("mean |rank shift| retuned mirror vs swapped start", CLAIMS["shift_ret_vs_start"],
          float(np.abs(ret["mirror"]["rank"] - start_m).mean()), tol=1e-6)

    # cross-check the study JSON quotes the same values it should
    sj = json.loads(STUDY_JSON.read_text(encoding="utf-8"))
    jm = sj["starts"]["mirror"]
    check("study JSON retuned mirror pct == oracle re-score",
          jm["retuned_ff_pct"], rb, tol=0.0)

    # ---------------- 4. red-team: stay-swapped contrast ----------------------
    print("\n== 4. Red-team: stay-swapped contrast ==")
    n_m, n_c = len(pairs_m), len(pairs_c)
    p_pool = (st_m * n_m + st_c * n_c) / (n_m + n_c)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_m + 1 / n_c))
    z = (st_m - st_c) / se
    print(f"  stay-swapped: mirror {st_m:.4f} (n={n_m:,}) vs c1b {st_c:.4f} (n={n_c:,})"
          f"  ->  two-proportion z = {z:.1f} (pairs treated as independent)")

    # displacement confound: stay fraction by pooled |rank displacement| decile
    dr_m = np.abs(rank[pairs_m[:, 0]] - rank[pairs_m[:, 1]])
    dr_c = np.abs(rank[pairs_c[:, 0]] - rank[pairs_c[:, 1]])
    a_m = ret["mirror"]["rank"]
    a_c = ret["c1b"]["rank"]
    stay_m_pair = ((a_m[pairs_m[:, 0]] < a_m[pairs_m[:, 1]])
                   != (rank[pairs_m[:, 0]] < rank[pairs_m[:, 1]]))
    stay_c_pair = ((a_c[pairs_c[:, 0]] < a_c[pairs_c[:, 1]])
                   != (rank[pairs_c[:, 0]] < rank[pairs_c[:, 1]]))
    edges = np.quantile(np.concatenate([dr_m, dr_c]), np.linspace(0, 1, 11))
    edges[0], edges[-1] = -1, np.inf
    bins_tbl = []
    print("  stay-swapped by pooled |rank displacement| decile "
          "(mirror n / stay  vs  c1b n / stay):")
    for i in range(10):
        mm = (dr_m > edges[i]) & (dr_m <= edges[i + 1])
        mc = (dr_c > edges[i]) & (dr_c <= edges[i + 1])
        row = {"bin": i,
               "dr_range": [float(edges[i]), float(edges[i + 1])],
               "mirror_n": int(mm.sum()),
               "mirror_stay": float(stay_m_pair[mm].mean()) if mm.any() else None,
               "c1b_n": int(mc.sum()),
               "c1b_stay": float(stay_c_pair[mc].mean()) if mc.any() else None}
        bins_tbl.append(row)
        print(f"    bin {i}: drank ({edges[i]:>9.0f}, {edges[i+1]:>9.0f}] "
              f" mirror {row['mirror_n']:6,} / {row['mirror_stay'] if row['mirror_stay'] is not None else float('nan'):.3f}"
              f"   c1b {row['c1b_n']:6,} / {row['c1b_stay'] if row['c1b_stay'] is not None else float('nan'):.3f}")
    print(f"  mean |drank|: mirror {dr_m.mean():,.0f}  vs c1b {dr_c.mean():,.0f} "
          f"(displacement distributions are NOT matched)")

    # displacement-standardised contrast: reweight c1b bins to the mirror
    # displacement profile (direct standardisation)
    wts = np.array([r["mirror_n"] for r in bins_tbl], dtype=float)
    c1b_rates = np.array([r["c1b_stay"] if r["c1b_stay"] is not None else np.nan
                          for r in bins_tbl])
    m_rates = np.array([r["mirror_stay"] if r["mirror_stay"] is not None else np.nan
                        for r in bins_tbl])
    ok_bins = ~np.isnan(c1b_rates) & ~np.isnan(m_rates) & (wts > 0)
    c1b_std = float(np.nansum(c1b_rates[ok_bins] * wts[ok_bins]) / wts[ok_bins].sum())
    m_std = float(np.nansum(m_rates[ok_bins] * wts[ok_bins]) / wts[ok_bins].sum())
    print(f"  displacement-standardised (mirror-profile weights): "
          f"mirror {m_std:.4f} vs c1b {c1b_std:.4f} "
          f"-> adjusted gap {m_std - c1b_std:+.4f} (raw gap {st_m - st_c:+.4f})")

    out = {
        "reviewer": "independent review script (biology/analysis/review_mirror_retune.py)",
        "date": "2026-09-30",
        "champion_int_score": champ_int,
        "retuned_int_scores": {k: v["int"] for k, v in ret.items()},
        "retuned_pcts": {k: v["pct"] for k, v in ret.items()},
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["ok"]),
        "redteam_stay_swapped": {
            "mirror_stay": st_m, "c1b_stay": st_c,
            "two_proportion_z_independent_pairs": float(z),
            "mean_abs_drank_mirror": float(dr_m.mean()),
            "mean_abs_drank_c1b": float(dr_c.mean()),
            "stay_by_pooled_drank_decile": bins_tbl,
            "displacement_standardised": {
                "mirror": m_std, "c1b_reweighted_to_mirror_profile": c1b_std,
                "adjusted_gap": m_std - c1b_std, "raw_gap": st_m - st_c},
        },
        "seconds": time.time() - t0,
        "rerun_command": "$PY biology/analysis/review_mirror_retune.py",
    }
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n{out['n_failed']} of {out['n_checks']} checks FAILED"
          if out["n_failed"] else
          f"\nall {out['n_checks']} checks PASSED")
    print(f"results -> {OUT_JSON}  ({out['seconds']:.0f}s)")


if __name__ == "__main__":
    main()

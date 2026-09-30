"""INDEPENDENT REVIEW of biology/studies/central_only.md (2026-09-30).

Re-derives every headline number of the central-only study WITHOUT running or
importing any code path from biology/analysis/exp_central_only.py:

  - the induced 'central' subgraph is rebuilt from
    layering.io_utils.load_graph_and_champion_order + node_ann.npz with review
    code, then compared array-by-array against the saved central_subgraph.npz;
  - the champion-restricted rank is recompressed with scipy.stats.rankdata
    (a different code path than the study's double argsort);
  - every saved rank vector is re-scored through the frozen oracle
    (src.mfas.metrics.score_from_order, CPU);
  - all Spearman/Kendall/displacement statistics are recomputed from the saved
    .npy vectors;
  - the CX / TuBu membership and every boundary-flow number are re-derived from
    the RAW annotation TSV (data/external/flywire_neuron_annotations.tsv.gz)
    joined on root IDs — i.e. independently of node_ann.npz as well.

Output: biology/results/review/review_central_only.json + a printed verdict
table (CONFIRMED / MISMATCH per claim).

Run:
  PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
  $PY biology/analysis/review_central_only.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RES = REPO_ROOT / "biology" / "results"
OUT_DIR = RES / "review"
OUT_JSON = OUT_DIR / "review_central_only.json"
ANN_TSV = REPO_ROOT / "data" / "external" / "flywire_neuron_annotations.tsv.gz"
ANN_NPZ = REPO_ROOT / "layering" / "outputs" / "fly_ann" / "node_ann.npz"
ANN_META = REPO_ROOT / "layering" / "outputs" / "fly_ann" / "node_ann_meta.json"

CHECKS: list[dict] = []


def check(name: str, claimed, got, tol: float = 0.0, note: str = "") -> None:
    """Record one claim-vs-recomputation comparison."""
    if isinstance(claimed, bool) or claimed is None:
        ok = claimed == got
    else:
        try:
            ok = abs(float(claimed) - float(got)) <= tol
        except (TypeError, ValueError):
            ok = claimed == got
    CHECKS.append(dict(name=name, claimed=claimed, got=got, tol=tol,
                       ok=bool(ok), note=note))
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: claimed={claimed} got={got}"
          f"{'  (' + note + ')' if note else ''}")


def rank_compress(r: np.ndarray) -> np.ndarray:
    """Dense 0..m-1 ranks preserving order — scipy path, independent of the
    study's double-argsort implementation. All inputs are distinct here."""
    return (stats.rankdata(r, method="ordinal") - 1).astype(np.int64)


def disp_stats(rank_a: np.ndarray, rank_b: np.ndarray, line_len: int) -> dict:
    """Normalized |rank displacement| statistics between two rank vectors."""
    d = np.abs(rank_a.astype(np.float64) - rank_b.astype(np.float64)) / (line_len - 1)
    q25, q50, q75 = np.percentile(d, (25, 50, 75))
    return dict(median=float(q50), q25=float(q25), q75=float(q75),
                frac_gt_5pct=float((d > 0.05).mean()))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from layering import io_utils                      # parity-gated loader
    from src.mfas.metrics import score_from_order, pct  # frozen oracle

    # ------------------------------------------------------------------ load
    g, _order, rank_full, ff_pct_full = io_utils.load_graph_and_champion_order(
        "connectome")
    n_full = int(g.n_nodes)
    src_f = np.asarray(g.src, dtype=np.int64)
    tgt_f = np.asarray(g.tgt, dtype=np.int64)
    w_f = np.asarray(g.weight, dtype=np.int64)
    node_ids = np.asarray(g.node_ids, dtype=np.int64)
    rank_full = np.asarray(rank_full, dtype=np.int64)

    ann = np.load(ANN_NPZ)
    meta = json.loads(ANN_META.read_text(encoding="utf-8"))
    print("== staleness gate (review's own) ==")
    check("node_ann.rank == pinned champion rank",
          True, bool(np.array_equal(np.asarray(ann["rank"], np.int64), rank_full)))

    # -------------------------------------------------- 1. subgraph rebuild
    print("== 1. induced central subgraph, rebuilt ==")
    central_code = meta["super_class_values"].index("central")
    mask = np.asarray(ann["super_class"]) == central_code
    central_nodes = np.flatnonzero(mask)
    m = int(central_nodes.size)
    keep = mask[src_f] & mask[tgt_f]
    remap = np.full(n_full, -1, dtype=np.int64)
    remap[central_nodes] = np.arange(m)
    my_src = remap[src_f[keep]]
    my_tgt = remap[tgt_f[keep]]
    my_w = w_f[keep]
    W_sub = int(my_w.sum())
    W_full = int(w_f.sum())

    check("nodes", 32294, m)
    check("nodes % of full", 23.63, 100.0 * m / n_full, tol=0.005)
    check("edges", 849937, int(keep.sum()))
    check("edges % of full", 15.02, 100.0 * keep.sum() / len(w_f), tol=0.005)
    check("subgraph weight", 12523698, W_sub)
    check("weight % of full", 29.88, 100.0 * W_sub / W_full, tol=0.005)
    check("full weight", 41912141, W_full)
    check("central mean edge weight / full mean ('~2x heavier')",
          2.0, (W_sub / keep.sum()) / (W_full / len(w_f)), tol=0.02)

    z = np.load(RES / "central_subgraph.npz")
    check("saved npz src == rebuilt", True,
          bool(np.array_equal(np.asarray(z["src"], np.int64), my_src)))
    check("saved npz tgt == rebuilt", True,
          bool(np.array_equal(np.asarray(z["tgt"], np.int64), my_tgt)))
    check("saved npz weight == rebuilt", True,
          bool(np.array_equal(np.asarray(z["weight"], np.int64), my_w)))
    check("saved npz node_ids == g.node_ids[central]", True,
          bool(np.array_equal(np.asarray(z["node_ids"], np.int64),
                              node_ids[central_nodes])))

    # ------------------------------------- 2. re-score through frozen oracle
    print("== 2. frozen-oracle re-scores of the saved rank vectors ==")
    r_champ = np.load(RES / "central_only_champ_restricted_rank.npy").astype(np.int64)
    r1 = np.load(RES / "central_only_run1_rank.npy").astype(np.int64)
    r2 = np.load(RES / "central_only_run2_rank.npy").astype(np.int64)
    for nm, r in (("champ_restricted", r_champ), ("run1", r1), ("run2", r2)):
        check(f"{nm} is a permutation of 0..{m-1}", True,
              bool(r.shape[0] == m and np.array_equal(np.sort(r), np.arange(m))))
    check("champ_restricted == my compression of champion|central", True,
          bool(np.array_equal(r_champ, rank_compress(rank_full[central_nodes]))))

    s_champ = score_from_order(r_champ, my_src, my_tgt, my_w)
    s1 = score_from_order(r1, my_src, my_tgt, my_w)
    s2 = score_from_order(r2, my_src, my_tgt, my_w)
    check("champ_restricted score", 10356669, s_champ)
    check("run1 score", 10449116, s1)
    check("run2 score", 10449826, s2)
    check("champ_restricted FF%", 82.6966, pct(s_champ, W_sub), tol=5e-5)
    check("run1 FF%", 83.4347, pct(s1, W_sub), tol=5e-5)
    check("run2 FF%", 83.4404, pct(s2, W_sub), tol=5e-5)
    check("gap run1 (pp)", 0.7382, pct(s1, W_sub) - pct(s_champ, W_sub), tol=5e-5)
    check("gap run2 (pp)", 0.7438, pct(s2, W_sub) - pct(s_champ, W_sub), tol=5e-5)

    # ------------------------------------------------- 3. order comparisons
    print("== 3. order comparisons recomputed from the saved vectors ==")
    def cmp_pair(tag, a, b, claims):
        rho = float(stats.spearmanr(a, b).statistic)
        tau = float(stats.kendalltau(a, b).statistic)
        d = disp_stats(a, b, m)
        check(f"{tag} Spearman", claims[0], rho, tol=5e-4)
        check(f"{tag} Kendall", claims[1], tau, tol=5e-4)
        check(f"{tag} median disp", claims[2], d["median"], tol=5e-4)
        check(f"{tag} IQR lo", claims[3], d["q25"], tol=5e-3)
        check(f"{tag} IQR hi", claims[4], d["q75"], tol=5e-3)
        check(f"{tag} frac >5%", claims[5], d["frac_gt_5pct"], tol=5e-3)
        return rho, tau, d

    cmp_pair("run1 vs champ", r1, r_champ,
             (0.7440, 0.6010, 0.0670, 0.025, 0.177, 0.578))
    cmp_pair("run2 vs champ", r2, r_champ,
             (0.7405, 0.5969, 0.0700, 0.027, 0.175, 0.616))
    cmp_pair("run1 vs run2 (noise)", r1, r2,
             (0.9602, 0.8867, 0.0141, 0.004, 0.032, 0.154))

    # ------------------------- 4. CX claims, re-derived from the RAW TSV join
    print("== 4. CX (and TuBu) re-derived from the raw annotation TSV ==")
    sc_by_root: dict[int, str] = {}
    cc_by_root: dict[int, str] = {}
    with gzip.open(ANN_TSV, "rt", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_root = header.index("root_id")
        i_sc = header.index("super_class")
        i_cc = header.index("cell_class")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            rid = int(p[i_root])
            sc_by_root[rid] = p[i_sc] or "NA"
            cc_by_root[rid] = p[i_cc] or "NA"
    sc_node = np.array([sc_by_root.get(int(r), "NA") for r in node_ids])
    cc_node = np.array([cc_by_root.get(int(r), "NA") for r in node_ids])
    n_unmatched = int(sum(1 for r in node_ids if int(r) not in sc_by_root))
    check("unmatched graph nodes in TSV join", 13, n_unmatched)
    check("TSV central set == node_ann central set", True,
          bool(np.array_equal(np.flatnonzero(sc_node == "central"), central_nodes)))

    cx_full = (sc_node == "central") & (cc_node == "CX")
    check("CX n (central, graph-joined)", 2848, int(cx_full.sum()),
          note="raw TSV has 2875 CX rows; 27 are not graph nodes")
    # Definition audit: every TSV CX row is super_class central; CX includes
    # columnar + tangential + ring neurons + LNOa (Schlegel et al. 2024 table's
    # own cell_class column — no re-labeling by the study).
    cx_sub = cx_full[central_nodes]                    # mask on the central line
    d1c = (r1.astype(np.float64) - r_champ) / (m - 1)  # signed, run1 vs champ
    d12 = (r1.astype(np.float64) - r2) / (m - 1)       # signed, run1 vs run2
    cx_disp = float(np.median(np.abs(d1c[cx_sub])))
    cx_noise = float(np.median(np.abs(d12[cx_sub])))
    check("CX median |disp| run1-vs-champ", 0.276, cx_disp, tol=5e-4)
    check("CX median |disp| noise (run1-vs-run2)", 0.0201, cx_noise, tol=5e-4)
    check("CX signal/noise ratio", 13.8, cx_disp / cx_noise, tol=0.06)
    check("CX median champ rank frac", 0.51,
          float(np.median(r_champ[cx_sub])) / (m - 1), tol=5e-3)
    check("CX median run1 rank frac", 0.79,
          float(np.median(r1[cx_sub])) / (m - 1), tol=5e-3)
    check("CX median run2 rank frac (direction reproduced)", 0.81,
          float(np.median(r2[cx_sub])) / (m - 1), tol=5e-3)
    cx_full_line_frac = float(np.median(rank_full[cx_full])) / (n_full - 1)
    print(f"    (context) CX median rank frac on the FULL-brain line: "
          f"{cx_full_line_frac:.3f} — 'mid-line' is true only OF THE CENTRAL "
          f"LINE, not of the whole brain")

    # Boundary flows from the full graph + TSV-derived classes.
    def flow(mask_nodes: np.ndarray, direction: str) -> dict:
        if direction == "out":
            emask, other = mask_nodes[src_f], sc_node[tgt_f]
        else:
            emask, other = mask_nodes[tgt_f], sc_node[src_f]
        tot = float(w_f[emask].sum())
        return {v: float(w_f[emask & (other == v)].sum()) / tot
                for v in np.unique(other[emask])}, tot

    cx_out, cx_out_w = flow(cx_full, "out")
    cx_in, cx_in_w = flow(cx_full, "in")
    check("CX out-weight frac to central", 0.9932, cx_out["central"], tol=5e-4)
    check("CX out-weight frac to descending", 0.0048,
          cx_out.get("descending", 0.0), tol=5e-4)
    check("CX in-weight frac from central", 0.9961, cx_in["central"], tol=5e-4)
    check("CX out total weight", 1032064, cx_out_w)
    check("CX in total weight", 1108535, cx_in_w)

    tubu_full = (sc_node == "central") & (cc_node == "TuBu")
    check("TuBu n", 150, int(tubu_full.sum()))
    tb_in, _ = flow(tubu_full, "in")
    tb_out, _ = flow(tubu_full, "out")
    check("TuBu in-weight frac from visual_projection", 0.6534,
          tb_in.get("visual_projection", 0.0), tol=5e-4)
    check("TuBu in-weight frac from central", 0.3466,
          tb_in.get("central", 0.0), tol=5e-4)
    check("TuBu out-weight frac to central", 1.0,
          tb_out.get("central", 0.0), tol=1e-6)
    tubu_sub = tubu_full[central_nodes]
    tb_disp = float(np.median(np.abs(d1c[tubu_sub])))
    tb_noise = float(np.median(np.abs(d12[tubu_sub])))
    check("TuBu median |disp| run1-vs-champ", 0.145, tb_disp, tol=5e-4)
    check("TuBu signal/noise ratio", 1.6, tb_disp / tb_noise, tol=0.05)

    # DAN spot-check (the 11.1x row with the bimodality note)
    dan_sub = ((sc_node == "central") & (cc_node == "DAN"))[central_nodes]
    check("DAN n", 320, int(dan_sub.sum()))
    check("DAN median |disp|", 0.005, float(np.median(np.abs(d1c[dan_sub]))),
          tol=5e-4)
    check("DAN frac moved >5% ('~36% minority')", 0.356,
          float((np.abs(d1c[dan_sub]) > 0.05).mean()), tol=5e-3)
    # Sensory-side backbone barely moves
    for cls, claimed in (("ALPN", 0.018), ("ALLN", 0.016), ("ALIN", 0.021),
                         ("LHLN", 0.034), ("Kenyon_Cell", 0.076), ("NA", 0.063)):
        sel = ((sc_node == "central") & (cc_node == cls))[central_nodes]
        check(f"{cls} median |disp|", claimed,
              float(np.median(np.abs(d1c[sel]))), tol=1e-3)
    na_sub = ((sc_node == "central") & (cc_node == "NA"))[central_nodes]
    check("NA n (67.9% of central)", 21912, int(na_sub.sum()))
    check("NA signal/noise ratio", 4.9,
          float(np.median(np.abs(d1c[na_sub])))
          / float(np.median(np.abs(d12[na_sub]))), tol=0.05)

    # --------------------------------------------- 5. excluding CX entirely
    print("== 5. excluding CX ==")
    non_cx = ~cx_sub
    check("non-CX n", 29446, int(non_cx.sum()))
    check("Spearman run1-vs-champ excluding CX", 0.778,
          float(stats.spearmanr(r1[non_cx], r_champ[non_cx]).statistic),
          tol=5e-4)

    # ------------------------------------ 6. full-brain super_class medians
    print("== 6. full-brain context medians ==")
    fr = rank_full.astype(np.float64) / (n_full - 1)
    for sc_name, claimed in (("ascending", 0.06), ("sensory", 0.07),
                             ("visual_centrifugal", 0.32), ("optic", 0.46),
                             ("visual_projection", 0.81), ("central", 0.82),
                             ("descending", 0.96), ("endocrine", 0.99),
                             ("motor", 0.999)):
        check(f"median full rank frac {sc_name}", claimed,
              float(np.median(fr[sc_node == sc_name])), tol=6e-3)

    # ------------------------------------------------------------- summary
    n_fail = sum(1 for c in CHECKS if not c["ok"])
    summary = dict(
        reviewed="biology/studies/central_only.md",
        review_date="2026-09-30",
        n_checks=len(CHECKS), n_failed=n_fail,
        cx_median_rank_frac_full_brain_line=cx_full_line_frac,
        checks=CHECKS,
    )
    OUT_JSON.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n{len(CHECKS)} checks, {n_fail} failed -> {OUT_JSON}")


if __name__ == "__main__":
    main()

"""Generate layering/layering_mouse.ipynb - the v1 mouse layering prototype.

Per repo convention the notebook is a BUILD ARTIFACT: edit this generator and
regenerate; do not hand-edit the .ipynb. Regenerate + execute with:

    $PY layering/create_notebook.py
    $PY layering/tools/exec_notebook.py layering/layering_mouse.ipynb
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT_NB = HERE / "layering_mouse.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = []

# ---------------------------------------------------------------------------
cells.append(md("""
    # Layered (ANN-style) representation of the MFAS-ordered mouse connectome — v1 prototype

    **Goal.** Take the champion MFAS order `pi` (H63) and re-draw the graph the way
    artificial neural networks are drawn: nodes assigned to **layers** with no
    intra-layer feedforward edge, edges allowed to skip layers, feedback drawn as
    separate red arcs, and nodes ordered inside each layer to keep the number of
    edge **crossings** low.

    Fixed decisions (v1) — full spec in `layering/README.md`:

    1. feedback edges are drawn as separate arcs, never reversed or dropped;
    2. HARD layering (zero intra-layer FF edges); the soft version is future work;
    3. edges may skip layers; 4. crossings are counted, not weighted;
    5. mouse first (148 nodes), fly connectome later; 6. deliverable = picture + structure;
    7. `pi` is the fixed input (joint layer/order optimization is deferred, T5);
    8. sources/sinks are just the first/last layers; 9. within-layer order = barycenter v1.

    Two layer-assignment methods are compared side by side (decision T2):

    - **A. FF-DAG longest-path** — earliest feasible layer; fewest layers; may
      reorder nodes relative to `pi` on FF-incomparable pairs, so some
      `pi`-feedback can be **reclaimed** (become forward-by-layer);
    - **B. pi-slices** — layers are contiguous slices of `pi`; `pi` fully
      preserved; feedback can never be reclassified.
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
    ## Configuration

    All hyperparameters live in `CONFIG` (no magic numbers). Seeds are set even
    though every step below is deterministic — reproducibility by convention.
"""))
cells.append(code("""
    CONFIG = {
        "dataset": "mouse",                 # v1 target (decision 5)
        "barycenter_sweeps": 10,            # within-layer ordering budget (decision 9)
        "figure_dpi": 120,
        "outputs_dir": "outputs",           # relative to layering/
        "global_seed": 42,
    }

    import sys
    from pathlib import Path

    REPO_ROOT = Path.cwd().resolve()
    while not (REPO_ROOT / "src" / "mfas").exists():
        if REPO_ROOT.parent == REPO_ROOT:
            raise RuntimeError("repo root not found above cwd")
        REPO_ROOT = REPO_ROOT.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from layering import core, draw, io_utils

    np.random.seed(CONFIG["global_seed"])
    OUT = REPO_ROOT / "layering" / CONFIG["outputs_dir"]
    OUT.mkdir(exist_ok=True)
    print("repo root:", REPO_ROOT)
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
    ## Load the graph and the champion order (parity-gated)

    The order comes from the **tracked** champion artifact
    `results/champions/H63_mouse_s42_cuda.npz` (pinned by `tools/pin_champion.py`).
    `io_utils` re-scores it through the frozen oracle and compares against
    `autoresearch/sota.json` — a stale or wrong artifact raises instead of
    silently layering a non-champion order.
"""))
cells.append(code("""
    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order(CONFIG["dataset"])
    print(f"dataset: {g.name}  nodes: {g.n_nodes}  edges: {g.n_edges}  "
          f"total weight: {g.total_weight:.6f}")
    print(f"champion feedforward pct (verified vs sota.json): {ff_pct:.11f}")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
    ## Step 0 — split edges by the order `pi`

    An edge `(u, v)` is **feedforward** iff `rank[u] < rank[v]`; everything else
    is feedback. Mutual pairs (both `u->v` and `v->u` present) put a hard floor
    under feedback: one direction of each pair is backward under ANY order and
    ANY valid layering.
"""))
cells.append(code("""
    ff = core.ff_mask_by_order(g.src, g.tgt, rank)
    mutual = core.mutual_edge_mask(g.src, g.tgt)
    w = np.asarray(g.weight, dtype=np.float64)
    total_w = w.sum()
    n_fb = int((~ff).sum())
    print(f"FF edges: {int(ff.sum())} ({100 * w[ff].sum() / total_w:.4f}% of weight)")
    print(f"FB edges: {n_fb} ({100 * w[~ff].sum() / total_w:.4f}% of weight)")
    print(f"mutual pairs: {int(mutual.sum()) // 2}; FB edges that are halves of "
          f"mutual pairs: {int(((~ff) & mutual).sum())} of {n_fb} "
          f"(unreclaimable by any layering)")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
    ## Layer assignment — methods A and B side by side

    Hard invariants are asserted, not assumed: every FF edge must strictly
    increase the layer (both methods), B must be monotone along `pi`, and A
    (earliest-feasible) must be pointwise ≤ B.
"""))
cells.append(code("""
    layer_A = core.layers_longest_path(g.src, g.tgt, ff, order)
    layer_B = core.layers_pi_slices(g.src, g.tgt, ff, order)

    for name, layer in (("A", layer_A), ("B", layer_B)):
        assert np.all(layer[g.src[ff]] < layer[g.tgt[ff]]), f"intra/backward FF edge in {name}"
    assert np.all(np.diff(layer_B[order]) >= 0), "B is not monotone along pi"
    assert np.all(layer_A <= layer_B), "A is not pointwise minimal"
    print("hard invariants OK")

    stats_A = core.layout_stats(g.src, g.tgt, g.weight, rank, layer_A, "A: FF-DAG longest-path")
    stats_B = core.layout_stats(g.src, g.tgt, g.weight, rank, layer_B, "B: pi-slices")
    scalar_keys = [k for k in stats_A if k not in ("widths", "span_hist")]
    pd.DataFrame([{k: s[k] for k in scalar_keys} for s in (stats_A, stats_B)]).set_index("method").T
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
    ## Feedback reclamation (decision T2)

    Edge classes on the drawing are **by layers**, not by `pi`. Method A can move
    an FF-incomparable node into an early layer, turning a `pi`-feedback edge
    into a forward-by-layer edge ("reclaimed"). Method B cannot, by construction.
    Here we measure how much that freedom is actually worth on the mouse graph.
"""))
cells.append(code("""
    rows = []
    for name, layer in (("A: FF-DAG longest-path", layer_A), ("B: pi-slices", layer_B)):
        cls = core.classify_by_layers(g.src, g.tgt, layer)
        reclaimed = (~ff) & (cls == core.FORWARD)
        intra = cls == core.INTRA
        rows.append({
            "method": name,
            "reclaimed_fb_edges": int(reclaimed.sum()),
            "reclaimed_fb_weight_pct": 100 * w[reclaimed].sum() / total_w,
            "intra_layer_edges": int(intra.sum()),
            "fwd_by_layer_weight_pct": 100 * w[cls == core.FORWARD].sum() / total_w,
            "ff_by_pi_weight_pct": 100 * w[ff].sum() / total_w,
        })
        if 0 < int(reclaimed.sum()) <= 20:
            for e in np.where(reclaimed)[0].tolist():
                print(f"  {name}: reclaimed edge {int(g.src[e])} -> {int(g.tgt[e])} "
                      f"(layers {int(layer[g.src[e]])} -> {int(layer[g.tgt[e]])}, "
                      f"w={float(w[e]):.6f})")
    pd.DataFrame(rows).set_index("method").T
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
    ## Within-layer ordering — crossings before and after barycenter

    Initial slots follow `pi` inside each layer; the barycenter heuristic then
    sweeps left-to-right / right-to-left, sorting each layer by the mean current
    y of its neighbours on the swept-from side. The crossing count is the number
    of **proper geometric crossings of the straight segments actually drawn**
    (forward-by-layer edges only; feedback arcs are outside the objective, T4).
"""))
cells.append(code("""
    layouts = {}
    rows = []
    for name, layer in (("A", layer_A), ("B", layer_B)):
        slot0 = core.initial_slots_from_order(layer, rank)
        mask_fwd = core.classify_by_layers(g.src, g.tgt, layer) == core.FORWARD
        x = layer.astype(np.float64)
        c0 = core.count_crossings(g.src, g.tgt, mask_fwd, x, core.y_coords(layer, slot0))
        slot, c_best = core.barycenter_order(g.src, g.tgt, mask_fwd, layer, slot0,
                                             CONFIG["barycenter_sweeps"])
        layouts[name] = (layer, slot)
        rows.append({"method": name,
                     "crossings_pi_order": c0,
                     "crossings_barycenter": c_best,
                     "reduction_pct": (100.0 * (c0 - c_best) / c0) if c0 else 0.0})
    pd.DataFrame(rows).set_index("method")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
    ## Drawing — method A (FF-DAG longest-path layers)

    Blue straight segments: forward-by-layer (always left → right, so no
    arrowheads). Red arcs: feedback-by-layer. Orange arcs (if any): intra-layer
    edges between FF-incomparable nodes (T1). Line width encodes edge weight
    (presentation only — the crossing objective stays unweighted).
"""))
cells.append(code("""
    layer, slot = layouts["A"]
    fig, ax = draw.draw_layered(
        g.src, g.tgt, g.weight, layer, slot, dpi=CONFIG["figure_dpi"],
        title="Mouse connectome, method A: FF-DAG longest-path layers "
              "(H63 champion order, barycenter within-layer)")
    fig.savefig(OUT / "layering_mouse_ffdag.png", dpi=CONFIG["figure_dpi"],
                bbox_inches="tight")
    print("saved:", OUT / "layering_mouse_ffdag.png")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
    ## Drawing — method B (pi-slices layers)

    Same graph, same conventions; layers are contiguous slices of the champion
    order, so reading the x-axis left to right reproduces `pi` exactly.
"""))
cells.append(code("""
    layer, slot = layouts["B"]
    fig, ax = draw.draw_layered(
        g.src, g.tgt, g.weight, layer, slot, dpi=CONFIG["figure_dpi"],
        title="Mouse connectome, method B: pi-slice layers "
              "(H63 champion order, barycenter within-layer)")
    fig.savefig(OUT / "layering_mouse_pislices.png", dpi=CONFIG["figure_dpi"],
                bbox_inches="tight")
    print("saved:", OUT / "layering_mouse_pislices.png")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
    ## Structure — width profiles and forward-edge span

    The "structure" half of the deliverable (decision 6): how wide each layer is,
    and how far forward edges jump (span 1 = adjacent layers, like a textbook
    ANN; larger span = skip connections). These profiles are what we will later
    compare against degree-matched random graphs (project research goal 2).
"""))
cells.append(code("""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=CONFIG["figure_dpi"])
    for j, stats in enumerate((stats_A, stats_B)):
        ax = axes[0][j]
        ax.bar(range(len(stats["widths"])), stats["widths"], color="#5b7fa6")
        ax.set_title(f"{stats['method']}: layer width profile", fontsize=10)
        ax.set_xlabel("layer")
        ax.set_ylabel("neurons")
        ax = axes[1][j]
        ax.bar(range(len(stats["span_hist"])), stats["span_hist"], color="#7a9b6d")
        ax.set_title(f"{stats['method']}: forward-edge span", fontsize=10)
        ax.set_xlabel("span (layers jumped)")
        ax.set_ylabel("edges")
    fig.tight_layout()
    fig.savefig(OUT / "layering_mouse_structure.png", dpi=CONFIG["figure_dpi"],
                bbox_inches="tight")
    print("saved:", OUT / "layering_mouse_structure.png")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
    ## Observations and next steps

    Headline structure of the result (numbers from the tables above, H63 order):

    - Method **A** compresses the 148-neuron line into **26 layers** vs **49**
      for method **B**, at the price of reordering FF-incomparable nodes
      relative to `pi`. Layer 0 under A is wide (43 neurons): every neuron with
      no FF in-edge slides to the front.
    - **Reclamation is exactly zero on the mouse graph** — method A's extra
      freedom recovered 0 of the 124 feedback edges, and produced 0 intra-layer
      edges, so both methods show the identical 93.1754% / 6.8246% split as the
      line `pi`. The floor from Step 0 explains 83 of those 124 (mutual pairs,
      unreclaimable in principle); the remaining 41 all have an FF path from
      target to source. Whether reclamation stays zero on the fly connectome is
      an open question for the scale-up.
    - The barycenter pass removes **~36%** of crossings under A
      (16,205 → 10,435) and **~38%** under B (18,517 → 11,444). The residual is
      dominated by long skip edges (mean span 8.2 layers under A, 16.6 under B;
      only 24% / 15% of forward edges connect adjacent layers) — exactly what
      the T3(a) upgrade (span-minimizing layer assignment) is meant to attack.
    - Both drawings and all statistics come from the same `(layer, slot)` arrays
      and the same classifier, so the pictures cannot disagree with the tables.

    Next steps (from the README roadmap): T3(a) span-minimizing layer
    assignment; the soft-layering variant (fixed L); a dedicated within-layer
    objective; scaling strategy for the fly connectome; null-model comparison
    (real connectome vs degree-matched random graphs); and the deferred T5
    joint 2D relaxation reusing the Rocket machinery.
"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9"},
})
nbf.write(nb, OUT_NB)
print(f"wrote {OUT_NB} ({len(cells)} cells)")

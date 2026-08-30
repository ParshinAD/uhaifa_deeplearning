# Layering — ANN-style layered representation of MFAS-ordered connectomes

A sub-project of `nn_backward_connections/`, separate from the Phase-7 MFAS
campaign (own branch family `layering/*`; the campaign's `autoresearch/` machinery
is not used here). The campaign's **frozen oracle and champion artifacts are
consumed read-only**.

## The task (spec v1, agreed 2026-08-31)

After MFAS has sorted the graph so that most edge weight points forward, re-draw
the same graph the way artificial neural networks are drawn:

> **Input:** a directed weighted graph `G` plus a linear order `pi` — the output
> of our MFAS optimizer (the sitting champion for the dataset).
> **Output:** (1) an assignment of every node to a **layer** such that no
> feedforward edge stays inside a layer; (2) an **order of nodes inside each
> layer** that keeps the number of edge crossings low ("first nodes of layer k
> connect mostly to first nodes of layer k+1"); (3) a **drawing** of all edges
> and a set of **structural statistics** (layer count, width profile, skip-length
> distribution, feedback share).

This is the classical Sugiyama framework for layered graph drawing, with step 1
(cycle removal) already solved by the MFAS campaign.

### Fixed decisions

| # | decision |
|---|---|
| 1 | Feedback edges are **drawn as separate arcs** — never reversed, never dropped. |
| 2 | **HARD layering** first (zero intra-layer FF edges). The soft version (fix the layer count `L`, minimize intra-layer weight) is future work. |
| 3 | Edges **may skip layers** (drawn as long straight segments). |
| 4 | Crossings are a **count**, not weighted. |
| 5 | **Mouse first** (148 nodes, 583 edges); the fly connectome (136k nodes) needs different internals later — keep the API, swap the implementation. |
| 6 | Deliverable = the picture first, plus the structure (profiles feed the "brains vs random graphs" track later). |
| 7 | `pi` is a **fixed input** (the pinned champion). A joint layer/order objective is future work (T5 below). |
| 8 | Sources/sinks are just the first/last layers — no special input/output layer. |
| 9 | Within-layer order: barycenter heuristic in v1; a dedicated objective later. |
| T1 | An intra-layer **feedback** arc is allowed (can occur only in method A, between FF-incomparable nodes). |
| T2 | Edge classes on the drawing are **by layers**, not by `pi` (see below). |
| T3 | Layer-assignment freedom: v1 uses "earliest feasible"; minimizing total edge span (graphviz-style) is the first planned upgrade. |
| T4 | Feedback arcs are **not** in the crossing objective (v1). |
| T5 | Deferred vision: a 2D generalization of Rocket where (layer, within-layer position) come out of ONE optimization instead of three sequential heuristics. |

### The two layering methods (decision T2)

The linear order and the layering are different objects: a layering only has to
respect the **partial order** of the FF DAG, and `pi` is just one linear
extension of it.

- **A. FF-DAG longest-path** (`core.layers_longest_path`): `layer[v]` = longest
  FF path ending at `v`. Fewest possible layers; on FF-incomparable pairs the
  layer order may disagree with `pi`, so a `pi`-feedback edge can land in an
  earlier layer than its target and become forward-by-layer — **reclaimed**
  feedback. Mouse: **26 layers**.
- **B. pi-slices** (`core.layers_pi_slices`): layers are contiguous slices of
  `pi` (cut before the first node that receives an FF edge from inside the
  current slice). `pi` fully preserved, no reclamation. Mouse: **49 layers**.

Both satisfy the same hard invariant (every FF edge strictly increases the
layer) and A is pointwise ≤ B; both are unit-tested.

## Architecture

```
layering/
├── README.md               ← this spec
├── core.py                 ← pure algorithms (numpy + loops, deterministic)
├── draw.py                 ← matplotlib rendering of a (layer, slot) layout
├── io_utils.py             ← dataset + pinned champion order, PARITY-GATED
├── create_notebook.py      ← generates layering_mouse.ipynb (nbformat)
├── layering_mouse.ipynb    ← BUILD ARTIFACT — regenerate, don't hand-edit
├── tools/exec_notebook.py  ← in-process notebook executor (no jupyter kernel
│                             in the `allen` env; the env is not mutated)
├── tests/test_core.py      ← unit + integration tests
└── outputs/                ← saved figures (PNG, dpi 120)
```

Data flow: `io_utils.load_graph_and_champion_order` → (`g`, `order`, `rank`) →
`core.ff_mask_by_order` → `core.layers_*` → `core.initial_slots_from_order` →
`core.barycenter_order` → `draw.draw_layered` + `core.layout_stats`.

### Provenance & invariants

- The champion order comes from the **tracked** `results/champions/*.npz`
  (pinned via `tools/pin_champion.py`, which re-scores through the frozen oracle
  before writing). Mouse: `H63_mouse_s42_cuda.npz`.
- `io_utils` re-scores the order with the frozen `src/mfas/metrics.py` and
  compares against `autoresearch/sota.json`'s `pct_mean_exact` (tol 1e-9): a
  stale artifact fails loudly. Note `score_from_order` consumes **ranks**.
- No scipy and no `np.dot`/matmul anywhere in `layering/` — the `allen` env's
  BLAS aborts the interpreter (campaign queue item P26). **Measured 2026-08-31
  on the Windows box: this also kills every `matplotlib` draw call** (figure
  transforms go through `np.dot`), so `savefig` dies silently (rc=127, empty
  log) in `allen` under any backend. Consequence: algorithm cells run under
  `allen`; the NOTEBOOK IS EXECUTED with the base Anaconda python
  (`/c/ProgramData/anaconda3/python.exe`, numpy 1.26.4 / matplotlib 3.8.4),
  which imports the same `layering` + frozen `src/mfas` code (numpy-only, no
  torch) and re-runs the sota.json parity gate inside the notebook itself.
- `core.count_crossings` counts **proper geometric crossings of the straight
  segments actually drawn** — the picture and the reported number can never
  disagree. O(E²): fine for mouse, not for the fly connectome.

## How to run

```bash
PY=/c/ProgramData/anaconda3/envs/allen/python.exe    # or the MacBook path
PY_VIZ=/c/ProgramData/anaconda3/python.exe           # base env: rendering only —
                                                     # matplotlib is dead in `allen`
                                                     # on this box (P26, see above)

$PY -m pytest layering/tests -q                      # tests (algorithms, no plotting)
$PY layering/create_notebook.py                      # regenerate the notebook
$PY_VIZ layering/tools/exec_notebook.py layering/layering_mouse.ipynb  # execute it
```

On a machine with a healthy BLAS (e.g. the MacBook), `$PY` can execute the
notebook too — the split exists only because of this box's broken MKL.

## Roadmap

1. **T3(a)** — spend the layer-assignment freedom on minimizing total edge span
   (network-simplex / graphviz-style) instead of "earliest feasible".
2. Soft layering: fixed `L`, intra-layer weight as a penalty (decision 2).
3. A real within-layer objective beyond barycenter (decision 9); possibly
   weighted crossings as a variant of decision 4.
4. Scaling to the fly connectome: aggregation (types/clusters), density
   matrices between layers instead of per-edge drawing, subgraph views.
5. Structure vs null models: layer-count / width-profile / span distribution of
   the real connectome against degree-matched random graphs (ties into the
   project's research goal 2).
6. **T5** — joint 2D relaxation: one differentiable objective producing layer
   and within-layer position together, reusing the Rocket machinery.

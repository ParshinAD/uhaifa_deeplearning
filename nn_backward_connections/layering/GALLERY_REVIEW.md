# Representation gallery — reviews and ranking

*2026-09-06. Ten representations of the same object (mouse graph, 148 nodes / 583 edges,
pinned H63 champion order, 93.18 % feedforward) were built and looked at. Static figures:
`outputs/gallery/V1..V7*.png` from `tools/gallery.py`; the v1.1 hard layerings A/B from
`layering_mouse.ipynb`; the interactive explorer from `tools/build_html.py`
(published as a private artifact, "Mouse Connectome Layers"). This file records what each
one shows, three reviews written from different seats, and the ranking that follows.*

## 1. The candidates

| id | representation | what it fixes | file |
|---|---|---|---|
| A | hard layering, FF-DAG longest path (26 layers) | v1 | `outputs/layering_mouse_ffdag.png` |
| B | hard layering, pi-slices (49 layers) | v1 | `outputs/layering_mouse_pislices.png` |
| V1 | arc diagram along pi, FF above / FB below, SCC span shaded, hub marked | order-native view | `V1_arc_diagram.png` |
| V2 | adjacency matrix in pi order, log colour, layer cuts, hub cross, SCC box | order-native, exact | `V2_sorted_adjacency.png` |
| V3 | soft layering L=8+hub, node level, ids, periphery greyed | fewer layers, hub isolated | `V3_soft_layers_nodes.png` |
| V4 | same layering, edges bundled per layer pair, arc height = skip length, % labels | scalable, ANN-like | `V4_bundled_blocks.png` |
| V5 | the L×L layer-to-layer weight matrix behind V4 | quantitative | `V5_layer_matrix.png`, `.csv` |
| V6 | trophic level vs pi rank | second hierarchy as control | `V6_trophic_vs_pi.png` |
| V7 | core–periphery: sources / SCC soft-layered / sinks | periphery separated | `V7_core_periphery.png` |
| V8 | interactive explorer: hover/pin a node, switch V3↔V7, hide classes, thin by weight | exploration | `explorer.html` |

Soft layering facts used below (exact DP, `tools/gallery.py`): cuts at pi ranks
0, 55, 56, 66, 94, 112, 119, 126, 133; widths 55 / 1 / 10 / 28 / 18 / 7 / 7 / 7 / 15;
4.10 % of weight intra-layer; crossings after barycenter 12 432 (vs 10 435 for hard A).
Core–periphery variant: widths 39 / 16 / 1 / 10 / 28 / 18 / 7 / 6 / 7 / 9 / 7, 3.70 % intra
inside the SCC, 14 035 crossings.

## 2. Scoring

Five criteria, 1–5 each. *Glance* = can a reader state the main structural fact within
ten seconds. *Fidelity* = the drawing shows where the feedback actually is, without a
layout artefact masquerading as structure. *Field standard* = maps onto a figure type
neuroscientists or graph people already read. *Scale* = survives the fly connectome
(136 k nodes, 5.7 M edges) without a new idea. *Null-ready* = yields a statistic with a
distribution under degree-preserving rewiring (Track C).

| id | glance | fidelity | field std | scale | null-ready | total | one-line verdict |
|---|---|---|---|---|---|---|---|
| **V5** matrix | 5 | 5 | 4 | 5 | 5 | **24** | The object everything else illustrates. FF 89.3 / intra 4.1 / FB 6.6 read off the title; the two 10 % cells (L0→L8, L3→L8) and the feedback column into the hub are visible at once. |
| **V4** bundled | 4 | 4 | 4 | 5 | 4 | **21** | V5 as a picture. Skip structure and the hub's feedback intake are legible; loses per-node detail by design. |
| **V1** arc | 4 | 5 | 4 | 2 | 3 | **18** | Best single view of *where the feedback is*: a fan from ranks 60–140 back into the hub. Standard in the ordering literature. Dies at n > ~2000. |
| **V2** adjacency | 3 | 5 | 5 | 4 | 4 | **21** | Exact, no layout freedom to mislead; the sorted-adjacency figure is what MFAS papers show. Needs the reader to know how to read a matrix; dense fly matrix needs binning. |
| **V8** explorer | 3 | 5 | 2 | 2 | 1 | **13** | Only tool that answers "what does node X connect to". Not a paper figure; scoring it on paper criteria is unfair, but it is not the deliverable either. |
| **V7** core–periphery | 3 | 4 | 3 | 1 | 2 | **13** | Correctly removes 46 periphery nodes from the core drawing; the core is still spaghetti at node level. |
| **V3** soft nodes | 2 | 3 | 3 | 1 | 2 | **11** | Better than A/B (9 columns, hub alone), but L0 = 55 nodes is a wall; the DP puts everything before the hub in one slice because those nodes barely talk to each other. |
| **V6** trophic | 3 | 3 | 3 | 5 | 4 | **18** | A control, not a representation: shows the MFAS hierarchy is not an artefact of the optimiser (ρ = 0.74 with a closed-form hierarchy) and that the graph is shallow (2.7 levels). |
| A hard longest-path | 1 | 2 | 2 | 1 | 2 | **8** | Layer index = hub distance; the star at layer 5 is a layout artefact, not biology. Keep as the invariant certificate. |
| B hard pi-slices | 1 | 3 | 2 | 1 | 2 | **9** | 49 layers for 148 nodes is a line with extra steps. Certificate only. |

## 3. Three reviews

### 3.1 Seat 1 — a systems neuroscientist reading the figures for a paper

- **None of these figures can be read without region names.** Every node is an integer.
  The hub is "92". A neuroscientist's first and only question is "what is 92?", and the
  figure cannot answer. Until the id → region mapping exists, the drawings are engineering
  validation, not science. (Verdict shared by all three seats; it is the blocking item.)
- **The hub needs a provenance check before it is interpreted.** In-degree 95 of 148:
  almost every region projects to it. In a normalised mesoscale connectivity table that
  pattern fits a large integrative target, but it also fits a data artefact (a merged
  or summary node, an "unassigned" bin). Ask the data provider which it is before the hub
  becomes a finding.
- **V4 and V5 are the right shape of figure.** They are the modern form of the
  Felleman–Van Essen / Harris et al. hierarchy figure: levels, feedforward mass between
  levels, feedback drawn separately. A reader of Harris 2019 will recognise V4 immediately.
  Two fixes for that audience: label layers by their dominant regions once names exist,
  and report feedback per *pair* of layers as a fraction of that pair's total (the field's
  "fraction feedforward" statistic), not only as % of the global total.
- **V1 says something the layered figures hide**: feedback is *long-range* and *convergent*.
  It is not "each area talks back to the previous one"; it is "the late half of the order
  projects back to one node". That is a biological claim worth a sentence, and V1 is the
  figure that makes it.
- **V3/V7 are not paper figures at this density.** 583 straight lines over 148 labelled
  nodes is unreadable in print at any size. They work only as the "supplementary
  full wiring" figure or as the interactive view.

### 3.2 Seat 2 — a graph-drawing / algorithms reviewer

- **The hard layerings are answering the wrong question.** With zero reclaimable feedback
  (proved in v1.1), a hard layering cannot show anything the order does not already show;
  it only chooses *how many* layers to spend. The soft curve (L=8 → 4.1 % intra, L=10 →
  3.4 %) says that 8–10 is the natural resolution; A's 26 and B's 49 are the cost of a
  constraint that buys nothing.
- **The soft DP has a degenerate first slice.** 55 nodes before the hub become one layer
  because the objective only counts *intra*-slice weight and those nodes are mutually
  sparse. Any objective of this form will do that. Fixes, in order of cost: (i) a width
  penalty or max-width constraint in the DP (still exact, O(L n² )); (ii) minimise total
  edge *span* instead of intra weight (the graphviz / network-simplex objective, T3a);
  (iii) split the periphery off first (V7 already does this and gets 39 + 16 instead of 55).
- **Crossing counts must not be compared across layerings** as if lower were better:
  12 432 (soft) vs 10 435 (hard A) is not "hard A is better"; a layering with more, thinner
  layers has fewer long edges. Report crossings only within a fixed layering when tuning
  the within-layer order.
- **The O(E²) crossing oracle and the O(L n²) DP both stop at the mouse.** For the fly:
  bundle first (V4/V5 need only the layer assignment, which is O(E) once cuts are known),
  and get cuts by a greedy sweep or by binning a continuous coordinate (Rocket position,
  trophic level). The DP is the certificate at small n, not the production path.
- **V2's white grid on a dark ground works; the earlier version (white on white) was
  invisible.** V4's first version stacked all arcs at one height and was unreadable; arc
  height ∝ span fixed it. Both are recorded because the reviewer should know the first
  drafts of the "good" figures were wrong too.

### 3.3 Seat 3 — a thesis methods reviewer (correctness, provenance, claims)

- **Provenance is clean.** Every figure consumes the pinned `H63_mouse_s42_cuda.npz`
  through `io_utils`, which re-scores it with the frozen oracle and checks `sota.json`.
  Nothing in `layering/` touches the optimiser (invariant 6). Numbers in this file come
  from `tools/gallery.py` / `tools/diagnostics.py` output, both re-runnable.
- **One bug shipped and was caught by eye**: the explorer initially coloured every edge as
  intra-layer because it assumed class codes 0/1/2 while `core.py` uses +1/0/−1. The
  static figures were unaffected (they call `classify_by_layers` directly). Lesson kept:
  the JSON should carry class *names*, not integers, or the HTML should import the
  constants. Not yet done.
- **Not verified**: hover/click pinning in the explorer. Browser automation could not
  deliver a click to the SVG circles (the text-box search path was verified and renders the
  focus correctly). Treat hover as untested until a human confirms it.
- **A free parameter was chosen by eye**: L=8. The soft curve justifies "8–10" but not 8
  over 10. Any statistic reported from V5 must state L and, for Track C, be computed at the
  same L on every null sample.
- **Do not over-read the 10 % cells.** L0 (55 nodes) → L8 (15 nodes) carrying 10.5 % of the
  weight is partly a *size* effect: L0 is the biggest layer. For a claim, normalise by the
  product of layer sizes or by a rewired null; the raw matrix is a description, not a test.
- **The trophic comparison is a control, keep it.** It is the one number here that does not
  depend on our optimiser at all, and it says the hierarchy is real (ρ = 0.74) but shallow
  (F0 = 0.71, 2.7 levels). That protects the thesis from "your hierarchy is an artefact of
  MFAS".

## 4. Ranking and what to do with it

1. **V5 layer matrix (with V4 as its picture)** — the paper figure and the Track C
   statistic. Next: width-constrained or span-minimising cuts; per-pair feedforward
   fractions; the same matrix on ≥ 20 degree-preserving rewirings.
2. **V2 sorted adjacency** — the exact companion figure and the MFAS-community standard;
   also the only one of the top three that needs no layering at all.
3. **V1 arc diagram** — the "feedback is long-range and convergent" figure; supplementary,
   small graphs only.
4. **V6 trophic control** — one panel in methods.
5. **V8 explorer** — the working tool for the researcher and the supervisor; not a figure.
   Blocked on region names for real usefulness.
6. **V7 / V3** — supplementary only, and only once names exist; prefer V7.
7. **A / B** — certificates in the notebook; not figures.

Blocking item for every seat: **node id → region name mapping**, plus the provenance of
node 92. Everything above is built so that names drop in as a label column without
touching the algorithms.

## 5. The fly connectome (added 2026-09-06, `tools/gallery_fly.py`)

The pinned H64 order (84.2582 %, parity-gated) was put through the scalable subset of the
gallery: binned adjacency (V2), bundled blocks (V4), layer matrix (V5), a span spectrum in
place of the arc diagram (V1'), the soft-layering curve, and the trophic control. Per-node
drawings and the crossing oracle do not exist at this size. Files:
`outputs/gallery_fly/`, numbers: `outputs/gallery_fly/report.json` and
`method_comparison.csv`. Runtime 45 s in the base Anaconda python.

### 5.1 Which slicing method is best on the fly

| L | equal count | equal weight | grid DP + exact refinement |
|---|---|---|---|
| 8 | 33.82 % intra | 30.16 % | **28.92 %** |
| 10 | 30.31 % | 28.61 % | **25.27 %** |

The DP wins by 3–5 pp; refining its grid cuts exactly (coordinate descent, cuts moved by
≤ 8 ranks) gains only 0.06 pp, so a 100-rank grid is fine at this scale. That is the
answer to "which method": **min-intra DP on a grid**, and the method choice is not the
story.

### 5.2 The story: the fly is not slice-layered, the mouse is

| | mouse | fly |
|---|---|---|
| intra-layer weight at L = 10 (best cut) | 3.4 % | 25.3 % |
| at L = 50 | 0.0 % | 9.6 % |
| feedback span in pi, median / n | 0.25 | 0.13 |
| share of FB weight on the top node | 40.5 % | 0.84 % |
| share of FB weight on the top 1000 nodes | — | 20.1 % |
| trophic Spearman vs pi / coherence F0 / depth | 0.74 / 0.71 / 2.7 | 0.76 / 0.36 / 6.7 |
| dominant V5 cells | L0→L8, L3→L8 (long skips) | L_k→L_{k+1} (a chain) |

Reading:
- **Locality.** The span spectrum shows 16 % of all fly weight sitting within 1.7 % of n
  in the order, on both sides of zero. Any contiguous slicing has to cut through that, so
  a quarter of the weight stays inside layers no matter where the cuts go. The mouse has no
  such short-range mass.
- **No hub.** Feedback in the fly is spread over thousands of neurons (top node 0.8 %,
  top 1000 = 20 %); the mouse story ("one node takes 40 % of the feedback") does not
  transfer at all.
- **A chain, not a fan.** The fly's layer matrix is banded: each layer projects mostly to
  the next one or two, and the thickest feedback bundles are also adjacent-layer (V4:
  L1→L0, L2→L1, ..., all 0.3–0.6 %). The mouse's matrix was dominated by the source layer
  projecting to the sink layer.
- **The trophic control agrees with pi about as well as in the mouse (ρ 0.76 vs 0.74) but
  says the fly hierarchy is deep (6.7 levels) and incoherent (F0 0.36)** — edges do not jump
  one level at a time. Consistent with the local-mass reading.
- **V2 shows block structure the layering ignores**: stripes and blocks across the whole
  order (cell-type-like groups that all project to the same targets), and one dense block
  near ranks 105k–134k with heavy near-diagonal feedback. 7,264 pure sources open the
  order, ~2,500 sinks close it (giant SCC spans ranks 7,264–134,103, and holds 1,173,079 of
  1,173,245 feedback edges).

### 5.3 What this means for the representation question

- For the **mouse**, layers = slices of pi is a good model (3 % loss at L = 10) and V5/V4
  are the figure.
- For the **fly**, slices of pi are the wrong axis: "layers" cut this way are thick local
  communities, and V4/V5 describe the chain between them but hide the 25 % inside. The
  representation the fly asks for is **two-dimensional: (group, depth)** — nodes grouped by
  community or cell type, groups placed by their pi position, and the layer matrix computed
  between groups (README roadmap item 4). The public FlyWire annotations (cell type,
  neuropil, hemilineage) are exactly such a grouping and, unlike the mouse region names,
  they are obtainable without asking anyone.
- V2 (binned adjacency) and V1' (span spectrum) need no layering and are the two fly
  figures that stand on their own today.

### 5.4 Honesty items

- The soft-layering curve for the fly is an upper bound on the true optimum (cuts on a
  100-rank grid, then exact local refinement); the refinement moved every cut by < 100
  ranks and changed the value by 0.06 pp, so the bound is tight in practice, not proven.
- The trophic solve first ran to the CG iteration cap; with a Jacobi preconditioner it
  converges (info = 0) to the same numbers, which is what is reported.
- `fb_weight_share_top_k_nodes` in `report.json` counts each FB edge at both endpoints and
  halves the sum for k > 1; it is a share-of-touch estimate, not an exact set cover.
- The H64 connectome order was pinned in this session
  (`results/champions/H64_connectome_s42_cuda.npz`, re-scored to 35,314,407 exactly by the
  pin tool) and registered in `io_utils.CHAMPION_ORDER_NPZ`. The earlier H42 pin stays.

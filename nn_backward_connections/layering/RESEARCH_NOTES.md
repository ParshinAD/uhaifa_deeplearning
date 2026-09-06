# Layering — research notes: what v1 shows, how to represent the graph, who it is for

*Written 2026-09-06 after reviewing the v1.1 deliverable (`layering_mouse.ipynb`, three
figures) and running `tools/diagnostics.py` on the pinned H63 mouse champion.
All numbers below come from that notebook's outputs or from
`outputs/diagnostics_mouse.txt`; the command to regenerate each is given in
`README.md` / the script docstring.*

## 1. What v1 actually shows

The v1 pictures are correct but they are not yet *readable*. Three things dominate them
and each is a structural fact about the graph, not a drawing bug:

| fact | number | consequence for the picture |
|---|---|---|
| One hub (node id 92, degree 151 of 583 edges) | 40.5 % of all feedback weight and 17.2 % of feedforward weight touches it | Method A collapses into a "star" at layer 5: 43 sources in layer 0 all pointing at one node, then 26 nodes in layer 6 fanning out of it. The longest-path layering is really "distance from the hub". |
| The whole feedback lives in one strongly connected component | 46 SCCs, one giant of 102 nodes; 123 of 124 FB edges are inside it; the giant spans pi-ranks 1..142 | Nothing outside the giant SCC carries feedback. 46 nodes are pure DAG periphery (sources/sinks). A drawing that does not separate periphery from core wastes 1/3 of the canvas. |
| Feedback is long-range, not local | FB span in pi: median 37.5 ranks, 90th percentile 85.7 of 148 | The red arcs are top-down projections across most of the hierarchy, not "recurrent loops between neighbouring layers". Sugiyama-style drawings assume most edges are short; here even the FF edges have mean span 8.2 layers (A) / 16.6 (B). |
| Zero reclaimable feedback | 124 of 124 FB edges have an FF path from target back to source (83 are halves of mutual pairs) | Any hard layering keeps exactly the champion split 93.18 / 6.82. The layering cannot "find" more feedforward structure than MFAS did — it is a *presentation* of pi, not a refinement. |

Crossing counts (A: 16 205 → 10 435 after barycenter, B: 18 517 → 11 444) are dominated by
the long skip edges; barycenter does what it can (−36 %/−38 %) but the residual is
intrinsic to drawing 459 edges of mean span 8–17 layers as straight lines.

**Verdict on the two methods.** Method B (pi-slices, 49 layers) preserves the champion
order exactly but produces 49 layers for 148 nodes — three nodes per layer on average,
which is a line, not a layered network. Method A (26 layers) is more compact but its
layer index is a hub-distance, which is misleading without saying so. Neither is the
final representation; both are useful *as a decomposition into (periphery DAG, core SCC)*
once that decomposition is made explicit.

## 2. How "layered" is this graph, really?

The soft-layering curve (exact DP: cut pi into `L` contiguous slices, minimise the weight
that ends up *inside* a slice) answers the question the hard layering could not:

| L slices | weight trapped inside slices |
|---|---|
| 3 | 20.2 % |
| 5 | 9.8 % |
| 8 | 5.0 % |
| 10 | 3.4 % |
| 15 | 1.4 % |
| 20 | 0.8 % |
| 26 | 0.4 % |
| 49 | 0.0 % |

So **8–10 layers already leave < 5 % of the weight intra-layer** (on top of the 6.8 % feedback
that no layering can remove). That is the layer count a human-readable drawing should use;
the 26/49 of the hard variants are the price of the *zero*-intra-layer constraint, and that
constraint buys almost nothing between L = 10 and L = 49.

A second, independent hierarchy — **trophic levels** (MacKay, Johnson & Jones 2020,
*How directed is a directed network?*, a Laplacian solve, no optimisation) — agrees with
the MFAS order only moderately: Spearman 0.74, and the trophic order leaves 89.98 % of the
weight feedforward versus the champion's 93.18 %. Trophic coherence F0 = 0.71 and the level
range is only 2.7 "hops": by that measure the mouse graph is a *shallow* hierarchy with a
lot of incoherence, which is again the hub plus the reciprocal pairs. This matters for the
representation question because trophic levels are the continuous analogue of a layer
index and cost one linear solve at any scale.

## 3. Representation options, ranked

The graph has three regimes — DAG periphery, hub, recurrent core — and the best
representation is the one that shows those three explicitly.

1. **Core–periphery layered view (recommended next).** Condense the 46 SCCs; draw the
   periphery (46 singleton nodes + the 2-cycle) as a proper DAG around a single block for
   the giant SCC, and draw the giant SCC *separately* with a soft layering of L ≈ 8–10
   slices of pi. Feedback arcs appear only in the second drawing, where they mean
   something. This is what Felleman & Van Essen (1991) did by hand for the macaque visual
   hierarchy, and what Harris et al. (2019, *Nature*, mouse cortico-thalamic hierarchy)
   did with an iterative feedforward/feedback consistency score — the field's standard
   "hierarchy figure" is exactly a soft layering with feedback drawn as separate arcs.
2. **Hub-aware drawing.** Node 92 should be drawn as its own layer or its own column
   with its 151 edges bundled, otherwise every layout is a star. Edge bundling by
   (source-layer, target-layer) pair, with width = summed weight, replaces 459 straight
   lines by at most L² bundles. This is the only way the fly connectome (5.7 M edges)
   can ever be drawn in this style, so building it on the mouse graph first is cheap
   insurance (README roadmap item 4).
3. **Layer-to-layer weight matrix (the "ANN view" proper).** For a fixed soft layering,
   the L × L matrix `W[i][j]` = weight from layer i to layer j. Upper triangle =
   feedforward (diagonal offset = skip length), lower triangle = feedback, diagonal =
   intra-layer. One heatmap says everything the arc drawing says, scales to any n, and is
   the natural object to compare against null models (Track C, G01) because it has a
   distribution under degree-preserving rewiring. This is the representation to compute
   *first*; the arc drawings are illustrations of it.
4. **Continuous 2-D embedding (deferred T5).** x = a hierarchy coordinate (pi rank,
   trophic level, or the Rocket position itself), y = a within-level coordinate from the
   Rocket machinery with a crossing/span penalty. Worth doing only after 1–3 exist,
   because without the SCC/hub decomposition it will reproduce the v1 star.
5. **Hard layering A/B (v1).** Keep as the *certificate* — they prove the hard invariant
   and the zero-reclamation result — not as the figure for a reader.

Node labels are the single biggest usability gap: `table_mouse.txt` has integer ids
1..149 and no names. Until the researcher supplies the id → region mapping, none of the
drawings can be read by a neuroscientist, and the hub (id 92) cannot be interpreted.
**Ask for the mapping before investing in drawing quality.**

## 4. What the representation can be used for, and by whom

| audience | what they get from it | which representation |
|---|---|---|
| **The thesis itself (Track C, "brains vs random graphs")** | A *structural fingerprint* of unavoidable feedback: layer count at a fixed intra-layer budget, skip-length distribution, feedback-span distribution, hub share of feedback, SCC condensation profile. Each has a null distribution under configuration-model / degree-preserving rewiring, so "does the real brain differ from random?" becomes a set of testable numbers rather than one MFAS score. | 3 (matrix) + the soft-layering curve |
| **MFAS algorithm work (Track A)** | Where the residual feedback sits: 40 % on one hub, the rest inside a 102-node SCC with long spans. A picture of *which* edges are feedback is what tells you whether a local search (window/SCC based) can reach them — the H21/H22 sizing questions had no picture behind them. Also a sanity check of a champion: a champion whose feedback is scattered randomly across pi looks different from one whose feedback is structured. | 1 (core–periphery) + 2 (hub bundling) |
| **Neuroscientists / the supervisor** | A Felleman–Van-Essen-style hierarchy of the 148 regions derived from an *optimal* order rather than a heuristic one, with the feedback projections named. Comparable directly to Harris et al. 2019 if the ids map to Allen regions. This is the figure that goes into a paper. | 1, with labels |
| **Connectome-constrained ANNs** (Lappalainen et al. 2024, *Nature*, fly visual system) | The layering *is* an architecture: layer assignment → module order, FF skip edges → skip connections, FB edges → recurrent connections across time steps. One can instantiate the mouse graph as a PyTorch module with exactly this wiring and ask whether the 6.8 % feedback weight changes what the network can compute. Ties the two halves of the thesis (initialisation geometry in the parent project; feedback in brains here). | 3 (matrix → module spec) |
| **Graph-drawing / Sugiyama practitioners** | A worked case where step 1 (cycle removal) is solved *optimally* and the remaining steps are stressed by long-span edges and a hub — a good benchmark for layer assignment that minimises total span (network simplex, T3a). | 5 + 2 |

## 5. Recommended next steps (in order)

1. **Get the node id → region name mapping** from the researcher (blocks everything for
   the neuroscience audience; costs nothing).
2. **Compute representation 3** (layer-to-layer weight matrix) for soft layerings
   L ∈ {5, 8, 10}, both for the whole graph and for the giant SCC alone. Save as CSV
   under `results/`. This is pure numpy and reuses the DP already in
   `tools/diagnostics.py`.
3. **Draw representation 1** (core–periphery with soft layers, hub as its own column,
   bundled edges). One figure per L. Keep v1 A/B as certificates in the notebook.
4. **Null-model fingerprint** (Track C): run the same statistics on ≥ 20
   degree-preserving rewirings of the mouse graph, each re-ordered by the campaign's
   pipeline (the estimator contract in `experiments/randomgraph.md`). Report where the
   real graph falls in each null distribution. This is the first thesis-grade result the
   layering track can produce.
5. Only then: T3a (span-minimising layer assignment), T5 (joint 2-D relaxation), and
   scaling to the fly connectome via the matrix representation.

## References

- Sugiyama, Tagawa & Toda (1981). Methods for visual understanding of hierarchical system structures. *IEEE TSMC*.
- Felleman & Van Essen (1991). Distributed hierarchical processing in the primate cerebral cortex. *Cerebral Cortex*.
- Harris et al. (2019). Hierarchical organization of cortical and thalamic connectivity. *Nature* 575.
- MacKay, Johnson & Jones (2020). How directed is a directed network? *Royal Society Open Science* 7.
- Gupte et al. (2011). Finding hierarchy in directed online social networks (agony). *WWW*.
- Lappalainen et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system. *Nature* 634.
- Bader et al. (2025). Rocket-crane algorithm for the Feedback Arc Set problem. *SNAM* 15:68.

# The fly as streams × depth — build log, four review rounds, final evaluation

*2026-09-10. Follow-up to GALLERY_REVIEW.md §5 ("the fly is not slice-layered")
and its §5.3 conclusion that the fly needs a two-dimensional (group, depth)
representation. This file records how that representation was built, reviewed
four times, verified, and accepted. Every number here is in
`outputs/fly_streams/report_streams.json` or `outputs/fly_ann/node_ann_meta.json`,
each regenerable by the `$PY` command in the producing tool's docstring.*

## 1. What was built

**The annotation join (the unlock).** The MFAS-challenge node IDs are FlyWire
root IDs. The public annotation table (Schlegel et al. 2024;
`flyconnectome/flywire_annotations`, commit `8587524`, CC-BY — cite Schlegel
*and* Dorkenwald 2024) joins at **136,635 / 136,648 = 99.99 %** with no ID
translation. That kills the mouse track's blocking item ("no node names") for
the fly: lanes are biologically named for free. Stored gzipped at
`data/external/flywire_neuron_annotations.tsv.gz` (sha256 of the raw TSV in
`tools/fly_ann_prep.py`); join is external data, oracle untouched (invariant 6).

**The representation.** 17 lanes (super_class refined by cell_class; optic
lanes named by input neuropil; olfactory AL·LH stage split out; optic dust in
its own lane) × equal-count contiguous bins of the pinned H64 champion order
(84.2582 %, parity-gated). Cells = (lane, bin); bundles = inter-cell weight
split FF/FB by the node-level order — bins are π-contiguous, so FF is always
rightward-or-vertical and no layout choice can relabel an edge.

| decomposition | weight hidden intra-group |
|---|---|
| Leiden communities (G-series, §5.3) | 83.3 % |
| best contiguous π-slicing, L=10 | 25.3 % |
| **streams × depth, B=16** | **14.4 %** (11.9 FF + 2.5 FB) |

**Deliverables.** Static: `outputs/fly_streams/S1_fly_streams_bio17_B{8,16,32}.png`
(headline B=16; B=8 is the feedback companion — 73 % of inter-cell FB inked),
`outputs/fly_cloud/C1_fly_cloud_{lanes,side}.png`, `lane_matrix.json` (17×17,
the honesty backstop that contains every gram of weight). Interactive:
**"Fly Connectome Streams"**
(https://claude.ai/code/artifact/1d49a8d0-8a16-440e-89fc-b7cb067e7027, sibling
of the mouse explorer) — streams grid with a live ink-threshold slider and
per-class coverage readout, hover/pin cell isolation with cell-type cards, a
136k-neuron canvas layer (stream / hemisphere colorings), the lane matrix
(% weight and FF-share diverging), and a methods panel carrying the full
provenance. Pipeline: `tools/fly_ann_prep.py → fly_streams.py → fly_cloud.py →
fly_artifact_export.py → build_fly_html.py`.

## 2. Findings the figures carry

- **The MFAS order recapitulates the sensory→motor hierarchy it was never told
  about**: lane medians photoreceptors 7.6k → lamina 24.9k → medulla 46.2k →
  medulla→lobula 69.6k → lobula 76.5k → MB 102.0k → VPN 110.6k → CX 113.1k →
  descending 130.8k → endocrine 135.8k → motor 136.5k; the neck motor neuron is
  the last node of the whole order. Flow-field test (afferent<intrinsic<efferent
  vs rank): Spearman ρ = 0.263, p ≤ 5·10⁻⁴ (2,000 permutations, 0 exceedances).
- **The inter-stream skeleton is a nearly pure FF cascade** (PR→lamina 100 %,
  lamina→medulla 91 %, medulla→VPN 95 %, VPN→central 94 %, central→descending
  97 %); the retro-projections VPN→optic are ~40–50 % FF — where the feedback
  actually lives, together with adjacent-bin chains inside the optic lanes.
- **Feedback is disproportionately inhibitory**: GABA+glutamate carry 55.8 % of
  FB weight vs 38.2 % of FF; within-NT feedback share: ACh 11.4 %, GABA 20.9 %,
  glutamate 22.3 %, serotonin 28.8 % (top_nt is a prediction, Eckstein 2024).
- **MFAS "resolves" the visual-centrifugal loop by placing those cells early**
  (median 43.5k, before medulla): their anatomically-feedback outputs become
  order-forward; the loop's backward half moves to their inputs.
- **~2.4k zero-out-weight fragments pile up at the end of π** and would read as
  late-stage neurons; they are hatched in every figure (genuine terminals for
  motor/endocrine, truncation artifacts elsewhere).
- The τ-regularized spectral coordinate (τ = mean weighted degree ≈ 138.9;
  unregularized vectors localize on sensory tails) separates hemispheres inside
  the optic lanes with no supervision — the cloud's within-lane y is its rank
  (ordinal, not metric).

## 3. The four review rounds (summaries; full texts in the session log)

1. **R1, three seats on v1**: right representation, wrong figure — six mandatory
   fixes (false caption, sink contamination drawn as biology, FB ink
   simultaneously over-inked and threshold-starved, collapsed verticals,
   olfaction buried in "central (other)", promised intra-cell encoding missing).
2. **R2 on v2**: all six verified landed; three new blockers — the cloud's
   spectral input was **unreproducible** (orphan `spec_reg.npz`, an invariant-3/5
   violation), same-lane bundles collapsed into false "highways" (33 % of
   inter-cell FF), stale 15-lane artifacts as a quotation hazard. Also fixed the
   CVD failure crimson-vs-medulla→lobula (ΔE deutan 2.8 → palette re-stepped,
   validated by computed dichromacy simulation).
3. **R3, verification pass on v3**: all blockers VERIFIED fixed; accounting
   identity closes exactly per variant (inter+intra = 84.2582 / 15.7418).
4. **R4 on the artifact page**: every quoted number traced to data (incl. the
   neck-motor root ID); 9 small bugs found (dark-theme handling, stale
   pin/hover, coverage readout vs checkboxes, …) — all fixed and republished.

**Final evaluation** (independent): **ACCEPT as final, 23/25** — glance 4,
fidelity 5, field standard 4, scale 5, null-ready 5. Three one-line caveats
(area wording, "every optic lane" quantifier, crimson hemispheres in the static
side-cloud PNG) — all applied before this commit.

## 4. What's next (single highest-value item per the evaluator)

**Track C on the lane matrix**: recompute the 17×17 W / FF-share on ≥ 20
degree-preserving rewirings (lanes fixed, each rewiring re-ordered by the
campaign pipeline per `experiments/randomgraph.md`), z-score each cell. That
turns "VPN→medulla is only 38 % FF" and the NT×direction skew from description
into test, and feeds research goal 2 directly.

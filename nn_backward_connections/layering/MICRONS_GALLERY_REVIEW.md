# MICrONS gallery — candidates, review, selection (2026-09-14)

The MICrONS analogue of `GALLERY_REVIEW.md`: which visualizations of the H42
champion order were built, what the review round said, what was fixed, and
which views made it into the interactive artifact. The fly gallery's verdicts
carried over where they apply (streams grid as the main representation, a
matrix as the un-thresholded backstop, per-neuron cloud as a layer, honesty
annotations on the drawing itself); this round only had to decide what is
DIFFERENT for a cortical patch with real 3-D anatomy.

Pipeline: `microns_ann_prep.py` → `microns_diagnostics.py` →
`microns_streams.py` (M2) + `microns_gallery.py` (M1, M3–M6) →
`microns_artifact_export.py` → `build_microns_html.py`.
All numbers referenced here live in `outputs/microns_ann/diagnostics.json`,
`outputs/microns_streams/report_streams.json`,
`outputs/microns_gallery/report_gallery.json`.

## The candidates

| id | figure | claim it carries |
|---|---|---|
| M1 | `microns_gallery/M1_depth_vs_rank.png` | rank follows cortical depth (E: ρ=0.356, p≤5·10⁻⁴), monotone — not canonical L4-first |
| M2 | `microns_streams/M2_microns_streams_ana10_B{8,16,32}.png` | the main streams view: anatomical lanes (pia top) × order bins |
| M3 | `microns_gallery/M3_lane_matrix.png` | 10×10 lane weights + FF share; every red cell is an E→I pair |
| M4 | `microns_gallery/M4_real_space.png` | somas at true positions painted by rank — depth gradient in anatomy |
| M5 | `microns_gallery/M5_ei_budget.png` | E/I direction budget; the fly chemistry finding reverses |
| M6 | `microns_gallery/M6_span_spectrum.png` | fly vs microns: order-local vs long-range feedback |

## Review round (independent reviewer, full report in the session log)

Claim checks: (a) depth hierarchy PASS to the digit; (b) early-inhibition
mechanism numbers PASS, but the quoted "4–16k earlier" range was wrong (real
per-band median gaps 6.7–12.9k) and "at every depth" is falsified by small-n
deep bins (true per layer band, not per 20-µm slice); (c) E/I backward shares
PASS; (d) ±1.7% shares needed one canonical rounding (5.5%, fly 19%);
(e) sink audit PASS.

Defects found and FIXED before the artifact:

1. **M5 had no visible labels** — class names and row labels were annotated
   outside the axes limits and clipped; the dashed graph-wide line was
   unlabeled. Rebuilt with y-ticks, row labels, a labeled reference line and a
   third panel showing the four class-pair channels (I→E / E→E / E→I / I→I,
   weight × FF share) plus the receiving-side numbers.
2. **M4 asserted a null its own numbers refute** ("no area/columnar rank
   pattern" vs rank–z ρ=−0.24). Recaptioned: the null is x-only (ρ=+0.05);
   the z trend is disclosed with its depth-residual component (−0.16).
3. **M2's caption misattributed the layer bands** to "public MICrONS tables".
   They are OUR density-landmark calibration below a fitted pia proxy; the
   caption now says so. (E/I labels *are* from the public Allen table.)
4. M1: median trend now starts at the L1/L2-3 band edge (the sub-60-µm hook
   was a pia-fit edge artifact), the excitatory median is overlaid on the
   inhibitory panel as a reference, per-panel log color scales are stated,
   and the headline says "in every layer band".
5. M3: masked cells are grey (previously white = ambiguous with the 50%
   diverging midpoint), the caption contradiction removed, and the
   truncation caveat added — M3 is the figure whose headline number (I→E =
   33.9% of all weight) is most inflated by sampling.
6. M6: "local" disambiguated to order-distance, exact 5.5/7.7/19/19 numbers,
   and the whole-brain-vs-patch completeness caveat added.

## Selection (per the reviewer's recommendation, adopted unchanged)

Interactive artifact **"MICrONS Cortex Streams"**
(https://claude.ai/code/artifact/d303903e-be1f-4f23-9e5b-875e7d24e347):

- **Streams** = M2 with bin switching, hover/pin cards, ink-threshold slider
  with live FF/FB coverage readout, neuron layer (lanes / E–I / reconstruction).
- **Anatomy** = M4 rebuilt as an interactive soma cloud (side + top view,
  color by rank / lanes / E–I, per-neuron hover cards) with the z-trend
  disclosed — the view the fly could never have (no soma positions there).
- **Matrix** = M3 (both metrics, masked-grey, tooltips).
- **Balance** = M5 rebuilt (labeled bars + the pair-channel mechanism +
  receiving-side numbers + the sampling caveat in-view).
- **Methods panel** = the caveats (proofread-249 first), the calibration
  recipe, the findings with exact statistics, provenance and licenses.
- M1 and M6 stay as static evidence figures in `outputs/microns_gallery/`
  (the artifact's Methods quotes their numbers; the PNGs are the print
  versions).

Nothing was dropped outright; M4's original *argumentative* form (the
unqualified top-view null) is the only thing that did not survive.

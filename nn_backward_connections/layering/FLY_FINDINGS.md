# What the fly-streams pictures actually tell us

*A plain-language reading of the figures in `outputs/fly_streams/`,
`outputs/fly_cloud/`, and the interactive artifact "Fly Connectome Streams".
For every finding: what you literally see, where to look, what it means, and
the honest caveat. Numbers come from `outputs/fly_streams/report_streams.json`
(re-runnable); the artifact's Methods panel shows the same tables.*

**The one-sentence summary:** the fly brain's wiring alone — no labels, no
anatomy — already encodes which way information flows; and the ~16 % of
synaptic weight that points "backward" is not leftover noise: it is local,
targeted, and disproportionately inhibitory.

---

## 1. An optimizer that knew nothing about biology rediscovered the sensory→motor pathway

**What you see.** In the streams grid (`S1_fly_streams_bio17_B16.png`, or the
artifact's default view), the colored mass runs diagonally from top-left to
bottom-right: photoreceptors occupy the first columns, then lamina, medulla,
lobula, visual projection neurons, the central brain, and finally descending,
endocrine and motor neurons in the last columns. Lane labels show the median
position of each stream: photoreceptors 7.6k → lamina 24.9k → medulla 46.2k →
lobula 76.5k → visual projection 110.6k → descending 130.8k → motor 136.5k
(of 136,648).

**What it means.** The horizontal axis is *not* anatomy — it is a pure
optimization: order the neurons so that as much synaptic weight as possible
points forward (84.26 % achieved). The optimizer saw only the connection
matrix, yet the order it found *is* the textbook processing pipeline — down to
a poetic detail: the very last neuron of the whole order is a neck motor
neuron. Direction of computation is recoverable from wiring alone.

**Caveat.** A statistical check backs the picture: the annotation's
input/intrinsic/output labels correlate with the order (Spearman ρ = 0.263,
p ≤ 5·10⁻⁴, 2,000 permutations). Modest ρ — because 87 % of neurons are
"intrinsic", so most of the axis has no label variance to explain.

## 2. Globally feedforward, locally recurrent

**What you see.** Crimson (feedback) in the streams grid does not arc across
the whole picture. It forms short bows *under* each optic lane between
neighboring columns — chains of adjacent-stage loops — plus a few long
leftward strands from the central brain back into the visual system. Compare
the mouse artifact, where feedback converges from everywhere onto one hub.

**What it means.** Recurrence in the fly lives *inside and between adjacent
processing stages* (medulla↔lamina, medulla↔lobula — known centrifugal
microcircuits), not as a global echo chamber. Earlier diagnostics agree: a
quarter of all weight sits within ±1.7 % of the order — dense local loops the
columns can't help but cut through. An engineer would say: a deep network with
recurrent blocks at every layer, not a fully recurrent net.

**Caveat.** Feedback is diffuse, so the default ink threshold shows only 53 %
of inter-cell feedback weight (the slider readout states this live). The lane
matrix view is the backstop: nothing is thresholded away there.

## 3. Feedback is disproportionately inhibitory — a control channel, not an echo

**What you see.** Not the picture — the table (artifact: Methods → "Feedback
is disproportionately inhibitory"; file: `report_streams.json → nt_direction`).
Each neuron carries a predicted transmitter; summing edge weight by the
sender's transmitter and by direction gives:

| transmitter (sender) | % of FF weight | % of FB weight | share of its own weight going backward |
|---|---|---|---|
| acetylcholine (excitatory) | 59.8 | 41.0 | 11.4 % |
| GABA (inhibitory) | 22.8 | 32.3 | 20.9 % |
| glutamate (inhibitory in fly) | 15.4 | 23.5 | 22.3 % |
| dopamine | 1.3 | 1.7 | 20.1 % |
| serotonin | 0.6 | 1.3 | 28.8 % |

**What it means.** The forward pass is excitatory drive; the backward pass is
enriched for inhibition and neuromodulation — gain control, normalization,
gating, valuation. Serotonin and dopamine are the most backward-skewed of all,
consistent with their role as state/learning signals broadcast to earlier
stages (the dopaminergic-neuron → Kenyon-cell teaching signal is the textbook
example). This is the reading predictive-coding theories would hope for.

**Caveat.** Transmitters are *predictions* (Eckstein et al. 2024), joined from
the annotation table — not ground truth, and not encoded anywhere in the
drawings themselves.

## 4. Two eyes, two nearly independent deep networks, merged late

**What you see.** The neuron cloud in hemisphere coloring
(`C1_fly_cloud_side.png`, or artifact → Neuron layer → "hemisphere"): inside
every optic lane the dot band splits cleanly — blue (right) on top, pink
(left) below — while the central-brain lanes stay mixed.

**What it means.** The vertical coordinate inside a lane is a spectral
"graph-affinity" score that was never told about hemispheres. If it separates
left from right by itself, left-optic and right-optic neurons must be wired
almost exclusively within their own side — two parallel copies of the visual
pipeline. The mixing in central lanes is where bilateral integration actually
happens. In ML terms: a two-tower (siamese) network with late fusion.

**Caveat.** This is a by-eye reading (stated as such in the artifact's
methods); no per-lane separation statistic has been computed yet.

## 5. Pipelines of very different depth, converging in the center

**What you see.** In the streams grid, vision crawls through five-plus lanes
(lamina → medulla → medulla→lobula → lobula → visual projection) before
reaching the central brain. The green "olfaction: AL·LH" lane (antennal lobe
and lateral horn, 1,708 neurons) sits mid-order and connects the early
sensory lane to the mushroom body in a short chain of bundles.

**What it means.** The brain is not one ladder but a bundle of conveyor belts
of different depths — vision is deep, olfaction reaches memory (the mushroom
body) in roughly two synapses (receptor → projection neuron → Kenyon cell) —
converging in the central brain. There are also direct skips (medulla →
visual projection, 95 % feedforward) alongside the stage-by-stage route.

**Caveat.** "Two synapses" is textbook anatomy that the short lane chain
corroborates; the figure shows lanes, not hop counts.

## 6. The order sorts the brain by cell type — types behave as computational units

**What you see.** In the cloud (`C1_fly_cloud_lanes.png`), dots do not spread
smoothly along x; they clump into narrow vertical stripes. Hover any cell in
the artifact: its card lists the cell types inside — e.g., the mushroom-body
cells around ranks 100–105k are one solid block of ~4.4k Kenyon cells.

**What it means.** Neurons of one type (same morphology/genetics label) have
wiring so similar that a connectivity-only optimizer places them in contiguous
rank blocks. The fly optic lobe is built of thousands of repeated columns, and
the graph "knows" it. For modeling, this backs treating a cell type — not a
neuron — as the computational unit.

## 7. How the optimizer "resolved" an anatomical feedback pathway — read hierarchies with care

**What you see.** The "visual centrifugal" lane (anatomically: central brain →
eye, i.e. feedback by every optics paper) sits *early* in the order — median
43.5k, before the medulla. Its outgoing bundles into the optic lanes are drawn
forward, in its own color; the crimson attached to it is on its *inputs*.

**What it means.** From a pure information-flow standpoint these cells behave
like part of the input cascade: they broadcast context into early vision and
collect their drive late. A loop must pay its backward cost somewhere, and the
optimizer chose the input side. Moral: "feedforward by order" and
"feedforward by anatomy" can disagree — any hierarchy metric will make such
choices, and readers of hierarchy figures should know it.

## 8. The hatched cells: data honesty on the picture itself

**What you see.** Grey hatched cells at the right edge of several lanes —
507 "photoreceptors", 579 sensory, 742 lamina cells sitting suspiciously next
to the motor neurons; also the motor and endocrine cells themselves.

**What it means.** Hatching marks cells where most neurons have *zero
outgoing weight in this graph*. For the sensory/optic ones this is a
reconstruction artifact — truncated fragments whose output synapses were
never traced, which any forward-maximizing order must dump at the end (2,447
pure sinks in the audit). For motor/endocrine neurons the same mark is honest
biology: their outputs leave the brain — muscles and hemolymph are outside
the graph. Without the hatch, a reader would "discover" photoreceptors at the
motor end of the hierarchy.

## 9. Fly vs mouse: unavoidable feedback is organized differently

**What you see.** Side by side with the mouse artifact: the mouse's feedback
converges on one hub (one node absorbs 40 % of feedback weight); the fly's is
hubless (top neuron 0.8 %, top 1000 ≈ 20 %) and chain-like along adjacent
stages. The retro-projections stand out in the lane matrix's FF-share view:
visual projection → medulla is only 38 % feedforward — the deep visual
stages talk back to early ones.

**What it means.** The *amount* of unavoidable feedback is only half the
story; its *shape* differs across brains and scales. That is exactly the
comparison the thesis's "brains vs random graphs" track can now quantify:
recompute the 17×17 lane matrix on degree-preserving rewirings and ask which
of these structures survive in a random graph with the same statistics.

---

*Provenance: order = pinned H64 champion (84.2582 %, oracle-verified);
annotations = Schlegel et al. 2024 / Dorkenwald et al. 2024 (FlyWire, CC-BY),
99.99 % join. Full build-and-review history: `FLY_STREAMS_REVIEW.md`.*

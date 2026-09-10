# What the fly-streams pictures actually tell us

*A plain-language reading of the interactive artifact **"Fly Connectome
Streams"** (https://claude.ai/code/artifact/1d49a8d0-8a16-440e-89fc-b7cb067e7027).
For every finding: exactly where to look on the page, what you will see, what
it means, and the honest caveat. The static PNGs in `outputs/fly_streams/` and
`outputs/fly_cloud/` are the print versions of the same views; every number
below lives in `outputs/fly_streams/report_streams.json` (re-runnable) and in
the page's Methods panel.*

**The one-sentence summary:** the fly brain's wiring alone — no labels, no
anatomy — already encodes which way information flows; and the ~16 % of
synaptic weight that points "backward" is not leftover noise: it is local,
targeted, and disproportionately inhibitory.

**Driving the page (30 seconds).** Three views top-left: **Streams** (the
layered drawing, default), **Matrix** (17×17 lane-to-lane weights),
**Chemistry** (transmitters vs direction). In Streams: *Depth bins* switches
8/16/32 columns; *Grid layers* toggles cells & bundles and the two edge
classes; the *Ink threshold* slider hides/reveals thin bundles and reports
live what share of weight is visible; *Neuron layer* adds all 136,648 neurons
as dots (colored by stream or by hemisphere) and has a type-search box.
Hovering a **cell** isolates everything in and out of it (click to pin);
hovering a **lane name** isolates that stream's outgoing bundles. "How it is
computed →" opens the methods panel.

---

## 1. An optimizer that knew nothing about biology rediscovered the sensory→motor pathway

**Where to look.** Open **Streams** (default view, 16 bins). Read the lane
labels top to bottom — each carries its median position in the optimizer's
order: photoreceptors 7.6k → lamina 24.9k → medulla 46.2k → lobula 76.5k →
visual projection 110.6k → descending 130.8k → motor 136.5k (of 136,648).
Now follow the colored mass: it runs in one diagonal from top-left
(photoreceptors' big 5.4k cell in the first column) to bottom-right (the
descending and motor cells in the last column). Hover the motor lane's last
cell: the card lists `brain_motor_neuron` — and the very last neuron of the
entire order is a neck motor neuron.

**What it means.** The horizontal axis is *not* anatomy — it is a pure
optimization (order the neurons so that as much synaptic weight as possible
points forward; 84.26 % achieved). The optimizer saw only the connection
matrix, yet the order it found *is* the textbook processing pipeline.
Direction of computation is recoverable from wiring alone.

**Caveat.** A statistical check backs the picture (Methods panel): the
annotation's input/intrinsic/output labels correlate with the order
(Spearman ρ = 0.263, p ≤ 5·10⁻⁴, 2,000 permutations). The ρ is modest because
87 % of neurons are labeled "intrinsic" — most of the axis has no label
variance to explain.

## 2. Globally feedforward, locally recurrent

**Where to look.** In **Streams**, untick **feedforward** under *Grid layers*
so only crimson remains. What is left is not a web spanning the whole page:
it is chains of short bows hugging each lane — feedback between *adjacent*
columns inside medulla, medulla→lobula, lobula — plus a handful of long
leftward strands from the central brain back into the visual system. Re-tick
feedforward and hover a mid-medulla cell: its card shows both directions of
its traffic at once.

**What it means.** Recurrence in the fly lives inside and between adjacent
processing stages (medulla↔lamina, medulla↔lobula — known centrifugal
microcircuits), not as a global echo chamber. Earlier diagnostics agree: a
quarter of all weight sits within ±1.7 % of the order. An engineer would say:
a deep network with recurrent blocks at every layer, not a fully recurrent net.

**Caveat.** Feedback is diffuse, so at the default threshold only 53 % of
inter-cell feedback weight is drawn — the slider's readout says so live, and
dragging it left reveals the thinner half. The Matrix view is the backstop:
nothing is thresholded away there.

## 3. Feedback is disproportionately inhibitory — a control channel, not an echo

**Where to look.** Open the **Chemistry** view. Top panel: each transmitter's
bar is its share of weight flowing backward, against the dashed line at the
brain-wide average (15.7 %). Serotonin (28.8 %), glutamate (22.3 %), GABA
(20.9 %) and dopamine (20.1 %) all sit right of the line; excitatory
acetylcholine (11.4 %) sits far left of it. Bottom panel: the two stacked
bars show composition — acetylcholine is 60 % of feedforward weight but only
41 % of feedback; GABA+glutamate together are 56 % of feedback.

**What it means.** The forward pass is excitatory drive; the backward pass is
enriched for inhibition and neuromodulation — gain control, normalization,
gating, valuation. Serotonin and dopamine are the most backward-skewed of
all, consistent with their role as state/learning signals broadcast to
earlier stages (the dopaminergic-neuron → Kenyon-cell teaching signal is the
textbook example). This is the reading predictive-coding theories would hope
for.

**Caveat.** Transmitters are per-neuron *predictions* (Eckstein et al. 2024)
joined from the annotation table — not ground truth. The drawings in the
Streams view do not encode transmitters at all; this view and the Methods
table are where that data lives.

## 4. Two eyes, two nearly independent deep networks, merged late

**Where to look.** In **Streams**, untick **cells & bundles** (the page
switches the neuron layer on by itself) and set *Neuron layer* to
**hemisphere**. Every optic band splits into two clean horizontal stripes —
blue (right eye) above, pink (left eye) below — while central (other),
central complex and descending stay salt-and-pepper mixed.

**What it means.** The vertical position inside a lane is a spectral
"graph-affinity" score that was never told about hemispheres. If it separates
left from right on its own, left-optic and right-optic neurons must be wired
almost exclusively within their own side — two parallel copies of the visual
pipeline. The mixing in the central lanes is where bilateral integration
actually happens. In ML terms: a two-tower (siamese) network with late fusion.

**Caveat.** A by-eye reading (stated as such in Methods); no per-lane
separation statistic has been computed yet.

## 5. Pipelines of very different depth, converging in the center

**Where to look.** In **Streams**, hover the lane name **"olfaction: AL·LH"**:
its outgoing green bundles jump almost straight into the mushroom body.
Now hover **"lamina"**: its traffic goes to medulla, which goes to
medulla→lobula, which goes to lobula, which goes to visual projection —
vision crawls through five lanes before reaching the central brain. For the
skip route, click the big medulla cells: cards show direct output into visual
projection as well (that lane pair is 95 % feedforward).

**What it means.** The brain is not one ladder but a bundle of conveyor belts
of different depths — vision is deep, olfaction reaches memory (the mushroom
body) in roughly two synapses (receptor → projection neuron → Kenyon cell) —
converging in the central brain, with skip connections alongside the
stage-by-stage route.

**Caveat.** One visible line is not "the only connection": a bundle is drawn
only if it passes the ink threshold. Example (computed): the first
sensory (other) cell shows a single bundle into olfaction, but its card's
"out below threshold" row reveals more traffic — smaller bundles (also mostly
into olfaction) hidden below the cut. Drag the slider left to watch them
appear. "Two synapses" itself is textbook anatomy that the short lane chain
corroborates; the page shows lanes, not hop counts.

## 6. The order sorts the brain by cell type — types behave as computational units

**Where to look.** With **cells & bundles** unticked and *Neuron layer* on
**streams**, look at any optic band: the dots clump into narrow vertical
stripes instead of spreading smoothly. Now type `KCg-m` into the type-search
box — 2,189 γ-Kenyon cells light up as one solid block around ranks 100–105k.
Try `Mi1` (1,584), `T4c` (1,706), `R1-6` (7,167): each type is a contiguous
band, not a smear.

**What it means.** Neurons of one type (same morphology/genetics label) have
wiring so similar that a connectivity-only optimizer places them in
contiguous rank blocks. The fly optic lobe is built of thousands of repeated
columns, and the graph "knows" it. For modeling, this backs treating a cell
type — not a neuron — as the computational unit.

**Caveat.** The search matches the annotation's `cell_type`; 1,233 neurons
without one appear as "… (class)" entries, and picking such an entry
highlights only the untyped members of that class (documented in Methods).

## 7. How the optimizer "resolved" an anatomical feedback pathway — read hierarchies with care

**Where to look.** In **Streams**, find the **"visual centrifugal"** lane —
anatomically these cells run central brain → eye, i.e. feedback by every
optics paper. Yet the lane sits *early* (median 43.5k, above medulla's 46.2k),
and if you click one of its small cells you see the split: its **outgoing**
pink bundles run rightward into medulla (drawn as feedforward, 99 % FF by the
order), while the crimson attached to it arrives on its **inputs** from the
central brain far to the right.

**What it means.** A loop must pay its backward cost somewhere, and the
optimizer chose the input side: from a pure information-flow standpoint these
cells behave like part of the input cascade — broadcasting context into early
vision, collecting their drive late. Moral: "feedforward by order" and
"feedforward by anatomy" can disagree — any hierarchy metric makes such
choices, and readers of hierarchy figures should know it.

## 8. The hatched cells: data honesty on the picture itself

**Where to look.** In **Streams**, the right edge: grey hatched cells in the
photoreceptors, sensory (other) and lamina lanes — sitting suspiciously next
to the motor neurons — and the motor/endocrine cells themselves. Hover the
hatched photoreceptor cell: "507 neurons (507 no-output)".

**What it means.** Hatching marks cells where most neurons have *zero
outgoing weight in this graph*. For the sensory/optic ones this is a
reconstruction artifact — truncated fragments whose output synapses were
never traced, which any forward-maximizing order must dump at the end (2,447
pure sinks in the audit). For motor/endocrine neurons the same mark is honest
biology: their outputs leave the brain — muscles and hemolymph are outside
the graph. Without the hatch, a reader would "discover" photoreceptors at the
motor end of the hierarchy.

## 9. Fly vs mouse: unavoidable feedback is organized differently

**Where to look.** Open **Matrix** and switch the metric to **FF share**.
Blue above the diagonal, crimson below: the crimson cells in the
visual-projection row are the retro-projections — hover
visual projection × medulla and the tooltip reads "38 % of it feedforward".
Then open the sibling mouse artifact ("Mouse Connectome Layers"): there, one
hub node absorbs 40 % of all feedback. The fly has no such hub (top neuron
0.8 %, top 1000 ≈ 20 %) — its feedback is the chain-and-retro-projection
pattern you just saw.

**What it means.** The *amount* of unavoidable feedback is only half the
story; its *shape* differs across brains and scales. That is exactly the
comparison the thesis's "brains vs random graphs" track can now quantify:
recompute the 17×17 lane matrix on degree-preserving rewirings and ask which
of these structures survive in a random graph with the same statistics.

---

*Provenance: order = pinned H64 champion (84.2582 %, oracle-verified);
annotations = Schlegel et al. 2024 / Dorkenwald et al. 2024 (FlyWire, CC-BY),
99.99 % join. Full build-and-review history: `FLY_STREAMS_REVIEW.md`; print
versions of every view: `outputs/fly_streams/S1_*.png`,
`outputs/fly_cloud/C1_*.png`.*

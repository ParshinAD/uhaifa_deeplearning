# What the fly-streams pictures actually tell us

*A plain-language reading of the interactive artifact **"Fly Connectome
Streams"** (https://claude.ai/code/artifact/1d49a8d0-8a16-440e-89fc-b7cb067e7027).
For every finding: exactly where to look on the page, what you will see, what
it means, and the honest caveat. The static PNGs in `outputs/fly_streams/` and
`outputs/fly_cloud/` are the print versions of the same views; every number
below lives in `outputs/fly_streams/report_streams.json` (re-runnable) and in
the page's Methods panel. Sections 1–9 are the findings; §10–§11 step back
and say what they add up to.*

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
brain-wide average (15.7 %). Serotonin (28.8 %), glutamate (22.2 %), GABA
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

## 10. What we re-discovered that biologists already knew — and why that is the point

These last two sections step back from single views and read the results as
a whole. This one lists what our method *re-derived* from wiring alone. When
a brand-new instrument reproduces the textbook, that is not a failure of
novelty — it is the calibration that makes everything else believable. Our
"instrument" is one sentence long: *line up all 136,648 neurons so that as
much synaptic weight as possible points forward.* No cell types, no anatomy,
no development — just the connection matrix. Here is what that one sentence
got right, and what re-deriving each fact adds.

**The brain runs sensors → processing → muscles.** The oldest claim in
neuroscience, drawn as arrows since Cajal. We see the full staircase at
neuron resolution, ending on a neck motor neuron as the literal last node
(ρ = 0.263, p ≤ 5·10⁻⁴ against the annotation's input/output labels).
*What re-deriving adds:* the hierarchy turns out to be a property of the
wiring itself — recoverable in one ~20-minute optimization run (the pipeline
behind that run took a long algorithm campaign to build) instead of a century
of experiments, and therefore portable to brains that have no textbook yet
(new species, new datasets, developmental stages).

**The visual conveyor lamina → medulla → lobula.** Classic optic-lobe anatomy,
including the transmedullary/T4 stage in between. We see not just the order
but the *spacing*: median depths lamina 24.9k → medulla 46.2k → the
in-between (T4) stage 69.6k → lobula 76.5k. *Adds:* depth becomes a measured
coordinate — every one of ~77,000 optic neurons gets its own number, where
anatomy assigns a stage to a whole region.

**Senses run in parallel and meet in the middle; smell is shallow, vision is
deep.** We see it as lane chains: olfaction reaches the mushroom body in
about two hops while vision crawls through five stages; two-fifths of all
forward weight (42.5 %) never leaves its own stream. *Adds:* all modalities
on one quantitative canvas — textbook figures usually draw one pathway at a
time.

**Two mirror half-brains, integrated in the center.** We see a coordinate
computed from wiring alone — never told about hemispheres (§4) — split every
optic lane into left/right halves while the central lanes stay mixed.
*Adds:* bilateral organization becomes something the wiring demonstrates by
itself — and, next, something scoreable (§11).

**Feedback modulates rather than drives.** Physiology dogma in cortex, and
known circuit-by-circuit in the fly (inhibitory centrifugal cells,
dopaminergic teachers). We see the brain-wide budget: GABA+glutamate carry
55.8 % of backward weight versus 38.2 % of forward; serotonin is the most
backward-leaning transmitter of all. *Adds:* dogma becomes a decomposition —
one table covering every connection in the graph.

**Cell types are real computational units.** Decades of typing by morphology,
genetics and connectivity clustering. The annotation's 8,000+ types appear
as fine stripes in an ordering that was never told types exist, and every
type we spot-checked (KCg-m, Mi1, T4c, R1-6 — among the largest) forms one
contiguous block; no across-all-types contiguity statistic has been computed
yet. *Adds:* an independent, embarrassingly simple corroboration — if types
were bookkeeping conveniences, the optimizer would have no reason to keep
each one together.

Why this matters, in two sentences. *Independence*: the textbook hierarchy
was pieced together from lesions, tracers and physiology — getting the same
answer from one mathematical rule means the hierarchy lives in the brain
itself, not in the methods that found it. *Transfer*: the same pipeline runs
unchanged on any future wiring map — brains that have a connectome but no
century of anatomy to lean on.

## 11. What our angle adds that the textbook could not — and where it can lead

Our lens differs from the classical ones in one word: **optimality**.
FlyWire's own rankings ask "how many steps from the senses is this neuron?";
"trophic" rankings (a method borrowed from food webs) give each neuron its
average level. We ask: *what is the best possible one-line arrangement of
the whole brain — and what does it refuse to do?* The refusals are where the
new information lives.

Honest framing first: ordering a connectome by feedforwardness is an old
idea — Durbin did it for the worm's ~280 neurons in 1987, Winding et al.
(2023) sorted the 3,016-neuron larva (66 % forward under their approximate
method), Borst (2024) lined up a 65-neuron optic column (~13–16 % recurrent —
close to our whole-brain number, though a 65-neuron column under a different
method is a coincidence to test, not a corroboration), and the algorithmic
problem on this very dataset is an open public challenge. What appears
genuinely unpublished is the *biological reading* of a near-optimal order at
adult whole-brain scale — the items below.

**1. A brain has an irreducible feedback number — and for the fly we now
have a tight ceiling on it.** Our arrangement proves that **at most 15.74 %**
of synaptic weight has to point backward; the public contest's best solution
(84.61 %, re-scored by our own oracle) tightens that ceiling to 15.39 %, and
no method has pushed it below ~15 %. Anatomy can say "there is feedback";
optimization puts a hard ceiling on how much of it is truly unavoidable. It
also reconciles two claims that sound contradictory: FlyWire's network paper
(Lin et al. 2024) called the fly "not predominantly feedforward" — and
indeed, on our copy of the graph 92.8 % of neurons (126,840 of 136,648) sit
in one giant loop in which every neuron can reach every other — while we
find that 84 % of the weight can still be pointed forward. Both are true at
once: *the brain is one giant loop that is ~84 % one-way by weight.*
→ Opens: the best-found one-way share as a comparable fingerprint (same
pipeline on all three: fly 84.3 %, mouse region-level map 93.2 %, mouse
cortex at synapse level 83.2 %), and the thesis's random-graph question: is
the fly's ~15.7 % more or less than what the same optimizer leaves
unavoidable in a random network where every neuron keeps its number of
connections but loses its identity — did evolution minimize feedback, or
protect it?

**2. Feedback has a shape, and shapes differ between brains.** Fly: hubless
(the busiest neuron touches 0.8 % of feedback), short chains between
adjacent stages, plus targeted retro-projections — the visual-projection →
medulla traffic is only 38 % forward (§9). Mouse region-level map: one node
absorbs 40 % of all feedback (pending a provenance check — that node may be
a summary region rather than real anatomy). → Opens: "what structure
explains unavoidable feedback" as a comparative program, with the lane
matrix as the statistic to compare against randomly rewired copies of each
brain.

**3. "Feedback pathway" splits into two different things.** The optimizer
must place every loop's cost somewhere, and it puts it on the lighter side —
which is exactly why its choice is informative. Visual centrifugal cells —
anatomically feedback, brain → eye — were placed *early*: their broadcasts
count as forward and the cost moved onto their inputs, meaning the eye-bound
direction of that loop carries more weight than the return. So anatomical
feedback may come in (at least) two computational kinds — on the evidence of
this one pathway so far: broadcast-like (behaves as input: context and state
delivered into early processing) and truly recurrent (the cost stays on the
output side). → Opens: a computable classifier of feedback pathways,
testable against what those pathways do in behavior.

**4. Forward excitation, backward inhibition — as one measured budget.**
Hints existed at smaller scales, in FlyWire's own papers and the larva
study. Our angle turns hints into numbers that cover every connection in the
graph: the backward share within each transmitter is 11.4 % (acetylcholine)
vs 20.9 / 22.2 % (GABA / glutamate) and 28.8 % (serotonin). → Opens: a
quantitative signature of predictive-coding-like organization, checkable
next in the MICrONS mouse-cortex dataset, where excitatory/inhibitory labels
come from anatomy rather than prediction.

**5. Depth at single-neuron resolution finds what region-level views cannot.**
Two uses already demonstrated: *quality control* — the optimizer swept the
2,447 zero-output neurons to the end of the order; most are truncated
reconstruction fragments, and once the genuine motor/endocrine terminals
among them are set aside (§8), the rest is an automatic proofreading
worklist; and *within-type structure* — a neuron ranked far from its type's
block is either an error or a subtype. → Opens: connectome-QC-by-ordering,
and a per-type depth atlas as a community resource.

**6. The right ANN metaphor for this brain — measured, not assumed.** The
naive "brain = a stack of layers" fails numerically: the best slicing into
10 global layers leaves 25.3 % of weight hidden inside layers. Our 17
streams × 16 depth bins leave 14.4 % — part of that drop is simply the finer
grid, but only the streams version has nameable parts, and the deeper point
stands (§5.2 of the gallery review): for the fly, slices of the order alone
are the wrong axis; streams × depth is the representation that fits. And the
layering *is* an architecture spec: 17 named streams, measured skip
connections, a measured 15.7 % backward budget under the pinned order.
→ Opens: build it — train neural networks with exactly this shape and ask
what the backward 15.7 % buys them. This is the bridge between the two
halves of the thesis.

**7. Symmetry as a score, not a description (proposed, not yet run).**
Because the whole arrangement is judged by a single number, mirror-swapping
paired left/right neurons turns "the hemispheres look similar" into a
measurement: the score drop indexes how far the wiring is from mirror
symmetry (given a left–right pairing, which exists for only part of the
brain), and it can be localized lane by lane.

---

*Provenance: order = pinned H64 champion (84.2582 %, oracle-verified);
annotations = Schlegel et al. 2024 / Dorkenwald et al. 2024 (FlyWire, CC-BY),
99.99 % join. Nearest published relatives of the approach: Durbin 1987
(C. elegans); Winding et al. 2023, Science (larval Drosophila); Borst 2024,
PLoS Comp Biol (optic-lobe column); Harris et al. 2019, Nature (mouse area
hierarchy); Lin et al. 2024, Nature (FlyWire network statistics). Full
build-and-review history: `FLY_STREAMS_REVIEW.md`; print versions of every
view: `outputs/fly_streams/S1_*.png`, `outputs/fly_cloud/C1_*.png`.*

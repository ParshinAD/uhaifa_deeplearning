# What the MICrONS pictures actually tell us

*A plain-language reading of the interactive artifact **"MICrONS Cortex
Streams"** (https://claude.ai/code/artifact/d303903e-be1f-4f23-9e5b-875e7d24e347),
the sibling of the fly's "Fly Connectome Streams". For every finding: exactly
where to look on the page, what you will see, what it means, and the honest
caveat. Static print versions: `outputs/microns_streams/M2_*.png` and
`outputs/microns_gallery/M1..M6_*.png`; every number below lives in
`outputs/microns_ann/diagnostics.json`,
`outputs/microns_streams/report_streams.json` and
`outputs/microns_gallery/report_gallery.json` (re-runnable), and in the page's
Methods panel. §7–§8 step back and compare the three brains we have now
ordered.*

**The one-sentence summary:** ordered by wiring alone, the mouse cortical
patch is a machine that flows *downward* — from the surface layers toward
white matter, with interneurons pushed to the front of the order and
excitatory feedback landing back on them — and its ~17 % of backward weight is
spread long-range, not chained locally like the fly's.

**The one big caveat, before anything else.** Only **249 of 67,534** neurons
in this release have a proofread axon; everyone else's outgoing wiring is a
truncated automatic fragment (proofread neurons carry a median 341 outgoing
synapses here vs 70 for the population), and 1,205 neurons have no outputs at
all. Long-range axons also leave the 1.4-mm volume entirely. Every directional
statement below inherits this; within-class *direction splits* are the robust
part, absolute class *weight shares* the fragile one. The page states this in
Methods and marks its consequences on the drawing (hatched cells, the
"reconstruction" dot layer).

**Driving the page (30 seconds).** Four views top-left: **Streams** (the
layered drawing, default), **Anatomy** (somas at their real positions),
**Matrix** (10×10 lane-to-lane weights), **Balance** (excitation/inhibition
vs direction). In Streams: *Depth bins* switches 8/16/32 columns; *Grid
layers* toggles cells & bundles and the two edge classes; the *Ink threshold*
slider reports live what share of FF and FB weight is visible; *Neuron layer*
adds all 67,534 neurons as dots (colored by lane, E/I, or reconstruction
status). Hovering a **cell** isolates its traffic (click to pin); hovering a
**lane name** isolates that lane; clicking it opens the layer guide. In
Anatomy, hover any dot to inspect one neuron. "How it is computed →" opens
the methods panel.

---

## 1. The optimizer's order descends the cortex — but it is NOT the textbook microcircuit

**Where to look.** Open **Streams** (default, 16 bins). The lanes are fixed
anatomy — pia at the top, white matter at the bottom — so the drawing cannot
manufacture a hierarchy; only the horizontal mass distribution is the
optimizer's. Follow the blue (excitatory) lanes: L2/3's mass sits left-of-center
(median rank 18.5k of 67.5k), L4 next (26.9k), L5 later (36.4k), L6 last
(46.9k) — a clean diagonal from top-left to bottom-right. Then open
**Anatomy**: the same gradient painted on real cortex, light (early) dots
shallow, dark (late) dots deep.

**What it means.** Ordered purely to maximize forward synaptic weight, the
intracortical graph reads as a top-down conveyor: superficial layers first,
deep layers last (Spearman ρ = +0.356 between excitatory rank and soma depth,
n = 55,855; 2,000 label permutations, 0 exceedances, p ≤ 5·10⁻⁴). That is a
real, statistically solid hierarchy — and it is *not* the canonical
thalamus→L4→L2/3→L5 circuit, and could not be: **the thalamus is outside the
graph**, so L4's defining property (being the input gate) is invisible to an
optimizer that sees only cortex-internal wiring. What remains visible is the
dominant intracortical drive L2/3→L5→L6 running down the column.

**Caveat.** The canonical circuit is not *refuted* — it is unmeasurable here.
This is a general lesson for connectome-derived hierarchies: what looks like
"the" hierarchy is the hierarchy of the subgraph you kept.

## 2. Interneurons are sorted to the front — and the reason is arithmetic

**Where to look.** In **Streams**, compare each green lane with the blue lane
above it: L1–L2/3 · inh median 11.5k vs L2/3 · exc 18.5k; L4 · inh 14.0k vs
L4 · exc 26.9k; L5 29.8k vs 36.4k; L6 40.0k vs 46.9k — every inhibitory lane
runs 6.7–12.9k ranks ahead of its excitatory partner. Then open **Balance**
and read the mechanism panel: **I→E carries 33.9 % of ALL synaptic weight and
the order makes 97.3 % of it feedforward; E→I carries 9.4 % at 48.4 %.**

**What it means.** Interneurons and their neighbours form dense loops
(E↔I). A linear order must pay every loop's backward cost somewhere, and the
optimizer pays it on the lighter side: I→E is ~3.6× heavier than E→I
(inhibitory axons are local, dense, and many-synapse), so inhibition goes
*first* and its massive output counts forward, while the return E→I
connections split the cost. This is exactly the mechanism the fly page
documented for one pathway (the visual-centrifugal cells, FLY_FINDINGS §7) —
here it operates on an entire cell class. In the fly's terms: cortical
inhibition behaves like *broadcast* (context delivered forward), not like
*true recurrence*, under a pure information-flow reading.

**Caveat.** "In every layer band", not "at every depth": in the deepest
20-µm slices the small-n inhibitory medians cross the excitatory ones. And
the 33.9 % figure is inflated by sampling (I axons complete, E axons
truncated) — the *direction* asymmetry is the robust part.

## 3. The fly's chemistry finding reverses: here, feedback is excitatory — and it lands on inhibition

**Where to look.** **Balance**, top panel: excitatory output is 26.6 %
backward, inhibitory output 5.2 % (graph-wide 16.8 %). Composition bars: the
backward weight is 78 % excitatory-sourced. Below: the receiving side —
**41 % of what interneurons receive is backward, vs 11 % for excitatory
cells.**

**What it means.** In the fly, backward weight was enriched for
inhibitory/modulatory transmitters — the "feedback = control channel"
reading. Here, with anatomy-based E/I labels (not predictions), the roles
flip: inhibition is forward-aligned (because it sorts early, §2), and the
feedback that excitation carries lands disproportionately *on* interneurons —
E→I is the only class pair the order leaves half-backward. The two results
are not contradictory; they are the same optimization resolving two different
circuit motifs. The fly's long-range neuromodulatory feedback has no analogue
inside a 1.4-mm patch; the cortex's tight E↔I loops have no analogue in the
fly's between-stage chains. What IS shared: in both brains, feedback is not a
mirror of feedforward — it is a structurally different channel.

**Caveat.** Weight = synapse counts; truncated E axons bias class shares
toward inhibition (stated on the view itself). The fly comparison also mixes
label types: predicted transmitters there, anatomical E/I here.

## 4. Feedback is long-range in the order, and hubless

**Where to look.** In **Streams**, untick **feedforward**: unlike the fly's
short bows hugging each lane, the crimson here spans the page — long leftward
arcs from late bins into early ones, much of it E→I and E→E across distant
bins. The static comparison figure `M6_span_spectrum.png` puts numbers on it.

**What it means.** Within ±1.7 % of the order the fly holds 19 % of its
weight (its feedback is local chains between adjacent stages); this graph
holds 5.5 %, and its feedback accumulates barely faster than its bulk. No
neuron is a feedback hub: the top receiver takes 0.13 % of backward weight
(the 148-node mouse regional map, by contrast, funnels 40 % of its feedback
through one node). With three datasets ordered by the same instrument we now
have three distinct feedback architectures: **local chains (fly), one hub
(regional mouse map), diffuse long-range (cortical patch)**.

**Caveat.** A whole brain vs a patch: long-range fly feedback pathways exist
*inside* its graph; the cortex patch's long-range loops (via thalamus, via
other areas) are cut at the boundary. The comparison is between what each
dataset contains, not between the animals.

## 5. The hierarchy is radial, not tangential: the order ignores visual areas

**Where to look.** **Anatomy**, top view (x–z plane): the rank colors are
salt-and-pepper — no left-right gradient along x, the axis that crosses from
V1 into the higher visual areas RL/AL. Rank vs x among excitatory cells:
ρ = +0.05.

**What it means.** The optimizer had every chance to discover the areal
hierarchy (V1 → higher areas is the textbook direction of cortical
feedforward) and did not: the weight that determines the order is dominated
by *within-column, across-layer* wiring, not by inter-area projections. In
this patch, depth beats area as the axis of information flow — consistent
with inter-area axons being long, thin on synapses locally, and heavily
truncated.

**Caveat.** A weak z-trend exists (ρ = −0.24, of which about a quarter is a
−0.16 residual of our pia-depth correction) — disclosed on the view; the
clean null is specifically the x axis. And truncation hits exactly the
inter-area axons hardest, so "the graph contains little areal signal" is
partly a statement about the reconstruction.

## 6. The reconstruction audits itself on the picture

**Where to look.** In **Streams**, the last column: hatched cells in several
lanes. Switch the neuron layer to **reconstruction**: crimson dots (zero
outputs) pile up at the far right of every lane; the 249 proofread-axon
neurons draw as larger dark dots scattered across the order.

**What it means.** As in the fly, a forward-maximizing order is a free
quality-control instrument: neurons with no outgoing weight can only be
placed at the end (median rank at the 99th percentile of the order), so
truncated fragments concentrate in the last bins and are marked as such
rather than being read as "the top of the hierarchy". The proofread neurons'
5× median out-weight advantage (341 vs 70) is the measured size of the
truncation bias.

**Caveat.** Unlike the fly (whose motor/endocrine terminals are *genuinely*
output-less in-graph), a cortical patch has no true output layer — everything
hatched here is reconstruction, not biology.

---

## 7. Same instrument, second brain: what transfers and what flips

The fly page ended with "the same pipeline runs unchanged on any future
wiring map". This page is that claim exercised, and the interesting part is
what did NOT transfer:

| | fly (whole brain) | MICrONS (cortical patch) |
|---|---|---|
| best forward share | 84.26 % | 83.24 % |
| what the order recovers | the sensory→motor pathway | the pia→white-matter axis |
| feedback locality (±1.7 % of order) | 19 % of weight | 5.5 % |
| feedback hub | none (top 0.8 %) | none (top 0.13 %) |
| backward-enriched class | inhibitory/modulatory transmitters | excitatory cells (landing on inhibition) |
| early-sorted "broadcasters" | visual centrifugal cells (one pathway) | the entire interneuron class |
| honesty layer | truncated sensory fragments at the end | 1,205 zero-out fragments; 249 proofread axons |

Two brains, one instrument, nearly the same forward share (~83–84 %) — but
opposite readings of what feedback is made of, and a mechanism (pay the loop
on the light side) that explains both. That mechanism is now a *predictive*
tool: given any strongly asymmetric reciprocal motif, the optimizer will
place the heavy sender early and label the light return as feedback.

## 8. What this opens for the thesis

1. **The null-model question now has three architectures to explain.** Local
   chains / one hub / diffuse long-range: recompute each dataset's span
   spectrum and lane matrix on degree-preserving rewirings and ask which
   architecture survives — the exact "brains vs random graphs" program, now
   with a cortical data point.
2. **A falsifiable claim about inhibition.** The early-I effect follows from
   the I→E/E→I weight asymmetry; a rewiring that preserves degrees but
   shuffles E/I identity should destroy it. Cheap to run, and it turns §2
   from a description into a test.
3. **The proofread-only sensitivity check.** The 245-node proofread-axon
   subgraph (`data/processed/microns_proofread.npz`) is the built-in control
   for every truncation caveat above: order it, and see which of §1–§5
   survive at full reconstruction quality (n is tiny, but the direction of
   each effect is checkable).
4. **Depth-hierarchy vs function.** MICrONS ships functional recordings for a
   subset of these neurons; whether the optimizer's rank correlates with
   functional response properties is an open, answerable question.

---

*Provenance: order = pinned H42 champion (83.2409 %, oracle-verified, score
12,819,555); graph = token-free v117 build (weight = synapse count); anatomy
= MICrONS minnie65 public tables (nucleus positions 100 % join; E/I from
allen_soma_ei_class_model_v1; 249 proofread axons), CC-BY 4.0 — cite the
MICrONS Consortium (2025, Nature 640) and Schneider-Mizell et al. (2025,
Nature 640). Layer bands are our density-landmark calibration below a fitted
pia proxy (relief 397 µm) — display annotation, not ground truth. Build and
review history: `MICRONS_GALLERY_REVIEW.md`; sibling artifact:
`FLY_FINDINGS.md`.*

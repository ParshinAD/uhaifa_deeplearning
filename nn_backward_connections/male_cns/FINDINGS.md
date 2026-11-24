# What the Male CNS order tells us, and how the male differs from the female

Companion to the interactive page `outputs/streams/malecns_streams.html` (open locally), in the
format of `layering/FLY_FINDINGS.md` for FlyWire. Each section says where to look on the page,
what we measured, and what it may mean. Female = FlyWire v783 brain (our main dataset); male =
Janelia Male CNS v1.0, brain + ventral nerve cord (VNC). Both graphs: weight = synapse count,
weight >= 2. All numbers come from `results/` (file named in each section).

Status: first pass, one MFAS seed for the male (MPS), one animal per sex. Numbers and code
were checked by two independent reviews. Lab, segmentation and synapse detection differ between
the two datasets, so "male vs female" below always means "this male dataset vs this female
dataset".

## 0. Main numbers (`results/comparison_table.csv`, `results/nulls/`)

| | female | male brain | male CNS |
|---|---|---|---|
| neurons / edges | 136,648 / 5.66M | 144,153 / 13.0M | 164,425 / 15.3M |
| FF %, champion pipeline H64 | 84.26 | | 81.81 |
| FF %, same G01 protocol | 83.66 | 80.68 | 80.83 |
| best random null (degree + lane blocks) | 81.90 | | 78.88 |
| real minus best null | +1.76 pp | | +1.95 pp |

The male is about 3 pp less feedforward. Its random nulls lose as much order or more (the male
graph is 2.2x denser), and the real graph beats the best null by the same ~2 pp in both. So the
gap needs no extra male feedback structure. The 2-cycle floor (a strict lower bound on feedback
within each graph) is 0.9 pp of W higher in the male CNS and 1.5 pp in the male brain, a third
to a half of the gap (section 10).

## 1. The optimizer again finds the sensory to motor axis, now through the nerve cord

**Where to look.** Streams view, 16 bins. Lanes are sorted by median position: VNC sensory and
sensory (other) on top, then photoreceptors, lamina, medulla ... central brain, descending, VNC
intrinsic, then motor (brain), motor (VNC) and endocrine at the bottom. The grey text under
each lane name gives the female median.

**What we measured.** Flow test (afferent < intrinsic < efferent vs rank,
`results/streams/report_streams.json`): rho 0.237, p <= 5e-4 (female 0.263 with the same
function). Medians: afferent 0.07, intrinsic 0.54, efferent 0.98 (female 0.07 / 0.55 / 0.96).
VNC sensory neurons are the earliest lane (median 0.05). Lane order of the brain is almost the
same as in the female (`outputs/fig1_lane_medians_female_vs_male.png`). Swaps: sensory (other)
now before photoreceptors, visual centrifugal after medulla, central (other) and central complex
tied (0.834 vs 0.836); ascending neurons move (section 9).

**What it means.** In the order, the cord forms both ends of one axis: its sensory neurons come
before the eye, its motor neurons after the descending neurons.

## 2. Globally feedforward, locally recurrent, in both flies (`results/analysis_*.json`)

**What we measured.** The giant strongly connected component holds 98.4 % of male neurons
(female 92.8 %) and more than 99.99 % of feedback weight in both. Feedback edges are short:
median span 0.085 n (female 0.133 n). 22.3 % of all weight lies within 1.7 % of n (female
19.2 %). Feedback is spread out: the 1,000 most loaded neurons touch 14.3 % of feedback weight
(female 20.1 %).

**What it means.** Same architecture as the female: one big loop, with nearly all backward
weight local. Male feedback is even more local.

## 3. Feedback is inhibitory, and dopamine is the most backward transmitter (`results/analysis_malecns.json`)

**Where to look.** Chemistry view.

**What we measured.** GABA + glutamate carry 50.2 % of male feedback weight and 34.8 % of
feedforward weight (female 55.8 / 38.2). Share of each transmitter's output that is feedback
(z vs label shuffle within superclass, 2,000 permutations): ACh 13.8 % (z -34.9), GABA 25.5 %
(+22.0), Glu 22.7 % (+12.3), DA 57.7 % (+12.2), histamine 15.8 % (+0.1). Female: ACh 11.3,
GABA 20.9, Glu 22.2, DA 51.7 %.

**What it means.** The female result holds in a second animal of the other sex from another
lab: backward traffic is mostly inhibitory, and dopaminergic neurons send more than half of
their output backward. The hubs are the same large inhibitory neurons: CT1 has the highest
degree in both, 10 of the top 12 hubs are GABAergic, and APL sends the most feedback in both.
5-HT and OA cannot be compared: the male table labels only 48 and 101 such neurons.

## 4. Left and right visual systems are nearly independent, now measured (`results/biology_diff.json`)

**Where to look.** Streams, untick cells & bundles, neuron layer = hemisphere.

**What we measured.** Share of weight that stays on the same side (both partners with known
side): optic lane to optic lane 97.2 % in the male, 96.3 % in the female; central (other) to
central (other) 73.9 % and 74.4 %; central complex about 50 % in both.

**What it means.** FlyWire section 4 ("two towers, late fusion") was read by eye. It is now a
measured number and it holds in both sexes. Left and right visual systems exchange about 3 % of
their weight; the sides mix in the central brain.

## 5. Pipelines of different depth converge in the center (`results/biology_diff.json`)

**What we measured.** Median positions in the mushroom-body circuit: olfactory projection
neurons 0.43, Kenyon cells 0.73, MBONs 0.79, DANs 0.90 (female 0.39 / 0.75 / 0.81 / 0.84).
Vision: photoreceptors 0.13, lamina 0.16, medulla 0.31, medulla->lobula 0.49, lobula 0.53,
visual projection 0.76.

**What it means.** The order of roles is the same in both: smell reaches memory in a short
chain, vision passes six lanes, both converge on the central brain. DANs sit later in the male
(one seed).

## 6. Cell types act as computational units, even more so in the male (`results/analysis_malecns.json`)

**Where to look.** Streams, untick cells & bundles, neuron layer = groups; type a cell type into
the search box.

**What we measured.** Median spread of positions within a type (IQR) 0.024 n vs 0.488 n for
random sets of the same size (female 0.044); 97.9 % of 532 types are tighter than the 5 % null
quantile (female 94.6 %).

## 7. Visual centrifugal cells: the same picture in both flies (`results/biology_diff.json`)

**Where to look.** Streams, lane "visual centrifugal".

**What we measured.** Anatomically these run from the central brain to the eye. In both orders
the lane sits early (male 0.42, female 0.32). Its outputs are mostly forward (male 87 %, female
94 %); its inputs only about half (55 % and 53 %).

**What it means.** In both flies the optimizer puts the loop's backward edges on the inputs of
this lane. A pathway that anatomy calls feedback can be forward in the order. Hierarchy plots
should be read with this in mind.

## 8. Neurons with no output (`results/streams/report_streams.json`)

**Where to look.** Streams, right edge, hatched grey cells.

**What we measured.** 2,163 neurons with zero output sit at the tail. 1,618 are medulla
neurons, almost all T1 (1,365) and Lawf2 (253); 301 are VNC motor neurons.

**What it means.** For motor neurons this is expected: their outputs go to muscles. For T1 the
guess (not checked) is that it has few chemical output synapses; Lawf2 output may fall below the
weight-2 floor. In the female the sinks were truncated sensory fragments instead.

## 9. New: where the order cuts the brain-cord loop (`results/vnc_loop.json`)

**What we measured.** FlyWire has no cord, so ascending neurons have no inputs there and sit at
the very front (female median 0.06). In the male they spread out (median 0.27, 16 % in the last
tenth). Several descending types (DNg03, DNge116, DNp17) and Johnston's organ afferents move
from the end of the female order to the front of the male one. Brain -> cord edges are 89.0 %
forward, cord -> brain 77.1 %; edges between brain and cord carry only 4.8 % of male feedback.

**What it means (guess).** Female positions of ascending and descending neurons are partly
artefacts of the volume boundary; the Johnston's organ move is not explained. Do not compare
these lanes across datasets.

## 10. New: where the male has more reciprocal connections (`results/biology_diff.json`, `lane_pair_change.csv`)

**What we measured.** The 2-cycle floor (minimum feedback forced by reciprocal pairs) is 7.6 %
of W in the male CNS and 8.3 % in the male brain vs 6.7 % in the female. Per lane, as a share
of the weight that stays inside the lane, the increase is in the central brain: mushroom body
23.7 % vs 18.7 %, olfaction 23.7 % vs 19.8 %, central complex 17.9 % vs 12.9 %, central (other)
6.8 % vs 4.8 %. Optic lanes change little (medulla 10.1 % vs 9.9 %, lobula 11.2 % vs 10.4 %).
The mushroom body also keeps more of its output inside itself (89 % vs 80 %). Among the lane
pairs that lost the most forward share are mushroom body -> central 80 -> 64 %, mushroom body
-> mushroom body 79 -> 67 %, central -> visual projection 53 -> 37 %; the list also holds neck
pairs (ascending -> descending 99 -> 84 %) and visual projection pairs. Forward shares depend on
the order; the male order here is the CNS order restricted to the brain.

**What it means.** The male has more reciprocal connections in memory and navigation centers;
the optic lanes barely change. Whether this is biology or synapse detection in dense neuropils
(Kenyon-cell axons are among the densest) cannot be told from one dataset per sex.

## 11. New: do the same cell types sit at the same place? (`results/comparison.json`, `noise_ceiling.json`, `type_fb_share_cross_sex.csv`)

**Where to look.** `outputs/fig2_type_rank_female_vs_male.png`.

**What we measured.**
- Position: 7,325 matched types; Spearman of median position 0.77 when both orders come from
  the same protocol (0.72 with the male CNS order). Two good orders of the same brain agree at
  0.92-0.95; this ceiling is optimistic, since all those orders start from the same greedy
  order.
- Feedback share per type (6,782 types with output >= 200 synapses): Spearman 0.61. High in the
  optic lanes (0.70-0.85), low in the mushroom body (0.40).
- 226 types are feedback-heavy (> 50 %) in both flies. Among the largest: optic-lobe feedback
  cells (LMa, Y, Li, TmY14, Lawf1), antennal-lobe local neurons (lLN) and APL.
- Types with sexual dimorphism or fru/dsx expression do not move more than other types of the
  same lane (p = 0.81). Their feedback share is not different from same-lane controls when whole
  types are shuffled (female z -1.7, p = 0.08; male z -0.3 to -0.5, p >= 0.58).

**What it means.** The two datasets differ well beyond optimizer noise, but the difference is
not concentrated in sex-specific circuits. Lab, segmentation, type-matching errors (2,751 male
neurons have combined type labels) and the cord are as likely causes as sex.

## 12. New: the nerve cord inside the order (`results/biology_diff.json`, male only)

**What we measured.** VNC intrinsic neurons of the three thoracic neuromeres sit together near
the end (T1 0.93, T2 0.94, T3 0.93). Posterior abdominal neuromeres A4-A9 sit in the middle
(0.46-0.56; about 85 % of these neurons are VNC intrinsic); A1-A3 sit with the thorax (0.83-0.93).
All motor subclasses (front, middle and hind leg, wing, neck, haltere, abdomen) are at
0.975-0.998.

**What it means (guess).** Leg and wing circuits are the last stage before the muscles. The
posterior abdominal neuromeres form a block in the middle of the order; whether that reflects
their own sensory input (abdomen, genitalia) is not tested.

## 13. Hemispheres

Swapping paired left and right neurons costs 1.62 pp in the male (female 2.19 pp). Both orders
put one side first more often (female left in 57 % of types, male right in 56 %, both
p = 3e-4), in opposite directions. In FlyWire the mirrored order is a near-equal optimum (A5);
this was not re-tested on the male, and there is one seed per fly. So the side is most likely
the optimizer's arbitrary choice.

## 14. New (2026-11-24): the cell-type null overshoots the real graph in the male too (`results/nulls/malecns__type_dcsbm__*.json`)

**What was measured.** A random graph that keeps, for every neuron, its in/out degree and
out-strength towards every cell type (11,768 types), run through the same optimizer protocol.
Twelve such graphs reach 81.228 +- 0.013 % forward weight; the real male graph reaches 80.83 %
(two seeds), below all twelve, a gap of -0.40 pp. FlyWire: 84.232 vs 83.658, -0.57 pp. At five
times the budget the gap for the probed draw does not shrink (-0.47 pp; FlyWire -0.65 pp). The
real graph keeps a reciprocal floor of 7.64 % of weight, its type-matched nulls 2.82 % (FlyWire
6.72 vs 2.02). A control with every untyped neuron as its own type stays above the real graph
(81.267 %, one draw).

**What it may mean.** The FlyWire result A7 (`biology/FINDINGS.md`) is not specific to that
dataset: in a second fly, from another lab and of the other sex, the type-level wiring diagram
again gives at least as much feedforward order as the real graph, while the real graph has 2.7
times the reciprocal weight of its type-matched null. Both numbers are heuristic values under
one protocol, not optima. Details and limits: `README.md` section 11.

## Caveats for all claims

- One MFAS seed for the male on MPS; female numbers are the pinned CUDA champion.
- One animal per sex; dataset differences (lab, segmentation, synapse detection, the cord, 20 %
  more neurons, 2.2x denser) cannot be separated from sex.
- Male lanes are a mapping to FlyWire lanes (optic via flywireType), not an official class.
- Rho in this file is the FlyWire-function value (0.237); `analysis/analyze.py` uses another tie
  convention (0.470). Compare only within one script.
- The sex flag is defined per dataset (female: dimorphic, female-specific or fru/dsx; male:
  any dimorphism label or fru/dsx); `analysis/dimorphism.py` uses only types present in both.

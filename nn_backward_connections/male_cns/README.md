# Male CNS v1.0: the project's analyses on a second fly, the other sex

Status 2026-11-17: first pass done, one MFAS seed, unreviewed by an independent run unless
marked. Everything here is separate from the FlyWire record; nothing in `src/`, `layering/`,
`biology/` or `autoresearch/` was changed.

Dataset: Janelia FlyEM Male CNS v1.0 (brain + ventral nerve cord, one male fly). Build and
provenance: `BUILD.md`. Graph used everywhere: traced-only neuron pairs, weight = synapse
count, weight >= 2 (same floor as our FlyWire graph): n = 164,425, m = 15,270,204,
W = 113,731,681. Brain-only subgraph (nerve-cord superclasses removed): n = 144,153,
m = 13,014,794.

## How to reproduce

```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
male_cns/.venv/bin/python -I male_cns/build_graph.py   # feather -> npz (see BUILD.md)
$PY male_cns/ann_prep.py                                # lanes / flow / NT in FlyWire vocabulary
$PY male_cns/run_mfas.py --seed 42                      # H64 with the connectome constants, ~62 min MPS
$PY male_cns/analyze.py --dataset flywire               # same analysis code on FlyWire (port check)
$PY male_cns/analyze.py --dataset malecns
$PY male_cns/analyze.py --dataset malecns_brain
$PY male_cns/run_nulls.py                               # null ladder, ~70 min MPS
$PY male_cns/vnc_loop.py
$PY male_cns/noise_ceiling.py
$PY male_cns/dimorphism.py
$PY male_cns/compare.py                                 # table + figures
```

| file | what |
|---|---|
| `build_graph.py`, `ann_prep.py` | graph and per-node annotations (lanes follow `layering/tools/fly_streams.py:lane_of`; optic neurons take the FlyWire `cell_class` of their matched `flywireType`) |
| `run_mfas.py` | the FlyWire champion pipeline H64, unchanged constants, scored by the frozen oracle |
| `analyze.py` | one dataset-agnostic script for findings A2, A3, A4, A5 (swap), B1, B2, B5, B7 and the 2-cycle floor of A1 |
| `run_nulls.py` | null ladder A1 with the FlyWire G01 protocol (4,000 epochs, 8 sift sweeps) |
| `vnc_loop.py`, `noise_ceiling.py`, `dimorphism.py` | new analyses that only make sense with two flies |
| `compare.py` | `results/comparison_table.csv`, `results/comparison.json`, `outputs/fig1..fig6` |
| `results/` | every number quoted below (JSON / CSV), the male order `malecns_H64_s42_rank.npy` |
| `page_ru/index.html` | Russian one-page visual summary for the supervisor (open locally; images from `outputs/`) |
| `logs/` | run logs |

## Port check

`analyze.py --dataset flywire` reproduces the logged FlyWire numbers: FF 84.258 %,
2-cycle floor 6.717 % of W (42.7 % of FB), 9,626 SCCs with a giant SCC of 126,840, 2,447
zero-output sinks, FB span median 0.133 n, GABA+Glu 55.8 % of FB vs 38.2 % of FF, type
contiguity IQR 0.044 vs null 0.489, NT backward shares (ACh 11.3 %, DA 51.7 %, 5-HT 28.8 %)
with z close to the published A3 values (GABA +15.7 vs +15.4, Glu +17.3 vs +17.2, DA +14.8 vs
+14.8, 5-HT +9.7 vs +9.5; ACh -39.8 vs -33.4). Small differences by definition: "types tighter
than the null" is 94.6 % of 429 types here vs 92.9 % of 437 published (different type set and
null); mirror pairs 65,158 (published 65,452) and swap cost -2.19 pp
(-2.21); flow rho is 0.454 here (average ranks for ties) vs 0.263 in `fly_streams.py`
(stable tie-break), so only compare rho within this folder.

## Results (one MFAS seed per dataset; observation first, then a cautious reading)

### 1. The male CNS is less feedforward, mostly as its random nulls are

| | female (FlyWire brain) | male brain only | male CNS |
|---|---|---|---|
| FF %, H64 champion pipeline | 84.26 | 81.16 (restricted order) | 81.81 |
| FF %, G01 protocol (4k epochs) | 83.66 (3 seeds) | 80.68 (1 seed) | 80.83 (2 seeds, 80.828 / 80.835) |
| greedy-FAS start | 68.91 | 65.76 | 65.88 |
| reciprocal edges | 18.4 % | 23.2 % | 22.3 % |
| 2-cycle floor, % of W | 6.72 | 8.26 | 7.64 |
| floor as share of FB | 42.7 % | 42.7 % (G01 order) | 42.0 % |

Rows other than the FF rows use the H64 order (restricted to the brain for the middle column).

Reading: the male graph is about 3 pp less feedforward at equal protocol. Only the 2-cycle
floor is a rigorous bound: it is 0.9-1.5 pp of W higher in the male, about half of the gap, and
it stays ~42 % of all feedback in both flies. The null ladder (section 10) shows the male nulls
lose as much order as the real graph or more (G(n,m) -4.5 pp, degree-matched -5.1 pp, lane
dcSBM -3.0 pp, real -2.8 pp). So the gap is consistent with size, density (93 vs 41 edges per
neuron) and degree differences and needs no extra male feedback structure. Density alone is
not isolated, and the male ladder includes the nerve cord while the female one is brain only.
Part of the gap may be optimizer budget: from G01 to H64 the male gains +0.98 pp, the female
+0.60 pp, so the male is less converged at a fixed epoch count. Lab and segmentation
differences cannot be separated from sex. One animal each.

### 2. The same architecture: globally feedforward, locally recurrent (B1, A2)

- Giant SCC: 98.4 % of male neurons (female 92.8 %); >99.99 % of FB weight inside it in both.
- FB is short: median span 0.085 n in the male vs 0.133 n in the female; 22.3 % of all weight
  lies within 1.7 % of n (female 19.2 %).
- FB is spread over many nodes: the top 1,000 nodes touch 14.3 % of FB (female 20.1 %).

### 3. Sensory to motor axis and cell types (A4)

- Lane order is almost the same (`outputs/fig1`, `fig6`): sensory and photoreceptors first,
  optic layers in anatomical order, mushroom body and central brain late, descending and motor
  last. Flow medians: afferent 0.07, intrinsic 0.54, efferent 0.98 (female 0.07 / 0.55 / 0.96).
- Nerve cord: VNC sensory neurons sit at the very front (median 0.05), VNC intrinsic at 0.93,
  VNC motor at 0.98. The cord is not a separate block; it is the input and output ends of
  one axis.
- Cell types are even tighter in the male: median within-type IQR 0.024 n vs 0.488 for random
  sets (female 0.044); 97.9 % of 532 types tighter than the 5 % null quantile (female 94.6 %).

### 4. Chemistry of feedback replicates (A3)

Share of a transmitter's output weight that is feedback (z vs within-superclass label shuffle):

| NT | female | male CNS |
|---|---|---|
| ACh | 11.3 % (z -39.8) | 13.8 % (z -34.9) |
| GABA | 20.9 % (+15.7) | 25.5 % (+22.0) |
| Glu | 22.2 % (+17.3) | 22.7 % (+12.3) |
| DA | 51.7 % (+14.8), 682 neurons | 57.7 % (+12.2), 392 neurons |
| 5-HT | 28.8 % (+9.7), 1,891 | 31.2 % (+1.9), 48 |
| OA | 16.7 % (+1.5) | 12.7 % (-0.6) |
| HA | not annotated | 15.8 % (+0.1) |

2,000 label permutations within superclass strata.

GABA+Glu carry 50.2 % of male FB vs 34.8 % of FF (female 55.8 / 38.2). Dopamine is the most
backward-biased transmitter in both flies. Caveat: the male NT table labels far fewer 5-HT and
OA neurons, so those rows are not comparable.

### 5. Hubs (B7): the same inhibitory giants

Top-degree hubs: 10 of 12 are GABAergic in the male (11 in the female); CT1 is the top hub in
both (the two male CT1 cells have degree 18,306 and 17,658; female top 16,784). The top feedback sender is APL in both flies (male 0.49 %
of FB).

### 6. Hemispheres (A5, swap only)

Swapping paired left/right neurons within type costs 1.62 pp in the male CNS (female 2.19 pp),
71,530 pairs (87 % of neurons). Which side comes first, counted over types with >= 5 pairs and
no tie: left earlier in 433 of 981 male types (44.1 %, binomial p = 3e-4) and in 418 of 736
female types (56.8 %, p = 3e-4). Both orders have a side bias, in opposite directions. Since
the mirrored brain is a twin optimum (A5), the sign is most likely the optimizer's arbitrary
choice; one seed each cannot tell. Re-tune after the swap (the other half of A5) was not run.

### 7. New with two flies: where the order cuts the brain-cord loop (`results/vnc_loop.json`)

- FlyWire lacks the cord, so ascending neurons (AN) have no inputs there and sit at the very
  front (female median 0.06). In the male, with their cord inputs present, AN spread out
  (median 0.27; 16 % in the last tenth). Several DN types (DNg03, DNge116, DNp17...) and the
  Johnston's organ afferents (JO-A, JO-C) jump from the end of the female order to the front of
  the male order.
- Brain->cord edges are 88.9 % feedforward, cord->brain 77.1 %; the brain-cord crossing holds
  only 4.8 % of all male FB. The cord's own FB (cord->cord) is 11.3 % of FB.
- Reading (guess, not tested): the female positions of AN and DN neurons are partly artefacts
  of the volume boundary (the order audits its dataset, as in B5). JO afferents are not
  obviously cut at the neck, so their jump is unexplained. Either way, do not compare these
  lanes across datasets.

### 8. New: cross-sex agreement of type positions (`results/comparison.json`, `noise_ceiling.json`)

- 7,325 FlyWire types matched by name. Spearman of per-type median position, female vs male
  (male full-CNS order restricted to the brain): 0.72; 0.74 without the neck lanes.
- Noise ceiling: two good orders of the same brain agree at 0.95 (female: three protocol orders
  vs the champion) and 0.92 (male: two protocol orders vs H64). These are optimistic ceilings,
  since all orders start from the same greedy-FAS order. With both orders from the same protocol
  (female G01 seed 42, male brain-only re-optimized) the cross-sex value is 0.77 (0.74 without
  neck lanes). So the two datasets differ well beyond pipeline noise; type-matching errors are
  part of that difference.
- Sexually dimorphic or fru/dsx types (617 matched) do NOT move more than other types of the
  same lane (median |shift| 0.028 vs 0.033 n, lane-stratified permutation p = 0.81). So the
  difference is not concentrated in sex-specific circuits; lab, segmentation, synapse
  detection and the cord are as likely causes. This is a negative result.

### 9. Dataset audit (B5)

2,163 zero-output sinks sit at the tail (median position 0.993). 1,618 of them are medulla
neurons, almost all T1 (1,365) and Lawf2 (253). My guess, not checked: T1 is known to have
few chemical output synapses; Lawf2 output may fall below the weight-2 floor.

### 10. Null ladder (A1)

Same G01 protocol as FlyWire (4,000 epochs, 8 sift sweeps, pipeline seed 42); FlyWire values
are the logged runs in `biology/results/nullmodels/runs/`. Male draws: 3 per rung (2 for
w_shuffled), female 12-20. Within a rung std <= 0.025 pp, range <= 0.10 pp, in both.

| rung | female | male CNS |
|---|---|---|
| G(n,m) | 69.82 | 65.36 |
| degree-matched, weights shuffled (w_shuffled) | 77.22 | 72.91 |
| degree-matched, source weights (w_source) | 79.98 | 74.84 |
| degree + lane blocks (L3 dcSBM) | 81.90 | 78.88 |
| real graph | 83.66 | 80.83 |

- The real graph is above every null draw in both flies; margin over the best rung (lane dcSBM):
  +1.76 pp female, +1.95 pp male.
- Degrees carry most of the order in both (+10.2 / +9.5 pp over G(n,m)).
- Lanes add more in the male (+4.0 vs +1.9 pp). Probable reason (guess): the male lanes include
  the nerve cord, whose sensory and motor lanes pin the two ends of the order.
- Figure: `outputs/fig5_null_ladder.png`.

## Limits

- One H64 seed on MPS for the male (non-deterministic device); FlyWire numbers are the CUDA
  champion. The male protocol runs used the same code on MPS.
- Weight floor 2 on both graphs; the male still has 2.7x the edges and 2.7x the synapses
  (2.3x synapses per neuron), so per-edge statistics differ in scale.
- Lanes for the male are a mapping, not an official class: optic lanes come from FlyWire type
  matches (249 optic neurons without a match go to "optic (other)").
- 2,751 male neurons carry combined FlyWire labels such as "FC1A,FC1B,FC1F"; 315 of these 420
  labels have no exact FlyWire match and drop out of the cross-sex comparison.
- Not ported: A7 type-dcSBM ladder, B6 central-only re-optimisation, A5 re-tune, B4 degraded
  order, the interactive streams artifact, MICrONS-only findings A6/B8.
- Aborted run, logged for completeness: H64 was first started on the graph with weight-1
  edges kept (25,563,096 edges); it was stopped during Rocket, before any score, when the
  weight floor was set to 2 to match FlyWire. Greedy-FAS on the all-weights graph was not recorded.

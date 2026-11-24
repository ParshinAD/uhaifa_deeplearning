# Male CNS v1.0: the project's analyses on a second fly, the other sex

Status 2026-11-17: first pass done, one MFAS seed, numbers and code reviewed once by an
independent agent. Everything here is separate from the FlyWire record; nothing in `src/`,
`layering/`, `biology/` or `autoresearch/` was changed.

Dataset: Janelia FlyEM Male CNS v1.0 (brain + ventral nerve cord, one male fly). Build and
provenance: `BUILD.md`. Graph used everywhere: traced-only neuron pairs, weight = synapse
count, weight >= 2 (same floor as our FlyWire graph): n = 164,425, m = 15,270,204,
W = 113,731,681. Brain-only subgraph (nerve-cord superclasses removed): n = 144,153,
m = 13,014,794.

## Start here

| want to | open |
|---|---|
| look at the network | `outputs/streams/malecns_streams.html` (interactive page, same as the FlyWire "Fly Connectome Streams"; open in a browser) |
| read the biology and the male vs female differences | `FINDINGS.md` (English), `FINDINGS.ru.md` (Russian) |
| check a number | this README (sections below) and the file in `results/` it names |
| rebuild everything | "How to reproduce" below; data provenance in `BUILD.md` |

## Folder map

| folder / file | contents |
|---|---|
| `BUILD.md` | download URLs, sha256, graph definition, reader venv |
| `FINDINGS.md`, `FINDINGS.ru.md` | biological readings, in the format of `layering/FLY_FINDINGS.md` |
| `pipeline/` | `build_graph.py` (feather -> npz), `ann_prep.py` (lanes / flow / NT in FlyWire vocabulary), `run_mfas.py` (H64 order), `run_nulls.py` (null ladder), `run_nulls_l4.py` (cell-type null) |
| `analysis/` | `analyze.py` (one script for both flies: A2, A3, A4, A5 swap, B1, B2, B5, B7, 2-cycle floor), `vnc_loop.py`, `noise_ceiling.py`, `dimorphism.py`, `biology_diff.py` (hemispheres, reciprocity by lane, per-type FB, lane-pair changes, MB roles, VNC neuromeres), `compare.py` (table + figures) |
| `streams/` | the interactive page pipeline (prep, layouts, export, template, build); its own `README.md` |
| `results/` | every number (JSON / CSV); male order `malecns_H64_s42_rank.npy`; `nulls/` null-ladder runs and orders; `streams/` page data |
| `outputs/` | figures `fig1..fig6`; `streams/` the page and its static previews |
| `logs/` | run logs |

## How to reproduce

```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
male_cns/.venv/bin/python -I male_cns/pipeline/build_graph.py   # feather -> npz (see BUILD.md)
$PY male_cns/pipeline/ann_prep.py                                # lanes / flow / NT in FlyWire vocabulary
$PY male_cns/pipeline/run_mfas.py --seed 42                      # H64 with the connectome constants, ~62 min MPS
$PY male_cns/analysis/analyze.py --dataset flywire               # same analysis code on FlyWire (port check)
$PY male_cns/analysis/analyze.py --dataset malecns
$PY male_cns/analysis/analyze.py --dataset malecns_brain
$PY male_cns/pipeline/run_nulls.py                               # null ladder, ~70 min MPS
$PY male_cns/pipeline/run_nulls_l4.py                            # cell-type null, 12 draws, ~60 min MPS
$PY male_cns/pipeline/run_nulls_l4.py --n-nulls 1 --epochs 20000 --sift-sweeps 40   # budget probe, ~19 min
$PY male_cns/pipeline/run_nulls_l4.py --real --epochs 20000 --sift-sweeps 40        # ~19 min
$PY male_cns/analysis/vnc_loop.py
$PY male_cns/analysis/noise_ceiling.py
$PY male_cns/analysis/dimorphism.py
$PY male_cns/analysis/compare.py                                 # table + figures
$PY male_cns/analysis/biology_diff.py                            # ~3 min
# interactive page, ~1 min total (see streams/README.md)
$PY male_cns/streams/malecns_streams_prep.py
$PY male_cns/streams/malecns_streams.py
$PY male_cns/streams/malecns_artifact_export.py
$PY male_cns/streams/build_malecns_html.py
```

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

Reading: the male graph is about 3 pp less feedforward at equal protocol. The 2-cycle floor
(a strict lower bound on feedback within each graph) is 0.9 pp of W higher in the male CNS and
1.5 pp in the male brain, a third (CNS) to a half (brain) of the gap; a difference of two lower
bounds does not bound the difference in feedback. The floor stays ~42 % of all feedback in both. The null ladder (section 10) shows the male nulls
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
- Brain->cord edges are 89.0 % feedforward, cord->brain 77.1 %; the brain-cord crossing holds
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
| cell-type dcSBM (L4, added 2026-11-24) | 84.23 | 81.23 |

- The real graph is above every draw of the first four rungs in both flies; margin over the
  lane dcSBM: +1.76 pp female, +1.95 pp male.
- Degrees carry most of the order in both (+10.2 / +9.5 pp over G(n,m)).
- Lanes add more in the male (+4.0 vs +1.9 pp). Probable reason (guess): the male lanes include
  the nerve cord, whose sensory and motor lanes pin the two ends of the order.
- Figure: `outputs/fig5_null_ladder.png`.

### 11. Cell-type null (A7): the inversion replicates in the male

`pipeline/run_nulls_l4.py`, runs `results/nulls/malecns__type_dcsbm__*.json`. Same generator
and protocol as the FlyWire L4 rung (`biology/studies/null_ladder_v2/REPORT.md` section 2.3):
every node keeps its in/out degree and out-strength profile over cell types, targets are
permuted inside each (source type, target type) block. Male types: `cell_type`, 11,768 labels
including fallbacks (2,069 nodes without a type get "<superclass> (class)"; 1.45 % of W touches
one). The reading rule was written in the script before the first run. Not in fig5.

| | female (v2 report) | male CNS |
|---|---|---|
| real graph, FF % | 83.658 (3 seeds) | 80.828 / 80.835 (2 seeds) |
| type dcSBM, 12 draws, FF % | 84.232 +- 0.009 | 81.228 +- 0.013 (81.211 to 81.254) |
| real minus null | -0.574 pp | -0.396 pp |
| draws above the real graph | 12 / 12 | 12 / 12 |
| 5x budget (20,000 epochs, 40 sweeps), real / one draw | 84.103 / 84.756 | 81.454 / 81.926 |
| real minus null at 5x budget | -0.653 pp | -0.471 pp |
| 2-cycle floor, % of W, real / null | 6.72 / 2.02 | 7.64 / 2.82 |
| FB above own floor, real / null | 9.62 / 13.75 | 11.53 / 15.95 |
| weight moved by one draw (gs8000; female value recomputed, not logged) | 77.7 % | 75.3 % |
| fallback control: untyped neurons as singleton types, one draw | not run | 81.267 |

Reading: in both flies, under this protocol and budget, the type-level wiring diagram reaches
the real level of order and passes it, and the real graph keeps 2.7x (male) to 3.3x (female)
the reciprocal floor of its type-matched null. For the one probed draw the gap does not shrink
at 5x budget (draw gs8000 vs real seed 42: -0.385 to -0.471 pp). On the FB-above-floor axis
the real graph is ahead by 4.4 pp (male) and 4.1 pp (female).

Limits:
- One pipeline seed per null, one probe draw, one real seed in the probe. Direction is
  protocol-conditional (identical heuristic budget on both sides). The greedy start has the
  same sign (null 66.19 vs real 65.88) but is not a direction control: greedy also puts the
  lane and degree nulls above the real graph.
- Fallback types are coarse (largest: `cb_intrinsic (class)` 880 nodes, `vnc_sensory (class)`
  697) and 1.45 % of W touches one, more than the gap as a worst-case bound. The control draw
  with every untyped neuron as its own type gives 81.267 %, still above the real graph, so the
  fallback labels do not drive the inversion (one draw).
- A draw is a multigraph: 4.19 % of slots fall on an already used pair (FlyWire 2.19 %). The
  scorer and the floor add parallel slots as weight on one edge. A simple-graph variant was
  not run.
- 2.68 % of W sits in one-slot blocks and stays exactly real; 24.7 % of W is unmoved in one
  draw. 5,358 of the 11,768 types have two nodes, so much of the permutation exchanges left
  and right homologues.
- Male types are finer (11,768 vs 8,857); 781 neurons carry combined `cell_type` labels.
- `gen_stats.variant` in these JSONs reads "L3_dcsbm" (generator reused); filter by `kind`.
- One animal per sex: the replication is across datasets, not a statement about sex.

## Limits

- One H64 seed on MPS for the male (non-deterministic device); FlyWire numbers are the CUDA
  champion. The male protocol runs used the same code on MPS.
- Weight floor 2 on both graphs; the male still has 2.7x the edges and 2.7x the synapses
  (2.3x synapses per neuron), so per-edge statistics differ in scale.
- Lanes for the male are a mapping, not an official class: optic lanes come from FlyWire type
  matches (249 optic neurons without a match go to "optic (other)").
- 2,751 male neurons carry combined FlyWire labels such as "FC1A,FC1B,FC1F"; 315 of these 420
  labels have no exact FlyWire match and drop out of the cross-sex comparison.
- Not ported: the other A7 rungs (W0, L2R), B6 central-only re-optimisation, A5 re-tune, B4 degraded
  order, static neuron-cloud PNGs (`fly_cloud.py`), MICrONS-only findings A6/B8.
- Aborted run, logged for completeness: H64 was first started on the graph with weight-1
  edges kept (25,563,096 edges); it was stopped during Rocket, before any score, when the
  weight floor was set to 2 to match FlyWire. Greedy-FAS on the all-weights graph was not recorded.

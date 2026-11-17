# Survey: other datasets for the MFAS pipeline (2026-11-17)

Web survey of directed weighted graphs where our pipeline (MFAS order, biology
analyses on annotations, random-graph nulls, streams x depth view) could run.
Five scoped searches (A-E), a gap hunt in two sessions (F, then G and H), an
independent fact-check of 15 load-bearing claims plus one more (V, V2), and a
provenance check of the mouse table (M). Raw notes with source URLs are in the files below; items
not checked on a primary page are marked UNVERIFIED there.

| file | scope |
|---|---|
| `A_invertebrate.md` | synaptic connectomes: fly, worm, annelid, Ciona, others |
| `B_vertebrate.md` | synaptic connectomes: MICrONS versions, H01, V1DD, zebrafish, songbird |
| `C_mesoscale.md` | directed region-level graphs (tract tracing), FF/FB ground-truth labels |
| `D_nonneural_benchmarks.md` | FAS/LOP benchmarks, FlyWire leaderboard, non-neural hierarchies |
| `E_prior_work.md` | papers computing feedforward order / hierarchy on connectomes |
| `F_gaps.md` | second pass for missed datasets (session 1, budget ran out) |
| `G_gaps2_aggregators_em.md` | session 2: KONECT, neuPrint/CAVE/BossDB/Codex lists, mouse cerebellum/thalamus/OB EM, human EM |
| `H_gaps2_species_ann_fc.md` | session 2: Aplysia, leech, Xenopus, lamprey; trained ANN/RNN weights; fly functional connectivity |
| `V_verification.md` | fact-check verdicts for 15 claims (10 checked in session 1) |
| `V2_verification.md` | session 2: the 5 remaining claims plus the Male CNS edge count |
| `M_mouse_table.md` | session 2: provenance check of `data/table_mouse.txt` against SEU-net, Allen-net, BRICseq-net, Oh 2014, Knox 2019 |

## Facts that change our own record

- **84.61 % is no longer the best FlyWire score** (V#1, confirmed on the live
  leaderboard). Top: Caleb Westfall, 2026-11-08, 35,544,233 = 84.807 % of
  41,912,141; places 2-5 (Oct-Nov 2026) at 84.71-84.80 %. Every row has a
  downloadable order, so the claims can be re-scored with our oracle. Our H64
  (84.258 %) is ~0.55 pp behind the top, not 0.36 pp. No write-ups found.
- Vahidi arXiv 2506.13799 (`2506.13799v1.pdf` in the repo root) is single-author; its 0.7524 is
  the initial order of its Algorithm 1, final 0.8461 = 35,462,925 (V#9).
- FlyWire/Codex has used a new synapse table (Yu et al. 2025) since July 2025;
  weights differ from v783 (F). Any external score must state its table.
- `FLY_FINDINGS.md` §11 quotes Winding 2023 as "66 % forward". The paper has no
  such number; its 66 % values mean other things; recurrence = 41 % of brain
  neurons (V#4). Needs correction.
- Durbin 1987 already did a max-forward sort of C. elegans (V2#7, confirmed from
  the thesis, chapter 8: 140 of 2,890 chemical synapses upward = 4.8 %; 8.4 % on
  the averaged data; class-level graph of the nerve ring, heuristic order). Feedback is "almost all
  ... in the form of reciprocal synapses", grouped in small modules. The thesis
  does not say "local"; that word was our paraphrase. Must be cited as prior work.
- Hoeller et al. (bioRxiv 10.64898/2025.12.22.696097): optic-lobe connections of
  the Male CNS are 45 % feedforward, 43 % lateral, 12 % feedback, and 69 % of
  feedback synapses are inhibitory (V2#8). Directions come from information-flow
  layers (+-0.5 layer), not an MFAS order, so the 12 % is not comparable with
  our feedback share.
- Male CNS preprint (10.1101/2025.10.09.680999 v2): "25.6M edges between 166,391
  neurons" (V2#16). Our own build of the traced-only table has 25,563,197 rows
  (`male_cns/BUILD.md`). The "124.2M synaptic connections" figure is still snippet-only.
- No paper found that compares a connectome's optimal feedback share with
  degree-matched or cell-type SBM nulls (E; absence in search, not proof).

## Ranked picks

Fit = how directly the whole pipeline applies (graph + annotations + access).

| # | dataset | why | access | status |
|---|---|---|---|---|
| 1 | FlyWire leaderboard orders | re-score top 5 with our oracle; diff vs H64 order shows where the 0.55 pp lives | download per row on codex | confirmed |
| 2 | Male CNS v1.0 (Janelia 2025/26) | second whole fly CNS, other sex/lab; 166,700 neurons, 11,710 types, NT, side | Feather files gs://flyem-male-cns, CC BY, no token (downloaded 2026-11-17) | counts confirmed on Janelia page (V#2: partly confirmed); no-token access confirmed by our download; analysed in `male_cns/` |
| 3 | MICrONS v1412 graph (Reimann, Zenodo 10.5281/zenodo.16744240) | larger/cleaner than our v117 build, coarse E/I, m-type, positions, proofreading | .h5 761 MB, CC BY 4.0, no token | confirmed |
| 4 | Drosophila larva (Winding 2023) | 3k neurons, L/R pairs; small enough for exact MIP optimum = true gap of our heuristic | GitHub; Netzschleuder `fly_larva` | confirmed |
| 5 | C. elegans Witvliet 2021 (8 stages) + Cook 2019 | paper: "brain becomes progressively more feedforward" with age — test with MFAS score | WormWiring / paper SI | confirmed (abstract) |
| 6 | Macaque Markov 2014 / Froudist-Walsh 2021, marmoset Majka 2020, mouse Harris 2019 | anatomical FF/FB labels (SLN, laminar) per connection = external ground truth for our order | Netzschleuder `macaque_markov`, core-nets.org, marmosetbrain.org, GitHub AllenInstitute/MouseBrainHierarchy | Markov confirmed; Harris README confirmed, files not opened |
| 7 | C. elegans signal propagation atlas (Randi 2023) | directed causal functional graph, compare with anatomical order | wormneuroatlas, DANDI 001075 | UNVERIFIED (F only) |
| 8 | BANC, MANC | fly brain+cord and VNC; Netzschleuder `fly_banc` for loader cross-check | Dataverse / neuPrint | partly UNVERIFIED |
| 9 | Fish1 zebrafish, V1DD | vertebrate contrast cases (whole larval brain; second mouse V1) | CAVE token; CAVE token | Fish1 confirmed (V2#13, CAVE `fish1_full`, no static edge list); V1DD UNVERIFIED |
| 10 | ISCAS circuit graphs (Hecht 2020) | exact FAS optima published -> algorithm benchmark (unweighted) | git.mpi-cbg.de/mosaic/FaspHeuristic | confirmed (most ISCAS exact; ISPD98 only ibm05) |
| 11 | Faculty hiring (Clauset 2015) | same objective (min violation ranking) + degree-preserving null, non-brain comparison | Netzschleuder | confirmed; 9-14 % up is Clauset only |
| 12 | flyvis (Lappalainen 2024) trained visual model | trained signed weights on the fly visual connectome topology (45,669 neurons, 1.5M connections): MFAS on trained vs synaptic weights | GitHub, MIT code; weights license UNVERIFIED | H |
| 13 | Songbird j0251 (SyConn2) | vertebrate loop circuit, static .npy with no token (18.4M synapse rows) | syconn.esc.mpcdf.mpg.de | confirmed (V2#12); data license not stated |
| 14 | Mouse CA3 (Zheng, Seung, Tank 2026) | recurrent mammalian region, 1,815 pyr + 229 inh, all soma neurons proofread (voxel share UNVERIFIED) | CAVE, free account | G; edge counts unknown |
| 15 | Trained RNN zoos (Driscoll 2024 multitask; Herrmann 2024 LSTM zoo) | "trained network vs brain" feedback; checkpoints over training | Quilt; Zenodo CC BY | H; sizes partly UNVERIFIED |

Weaker: mouse cerebellum Park 2023 (open 2.3 MB, near-DAG control), neuPrint
`mushroombody` (983 neurons), MICrONS Basil (automated only), mosquito antennal
lobe (size unknown), Allen V1 biorealistic model, Xenopus tadpole model connectome
(generated, not traced), Franconville 2018 CX functional pairs, KONECT dominance
networks. Not found: mouse olfactory bulb or thalamus EM graphs beyond Morgan 2016,
a second human EM connectome, released Aplysia/leech/lamprey graphs, whole-brain fly
functional connectivity after Pospisil 2024. Also weaker: hemibrain and optic lobe (truncated volumes), FANC (partly proofread),
H01 (104 proofread cells), web/LAW graphs (weak baselines), LOLIB (dense, LOP
solvers win), F-TRACT (missing pairs), TRRUST (unweighted).

## 148-node mouse table

Provenance still unknown (`M_mouse_table.md`). Ruled out with a positive control
(the test finds a planted scaled copy): Oh 2014 (raw, p-thresholded, 4 simple
normalizations), all SEU-net variants, Allen-net (= Oh 2014 W_ipsi +
W_contra on 213 regions), BRICseq-net (38x41), Knox 2019 regionalized matrices.
Ruled out earlier on size: Harris 2019 (43 areas), Calabrese 2015 dMRI (296 ROIs).
No candidate matches more of our weights than chance. Remaining leads: a
nonlinear transform of a known matrix, Zingg 2014, Wang 2022 chemo-connectome,
BAMS, Rubinov 2015; cheapest is to ask whoever supplied the file.

## Limits

- Web search had a shared per-session budget. Session 2 finished the planned F
  areas (G, H) except MouseLight details and Gao 2022; some items in G/H stay UNVERIFIED (marked there), and the full
  CAVE datastack list needs a login.
- V/V2: all 16 claims have a verdict; #14 not found (unsupported). Still
  snippet-only: the Cell version's "124.2M synaptic connections / 166,483 neurons" for the Male CNS (the Cell page
  returns 403). The November 2026 C. elegans "3x recurrence" preprint (E) was not
  found and is treated as unsupported.
- Only the Male CNS was downloaded into `data/` and run through the pipeline
  (see `male_cns/README.md`); small files (Park 2023 zip, SEU-net xlsx, M
  candidates) were fetched for checks only. Male CNS neuron counts differ by
  source: 166,691 (preprint), 166,391 (preprint graph), 166,700 (v1.0 release).

## Resume point

Survey done (A-H, V, V2, M). Do not redo it. Open follow-ups, not survey:
re-score the FlyWire leaderboard orders with the frozen oracle; fix the "66 %"
Winding quote in `layering/FLY_FINDINGS*.md` §11; cite Durbin 1987 with the
corrected wording; next datasets after the Male CNS: MICrONS v1412, larva (exact
MIP optimum), flyvis trained weights.

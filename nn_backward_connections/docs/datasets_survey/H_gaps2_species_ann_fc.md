# H. Gap-hunt part 2: small-species circuits, trained ANN weights, fly functional connectivity (2026-11-17)

Second pass for three areas that F could not cover: (a) circuit-level graphs in Aplysia, leech, Xenopus,
lamprey and other small nervous systems; (b) released trained RNN / ANN weight matrices; (c) Drosophila
optogenetic or functional connectivity that gives a directed graph. Only items not already in A-F are listed.
About 20 WebSearch calls and 20 WebFetch calls were used. Several fetches failed (bioRxiv rate limit 429,
eLife and PMC pages that render through JavaScript, Nature login redirect), so some numbers stay UNVERIFIED.

"VERIFIED" means the number was read on a page fetched in this session.
"UNVERIFIED (snippet)" means it was only in a search-result snippet.
"UNVERIFIED" means memory or not found.

Fit: 5 = use next, 4 = strong, 3 = useful with caveats, 2 = marginal, 1 = not usable as a graph.

## (a) Small-species circuits

| Name | Citation / DOI | URL | Nodes / edges | W / D | Annotations | Access, license | Fit | Why |
|---|---|---|---|---|---|---|---|---|
| Xenopus hatchling tadpole spinal cord, generated "anatomical" connectome | Borisyuk, Al Azad, Conte, Roberts, Soffe, Front Neuroinform 5:20 (2011), 10.3389/fninf.2011.00020; Roberts et al., J Neurosci 34:608 (2014); Ferrario et al., eLife 7:e33281 (2018), 10.7554/eLife.33281 | ModelDB 238332 (modeldb.science/238332), github.com/ModelDBRepository/238332 | 2011 model: ~1,500 neurons, 6 cell types, 122,221 connections (51,530 ascending, 70,691 descending) (UNVERIFIED (snippet)). Later versions ~1,400-2,000 neurons, ~90,000 synapses (UNVERIFIED (snippet)) | D; synapse presence (binary per pair as far as seen, weight UNVERIFIED) | cell type (6 classes incl. cIN, dIN, aIN, MN), side, rostro-caudal position | open code (NEURON-python + MATLAB, VERIFIED on ModelDB page). License not stated (VERIFIED absence on ModelDB page) | 3 | Not EM: each run of the axon-growth model gives one random "connectome" from measured anatomy. Useful as a biology-grounded generative null with a known rostro-caudal axis (ascending vs descending is a built-in FF/FB label). Must run their code to get edges |
| Leech segmental ganglion, SBEM + voltage imaging (Hirudo verbana) | Ashaber, Tomina, Kassraian, Bushong, Kristan, Ellisman, Wagenaar, eLife 10:e61881 (2021), 10.7554/eLife.61881; earlier Pipkin et al., Front Neuroanat 12:95 (2018) | github.com/wagenadl/leechem (Python access to SBEM tracing and VSD data) | 531 synapses onto one motor neuron DE-3R, presynaptic partners traced back to somata (UNVERIFIED (snippet)). Ganglion has ~400 neurons (UNVERIFIED (snippet)) | D, W (synapse count) but only inputs to one cell | VSD activity per neuron during swim, crawl, local bend | open via the repo, license UNVERIFIED | 1 | Star graph around one neuron, not a ganglion connectome. Watch only, in case the full ganglion tracing is released |
| Aplysia buccal ganglion, directed-information functional graph | "Inferring neuronal network functional connectivity with directed information", J Neurophysiol (2017), 10.1152/jn.00086.2017, PMID 28468991 (authors and volume UNVERIFIED) | journals.physiology.org | tens of VSD-recorded neurons (UNVERIFIED) | D, W, signed (E/I inferred) | identified feeding-CPG neurons | matrices only shown as figures; no deposit found | 1 | No downloadable Aplysia wiring diagram found. Classic pairwise maps (Gardner 1977, Fiore 1975) are scattered across papers |
| Lamprey spinal CPG | Kozlov et al., PNAS 106:20027 (2009); Hellgren-Kotaleski, Grillner, Lansner models | pnas.org/doi/10.1073/pnas.0906722106 | model of ~10,000 neurons (UNVERIFIED (snippet)) | D, rule-based | cell classes E, C, L, M, reticulospinal | no connectivity file found | 1 | Cell-type rule models only; no released adjacency |
| Lobster / crab stomatogastric ganglion | review: J Crustacean Biol 46:ruag009 (2026) | academic.oup.com | ~30 neurons (UNVERIFIED (snippet)) | D, known from paired recordings | identified neurons | no EM connectome; review states "we are lacking a full crustacean connectome" (UNVERIFIED (snippet)) | 1 | Too small and only a textbook diagram |

No 2024-2026 EM connectome was found for Aplysia, other molluscs, leech (full ganglion), Xenopus, lamprey,
amphioxus, crustaceans, Nematostella, Trichoplax, tardigrades, Daphnia or jellyfish. Searches covered these
names directly; absence in search is not proof.

## (b) Trained ANN / model weight matrices usable as directed weighted graphs

| Name | Citation / DOI | URL | Size | W / D | Annotations | Access, license | Fit | Why |
|---|---|---|---|---|---|---|---|---|
| flyvis: connectome-constrained fly visual system model, 50 trained models | Lappalainen, Tschopp, ... Macke, Turaga, Nature 634:1132 (2024), 10.1038/s41586-024-07939-3; bioRxiv 10.1101/2023.03.11.532232 | github.com/TuragaLab/flyvis; turagalab.github.io/flyvis | 45,669 neurons, 1,513,231 connections, 65 cell types (64 + CT1 split), 721 columns; 734 free parameters; ensemble of 50 models (VERIFIED, bioRxiv full text) | D, W, signed (trained per cell-type pair strength x synapse count) | cell type, column, hex coordinates | MIT (VERIFIED for the repo). Pretrained models via the package's download command; weight license reused as MIT by third parties (UNVERIFIED for weights) | 4 | Best "trained network on a real connectome" case: same topology as the anatomy, 50 independent trainings. MFAS on trained signed weights vs on raw synapse counts tests whether training changes the FF/FB split. Caveat: weights are shared per type pair, so the effective graph is close to a 65-node type graph |
| Allen biorealistic V1 model, trained (Billeh lineage) | Allen Institute 2026 bioRxiv "Deep-learning-assisted simulation of a cortical circuit" 10.64898/2026.03.13.711751 (details UNVERIFIED, page returned 429) | github.com/AllenInstitute/biorealistic-v1-model | full ~203k neurons, core ~67k neurons; 10 prebuilt core networks core_nll_0..9; trained edge files `v1_v1_edges_bio_trained.h5` (VERIFIED, repo page). Edge count not given | D, W, signed (E/I) | 17+ cell classes by layer (UNVERIFIED), positions | SONATA h5 on Dropbox; BSD-3-Clause (VERIFIED) | 3 | Before/after training on the same model connectome, 10 instances. Model is built from connection-probability rules, not EM, so feedback there reflects the rules |
| Billeh et al. 2020 mouse V1 model (untrained) | Billeh et al., Neuron 106:388 (2020) | portal.brain-map.org/explore/models/mv1-all-layers | 230,924 cells, 51,978 in the core (UNVERIFIED (snippet)) | D, W | cell type, layer, position | SONATA via Dropbox (license UNVERIFIED) | 2 | Rule-generated cortical graph. Pair with the trained version above, or skip |
| Chen, Scherr, Maass 2022 trained V1 model support files | Science Advances 8:eabq7592 (2022), 10.1126/sciadv.abq7592 | Zenodo 10.5281/zenodo.19965055; github.com/ifgovh/Training-data-driven-V1-model | one file GLIF_network.zip, 3.4 GB (VERIFIED). Trained weights inside: not stated (VERIFIED absence on Zenodo and repo page) | D, W | as Billeh | CC BY 4.0 (VERIFIED). Repo has no license (VERIFIED) | 2 | Probably only the untrained Billeh core; trained weights may need retraining |
| RNN weight model zoo (LSTM) | Herrmann, Faccio, Schmidhuber, ICML 2024 (PMLR 235:18205), arXiv 2403.11998 | Zenodo 10.5281/zenodo.10835106; github.com/vincentherrmann/rnn-weights-representation-learning | 2 zips: formal languages 558.3 MB, sequential MNIST 1.4 GB (VERIFIED). ~1000 LSTMs per zoo, 2 layers x 32 hidden, 9 checkpoints each (UNVERIFIED (snippet)) | D, W, signed (gated, 4 weight blocks per layer) | task id, training step, performance | CC BY 4.0 (VERIFIED) | 3 | Many trained recurrent nets with checkpoints: FF share vs training step. Small (32 units) and LSTM gates make "the" weight matrix ambiguous; take |W_hh| summed over gates as the graph |
| Driscoll multitask RNNs | Driscoll, Shenoy, Sussillo, Nat Neurosci 27:1349 (2024), 10.1038/s41593-024-01668-6 | github.com/lauradriscoll/flexible_multitask; Quilt bucket aind-trained-networks, packages "multitask_shared", "multitask_processed" (VERIFIED, repo README) | units per net and number of nets UNVERIFIED (Yang 2019 lineage uses 256 units, UNVERIFIED) | D, W | task set, init seed, activation | open; license not stated (VERIFIED absence) | 3 | Vanilla RNNs trained on 15 cognitive tasks: the simplest "trained recurrent net vs brain" comparator; vanilla W_hh is one directed matrix |

Not found: a large zoo of trained vanilla RNNs (hundreds of units) with an explicit license. Herrmann's zoo is
LSTM; Driscoll's is the closest vanilla-RNN release. D already notes the option of training our own.

## (c) Drosophila functional / optogenetic connectivity

| Name | Citation / DOI | URL | Size | W / D | Access, license | Fit | Why |
|---|---|---|---|---|---|---|---|
| Central complex functional connectome (optogenetics + Ca imaging) | Franconville, Beron, Jayaraman, eLife 7:e37017 (2018) | elifesciences.org/articles/37017 | 70 presynaptic-postsynaptic cell-type pairs tested (UNVERIFIED (snippet)) | D, signed (excitatory / inhibitory), cell-type level | eLife CC BY (UNVERIFIED for data) | 2 | The only systematic fly optogenetic connectivity map found. Tiny, one region, cell-type level; could check a few FF/FB calls of our order |
| opto-HI-FISH whole-brain activation maps | Watanabe, Chiu, Anderson, eLife RP92380 (2024) | Dryad 10.5061/dryad.dr7sqvb7d; github kiichiwatanabelab/Hr38_eLife | few stimulated lines, brain-wide Hr38 snapshots (UNVERIFIED (snippet)) | D (stimulated line -> activated regions), not a matrix | Dryad (CC0 typical, UNVERIFIED) | 1 | Candidate downstream targets only, not a connectivity matrix |
| Whole-brain Ca imaging FC vs FlyWire/hemibrain SC | Okuno, Woodward, Okano, Hata, bioRxiv 10.1101/2025.07.01.662601; eLife RP107990 (2025) | DANDI 000727 (imaging data) | region level, ROI counts varied (UNVERIFIED) | undirected correlation FC | DANDI open | 1 | Not directed. Listed so it is not searched again |

Pospisil et al. 2024 (Nature 634:201) is already in E. Search found no experimental fly effectome dataset
released after it (2024-2026); that work is simulation-based. No whole-brain fly optogenetic connectivity
matrix exists as of these searches (absence in search, not proof).

## Takeaways for the pipeline

1. flyvis is the strongest new item: trained signed weights on a real connectome topology, 50 models, MIT.
   Directly answers "does training change the feedforward share" on fly optic lobe data we already understand.
2. For "trained recurrent net vs brain", Driscoll (vanilla RNN, Quilt) and Herrmann (LSTM zoo, Zenodo, CC BY
   4.0, checkpoints over training) are ready; the Allen trained V1 gives a cortical model before/after training.
3. Xenopus tadpole spinal cord is a generated connectome with a built-in ascending/descending label; usable
   as a biology-grounded null, not as a measured brain.
4. Aplysia, leech, lamprey, stomatogastric: no usable released graph.

## Sources

- https://github.com/TuragaLab/flyvis
- https://turagalab.github.io/flyvis/
- https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1.full
- https://github.com/AllenInstitute/biorealistic-v1-model
- https://portal.brain-map.org/explore/models/mv1-all-layers
- https://zenodo.org/records/19965055
- https://github.com/ifgovh/Training-data-driven-V1-model
- https://zenodo.org/records/10835106
- https://arxiv.org/abs/2403.11998
- https://github.com/vincentherrmann/rnn-weights-representation-learning
- https://github.com/lauradriscoll/flexible_multitask
- https://modeldb.science/238332
- https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2011.00020
- https://elifesciences.org/articles/33281
- https://elifesciences.org/articles/61881
- https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2018.00095/full
- https://journals.physiology.org/doi/full/10.1152/jn.00086.2017
- https://www.pnas.org/doi/10.1073/pnas.0906722106
- https://academic.oup.com/jcb/article/46/1/ruag009/8524268
- https://elifesciences.org/articles/37017
- https://elifesciences.org/articles/92380
- https://www.biorxiv.org/content/10.1101/2025.07.01.662601v2

## Not checked

- Edge counts of flyvis exported graph and the size of the pretrained download (docs pages did not show them).
- Driscoll network size and number of trained nets (Nature redirect, bioRxiv 429, Quilt renders through JavaScript).
- Allen 2026 trained V1 paper text (bioRxiv 429); Billeh edge counts.
- Xenopus model numbers on the primary pages (eLife and PMC did not render); whether the model code writes a weighted edge list.
- Leech full-ganglion tracing beyond DE-3 in the leechem repo.
- Winsor and Katz, eLife 14:e108573 (2025) commentary "Connectomes: The wiring diagram of an entire animal" (page did not render; probably about Platynereis, already in A).
- Other trained-model releases: Yang 2019 multitask (gyyang/multitask), Song/Yang/Wang 2016 Dale's-law RNNs, connectome-constrained C. elegans models, Shiu 2024 fly brain model (uses FlyWire weights unchanged, so no new graph).
- Fly larva all-optical connectivity screens.

# B. Vertebrate synapse-resolution connectomes: survey for MFAS ordering + biology

Survey date: 2026-11-17. Method: WebSearch + WebFetch. Every number has a source URL. "UNVERIFIED" means the number came from memory, a secondary source, or a truncated snippet and was not confirmed on a primary page. Counts that I could not find are written "not found", not estimated.

Fit score (1-5) means fit for our pipeline: whole-graph MFAS ordering + per-neuron annotations (type, E/I, layer, position, proofreading, function) + random-graph nulls + layered plot.

## Summary table

| # | Dataset | Species / tissue | Cells / neurons | Synapses | Proofread | Key annotations | Access | Edge list effort | Fit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | MICrONS minnie65, current CAVE (v1300 / v1507 / v1621 / v1718 / v1822 / v1926) | mouse V1 + AL/RL/LM, P87, ~1 mm3 | 144,120 nuclei detected; 94,014 rows in metamodel cell types (63,821 exc, 7,866 inh, 18,775 non-neuron) | 337,312,429 rows in `synapses_pni_2` | 2,182 cells in `proofreading_status_and_strategy`; ~1,800-2,300 with usable axons | E/I, m-types, area, column manual types, 19,181 manual 2P coregistrations, digital-twin tuning | CAVE token (free) or static GCS dumps; CC license not stated | low (static CSV.GZ 20 GB, or Zenodo h5) | 5 |
| 1b | MICrONS v1412 graph on Zenodo (Reimann) | same | "most neurons" of minnie65 minus ~20,000 peripheral | per-pair counts | `proofreading_status_and_strategy` field | E/I, m-type, xyz, area | Zenodo, CC BY 4.0, 864 MB, no token | minimal | 5 |
| 1c | MICrONS v117 (what we use) | same | - | 47.5 GB `synapses_pni_2.csv` (BossDB) | `proofreading_status_public_release.csv` (56 KB) | few | static | done | (baseline) |
| 2 | MICrONS minnie35 | mouse visual cortex, other part of same block | not found | 14.5 GB `assigned.csv.gz` | "proofread v0" seg only | none found | static BossDB | medium | 2 |
| 3 | MICrONS Phase 1 L2/3 ("pinky100") | mouse V1 L2/3, ~3e6 um3 | UNVERIFIED (hundreds of PyCs) | UNVERIFIED | proofread PyC/inh. | Ca imaging >100 PyCs | layer23.microns-explorer.org | medium | 2 |
| 4 | V1DD (Allen V1 Deep Dive) | mouse V1, 800x800x800 um, all layers | 207,455 nuclei; 88,030 predicted neurons | not found (165,533,529 rows in `synapse_target_predictions_ssa`) | 852 fully proofread (CAVE v742, per one paper) | cell types (`cell_type_multifeature_v1`), 2P/3P functional, 571 manual coregistrations | CAVE `v1dd_public` on separate Allen server, token; S3 analysis asset | medium | 4 |
| 5 | H01 human temporal cortex | human MTG L1-6 + WM, ~1 mm3 | 57,216 cells; ~16,087 neurons (secondary) | 149,871,669 (paper; 133.7M "connections" in preprint abstract) | 104 cells | E/I per synapse; layers as volumes; `cells` table with `cell_type` | GCS static (Avro export), CAVE `h01_c3_flat` (token by email) | medium-high | 3 |
| 6 | Fish1 (Petkova, Januszewski et al.) | larval zebrafish 7 dpf, whole brain + ant. spinal cord | 187,053 somas; >40,000 molecular (26,915 vglut2a+, 14,510 gad1b+) | ~30 million | "over 42,000 edited segments"; no proofread-neuron count | E/I (vglut2a/gad1b), per-synapse e/i, positions, LM atlas | CAVE `fish1_full` on brain-wire-test.org, token; no static synapse dump found | medium | 4 |
| 7 | Fish-X (Li ... Du 2025) | larval zebrafish 6 dpf, whole brain + SC + retina | 176,810 cells (brain+SC) | 25.25 M PSDs | 523 manually reconstructed neurons | 5 neuromodulator types (APEX2), Gly | not found (data statement not read) | high/unknown | 2 |
| 8 | Zebrafish hindbrain (Vishwanathan 2017, 2024; Yang 2023) | larval zebrafish brainstem, 0.0024 mm3 (secondary) | UNVERIFIED (~3,000) | not found | proofread (Eyewire) | cell classes (ABD, integrator, vestibular...) | seunglab.org/zebrafish | low-medium | 3 (small, but prior cycle work) |
| 9 | Boulanger-Weill 2025 hindbrain CLEM | larval zebrafish anterior hindbrain | 1,246 reconstructed neurons | not found | manual | functional classes from Ca imaging | preprint, access not found | unknown | 2 |
| 10 | Svara 2022 whole-brain SBEM | larval zebrafish 5 dpf, no retina | not found | not found | automated (SyConn2) | Mauthner circuit, some 2P | probably syconn.esc.mpcdf.mpg.de, not verified | unknown | 2 |
| 11 | Wanner & Friedrich 2016/2020 olfactory bulb | larval zebrafish OB | ">1,000" neurons, ~98% reconstructed (UNVERIFIED wording) | not found | dense manual | odor responses (2020) | BossDB/neurodata, Zenodo 10.5281/zenodo.58985 (2016) | low-medium | 3 |
| 12 | Hildebrand 2017 | larval zebrafish 5.5 dpf whole brain | myelinated projectome only | none | - | - | BossDB, CC BY 4.0 | n/a | 1 |
| 13 | Songbird Area X j0251 (Rother ... Kornfeld 2025) | zebra finch basal ganglia, 256x256x384 um | >8,500 neurons with soma; neuron table 4,261,298 rows (fragments) | 18,361,224 synapses in static table | "minimal manual proofreading" | 18-20 cell types (v8), synapse area, post-morphology (spine/shaft/soma) | static `.npy` on syconn.esc.mpcdf.mpg.de, no token | low | 4 |
| 14 | Songbird HVC j0256 (Kornfeld 2017) | zebra finch HVC | small, skeletons | 44 HVC(RA)-HVC(RA) synapses reported | manual skeletons | HVC(RA)/interneuron | GitHub + webKnossos | high | 1 |
| 15 | Motta 2019 L4 | mouse S1 L4, ~500,000 um3, P28 | 89 neurons with soma | UNVERIFIED (~153k synapses, ~6,979 axons x 3,719 dendrites) | semi-auto | axon/dendrite types | l4dense2019.brain.mpg.de CSV (host was down on fetch) | low (CSV) but axon-to-dendrite, not neuron-to-neuron | 2 |
| 16 | Sievers et al. 2024 cortical column | mouse barrel cortex column | ~9,500 neurons (snippet) | ~92 million (snippet) | automated (RoboEM) | cell types | preprint; data access not found | unknown | 3 if released |
| 17 | Loomba 2022 mouse/macaque/human | L2/3 cortex volumes | not found | not found | - | E/I axon types | Edmond 1.3 GB zip (analysis data) | high | 1 |
| 18 | Kasthuri 2015 | mouse S1, ~1,500 um3 | fragments | 1,700 | saturated | - | NeuroData/BossDB | n/a | 1 |
| 19 | Lee 2016 | mouse V1 L2/3 | 201 neurons in "network core" (snippet) | not found | manual | 2P tuning | neurodata.io/data/lee16 | medium | 1 |
| 20 | Helmstaedter 2013 e2006 | mouse retina IPL | 950 neurons | contacts, not synapses | dense | cell types | MPI | medium | 2 |
| 21 | Bae 2018 Eyewire museum e2198 | mouse retina | ~400 RGCs | not a connectome | - | Ca responses for 82% | museum.eyewire.org | n/a | 1 |
| 22 | Eyewire II (2026) | mouse retina, ~1 mm2 | ~100,000 neurons, >37,000 reconstructed | synapses annotated only in a 14x20 um sub-volume | ~37% of cells | types, ~400 Ca-imaged cells | sign-up (CAVE), Hugging Face skeletons | high (no synapse table yet) | 1 now |
| 23 | Nguyen 2023 cerebellum | mouse lobule V | GrC/PC subsets | 9,012 MF->GrC + 19,761 GrC->PC analysed | partial | cell classes | not checked | medium | 1 (bipartite, near-acyclic by anatomy) |
| 24 | MouseConnects / HI-MC | mouse hippocampal formation, 10 mm3 | in progress (2023-2028) | - | - | - | none yet | - | 0 (not available) |
| 25 | ZAPBench fish ("Fish2") EM | larval zebrafish | EM connectome "upcoming" as of Feb 2026 | - | - | 2 h whole-brain activity of >70,000 neurons | not released | - | watch list |

## Per-dataset notes

### 1. MICrONS minnie65 (mouse visual cortex, 1 mm3) - top pick

- Citation: MICrONS Consortium et al. (2025) "Functional connectomics spanning multiple areas of mouse visual cortex", Nature 640:435-447, https://doi.org/10.1038/s41586-025-08790-w. Companion Nature 2025 papers: Schneider-Mizell et al. https://doi.org/10.1038/s41586-024-07780-8 (inhibitory census, the "Minnie column"); Elabbady et al. https://doi.org/10.1038/s41586-024-07765-7 (cell types); Celii et al. https://doi.org/10.1038/s41586-025-08660-5; Wang et al. https://doi.org/10.1038/s41586-025-08829-y; Ding et al. https://doi.org/10.1038/s41586-025-08840-3 (functional wiring rule). CAVE: Dorkenwald et al. 2025 Nature Methods.
- Tissue: P87 mouse, V1 + AL, RL, LM higher visual areas, ~1 mm x 0.5 mm depth (https://allenswdb.github.io/anatomy/microns-em/em-background.html).
- Size (https://www.microns-explorer.org/cortical-mm3): ">523 million" synapses detected automatically; ~200,000 cells, ~120,000 neurons (estimated); 75,902 neurons functionally recorded.
- CAVE tables (https://tutorial.microns-explorer.org/annotation-tables.html): `synapses_pni_2` 337,312,429 rows; `nucleus_detection_v0` 144,120; `aibs_metamodel_celltypes_v661` 94,014; `aibs_metamodel_mtypes_v661_v2` 72,158 (layer-prefixed m-types, i.e. layer comes from here); `allen_v1_column_types_slanted_ref` 1,357 (manual); `baylor_log_reg_cell_type_coarse_v1` 55,063 (E/I); `baylor_gnn_cell_type_fine_model_v2` 49,051; `nucleus_functional_area_assignment` (V1/AL/RL/LM); `proofreading_status_and_strategy` 2,182; `coregistration_manual_v4` 19,181 (15,439 unique EM neurons per explorer page); `coregistration_auto_phase3_fwd_apl_vess_combined_v2` 83,046; `digital_twin_properties_bcm_coreg_v4` (OSI/DSI etc.); `synapse_target_predictions_ssa` 204,331,842 (spine/shaft/soma).
- Automated cell class counts (tutorial notebook at v1926, https://tutorial.microns-explorer.org/quickstart_notebooks/04-cave-query-synapses.html): excitatory_neuron 63,821; inhibitory_neuron 7,866; nonneuron 18,775.
- Proofreading: axon strategy counts at the current version: axon_partially_extended 1,382, axon_fully_extended 802, axon_interareal 120 (from notebook output; overlapping tutorial texts say "~2000" and "~1800" usable axons). Synapses among proofread neurons: 389,520 (same notebook, `remove_autapses=True`). The consortium paper says proofread outputs carry ">900,000 synaptic connections" (search snippet, not opened). Dendrites onto single-soma cells are considered trustworthy without proofreading; unproofread axons are "not analyzable" (https://tutorial.microns-explorer.org/proofreading.html).
- Versions (https://tutorial.microns-explorer.org/materialization-version.html): 117 (2021-06-11, long-term), 343, 661, 795, 943 (long-term), 1078, 1181, 1300 (2025-01-13, long-term, flat segmentation `seg_m1300` and v1300 SWC skeletons), 1412 (2025-04-29, expired in CAVE, static dump exists), 1507, 1621, 1718, 1822, 1926 (2026-09-28, current). Note: the explorer front page still says "8 public versions ... latest 1300"; the tutorial is newer.
- What changed vs v117 (our current graph): (a) v117 static synapse CSV is the original segmentation; FAQ says "This connectivity table represents the original v117 segmentation" and proofreading has since improved it; (b) proofreading table grew from the 56 KB `proofreading_status_public_release.csv` (v117, deprecated) to 2,182 cells with strategy labels; (c) cell-type tables from v661 on (metamodel, m-types, GNN); (d) manual coregistration v4 (19,181) and digital-twin functional properties; (e) spine/shaft/soma target labels; (f) Gamlin 2025 Martinotti tables (`cg_cell_type_calls`) from v1300. Root IDs change between versions, so annotations must be joined by nucleus id, not root id.
- Static access, no token (https://tutorial.microns-explorer.org/static-repositories.html): v117 synapse graph `https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/synapse_graph/synapses_pni_2.csv` (47.5 GB); archived per-version tables under `gs://mat_dbs/public/minnie65_phase3_v1/v{343,661,795,943,1078,1181,1412}/` incl. `synapses_pni_2_v1_filtered_view.csv.gz` (20 GB) with header file, plus cell type tables. Live versions (1300+) need a free CAVE token.
- License: no license stated on explorer or tutorial pages; there is a Terms and Conditions + Citation Policy link; FAQ says "public and open access" and asks to cite and link microns-explorer.org. Mark license as UNVERIFIED (not CC-stated).
- Prior ordering / hierarchy work: Reimann, Egas Santander et al. directed-simplex analyses on v117, v1181 and v1412 (bioRxiv 10.1101/2023.12.22.573036; PMC13253143; Zenodo 13849415 for v1181). Directed simplices are all-to-all feedforward cliques, so this is the closest prior art to "feedforwardness" in MICrONS. Ding et al. 2025 study like-to-like rule across feedforward (V1 to HVA) and feedback (HVA to V1) connections. No MFAS / global ordering on MICrONS found (search for MFAS on MICrONS/H01/zebrafish returned only FlyWire work, e.g. arXiv 2506.13799, which is FlyWire-only).
- Fit 5: same dataset family as our v117 graph, so upgrading to v1412 (Zenodo) or v1926 gives cleaner edges plus E/I, m-type/layer, area, proofreading, function, all keyed by nucleus id.

### 1b. MICrONS connectome as a ready graph on Zenodo (best shortcut)

- "Connectomics of (part of) the MICrONS mm3 dataset", Zenodo, https://doi.org/10.5281/zenodo.16744240, v3.0.0, 2025-08-05, CC BY 4.0. Author shown on the fetched page: M. W. Reimann (EPFL); a search snippet instead named B. Danskin with F. Collman and C. Schneider-Mizell, so authorship is UNVERIFIED.
- Content (fetched page): internal connectivity among most neurons of minnie65, based on MICrONS v1412; ~20,000 peripheral neurons excluded to avoid edge effects; neuron xyz in um and nm; distance-dependent control model included. Files `microns_mm3_connectome_v1412.h5` (761.0 MB) and `microns_mm3_v1412_distance_dependent_control.h5` (103.1 MB). Format for Connectome-utilities (`conntility`). Snippet adds neuron table fields: nucleus id, coarse E/I `classification_system`, `cell_type` (m-type), `proofreading_status_and_strategy`. Earlier versions: 1.0.0 = v117, 2.0.0 = v1181.
- Neuron / edge counts: not found on page (must load the file).
- Fit 5: one 761 MB download, no token, comes with E/I, m-type, position, proofreading, and a ready distance-dependent null model, which fits our random-graph comparison directly.

### 2. MICrONS minnie35

- Second part of the same EM block. Static only: EM 55 TB, autoseg and "proofread v0" seg (10 TB each), clefts 94 GB, synapse table `minnie35/synapse_graph/assigned.csv.gz` (14.5 GB) (static-repositories page). No CAVE tables, cell types or proofreading tables found for minnie35. Counts not found.
- Fit 2: edges obtainable, but almost no annotations.

### 3. MICrONS Phase 1 L2/3 ("pinky100")

- Turner, Macrina, Bae et al. (2022) "Reconstruction of neocortex: Organelles, compartments, cells, circuits, and activity", Cell 185(6):1082-1100, https://doi.org/10.1016/j.cell.2022.01.023. Companion: Dorkenwald et al. 2022 (binary and analog synapse variation).
- ~3 million um3 L2/3 V1; calcium imaging of >100 PyCs; analysed subgraphs of 111 and 100 neurons (snippets). Data at layer23.microns-explorer.org. Total neuron and synapse counts not found (UNVERIFIED).
- Fit 2: small, superseded by minnie65.

### 4. V1DD (Allen Brain Observatory V1 Deep Dive)

- No resource paper found. Allen SWDB Data Book: collected 2020-2022, "publicly available since August 2025"; 800x800 um region, all layers (https://allenswdb.github.io/anatomy/microns-em/em-background.html).
- CAVE tables (SWDB key-tables page, labeled 2024 Data Book, may be dated): `nucleus_detection_v0` 207,455; `neurons_soma_model` 88,030; `synapse_target_predictions_ssa` 165,533,529; `functional_coregistration_manual_2` 571. Recommended cell-type table `cell_type_multifeature_v1`.
- Paper using V1DD (PMC13722110, "Functionally identified neuronal assemblies robustly encode stimuli and are structurally delineated by inhibition"; bioRxiv 2025.04.24.649900): CAVE version 742, 207,455 nuclei, 8,812 typed neurons (7,864 pyramidal, 948 inhibitory), 852 fully proofread cells, 400 verified 2P-EM matches.
- Access: `CAVEclient("v1dd_public", server_address="https://global.em.brain.allentech.org")`, separate token; no static meshes; analysis asset `s3://aind-open-data/v1dd-analysis-1196-1_2025-08-14_16-38-00/` (no-sign-request) (https://github.com/AllenSWDB/SWDB_2026_Connectomics). Physiology: https://github.com/AllenInstitute/v1dd_physiology.
- License not found. Total synapse count not found.
- Fit 4: second mouse V1 cubic-mm volume with all layers and function, good replication of MICrONS results; access only through CAVE (token) plus an analysis asset whose contents I did not open.

### 5. H01 human temporal cortex

- Shapson-Coe, Januszewski, Berger et al. (2024) "A petavoxel fragment of human cerebral cortex reconstructed at nanoscale resolution", Science 384:eadk4858, https://doi.org/10.1126/science.adk4858 (403 on fetch; numbers below from preprint abstract and secondary pages). Preprint bioRxiv 10.1101/2021.05.29.446289.
- 45-year-old female, left anterior middle temporal gyrus, epilepsy surgery; ~1 mm3, 1.4 PB (search snippets).
- Preprint abstract (fetched): 57,216 cells, 133.7 million synaptic connections, 104 manually proofread cells, glia outnumber neurons 2:1. Published-paper synapse count via snippet: 149,871,669 (111,272,315 excitatory, 38,599,354 inhibitory). Neuron count 16,087 from a search snippet (UNVERIFIED on primary page).
- Data (https://h01-release.storage.googleapis.com/data.html): c3 segmentation recommended for connectomics; c3 synapses with E/I type; "Synaptic connections database" export in Apache Avro at `gs://h01-release/data/20210729/c3/synapses/exported/`; cortical layer volumes; 50k soma locations; 104 proofread cells (SWC). License CC BY 4.0 (stated on data page).
- CAVE: datastack `h01_c3_flat` on `https://global.brain-wire-test.org`, account via h01-proofreading@g.harvard.edu; versions seen [1172, 1187, 1189, 1190]; tables `cells` (with `cell_type`), `nucleus`, `synapses`, `proofreading_status_test` (navis tutorial).
- Proofread fraction "<1%" (ConnectomeBench2 table, arXiv 2606.21116, snippet).
- Fit 3: only human cellular connectome; edges available from Avro, E/I from synapse labels and layers from volumes, but with 104 proofread cells most neuron-neuron edges come from unproofread axons, so the graph is noisy and sparse between somas.

### 6. Fish1 - larval zebrafish whole brain (Petkova, Januszewski ... Engert 2025)

- "A connectomic resource for neural cataloguing and circuit dissection of the larval zebrafish brain", bioRxiv 10.1101/2025.06.10.658982 (June 2025). No journal version found as of this search.
- 7 dpf larva, CLEM: confocal vglut2a/gad1b + ssEM 4x4x30 nm. 187,053 cell bodies (brain, spinal cord, ganglia); 26,915 vglut2a+ and 14,510 gad1b+ (https://fish1-release.storage.googleapis.com/index.html); ~30 million synapses, mostly labeled exc/inh/modulatory (abstract).
- Access (https://fish1-release.storage.googleapis.com/programmatic.html): CAVE datastack `fish1_full` on `https://global.brain-wire-test.org/`, token required; tables `somas` (cell type, root id, position), `synapses_axde_label` (pre/post root + e/i), `synapses_axon_to_dendrite_size`, `synapses_axax`, `nucleus_table` ... ; "over 42,000 edited segments". Static: VAST image/segmentation files and two analysis zips; no static synapse table found.
- Data policy: free to publish independently, cite Petkova et al. 2025; no named license.
- Fit 4: first vertebrate whole-brain cellular graph with molecular E/I and positions; but proofreading is partial and edges must be pulled through CAVE.

### 7. Fish-X (Li, Liu, ... Du 2025)

- "Multiplexed neuromodulatory-type-annotated whole-brain EM-reconstruction of larval zebrafish", bioRxiv 10.1101/2025.06.12.659365 (v3).
- 6 dpf, ATUM-SEM 4x4x33 nm, 22,887 sections; 176,810 cells in brain + anterior SC; 25.25 M PSDs; 523 manually reconstructed neurons; 4,328 APEX2-labeled neurons (NE, DA, 5-HT, Hcrt, Gly). Authors say full proofreading is still needed (secondary summary).
- Data availability statement: not found in the parts I could read.
- Fit 2: neuromodulatory labels are unique, but no neuron-neuron graph is available yet.

### 8. Zebrafish hindbrain / oculomotor integrator (Seung lab)

- Vishwanathan et al. 2017 Curr Biol (UNVERIFIED citation details); Vishwanathan, Sood, Wu et al. (2024) "Predicting modular functions and neural coding of behavior from a synaptic wiring diagram", Nat Neurosci 27:2443-2454, https://doi.org/10.1038/s41593-024-01784-3; Yang, Vishwanathan ... Seung (2023) "Cyclic structure with cellular precision in a vertebrate sensorimotor neural circuit", Curr Biol 33(11):2340-2349, https://doi.org/10.1016/j.cub.2023.05.010.
- Data: https://seunglab.org/zebrafish (proofread neurons and synaptic contacts; ABD, integrator, vestibular, reticulospinal types). Neuron/synapse counts not found (~3,000 neurons is UNVERIFIED). Volume 0.0024 mm3 (secondary).
- Prior work highly relevant to us: Yang 2023 argues against a feedforward picture: 3-cycles are strongly over-represented in the oculomotor module; Vishwanathan 2024 finds two 3-block cycles (positive feedback for the integrator).
- Fit 3: small, but a published vertebrate claim of strong cyclic (anti-feedforward) structure; a good contrast case.

### 9. Boulanger-Weill et al. 2025 (hindbrain CLEM)

- bioRxiv 10.1101/2025.03.14.643363 (v3 title "Correlative light and electron microscopy identifies circuit motifs underlying processing of noisy visual motion in larval zebrafish"). 1,246 neurons reconstructed in and connected to the anterior hindbrain; functional classes; recurrent excitation and inhibitory motifs. Data access not found. Fit 2.

### 10. Svara et al. 2022 whole-brain SBEM

- Svara, Förster, Kubo ... Baier (2022) Nat Methods 19:1357-1366, https://doi.org/10.1038/s41592-022-01621-0. 5 dpf, SBEM 14x14x25 nm, ~29,000 sections (secondary), SyConn2 synapses, Mauthner connectivity. Counts and data URL not found (SyConn2 paper says its datasets are at syconn.esc.mpcdf.mpg.de, CC BY; the data page there lists only j0251 downloads, so zebrafish availability is UNVERIFIED). Fit 2.

### 11. Zebrafish olfactory bulb (Wanner, Friedrich)

- Wanner et al. 2016 (Sci Data, "3-dimensional electron microscopic imaging of the zebrafish olfactory bulb and dense reconstruction of neurons", 1,022 neurons per secondary snippet; Zenodo 10.5281/zenodo.58985); Wanner & Friedrich (2020) Nat Neurosci 23:433-442, https://doi.org/10.1038/s41593-019-0576-z (complete wiring diagram + odor responses). Synapse counts not found. Fit 3: dense, almost complete small circuit with function; check whether the 2020 matrix is public.

### 12. Hildebrand et al. 2017

- Nature 545:345-349, https://doi.org/10.1038/nature22356. Whole-brain ssEM, myelinated-axon projectome, no synapses. BossDB https://bossdb.org/project/hildebrand2017 (CC BY 4.0) and neurodata. Fit 1.

### 13. Songbird basal ganglia Area X (j0251) - strong static option

- Rother, Januszewski, Jain, Fee, Kornfeld (2025) "The songbird basal ganglia connectome", bioRxiv 10.1101/2025.10.25.684569 (PMC12636473). Follow-up: Shvedov ... Kornfeld, Scott (2026) Curr Biol 36(9):2242-2254, https://doi.org/10.1016/j.cub.2026.03.057.
- Adult male zebra finch, SBEM 10x10x25 nm, 256x256x384 um; >8,500 automated neuron reconstructions, ~20 million synapses, minimal manual proofreading (https://www.songbird-connectomics.org).
- Static download, no token (https://syconn.esc.mpcdf.mpg.de/data.html): `j0251_synapses.npy` 613 MB, 18,361,224 synapses (v8 cell types); `j0251_neurons.npy` 94 MB, 4,261,298 rows (includes fragments) with cell type, certainty, skeleton length; `celltype_mapping.txt` (20 entries); post-morphology mapping (spine neck/head, shaft, soma); REST API guide. License not stated.
- Fit 4: easy, compact, with cell types and synapse sizes; a non-cortical vertebrate circuit (basal ganglia loop) with an expected feedforward direction (HVC/LMAN inputs -> MSN -> GP), nice test of ordering biology. Need to filter to soma-bearing cells.

### 14. Songbird HVC (Kornfeld et al. 2017)

- eLife 6:e24364, https://doi.org/10.7554/eLife.24364; data https://github.com/jmrk84/HVC_paper and webKnossos j0256. Skeleton-level, small. Fit 1.

### 15. Motta et al. 2019, mouse S1 L4

- Science 366:eaay3134, https://doi.org/10.1126/science.aay3134. ~500,000 um3, P28, 89 neurons, 2.7 m neurite; connectome of Fig. 3 as CSV (axon-target pair synapse counts) at https://l4dense2019.brain.mpg.de/ (host unreachable when fetched). Synapse/axon counts UNVERIFIED. Most axons have no soma in volume, so a neuron-neuron graph is tiny. Fit 2.

### 16. Sievers et al. 2024, cortical column

- bioRxiv 10.1101/2024.03.22.586254, Helmstaedter lab. Snippets: ~9,500 neurons and ~92 million synapses in a mouse barrel-cortex column. No journal version or data release found. Fit 3 if data appear (dense, column-complete, automated proofreading).

### 17-21. Older small mouse datasets

- Loomba et al. 2022 Science 377:eabo0924, https://doi.org/10.1126/science.abo0924; Edmond dataset https://doi.org/10.17617/3.RW12N6 (thesis_data.zip, 1.3 GB). Human/macaque/mouse L2/3 axon statistics. Fit 1.
- Kasthuri et al. 2015 Cell 162:648-661, https://doi.org/10.1016/j.cell.2015.06.054; 1,700 synapses. Fit 1.
- Lee et al. 2016 Nature 532:370-374, https://doi.org/10.1038/nature17192; https://neurodata.io/data/lee16/. Fit 1.
- Helmstaedter et al. 2013 Nature (mouse retina IPL e2006): 950 neurons, contacts not synapses (a later analysis used 236,020 putative synapses = 41% of contacts, PMC4944929). Fit 2.
- Bae et al. 2018 Cell 173:1293-1306, https://doi.org/10.1016/j.cell.2018.04.040; ~400 RGCs, museum.eyewire.org. Not a connectome. Fit 1.

### 22. Eyewire II (mouse retina, 2026)

- Eyewire II consortium (Ströh et al.), bioRxiv 10.64898/2026.05.28.727403 (PMC13252113). ~1 mm2 adult retina, 8x8x40 nm. >8,000 BCs, >13,000 ACs, >4,000 RGCs proofread; ~37% of cells by Sept 2026; synapses manually annotated only in a 14x20 um sub-volume; ~400 cells with Ca imaging. Access by sign-up (CAVE); skeletons on Hugging Face `eulerlab/eyewire2-data`. Fit 1 now (no network), watch list.

### 23. Mouse cerebellum (Nguyen et al. 2023)

- Nature 613:543-549, https://doi.org/10.1038/s41586-022-05471-w. 9,012 MF->GrC and 19,761 GrC->PC synapses analysed in lobule V. Feedforward by anatomy, bipartite-ish. Fit 1.

### 24-25. Not yet available

- MouseConnects / HI-MC (Google + Harvard + others, BRAIN CONNECTS): 10 mm3 mouse hippocampal formation, 2023-2028, no data release found.
- ZAPBench fish (Fish2) EM connectome: ZAPBench (Lueckmann et al., ICLR 2025, arXiv 2503.02618) pairs >70,000-neuron whole-brain activity with an "upcoming" EM connectome of the same fish (still upcoming in a Feb 2026 arXiv paper). If released, it would be the only vertebrate whole-brain connectome with whole-brain activity of the same animal.

### Not found

- No synapse-resolution connectome released for rat, macaque or marmoset beyond the Loomba 2022 L2/3 volumes. Primate results returned only tract-tracing / MRI connectomes.
- No second large human EM connectome beyond H01.
- No application of MFAS / global feedforward ordering to any vertebrate cellular connectome. Closest prior art: directed-simplex analyses on MICrONS (Reimann group) and 3-cycle over-representation in the zebrafish oculomotor module (Yang 2023).

## Sources (main)

- https://tutorial.microns-explorer.org/materialization-version.html
- https://tutorial.microns-explorer.org/static-repositories.html
- https://tutorial.microns-explorer.org/annotation-tables.html
- https://tutorial.microns-explorer.org/proofreading.html
- https://tutorial.microns-explorer.org/quickstart_notebooks/04-cave-query-synapses.html
- https://tutorial.microns-explorer.org/release_manifests/version-1300.html
- https://tutorial.microns-explorer.org/faq.html
- https://www.microns-explorer.org/cortical-mm3
- https://zenodo.org/records/16744240
- https://www.biorxiv.org/content/10.1101/2023.12.22.573036v3.full
- https://allenswdb.github.io/anatomy/microns-em/em-background.html
- https://allenswdb.github.io/anatomy/microns-em/key-tables.html
- https://github.com/AllenSWDB/SWDB_2026_Connectomics
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13722110
- https://h01-release.storage.googleapis.com/landing.html
- https://h01-release.storage.googleapis.com/data.html
- https://www.biorxiv.org/content/10.1101/2021.05.29.446289v4
- https://navis-org.github.io/navis/stable/generated/gallery/4_remote/tutorial_remote_04_h01/
- https://github.com/VCG/cave-scripts
- https://fish1-release.storage.googleapis.com/index.html
- https://fish1-release.storage.googleapis.com/programmatic.html
- https://fish1-release.storage.googleapis.com/data_policy.html
- https://www.biorxiv.org/content/10.1101/2025.06.10.658982v1
- https://www.biorxiv.org/content/10.1101/2025.06.12.659365v3.full
- https://seunglab.org/zebrafish/home/
- https://pubmed.ncbi.nlm.nih.gov/37236180/
- https://www.biorxiv.org/content/10.1101/2025.03.14.643363v3
- https://www.songbird-connectomics.org
- https://syconn.esc.mpcdf.mpg.de/data.html
- https://www.biorxiv.org/content/10.1101/2025.10.25.684569v1
- https://bossdb.org/project/hildebrand2017
- https://neurodata.io/data/hildebrand17/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13252113/
- https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.RW12N6
- https://research.google/blog/improving-brain-models-with-zapbench/

# C. Directed, weighted mesoscale / region-level connectomes (survey, 2026-11-17)

Scope: directed region-level graphs from tract tracing (or single-neuron reconstructions) usable for weighted
MFAS ordering and real-vs-random comparison. Numbers come from the cited page or from my own count of a downloaded
file (marked "counted"). Anything I could not confirm is marked UNVERIFIED. Fit score: 1 (useless) to 5 (ideal) for
our pipeline: directed + weighted + square (edge-complete) + FF/FB ground truth is the ideal.

## Summary table

| # | Dataset | Species | Nodes | Directed edges | Weight | Complete? | FF/FB ground truth | Access | Fit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Markov et al. 2014 (Cereb Cortex) 91x29 | macaque | 91 sources x 29 injected; 29x29 edge-complete | 1,615 pathways (91x29); 536 in 29x29 | FLNe (fraction of labeled neurons, sums to 1 per target) | 29x29 complete | SLN for 11 visual targets (628 edges, Markov 2014 JCN) | core-nets.org (register); Netzschleuder `macaque_markov` (flne / sln / distances) | 5 |
| 2 | Froudist-Walsh et al. 2021 (Neuron) 40-area release | macaque | 91 x 40 injected; 40x40 edge-complete | UNVERIFIED (density 0.62 per 2024 review) | FLNe | 40x40 complete | SLN (89 source areas; SUB, PIR excluded) | core-nets.org, Download, "Froudist-Walsh et al." | 5 |
| 3 | Majka et al. 2020 (Nat Commun) / Theodoni 2021 | marmoset | 116 areas x 55 injected; 55x55 edge-complete | 55x55: 1,861 (one source) ; density 62.43% | FLNe (log10 mean over injections) | 55x55 complete | SLN per connection; hierarchy from beta regression of SLN | marmosetbrain.org "Connectivity Matrix" tab; CC-BY-SA 4.0 (paper also mentions GPL for software) | 5 |
| 4 | Harris et al. 2019 (Nature) hierarchy | mouse | 43 isocortical targets / 40 sources (counted); thalamus 24-29 nuclei | CC ipsi: 1,180 unique src->tgt pairs (counted); TC: 400 pairs (counted) | layer densities per Cre line; projection volume via Allen API | edge list, not complete matrix | YES: every connection (per Cre line) labeled FF/FB via 9 laminar clusters -> `clustermapping.xlsx`; global hierarchy score | github.com/AllenInstitute/MouseBrainHierarchy (Allen terms of use) | 5 (labels) |
| 5 | Oh et al. 2014 (Nature) linear model | mouse | 213 per hemisphere (426 nodes) | 32,822 (16,954 ipsi) | fitted normalized connection strength + p-value + distance | full model, p-value filter needed | no | Netzschleuder `mouse_mesoscale/model`; Nature Suppl. Table 3 | 4 |
| 6 | Knox et al. 2019 (Netw Neurosci) voxel model, regionalized | mouse | voxel 100 um, regionalized to Allen CCF regions (e.g. 291/577 UNVERIFIED) | dense (all pairs nonzero) | normalized connection density / strength | complete, needs threshold | no | github.com/AllenInstitute/mouse_connectivity_models (`VoxelModelCache`) | 4 |
| 7 | Gamanut et al. 2018 (Neuron) | mouse | 47 areas x 19 injected; 19x19 complete | 19x19 at ~97% density | FLNe (Suppl. Table 6) | 19x19 complete | SLN UNVERIFIED for this paper | paper supplement; maybe core-nets.org (UNVERIFIED) | 3 |
| 8 | D'Souza et al. 2022 (Nat Commun) | mouse visual cortex | ~10 visual areas (UNVERIFIED) | UNVERIFIED | laminar retrograde fractions | small, complete-ish | YES: FF/FB hierarchical distance via beta regression | Nature source data (UNVERIFIED) | 3 |
| 9 | SEU-net, Xiong et al. 2025 (Nat Methods) | mouse | 148 (regions incl. cortical layers); 90 without layers | 1,365 nonzero incl. 66 self-loops (counted, with-layers) | "connection strength" from single-neuron axon/bouton, range 0.18-14.9 (counted) | square 148x148 | no | Zenodo 10.5281/zenodo.10677899 (latest 17182254), `SEU_net_w_corticalLayers.xlsx`, CC-BY 4.0 | 4 |
| 10 | Cell-type-specific Allen model, 2023 (Netw Neurosci 7(4):1497) | mouse | whole brain, per Cre line | UNVERIFIED | projection model tensor | model w/ missing values | no | paper (bioRxiv 2023.05.02.539079) | 3 |
| 11 | BRICseq, Huang et al. 2020 (Cell) | mouse cortex | UNVERIFIED (~40 areas) | UNVERIFIED | barcode counts | single animal | hierarchy-related claims UNVERIFIED | Cell supplement | 3 |
| 12 | Zingg et al. 2014 (Cell), Mouse Connectome Project | mouse cortex | UNVERIFIED | 240 intracortical connections reconstructed (binary?) | UNVERIFIED | - | no | mouseconnectome.org / UCLA (matrix availability UNVERIFIED) | 2 |
| 13 | BAMS rat (2011/12/13 releases) | rat | 503 / 503 / 496 | 27,667 / 30,088 / 33,020 | ordinal 1..9 (~80% are 1) | literature collation | no | Netzschleuder `rat_brain_bams` | 4 |
| 14 | Swanson-Hahn-Sporns series (2016-2026, PNAS) | rat | cortex 2017: 154 (77/hemi); 2024 rostral: ~790 regions / ~77,000 conn. (UNVERIFIED garbled excerpt); whole CNS: 924 regions / 81,582 conn. | see left | ordinal, log10-scaled qualitative | literature collation, bilateral | no | PNAS Datasets S1-S3; Google Sheets on the Neurome Project site | 4 |
| 15 | Scannell et al. 1995 / 1999 | cat | 65 (1995); 95 = 52 ctx + 43 thal (1999); cortex-only 52 | 1,139 ; 2,126 ; 818 | ordinal 1-3 | literature collation | no | Netzschleuder `cat_cortex` (unweighted), `cat_corticothalamic` (weighted); BCT | 3 |
| 16 | Felleman & Van Essen 1991 | macaque visual | 32 | 315 (Netzschleuder) | binary (original has FF/FB/lateral classes) | literature | YES (original: ascending/descending/lateral pathway labels; on Netzschleuder UNVERIFIED) | Netzschleuder `macaque_visual_cortex` | 4 (validation) |
| 17 | CoCoMac derivatives | macaque | Modha & Singh 2010: 383; Netzschleuder: 242; Young 1993: 71; Kaiser: 94 | 6,602; 4,090; 746; 2,390 | binary | literature, overlapping parcels (Modha) | no | Netzschleuder; Modha-Singh PNAS SI text files | 2 |
| 18 | FlyWire projectome (Dorkenwald et al. 2024) | Drosophila | 78 neuropils | UNVERIFIED (build yourself) | fractional neuron weighting (each neuron contributes total 1) | square | no | Codex download; fafbseg-py | 4 (same data as our fly graph) |
| 19 | Hemibrain ROI connectivity (Scheffer 2020) | Drosophila | ROI hierarchy (primary ROIs) | build via neuprint | synapse counts per ROI | partial brain | no | neuprint / vfb-region-connectivity | 3 |
| 20 | Pigeon telencephalon, Shanahan et al. 2013 | pigeon | 52 | 344 | binary | literature | no | paper Table 2 | 2 |
| 21 | OSCINE-NET, Savoy et al. 2024 (BMC Neurosci) | songbirds | 187 | 1,107 | binary directed | literature | no | Additional file 1 (xlsx), oscine-net.org; article CC BY-NC-ND | 3 |
| 22 | Kunst et al. 2019 (Neuron), mapzebrain | larval zebrafish | 73 (36 per side + 1) | matrix reported as undirected (UNVERIFIED) | normalized strength | - | no | mapzebrain.org / bi.mpg.de/baier/connectome | 2 |
| 23 | Du et al. 2025 bioRxiv (10.1101/2025.06.06.658008) | larval zebrafish | UNVERIFIED | UNVERIFIED | "weighted and directed inter-region connectome" | - | no | UNVERIFIED | 3 (if released) |
| 24 | Venkadesh et al. 2025 bioRxiv | mouse/marmoset/macaque/human | 22 homologous regions | <= 462 | "effective connection cost"; direction imported from mouse tracer | - | no | Zenodo 10.5281/zenodo.17653309 | 2 |
| 25 | F-TRACT (Lemaréchal et al. 2022, Brain) | human | 130 (Lausanne2008) / 360 (HCP-MMP1) | UNVERIFIED | CCEP response probability, latency (effective, not anatomical) | - | no | Zenodo 10.5281/zenodo.7015415 (CC-BY 4.0); EBRAINS | 3 |
| 26 | Brain/MINDS marmoset (Skibbe 2023 PLoS Biol; Watakabe 2023 Neuron) | marmoset | PFC injections only (52 antero + 19 retro) | - | projection density | PFC rows only | no | dataportal.brainminds.jp | 2 |
| - | Human dMRI (HCP, Budapest, etc.) | human | many | undirected | streamline counts | - | no | Netzschleuder `budapest_connectome`, `human_connectomes_multiscale` | 1 |

## Notes per dataset

### Macaque, Kennedy lab (core-nets)
- Markov N.T. et al. 2014, "A weighted and directed interareal connectivity matrix for macaque cerebral cortex", Cereb Cortex 24(1):17-36, doi:10.1093/cercor/bhs270. 29 injected areas in the 91-area M132 atlas, 1,615 pathways; FLNe per pathway; incoming FLNe of each target sums to 1. 29x29 subgraph is edge-complete, density about 0.66, 536 directed edges (arXiv 1411.0431).
- Markov N.T. et al. 2014, "Anatomy of hierarchy: feedforward and feedback pathways in macaque visual cortex", J Comp Neurol 522(1):225-259, doi:10.1002/cne.23458. SLN = fraction of supragranular labeled neurons; SLN > 0.5 = FF. Netzschleuder `sln` subnetwork: 628 edges into 11 visual areas.
- Markov et al. 2013 Science 342:1238406, doi:10.1126/science.1238406: counterstream, SLN hierarchy, core-periphery.
- Froudist-Walsh S. et al. 2021, Neuron (dopamine gradient / distributed working memory), doi UNVERIFIED (S0896-6273(21)00621-8). 40 injected areas (adds 1, 3, V6, F4, F3, 25, 32, 9, 45A, OPRO). FLN + SLN downloadable from core-nets.org after registration (`index.php?action=download`). Used again in bioRxiv 2024.08.07.606907 (community detection) and 2022.02.20.481230 (conscious access model).
- Planned 139-area atlas (NHP Neuroimaging and Neuroanatomy Project, arXiv 2010.00308, "complete by 2025"): no public release found.
- Netzschleuder `macaque_markov`: flne (91 nodes, 1,989 multi-edges = 1,615 distinct), sln (91, 628), distances (93, 2,667). Formats gt/GraphML/GML/CSV.
- Prior hierarchy work: Markov 2014 JCN (beta-regression hierarchy from SLN), Chaudhuri et al. 2015 and Mejias et al. 2016 dynamics models, Froudist-Walsh 2021 (40-area hierarchy).
- Why 5: edge-complete weighted square subgraph plus quantitative FF/FB per edge. We can check how many SLN-FF edges our MFAS order puts forward, and compare FB weight with random nulls. Caveat: graph is dense (~66%), so MFAS on 29-40 nodes is small (exact solvers feasible, which is a plus: we can get the true optimum).

### Marmoset (Rosa / Majka)
- Majka P. et al. 2020, "Open access resource for cellular-resolution analyses of corticocortical connectivity in the marmoset monkey", Nat Commun 11:1133, doi:10.1038/s41467-020-14858-0. 143 retrograde injections, 52 animals, 55 injected areas, 116-area Paxinos parcellation, FLNe 55x116, SLN per connection.
- Theodoni P. et al. 2021, "Structural attributes and principles of the neocortical connectome in the marmoset monkey", Cereb Cortex (PMC8634603). 55x55 edge-complete, density 62.43%, k = 1,861 edges (from a search snippet; 0.6243 x 55 x 54 = 1,854, small mismatch, check on download). Hierarchy from SLN beta regression, 6Va at the top.
- Portal: marmosetbrain.org (Connectivity Matrix tab: FLNe, SLN, distances); API analytics.marmosetbrain.org. Data CC-BY-SA 4.0.
- Majka et al. 2022 Nat Commun 13:7416: integrated functional + structural resource (not checked further).

### Mouse, Allen Institute
- Oh S.W. et al. 2014, "A mesoscale connectome of the mouse brain", Nature 508:207-214, doi:10.1038/nature13186. 469 injections; linear model on 213 regions per hemisphere. Suppl. Table 3 = model strengths + p-values. Netzschleuder `mouse_mesoscale/model`: 426 nodes, 32,822 edges (16,954 ipsi), weight / pvalue / distance (downloaded and checked). Density high unless filtered by p-value.
- Knox J.E. et al. 2019, "High-resolution data-driven model of the mouse connectome", Netw Neurosci 3(1):217-236, doi:10.1162/netn_a_00066. 428 wild-type experiments, 100 um voxel model, regionalized matrices (normalized connection strength/density). Code + cache: github.com/AllenInstitute/mouse_connectivity_models. Coletta et al. 2020 Sci Adv (doi:10.1126/sciadv.abb7187) built a 15,314-node directed voxel graph from it and reported a sender/receiver hierarchy.
- Harris J.A. et al. 2019, "Hierarchical organization of cortical and thalamic connectivity", Nature 575:195-202, doi:10.1038/s41586-019-1716-z (bioRxiv 292961). Final hierarchy: 37 cortical areas + 24 thalamic nuclei. Method: laminar termination patterns clustered (9 clusters), each cluster mapped to FF or FB; CT connections FF/FB by L5/L6 source (LDA); hierarchy score = (FB - FF)/(FB + FF) style, iterated. Repo files (checked): `Input/CC_TC_CT_clusters.xlsx` (7,063 rows: source, Cre line, target, hemi, L1..L6a densities, Cluster ID), `Input/clustermapping.xlsx` (cluster -> FF/FB), `Output/inputexpanded_CC9.xls` (connections with FF/FB), `Output/ghs_*.xls` (global hierarchy scores). My count from the input file: 40 isocortex sources, 43 targets, 1,180 unique ipsi source->target pairs, 902 of them reciprocated; 400 thalamus->cortex pairs. License: "Other" (Allen terms of use). Very relevant: per-connection anatomical FF/FB labels for mouse, the direct counterpart of SLN. Weights must be joined from Allen projection data or Knox model.
- Cell-type-specific model 2023: "Modeling the cell-type-specific mesoscale murine connectome with anterograde tracing experiments", Netw Neurosci 7(4):1497 (PMC10745083), 1,751 experiments, directed weighted multigraph per Cre line. Authors and sizes UNVERIFIED.

### Mouse, other labs
- Gamanut R. et al. 2018, "The mouse cortical connectome, characterized by an ultra-dense cortical graph, maintains specificity by distinct connectivity profiles", Neuron 97:698-715, doi:10.1016/j.neuron.2017.12.037 (doi UNVERIFIED, S0896-6273(17)31185-6). 27 retrograde injections into 19 of 47 areas; FLNe in Suppl. Table 6; 19x19 density ~97%. Too dense and small for MFAS to be informative, but exact optimum is cheap.
- D'Souza R.D. et al. 2022, "Hierarchical and nonhierarchical features of the mouse visual cortical network", Nat Commun 13:503, doi:10.1038/s41467-022-28035-y. Retrograde laminar data in mouse visual cortex, FF/FB via beta regression (Kennedy/Knoblauch method). Size and data files UNVERIFIED.
- Xiong F., Liu L., Peng H. et al. 2025, "Reconstruction of a connectome of single neurons in mouse brains by cross-validating multi-scale multi-modality data", Nat Methods 22:2670-2683, doi:10.1038/s41592-025-02784-2. SEU-net from 1,877 fully reconstructed neurons. Zenodo files (downloaded and checked): `SEU_net_w_corticalLayers.xlsx` 148x148, rows = source, columns = target, 0 = no connection; 1,365 nonzero including 66 diagonal; 126 regions have outgoing, 135 incoming edges; weights 0.184-14.89, median 0.93. `SEU_net_wo_corticalLayers.xlsx` 90 regions. CC-BY 4.0. Paper also compares with "Allen-net" and "BRICseq-net".
- Huang L. et al. 2020, "BRICseq bridges brain-wide interregional connectivity to neural activity and gene expression in single animals", Cell 182:177-188, doi:10.1016/j.cell.2020.05.029. Cortical region-to-region matrix from barcode sequencing; size UNVERIFIED.
- Zingg B. et al. 2014, "Neural networks of the mouse neocortex", Cell 156:1096-1111, doi:10.1016/j.cell.2014.02.023. 600+ pathways, 240 intracortical connections; matrix availability UNVERIFIED.
- Foster N.N. et al. 2021, "The mouse cortico-basal ganglia-thalamic network", Nature 598:188-194, doi:10.1038/s41586-021-03993-3. Subnetwork-level, no ready region matrix found.

### Rat
- BAMS (Bota and Swanson 2007, J Comp Neurol 500:807-814, doi:10.1002/cne.21209). Netzschleuder `rat_brain_bams`: 2011 503/27,667; 2012 503/30,088; 2013 496/33,020; ordinal strength 1..9 (mapping undocumented).
- Swanson L.W., Hahn J.D., Sporns O. series: 2015 cortex association (doi:10.1073/pnas.1504394112), 2016 cerebral nuclei (doi:10.1073/pnas.1613184113), 2017 cortex 154 regions, log10 weights (doi:10.1073/pnas.1712928114), 2018 endbrain (PNAS 115:E6910), 2019 thalamus / interbrain (PubMed 31806763), 2022 rostral sector (doi:10.1073/pnas.2210931119), 2024 "Neural network architecture of a mammalian brain" (doi:10.1073/pnas.2413422121; ~790 regions / ~77,000 connections, UNVERIFIED), later whole CNS 924 regions / 81,582 connections, density 9.6% (from a search snippet about PNAS 2025-2026 papers, exact paper UNVERIFIED). Tables: PNAS Datasets and Google Sheets at sites.google.com/view/the-neurome-project/connections. Weights are qualitative (ordinal, log10 mapped), directed, bilateral. Good large directed area-level graph for the neuron vs area question; FF/FB labels absent.

### Cat
- Scannell J.W., Blakemore C., Young M.P. 1995, J Neurosci 15(2):1463-1483, doi:10.1523/JNEUROSCI.15-02-01463.1995: 65 areas, 1,139 connections, strengths 1-3.
- Scannell J.W. et al. 1999, Cereb Cortex 9(3):277-299, doi:10.1093/cercor/9.3.277: Netzschleuder `cat_corticothalamic` all 95/2,126, cortex 52/818, weights 1-3. Also in Brain Connectivity Toolbox (bctnet datasets page now 404).

### Macaque literature collations
- Felleman D.J., Van Essen D.C. 1991, Cereb Cortex 1:1-47: 32 visual areas, Netzschleuder `macaque_visual_cortex` 315 edges, binary. Original paper classifies pathways as ascending/descending/lateral (FF/FB/L) by laminar pattern: ground truth for ordering. Hilgetag C.C., O'Neill M.A., Young M.P. 1996, Science 271:776, doi:10.1126/science.271.5250.776: computed 150,000 hierarchies all violating fewer FVE constraints than FVE itself; only 2 of 30 areas had fixed levels. This is exactly an "optimal ordering is indeterminate" result, directly comparable to our seed-stability findings. Follow-up: Hilgetag et al. 2000 Phil Trans B 355:71-89; Reid et al. 2009 NeuroImage (continuous hierarchy optimization).
- Modha D.S., Singh R. 2010, PNAS 107:13485, doi:10.1073/pnas.1008054107: CoCoMac, 383 regions (overlapping parcels), 6,602 directed binary edges, text edge lists in SI.
- Netzschleuder `macaque_cocomac` (242/4,090), `macaque_neural` (71/746, Young 1993), `macaque_cortex_kaiser` (94/2,390), all binary.

### Drosophila region level
- Dorkenwald S. et al. 2024, "Neuronal wiring diagram of an adult brain", Nature 634:124-138, doi:10.1038/s41586-024-07558-y. Projectome over 78 neuropils (Ito 2014 atlas), built from intrinsic neurons; each neuron contributes total weight 1 split by its pre/post sites. Build from Codex (codex.flywire.ai/api/download) + fafbseg-py. Lin et al. (PMC10402125) and Pentek & Ercsey-Ravasz 2024 (arXiv 2410.01269) built the same projectome.
- Why useful: same underlying data as our neuron-level FlyWire graph. Coarse-graining it to 78 neuropils gives a within-dataset neuron-vs-area comparison that removes the species/method confound. Same idea works for MICrONS (area x layer). Caveat: a fly neuron spans several neuropils, so region assignment is a modeling choice.
- Hemibrain (Scheffer et al. 2020 eLife): ROI-level matrix via neuprint `fetch_roi_hierarchy` / neuprintr `neuprint_ROI_connectivity`; VirtualFlyBrain/vfb-region-connectivity on GitHub.

### Birds, fish
- Shanahan M. et al. 2013, Front Comput Neurosci 7:89, doi:10.3389/fncom.2013.00089: pigeon telencephalon 52x52, 344 directed binary connections, in Table 2.
- Savoy A. et al. 2024, BMC Neurosci 25:79, doi:10.1186/s12868-024-00919-3: OSCINE-NET, 187 nodes, 1,107 directed connections, unweighted, 14 songbird species (zebra finch 78.7% of entries); Additional file 1 xlsx.
- Kunst M. et al. 2019, Neuron 103:21-38, doi:10.1016/j.neuron.2019.04.034: mapzebrain, 73 regions, matrix described as undirected (from search snippet, UNVERIFIED). Single-neuron tracings (>4,000 now) could give a directed matrix by soma/dendrite -> axon region.
- Du X.-F. et al. 2025 bioRxiv 10.1101/2025.06.06.658008: larval zebrafish CNS atlas, "weighted and directed inter-region connectome"; sizes and data UNVERIFIED.

### Human (brief)
- dMRI tractography is undirected (HCP, Budapest Reference Connectome, Netzschleuder `human_connectomes_multiscale`): not usable for FAS.
- F-TRACT CCEP atlas (Lemaréchal et al. 2022 Brain 145:1653, doi:10.1093/brain/awab362; Zenodo 10.5281/zenodo.7015415, CC-BY 4.0): directed effective connectivity (response probability) on 130/360 parcels. Directed but functional, not anatomical.
- Venkadesh S. et al. 2025 bioRxiv 10.1101/2025.09.07.674762: 22 homologous regions in 4 species, direction taken from mouse tracer data (so not independent per species). Zenodo 10.5281/zenodo.17653309.
- 2026 bioRxiv 10.64898/2026.06.16.732559: "directed human structural connectome" using macaque tract tracing to orient human tractography, analysed with trophic levels (MacKay, Johnson, Sansom 2020). Direction borrowed, not measured.

### Repositories
- Netzschleuder (networks.skewed.de, tag Connectome): 36 entries. Directed + weighted region-level: `macaque_markov`, `cat_corticothalamic`, `rat_brain_bams`, `mouse_mesoscale/model`. Directed binary: `cat_cortex`, `macaque_cocomac`, `macaque_neural`, `macaque_visual_cortex`, `macaque_cortex_kaiser`. Neuron-level also present: `flywire`, `fly_hemibrain`, `fly_banc`, `fly_manc`, `microns_mm3`, `fly_larva`, `platynereis_connectome`, C. elegans sets.
- Brain Connectivity Toolbox datasets page (sites.google.com/site/bctnet/datasets) returned 404 today; its cat/macaque sets are mirrored in Netzschleuder.
- Open Connectome / NeuroData graph collection: archived; Netzschleuder took BAMS and Markov from it.
- core-nets.org (Kennedy lab, registration), marmosetbrain.org, Allen GitHub repos, Zenodo.

### Prior hierarchy / feedforward-ordering work on these graphs
- Felleman & Van Essen 1991 -> Hilgetag 1996 / 2000 (optimal hierarchies, constraint violations: essentially an MFAS-like problem on FF/FB constraints), Reid 2009.
- Markov 2014 JCN, Chaudhuri 2015, Froudist-Walsh 2021, Theodoni 2021: SLN beta-regression hierarchy.
- Harris 2019: iterative global hierarchy score from FF/FB labels; Ma et al. (SJTUZhangLab/HierarchyAnalysis) reuse it.
- Kale P., Zalesky A., Gollo L.L. 2018, Netw Neurosci, doi:10.1162/netn_a_00040: how undirected approximations distort 6 directed connectomes (cat, mouse, C. elegans, 3 macaque). Mouse version N=213, E=2,105 after disparity filter p<0.05 (from search snippet).
- Coletta 2020 Sci Adv: sender/receiver hierarchy on voxel mouse graph.
- Pathak, Menon, Sinha 2024 PNAS 121:e2314291121 (arXiv 2308.08898): hierarchy index for directed brain networks (macaque 266 regions / 2,602 links mentioned in snippet, source UNVERIFIED).
- Trophic levels (MacKay et al. 2020) applied to a directed human connectome (2026 bioRxiv above). No trophic analysis of Markov/Allen matrices found.

## Side task: provenance of `data/table_mouse.txt` (148 nodes, 583 edges)

Fingerprint: ids 1..149 without 37, 583 edges, no self-loops, weights 0.000184-0.614 (median 0.0063), max in-degree 95 (node 92), max out-degree 56; 144 nodes have outgoing, 114 incoming edges; no left/right hemisphere shift symmetry (checked shifts 60-89).

Checked today:
1. Oh 2014 linear model (Netzschleuder `mouse_mesoscale/model`, 32,822 weights). Our 583 weights: 0 matches at 8+ significant digits, 12 at 6 digits (chance level for this many values). Not taken from that matrix as-is.
2. SEU-net 2025 (Xiong et al.), which is "148 regions including cortical layers": 148 matches our node count, but it has 1,365 edges, weights 0.18-14.9, max in-degree 66 and max out-degree 55. Our hub has in-degree 95, larger than any SEU node in either direction, so our table cannot be a thresholded subgraph or transpose of SEU-net with layers. The 148 is a coincidence as far as this file goes.
3. Harris 2019 cortical file: 43 areas, wrong size.
4. Earlier repo check (biology/studies/untapped.md §C): Calabrese 2015 dMRI (296 ROIs) refuted; Bader 2025 and Vahidi 2025 papers contain no mouse data.

Not checked (open leads): the "Allen-net" and "BRICseq-net" built in the SEU-net paper (may be on the same 148-region parcellation; probably inside the 975 MB `jupyter_tutorial.zip` on Zenodo, not downloaded); Knox 2019 regionalized matrices; Swanson-style or BAMS mouse collations; any thresholded/normalized version of Oh 2014 Suppl. Table 3 (row-normalized values would not match the raw model weights). Conclusion: provenance still unknown; the heavy-edge floats (51->53 0.613959345844929, 105->123 0.594344938173106) remain the way to identify it once a candidate matrix is found.

## Sources
- https://networks.skewed.de/?tags=Connectome ; /net/macaque_markov ; /net/mouse_mesoscale ; /net/rat_brain_bams ; /net/cat_corticothalamic
- https://arxiv.org/pdf/1411.0431 ; https://sbri.fr/wp-content/uploads/marko14jcn.pdf ; https://www.cell.com/neuron/fulltext/S0896-6273(21)00621-8
- https://www.biorxiv.org/content/10.1101/2024.08.07.606907v2.full ; https://arxiv.org/pdf/2010.00308
- https://www.nature.com/articles/s41467-020-14858-0 ; https://www.biorxiv.org/content/10.1101/2020.02.28.969824v1.full ; https://www.marmosetbrain.org
- https://www.nature.com/articles/s41586-019-1716-z ; https://github.com/AllenInstitute/MouseBrainHierarchy ; https://www.biorxiv.org/content/10.1101/292961v1.full
- https://www.nature.com/articles/nature13186 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC6372022/ ; https://github.com/AllenInstitute/mouse_connectivity_models ; https://www.science.org/doi/10.1126/sciadv.abb7187
- https://www.cell.com/neuron/fulltext/S0896-6273(17)31185-6 ; https://www.nature.com/articles/s41467-022-28035-y
- https://www.nature.com/articles/s41592-025-02784-2 ; https://zenodo.org/records/10677899
- https://direct.mit.edu/netn/article/7/4/1497/117562 ; https://www.cell.com/cell/fulltext/S0092-8674(20)30624-3 ; https://resource.loni.usc.edu/wp-content/uploads/2015/10/Zingg_Neural_Networks.pdf ; https://www.nature.com/articles/s41586-021-03993-3
- https://www.pnas.org/doi/10.1073/pnas.1712928114 ; https://www.pnas.org/doi/10.1073/pnas.2413422121 ; https://sites.google.com/view/the-neurome-project/connections
- https://www.jneurosci.org/content/15/2/1463 ; https://www.pnas.org/doi/10.1073/pnas.1008054107 ; https://www.semanticscholar.org/paper/Indeterminate-Organization-of-the-Visual-System-Hilgetag-O'Neill/a4f32c8f294b2181d86a8b2bf1a58a62028bfa0d
- https://www.nature.com/articles/s41586-024-07558-y ; https://pmc.ncbi.nlm.nih.gov/articles/PMC10402125/ ; https://arxiv.org/abs/2410.01269 ; https://github.com/VirtualFlyBrain/vfb-region-connectivity
- https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2013.00089/full ; https://bmcneurosci.biomedcentral.com/articles/10.1186/s12868-024-00919-3
- https://www.cell.com/neuron/fulltext/S0896-6273(19)30391-5 ; https://www.biorxiv.org/content/10.1101/2025.06.06.658008v1
- https://www.biorxiv.org/content/10.1101/2025.09.07.674762.full ; https://zenodo.org/records/7015415 ; https://www.biorxiv.org/content/10.64898/2026.06.16.732559v1.full
- https://www.biorxiv.org/content/10.1101/2022.06.06.494999v1 ; https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3002158
- https://arxiv.org/abs/1801.01651 ; https://arxiv.org/abs/2308.08898 ; https://academic.oup.com/cercor/article/34/5/bhae174/7676598

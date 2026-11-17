# F. Gap-hunt: directed weighted datasets missing from the survey (2026-11-17)

Stop condition: this pass did NOT end because searches ran dry. The session-wide WebSearch budget
(200 calls, shared across agents) ran out after 9 searches by this agent. Further checks used WebFetch
on known URLs only. Areas (d) ANN weights and (e) the neuPrint, CAVE and BossDB lists are therefore
only partly covered (see "Not checked").
"VERIFIED" means the number was read from a fetched page or search snippet in this session.
"UNVERIFIED" means it comes from memory or an unconfirmed snippet.

## (a) New connectome releases / versions

| Name | Citation / DOI | URL | Nodes / edges | W / D | Annotations | Access, license | Fit | Why |
|---|---|---|---|---|---|---|---|---|
| Fish-X larval zebrafish whole-brain EM, neuromodulator-annotated | Li, Liu, Shi et al., bioRxiv 10.1101/2025.06.12.659365 (v1 Jun 2025, v3) | biorxiv link | >176k cells, >25M synapses (VERIFIED from abstract); retina to anterior spinal cord, 6 dpf | D, W (synapse counts) if the synapse table is released | NE/DA/5-HT/hypocretin/glycinergic labels from APEX2 | Data availability UNVERIFIED. Automated segmentation; authors say whole-brain proofreading is still needed | 2 | A second vertebrate whole-brain EM volume next to Fish1, but no proofread edge list is confirmed |
| FlyWire, new default synapse table ("Princeton synapses", Yu et al.) | bioRxiv 10.1101/2025.07.11.664377 | codex.flywire.ai | ~140k neurons, >50M synapses (snippet) | D, W | same as FAFB | Codex default since Jul 2025 (VERIFIED from Codex about page) | 4 | NEW VERSION of the covered FAFB data. Edge weights differ from v783 Buhmann+Heinrich synapses, so MFAS scores computed on the two tables are not comparable. Record which table was used |
| Netzschleuder `fly_banc` (BANC packaged) | Bates et al., Nature 2026 (published 8 Jun 2026) | networks.skewed.de/net/fly_banc | 150,802 / 1,861,418 (VERIFIED, Netzschleuder) | D, W | per Netzschleuder | open | 4 | Ready-made graph-tool/CSV version of the covered BANC. Useful as an independent loader check |
| Netzschleuder `fly_optic_lobe` (2025) | Nern et al. 2025 | networks.skewed.de | 10,318,164 nodes / 22,258,367 edges (VERIFIED). The node count suggests it includes segment fragments; needs filtering | D, W | | open | 2 | Covered dataset; the packaging has caveats |
| Netzschleuder `microns_mm3` | MICrONS 2023-2025 | networks.skewed.de | 71,981 / 9,005,438 (VERIFIED) | D, W | | open | 3 | Packaged version of the covered MICrONS, probably all segments including unproofread ones; check before use |
| C. elegans male posterior connectome | Jarrell et al., Science 2012 | networks.skewed.de/net/celegans_male | 272 / 4,451 (VERIFIED) | D, W | cell classes | open | 2 | Small. Largely superseded by Cook 2019 male, but an independent reconstruction |
| C. elegans dauer connectome | Yim et al., Nature 2024 (UNVERIFIED citation details) | OpenWorm ConnectomeToolbox "Yim 2024" | UNVERIFIED | D, W (normalized and raw), plus contactome | developmental stage | open via the toolbox | 3 | Another developmental state, comparable with Witvliet L1..adult on the same neuron set |
| Drosophila optic medulla, 7-column | Takemura et al., Nature 2013 | networks.skewed.de/net/fly_medulla | 1,781 / 33,641 (VERIFIED). Tagged unweighted on Netzschleuder; the original has synapse counts (UNVERIFIED) | D | cell types | open | 1 | Superseded by the optic lobe data |
| Mouse V1 EM (Bock 2011) | Bock et al., Nature 2011 | networks.skewed.de/net/mouse_visual_cortex | 195 / 214 (VERIFIED) | D, W | | open | 1 | Too sparse for MFAS |
| Comparative insect central complex (6 species, SBEM projectomes + local connectomes) | Gillet, Sayre, ..., Funke, Heinze, bioRxiv 10.64898/2026.02.25.708065 | biorxiv | not given in the text (VERIFIED absence) | D, W locally | head-direction cell types across species | data availability not found | 2 (watch) | The only cross-species insect connectomics seen. Species and data access unknown |
| Larval Drosophila CNS, eFIB-SEM + brain-wide activity | bioRxiv 10.1101/2025.09.25.678485 (v2 19 Mar 2026) | biorxiv | UNVERIFIED | D, W | functional activity matched to EM | UNVERIFIED | 3 (watch) | Possibly a complete larval CNS (brain + VNC) EM volume with activity. Edge list status unknown |

No 2025-2026 whole-brain connectome of a new species (ant, bee, mosquito, Xenopus, lamprey, mollusc, echinoderm) turned up in the searches that ran. A full sweep was not possible within the search budget.

## (b) Single-neuron projectomes (turn into directed region-to-region graphs)

Building the graph: soma region -> axon-terminal region, weighted by axon length or terminal count, summed over neurons. Region level only; projectomes give no synapses.

| Name | Citation / DOI | URL | Size | Annotations | Access, license | Fit | Why |
|---|---|---|---|---|---|---|---|
| Peng et al. 2021, 1,741 mouse full morphologies | Nature 598:174, 10.1038/s41586-021-03941-1 | Brain Image Library 10.35077/g.25 | 1,741 SWC files (original + CCFv3-registered) (VERIFIED) | Cre lines, soma region, molecular types | open; Allen Terms of Use (VERIFIED) | 3 | CCF-registered, so easy to aggregate to a region graph. Sampling is biased toward the labelled lines |
| Qiu et al. 2024, hippocampal projectomes | Science 383:eadj9198, 10.1126/science.adj9198 | Dryad 10.5061/dryad.wh70rxwv4 (3.28 GB) | 10,100 neurons, 43 subtypes (VERIFIED) | subregion, spatial transcriptomics | Dryad (CC0 is typical for Dryad, UNVERIFIED) | 2 | Somata are only in the hippocampus, so it gives an out-star, not a closed graph |
| Gou et al. 2025, macaque PFC projectomes | Cell 188:3806, 10.1016/j.cell.2025.06.005 | Zenodo 10.5281/zenodo.15128612 (330.6 MB zip, NMT space); macaque.digital-brain.cn | 2,231 neurons, 19 injection sites, 7 monkeys, 32 subtypes (VERIFIED) | IT/PT/CT subtypes | open; license not stated on Zenodo | 2 | First primate single-neuron data. Somata only in PFC, so it is a partial graph |
| MouseLight (Winnubst et al., Cell 2019) | 10.1016/j.cell.2019.07.042 (UNVERIFIED DOI) | ml-neuronbrowser.janelia.org | ~1,000-1,200 neurons (snippet says 1200, UNVERIFIED) | CCF region | open | 3 | Combine with Peng 2021 (CCF-registered, commonly done together) into one directed region graph |
| Gao et al. 2022 mouse PFC projectomes | Nat Neurosci 2022 (UNVERIFIED, ~6,000 neurons) | UNVERIFIED | UNVERIFIED | | | 2 | Same group as Qiu 2024. Not checked |

## (c) Directed functional / effective connectivity

| Name | Citation / DOI | URL | Size | W / D, sign | Access, license | Fit | Why |
|---|---|---|---|---|---|---|---|
| C. elegans signal propagation atlas | Randi, Sharma, Dvali, Leifer, Nature 623:406 (2023), 10.1038/s41586-023-06683-4 | DANDI 001075 (4.1 TB raw NWB, CC-BY-4.0, VERIFIED); processed matrix via `wormneuroatlas` / OpenWorm ConnectomeToolbox Randi2023_data | 10,438 pairs probed; matrix 181x181 with 1,150 nonzero at q<0.05 (VERIFIED) | D, W, signed (excitatory or inhibitory) | open | 5 | The best direct test: a causal directed graph on the same neurons as Cook/Witvliet. Compare the MFAS order of the functional graph with the anatomical order |
| C. elegans extrasynaptic monoamine/neuropeptide | Bentley et al., PLoS Comput Biol 2016 | Netzschleuder celegans_extrasynaptic; OpenWorm toolbox | 251 / 3,162 (VERIFIED) | D, unweighted | open | 3 | Wireless layer. Is feedback different in the neuromodulatory graph? |
| C. elegans neuropeptide connectome | Ripoll-Sánchez et al., Neuron 2023 (UNVERIFIED details) | OpenWorm toolbox (short-, mid- and long-range versions) | UNVERIFIED (~300 neurons) | D, ligand-receptor based weights UNVERIFIED | open | 3 | Dense directed graph with 3 range variants, a natural ladder for feedback analysis |
| C. elegans NT identity atlas | Wang et al. 2024 (hermaphrodite and male) | OpenWorm toolbox | node annotations | n/a | open | 4 as annotation | Gives E/I sign labels for the Cook/Witvliet graphs |
| C. elegans whole-brain activity (Atanas 2023) | Cell 2023 | linked from the toolbox | UNVERIFIED | needs a model to get a directed graph | open | 1 | Activity only, not a graph |
| F-TRACT human CCEP atlas | Trebaul 2018 NeuroImage; Lemaréchal 2022 Brain (780 patients, 34,354 stimulations) | f-tract.eu/atlas; Zenodo 7015415 (v2112, 915 MB, CC-BY-4.0); newer releases on EBRAINS (2023+) | parcellations: Lausanne2008-60/125/250, HCP-MMP1, MarsAtlas, AAL, ... (VERIFIED) | D (stimulated -> recorded), W = response probability, plus latency features | open; EBRAINS may need a login (UNVERIFIED) | 4 | The only human directed whole-cortex weighted graph. Has missing pairs (sampling by electrodes), so MFAS needs a mask for unmeasured pairs |
| Allen Synaptic Physiology (Campagnola et al., Science 2022) | 10.1126/science.abj5861 (UNVERIFIED DOI) | aisynphys package (Allen Institute Software License: BSD-2 + non-commercial clause, VERIFIED) | UNVERIFIED (tens of thousands of probed pairs, mouse V1 + human cortex) | D, W connection probability and strength between cell-type x layer classes, signed by E/I | open | 3 | Cell-class-level directed graph (~tens of nodes) with probabilities; a small but clean E/I-annotated MFAS case |

## (d) Gene regulatory / ANN

| Name | Source | Size | W / D | License | Fit | Why |
|---|---|---|---|---|---|---|
| TRRUST v2 | grnpedia.org/trrust; Han et al. NAR 2018 | human 8,444 edges / 800 TFs; mouse 6,552 / 828 TFs; sign known for 59.8% (VERIFIED) | D, signed, unweighted | CC BY-SA 4.0 (VERIFIED) | 2 | Signed directed control network. Unweighted, so MFAS reduces to unweighted FAS |
| RegulonDB (E. coli) | regulondb.ccg.unam.mx | UNVERIFIED (several thousand TF-gene edges) | D, signed, with evidence levels | academic terms UNVERIFIED | 2 | Classic near-DAG transcriptional hierarchy, a useful low-feedback reference point |
| Trained RNN weight matrices | not found (search budget ran out) | | | | | |

## (e) Aggregators: checked

- Netzschleuder Connectome tag (36 entries, VERIFIED): new relative to the survey are celegans_male, celegans_extrasynaptic, fly_medulla, mouse_visual_cortex, macaque_cortex_kaiser (94 areas, unweighted), macaque_neural (Young 1993, unweighted), plus packaged versions of covered sets (fly_banc 2026, fly_optic_lobe, microns_mm3, platynereis_connectome 2025, rat_brain_bams). The remaining entries are undirected and not useful.
- OpenWorm ConnectomeToolbox (VERIFIED): new relative to the survey are Randi 2023 functional, Bentley 2016, Ripoll-Sánchez 2023, Yim 2024, Wang 2024 NT, Cook 2020, Durbin 1987.

## Not checked (search budget exhausted)

KONECT; the neuPrint dataset list (the Janelia datasets page returned 404); CAVE public datastacks; BossDB project list (the page renders through JavaScript); Codex dataset list; ANN weight releases; Drosophila optogenetic functional connectivity (e.g. Pospisil/Brezovec-style datasets); Aplysia, leech, Xenopus, lamprey circuits; mouse olfactory bulb, cerebellum and thalamus EM; human 2025-2026 EM; MouseLight details; Gao 2022. These need a second pass once the budget refills.

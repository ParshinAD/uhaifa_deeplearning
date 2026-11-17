# G. Gap-hunt part 2: aggregator lists and missing EM regions (2026-11-17)

Continues F. Scope: KONECT; dataset lists of neuPrint, CAVE, BossDB and FlyWire Codex; mouse
cerebellum, thalamus and olfactory bulb EM; human EM 2025-2026. Only items not already in A-F are
reported. 14 WebSearch calls were used; most facts come from fetched pages, public APIs and one small
download (2.3 MB annotation zip, kept in the session scratchpad, not in `data/`).
"VERIFIED" means the number was read on a fetched page, an API answer or a downloaded file in this
session. "UNVERIFIED" means it comes from a search snippet or memory.
Fit 1-5 as in A-F (5 = use next, 1 = not usable as a graph, 0 = nothing released).

## (a) Aggregator lists: what they contain

| Aggregator | How checked | What is there | New relative to A-F |
|---|---|---|---|
| neuPrint (neuprint.janelia.org) | `GET /api/dbmeta/datasets` answers without a token (VERIFIED); queries still need a token | hemibrain:v1.2.1, male-cns:v0.9, male-cns:v1.0 (description says 167k neurons, brain + nerve cord), manc:v1.0, manc:v1.2.1, manc:v1.2.3, mushroombody, optic-lobe:v1.0.1, optic-lobe:v1.1 (VERIFIED) | `mushroombody` (Takemura 2017, row G4); `manc:v1.2.3` (segment property update dated 2024-08-31, newer than the v1.2.1 in A). FANC, BANC and FAFB are not on this server |
| FlyWire Codex (codex.flywire.ai) | landing and download pages | 5 datasets: FAFB v783 139,255 neurons / 3,732,460 connections; BANC v888 158,262 / 3,037,361; MANC v1.2.1 23,665 / 5,305,638; MAOL v1.1 52,445 / 6,484,936; MCNS v1.0 166,700 / 6,242,118 (VERIFIED). Download needs Google sign-in (VERIFIED for apps). MFAS leaderboard exists only for FAFB (VERIFIED) | Codex "connections" are fewer than our FAFB edge count (3,732,460 vs 5,657,719), so Codex applies a synapse threshold (probably >= 5, UNVERIFIED). The MCNS 6.24M is therefore not comparable with the 25.6M edge count quoted in README (still UNVERIFIED). Codex also links a mouse CA3 explorer (row G2), so it is no longer fly-only |
| CAVE global server (global.daf-apis.com) | `/info/` redirects to Google OAuth (VERIFIED) | full list not visible without login. A March 2024 printout in navis issue #146 (VERIFIED on page): fanc_production_mar2021, fanc_sandbox, flywire_fafb_production, flywire_fafb_public, flywire_fafb_sandbox, minnie35_public_v0, minnie65_public, minnie65_public_v117/v343/v661, minnie65_sandbox, mrgd, pinky_sandbox | `mrgd` (content unknown, not checked). Other public stacks live on other servers: v1dd_public (allentech, in B), h01_c3_flat (in B), CA3 via pyr.ai (row G2) |
| BossDB (bossdb.org/projects) | the page is a JS app; its data source is a public JSON snapshot `https://bossdb-metadata-snapshot.s3.amazonaws.com/mongo-data.json` (798 KB, VERIFIED). Notebook list also in GitHub aplbrain/bossdb-collab-notebooks | ~75 projects. Only 7 have a "connectomes" link (VERIFIED): microns_minnie2021 (synapse CSVs), microns_pinky2021, microns_basil2019 (synapse CSV), witvliet2020, kasthuri2015, takemura2013, bock2011. The rest are images and segmentations only | Basil synapse graph (row G1); cerebellum Park 2023 annotations (row G5); mosquito antennal lobe (row G3); dLGN Morgan (row G8); 2026 additions without papers: xiang2026 mouse spinal cord dorsal horn TEM, singer2026 retina amacrine APEX2, han2026 nematode esophagus, chandok2026 RAEM method (all fit 0 for now) |
| KONECT (konect.cc) | category pages Neural, Animal, Trophic, Metabolic | Neural: only C. elegans dimacs10-celegansneural 297 / 4,296, directed, weighted (VERIFIED; covered). Animal, directed: Japanese macaques dominance 62 / 1,187 W; bison 26 / 314 W; sheep 28 / 250 W; cattle 28 / 217 W; hens 32 / 496 unweighted tournament; rhesus grooming 16 / 111 W (VERIFIED). Trophic: Florida wet/dry (covered in D), Little Rock Lake 183 / 2,494 unweighted (VERIFIED). Metabolic: undirected or unweighted only | Nothing important. The dominance nets are small extra cases for D section 2.5 (DomArchive already covers the type). KONECT is not a source of new brain graphs |

## (b) New datasets found

| # | Name | Citation / DOI | URL | Nodes / edges | W / D | Annotations | Access, license | Fit | Why |
|---|---|---|---|---|---|---|---|---|---|
| G1 | MICrONS Basil synapse graph | BossDB 10.60533/BOSS-2019-QF8A; data used in Turner et al., Cell 2022 (10.1016/j.cell.2022.01.023) and for the v117 neuron classifier | bossdb.org/project/microns_basil2019; `s3://bossdb-open-data/iarpa_microns/basil/synapse_graph/cleaned_final.csv.gz` | one row per synapse; file 2,153,244,104 bytes gz (VERIFIED). Columns pre/post segment id, xyz, size (VERIFIED from first rows). Neuron count and volume size not found | D, W after summing synapses per pair | none in the file; nucleus segmentation separate | open S3, no token; BossDB terms (license not read) | 2 | A third mouse V1 EM volume from the MICrONS program. Automated segments only, no proofreading table seen, so it needs nucleus-based filtering before MFAS. BossDB text says "all cortical layers", search snippet tags say "layer 2/3": scope UNVERIFIED |
| G2 | Mouse hippocampal CA3 (Zheng, Park, Hammerschmith ... Seung, Tank) | bioRxiv 10.1101/2025.07.09.663979; Nature Neuroscience 2026, s41593-026-02388-9 | pyr.ai; codex.flywire.ai/research/mouse_ca3_explorer; code github.com/seung-lab/ca3_paper | ~1 x 1 x 0.1 mm3; 1,815 pyramidal + 229 inhibitory soma-containing neurons, >55,000 mossy fibers (VERIFIED, Europe PMC full text). MF-Pyr bipartite graph of 636 Pyr and ~25,200 MFs (VERIFIED). Synapse total not found | D, W (synapse counts) | cell types (thorny / sparsely thorny Pyr, perisomatic / dendritic inhibitory), proofreading of all soma neurons | "An account is required, but access is open to all users" (VERIFIED); CAVE + cloudvolume. Proofread share ~32 % (preprint) / ~40 % (journal) of voxels (UNVERIFIED snippet) | 3 | First large mammalian region known for strong recurrent excitation (CA3 collaterals), with full soma proofreading. Pyr-Pyr recurrence plus MF input layer gives a natural "input layer + recurrent core" test for MFAS. Paper already compares with degree-preserving and proximity-preserving nulls on the MF-Pyr graph |
| G3 | Aedes aegypti antennal lobe, maxillary-palp glomeruli | Bao, Alford et al., bioRxiv 10.1101/2025.07.29.667487 | BossDB 10.60533/boss-2025-gcyr (ssTEM GridTape, 4x4x40 nm, 1,747 slices, VERIFIED) | not given in abstract | D, W | OSN / PN / LN classes, CO2 glomerulus | open BossDB; graph release not confirmed | 2 | Only released mosquito connectome piece. Abstract (VERIFIED): CO2 OSNs make "high levels of recurrent synaptic connections with one another" and more recurrence than any D. melanogaster glomerulus. Small, but a direct cross-species recurrence comparison with FlyWire glomeruli |
| G4 | Drosophila mushroom body alpha lobe | Takemura et al., eLife 2017, 10.7554/eLife.26975 | neuPrint dataset `mushroombody` | 983 neurons (VERIFIED, neuPrint description and abstract) | D, W | KC / MBON / DAN, compartments | neuPrint token (free) | 2 | Small, fully reconstructed recurrent module (KC>DAN, DAN>MBON reported). Mostly superseded by hemibrain and FlyWire, but useful as an exact-MIP size test case |
| G5 | Mouse cerebellum molecular layer (Park et al.) | bioRxiv 10.1101/2023.09.15.557147 ("A cerebellar disinhibitory circuit supports synaptic plasticity"); BossDB 10.60533/BOSS-2023-3DLD | bossdb.org/project/park2023; annotations zip `https://bossdb-assets.s3.amazonaws.com/files/cell_syn_annotations.zip` (2.3 MB) | 175 x 122 x 50 um, P56. Cells: 24 CF, 1,035 IN, 381 PC, 33,900 PF. Synapses: PF>PC 148,781 (119,289 pairs), PF>IN 48,943, IN>PC 10,652, IN>IN 7,626 (4,456 pairs), CF>PC 3,765, CF>IN 56 (all VERIFIED from the downloaded files) | D, W (count synapse points per pair) | cell class per id | open, no token; license not stated | 2 | Ready-to-use small graph. Feedforward by anatomy (PF/CF to IN/PC, PC has no output inside) except the IN>IN layer, so it is a clean "near-DAG with one recurrent stage" control. Synapses are manual neuroglancer points with "id-id" in the Description field |
| G6 | Mouse cerebellum, developmental series | Dhanyasi et al., bioRxiv 10.1101/2025.09.15.676403 | biorxiv | P0, P3, P7, P10, P14 ssEM; PC and parallel-fiber cohorts (snippet) | D | ages | data release UNVERIFIED | 1 | Partial reconstructions, mostly bipartite PF>PC |
| G7 | Mouse cerebellum granular layer vCLEM | bioRxiv 10.1101/2025.10.02.679866 (data from Han et al.) | biorxiv | 364 granule cells, 1 PC, 2 basket cells (snippet, UNVERIFIED) | D | VGluT1+/- mossy fibers | UNVERIFIED | 1 | Bipartite MF>GrC, no recurrence |
| G8 | Mouse dLGN (visual thalamus) | Morgan, Berger, Wetzel, Lichtman, Cell 2016, 10.1016/j.cell.2016.02.033; Morgan & Lichtman, Neuron 2020, 10.1016/j.neuron.2020.02.001 | BossDB 10.60533/BOSS-2020-OMCN | volume 500 x 400 x 300 um, P32, 10,000 sections (VERIFIED, BossDB text). RGC>TC network size UNVERIFIED | D | RGC / TC / interneuron | open images; connectivity table not found | 1 | The only large thalamus EM volume found. The published network is RGC>TC bipartite, so no feedback to order. No TRN or corticothalamic EM connectome found |
| G9 | Mouse hippocampal CA1 large volume (Schmidt lab) | Corteze, Grasso, Vezir, Schmidt, bioRxiv 10.1101/2025.04.04.647285; companion Grasso et al. 10.1101/2025.04.07.647631 | "available online" (link not read) | 1.1 x 0.92 x 0.143 mm3 (VERIFIED, as quoted in Zheng 2025) | D, W | none; example automated segmentations only | open browsing, UNVERIFIED | 1 | Image volume, sparse reconstruction, no neuron graph |
| G10 | Mouse CA3 module (Sammons et al.) | PNAS 121:e2312281120 (2024), 10.1073/pnas.2312281120 | webKnossos wklink.org/7823 | 986 neurons in volume, 55 Pyr dendrites and 7 axons reconstructed (VERIFIED, Europe PMC full text) | D | Pyr / interneuron | browse only; other data in SI | 1 | Connection rate study, not a graph |
| G11 | LICONN expansion light-microscopy connectome | Tavakoli et al., Nature 642:398 (2025), 10.1038/s41586-025-08985-1 | janelia.org/lab/tavakoli-lab/liconn | not found | D | molecular labels | release UNVERIFIED | 1 | New modality, small volumes, no released graph seen |

## (c) Searched, nothing found

- Mouse olfactory bulb: no EM connectome of a glomerulus or mitral/tufted circuit found. Closest is
  synchrotron X-ray nanotomography of the OB (bioRxiv 10.1101/2025.04.24.650439), which has no
  synapses; the authors list targeted vEM as future work. Zebrafish OB (Wanner) is already in B.
- Mouse thalamus beyond dLGN: no TRN or higher-order nucleus EM connectome found. The 2025 TRN CLEM
  paper (J Microsc Res Tech, 10.1002/jemt.70196) is a method paper.
- Human EM 2025-2026: no second human connectome after H01 found in 4 searches. 2026 human preprints
  are X-ray / dMRI scale (bioRxiv 10.64898/2026.04.02.716198, 10.64898/2026.03.31.715699). Helmstaedter
  2025-2026 items are reviews. BossDB chandok2026 has human tissue but is an imaging-method dataset.
- Mouse cochlear nucleus (Spirou, Manis 2023, BossDB) and Maher/Briegel 2023 dLGN bouton sampling:
  input-terminal studies, not networks (fit 0).
- Older larval fly CATMAID sets on BossDB (Ohyama 2015, Gerhard 2017): partial, superseded by Winding 2023.

## Sources

- https://neuprint.janelia.org/api/dbmeta/datasets (API answer, read in full)
- https://codex.flywire.ai/api/download ; https://codex.flywire.ai/research/mouse_ca3_explorer ; https://codex.flywire.ai/app/mfas_challenge?dataset=fafb ; https://codex.flywire.ai/about_flywire
- https://global.daf-apis.com/info/ (login redirect) ; https://github.com/navis-org/navis/issues/146
- https://bossdb-metadata-snapshot.s3.amazonaws.com/mongo-data.json ; https://github.com/aplbrain/bossdb-collab-notebooks
- https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/basil/synapse_graph/cleaned_final.csv.gz (HEAD + first 20 KB)
- https://bossdb-assets.s3.amazonaws.com/files/cell_syn_annotations.zip (downloaded, 2.3 MB)
- http://konect.cc/categories/Neural/ ; /Animal/ ; /Trophic/ ; /Metabolic/
- Europe PMC REST: PMC12338526 full text (Zheng CA3), PMC10861929 full text (Sammons CA3); abstracts for 10.1101/2025.04.04.647285, 10.1101/2025.04.07.647631, 10.1101/2025.07.29.667487, 10.1101/2023.09.15.557147, 10.1016/j.cell.2016.02.033, 10.7554/eLife.26975, 10.1101/2025.09.25.678485
- Search results (snippets only): biorxiv 10.1101/2025.09.15.676403, 10.1101/2025.10.02.679866, 10.1101/2025.04.24.650439; nature.com s41586-025-08985-1; arXiv 2510.15745 (State of Brain Emulation 2025)

## Not checked

- Full CAVE datastack list as of 2026 (needs a Google-authenticated token); content of `mrgd`.
- Basil volume size, neuron count and proofreading status; whether `cleaned_final.csv` is all-layer.
- CA3 synapse table export format, total synapse and Pyr-Pyr edge counts, the exact pyr.ai / CAVE datastack name.
- Mosquito AL neuron and edge counts and whether a connectivity table is released.
- Codex synapse threshold behind the "connections" counts.
- Out of this part's scope (still open from README item 1): Aplysia, leech, Xenopus, lamprey; trained-RNN weights; fly optogenetic functional connectivity.

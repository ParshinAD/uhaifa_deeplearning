# Survey A: synapse-resolution connectomes of invertebrates and small chordates

Compiled 2026-11-17 from web sources (WebSearch, WebFetch, Europe PMC REST full texts).
Every number below was read on a source page that is linked in the notes. Anything I could not confirm on a primary or
official page is marked **UNVERIFIED**. "n/f" means the number was not found during this search; it was not estimated.

Scope: cellular (synapse-resolution) connectomes of nematodes, Drosophila (larva and adult), annelid, tunicate,
cnidarian, cephalopod and other invertebrates. Vertebrates (zebrafish etc.) are out of scope.

Fit score for us (MFAS ordering + per-neuron annotations + degree-preserving nulls + streams-x-depth plot):
5 = use next, 4 = strong, 3 = useful with caveats, 2 = marginal, 1 = not usable as a graph.

## Summary table

| # | Dataset | Species / tissue | Neurons (nodes) | Directed edges | Weight | Annotations | Access | License | Fit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Male CNS v1.0 (Berg et al. 2025 bioRxiv; Cell 2026) | D. melanogaster male, brain + optic lobes + VNC | 166,691 | n/f (weights file 1.1 GB) | synapse count (minconf 0.5) | type (11,691 types), class, side, NT prediction, fru/dsx, proofread | flat Feather on gs://flyem-male-cns, neuPrint `male-cns:v1.0` | CC BY 4.0 | 5 |
| 2 | BANC v888 (Bates ... Lee, bioRxiv 2025; Nature 2026) | D. melanogaster female, brain + VNC | 158,262 (VFB, v888); 142,719 proofread (preprint, v821) | n/f (edgelist_simple_v3 ~336 MB) | synapse count | super/class/subclass, cell type, NT (8 classes), body part / effector, proofread status | Codex (token), Harvard Dataverse, GCS bucket | CC BY 4.0 (VFB) | 4 |
| 3 | FlyWire FAFB v783 (Dorkenwald 2024; Schlegel 2024) | D. melanogaster female whole brain | 139,255 | ~15.1 M weighted edges (Schlegel); 2,701,601 thresholded (Lin) | synapse count | flow/superclass/class/type (8,453 types), side, hemilineage, NT | Zenodo 10676866, Codex | CC BY 4.0 | already used (5) |
| 4 | MANC v1.2.1 (Takemura 2024 eLife; Marin 2024) | D. melanogaster male VNC | ~23,000 (VFB: 23,759 entries); ~15.8k central + ~6.5k sensory | n/f (simple edgelist ~83 MB) | synapse count | type, hemilineage, serial homology, side, NT prediction (ACh/GABA/Glu) | neuPrint, gs://flyem-manc-exports | CC BY 4.0 | 4 |
| 5 | FANC (Azevedo 2024 Nature) | D. melanogaster female VNC | ~14,600 cell bodies | n/f | synapse count | partial; motor neuron atlas | CAVE / gs://lee-lab_female-adult-nerve-cord, CATMAID | not stated | 2 |
| 6 | Hemibrain v1.2.1 (Scheffer 2020 eLife) | D. melanogaster female, ~half central brain | ~25,000 traced | ~3 M | synapse count | type, instance, ROI | neuPrint; 44 MB compact export | UNVERIFIED (not stated on VFB) | 3 |
| 7 | Optic lobe v1.1 (Nern 2025 Nature) | D. melanogaster male right optic lobe | ~53,000 (732 types) | ~49 M connections (synapse-level, not edges) | synapse count | type, NT, spatial maps | neuPrint `optic-lobe:v1.1`, gs://flyem-optic-lobe | CC BY 4.0 | 3 |
| 8 | Larval brain (Winding et al. 2023 Science) | D. melanogaster L1 larva brain | 3,016 (480 input + 2,536 brain); Netzschleuder graph 2,956 nodes | 116,922 (multigraph, per edge type) | synapse count, split a-d / a-a / d-a / d-d | hemisphere, L/R homologue pairs, cell class, 93 clusters | Science Data S1/S2; Netzschleuder `fly_larva` csv; CATMAID L1EM | Netzschleuder page not explicit | 5 |
| 9 | Cook et al. 2019 Nature (WormWiring) | C. elegans adult hermaphrodite + male, whole animal | herm. graph 460 nodes (302 neurons + 132 muscles + 26 end organs) | herm. chemical 4,879; male chemical 5,306 (Netzschleuder "connectivity" nets) | EM-based strength (see note) | cell name (encodes side), cell type, class | WormWiring xlsx (SI 2-7, corrected July 2020), Netzschleuder `celegans_2019`, OpenWorm `cect` | n/f | 5 |
| 10 | Witvliet et al. 2021 Nature (8 stages) | C. elegans brain (nerve ring + ventral ganglion), L1 to adult | 162 neurons (L1) to 180 (adult) + 10 glia + 32 muscles | ~800 connections at birth; synapses ~1,300 to ~8,000 | synapse count | type (sensory/inter/motor/modulatory), NT, connection class (stable/variable/dynamic) | nemanode.org, Nature Supplementary Tables, OpenWorm `cect` | Zenodo part CC BY 4.0 | 5 |
| 11 | Varshney et al. 2011 PLoS CB (WormAtlas) | C. elegans somatic nervous system (compiled from White 1986) | 279 | 2,194 chemical directed; 514 gap-junction pairs | synapse count | type via name | WormAtlas NeuronConnect.xls | CC BY (paper) | 4 |
| 12 | White et al. 1986 / Durbin 1987 (N2U, JSH, JSE) | C. elegans | n/f | n/f | synapse count | type, NT (NemaNode) | NemaNode, OpenWorm `cect` | n/f | 3 |
| 13 | Brittin et al. 2021 Nature | C. elegans nerve ring L4 + adult (JSH, N2U) | n/f | n/f | contact area + Cook 2019 synapses | contactome | Zenodo 4763083, GitHub cabrittin/elegansbrainmap | n/f | 3 |
| 14 | Yim et al. 2024 Nat Commun (dauer) | C. elegans dauer nerve ring | 220 cells listed in OpenWorm (incl. some muscles) | n/f | normalized and raw synaptic weights | names | Supplementary MOESM6 xlsx, BossDB, GitHub jabae | n/f | 4 |
| 15 | Cook et al. 2025 Science (P. pacificus head) | Pristionchus pacificus, 2 adult heads (nerve ring to RVG) | n/f | n/f | synapse count (chemical only analysed) | homology to C. elegans names | Supplementary Data S3/S4, GitHub stevenjcook/cook_et_al_2025_pristionchus | n/f | 4 |
| 16 | Hong et al. 2019 eLife | P. pacificus amphid circuit | n/f | n/f | synapse count | homologs | WormWiring | n/f | 2 |
| 17 | Verasztó et al. 2025 eLife | Platynereis dumerilii 3-day larva, whole body | 966 neurons; graph 2,675 nodes (incl. 467 fragments) | 14,066 | synapse count (26,881 in-graph synapses) | cell type (202 neuronal), class, side, segment, NT/peptide for 26% of types | CATMAID catmaid.jekelylab.ex.ac.uk; source data (R tbl_graph .rds); Zenodo code | CC BY (article) | 4 |
| 18 | Ryan, Lu, Meinertzhagen 2016 eLife | Ciona intestinalis larva CNS | 177 | n/f | synapse count (6,618 synapses incl. 1,772 NMJ; plus 1,206 gap junctions) | cell type, side, position | eLife source data xlsx (Fig 16 full matrix) | CC BY (eLife) | 4 |
| 19 | Bidel et al. 2023 eLife | Octopus vulgaris vertical lobe, small volume | n/f | n/f | synapses | cell classes | n/f | CC BY (eLife) | 1 |
| 20 | Zhang et al. 2025 Curr Biol | Hydra vulgaris endodermal nerve net (half) | 20 complete neurons | no synapses found | n/a | none | Mendeley Data, BossDB | n/f | 1 |
| 21 | Aedes aegypti whole brain, ant (Ooceraea biroi), six-insect central complex | mosquito, ant, other insects | in progress | n/a | n/a | n/a | not released | n/a | 0 |

## Per-dataset notes

### Drosophila adult

**1. Male CNS (MCNS) v1.0.** Berg et al., "Sexual dimorphism in the complete connectome of the Drosophila male central
nervous system", bioRxiv 10.1101/2025.10.09.680999 (v2 Oct 2025); Cell 2026 (S0092-8674(26)00942-6). 166,691 neurons,
fully proofread, 11,691 types; 7,205 isomorphic / 114 dimorphic / 262 male-specific / 69 female-specific types vs FlyWire
(abstract). v0.9 released 2025-10-03, v1.0 released 2026-06-08. Flat files under
`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`: `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (1.1 GB,
"full connection graph"), `body-annotations-...feather` (13 MB, classes, types, sides), `body-neurotransmitters-...feather`
(42 MB), synapse tables (6.8-12.7 GB). Also neo4j dump and neuPrint (`male-cns:v1.0`, token needed). License CC BY 4.0.
The connection-weights file is segment-to-segment, so it must be filtered to proofread neurons. Total synapse and edge counts
were not found on the pages read. The sjcabs tutorial also mirrors v0.9 as `malecns_09_simple_edgelist.feather` (~3.2 GB).
Prior hierarchy work: "The organization of visual pathways in the Drosophila brain" (Cell 2026, 10.1016/j.cell.2026.08.014;
bioRxiv 10.64898/2025.12.22.696097) assigns layers with an extension of the Schlegel information-flow algorithm: of synapses
from optic lobe neurons, 45% feedforward, 43% lateral, 12% feedback (quoted from search snippet of the bioRxiv full text;
treat as UNVERIFIED until read in full).
Fit 5: same scale as FlyWire, fully proofread, CC BY flat files, full CNS with side and NT; a male-vs-female and
brain-vs-CNS comparison with our existing FlyWire run is natural.
Sources: https://male-cns.janelia.org/download/ , https://www.janelia.org/project-team/flyem/male-cns-connectome ,
https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2.full

**2. BANC (Brain And Nerve Cord).** Bates ... Lee, "Distributed control circuits across a brain-and-cord connectome",
bioRxiv 10.1101/2025.07.31.667571 (v3); Nature 2026 (s41586-026-10735-w). Female adult, GridTape TEM 4x4x45 nm, 7,010
sections. Neuron counts differ by source and version: 142,719 proofread (v821, preprint); 158,262 (v888, VFB); 142k
(flyconnecto.me table); 188,153 rows in `banc_888_meta.feather` (sjcabs tutorial, includes non-neurons/fragments?
UNVERIFIED). Abstract says ~10^8 synaptic connections. Annotations: super class / class / subclass / cell function / body
part, cell types matched to FAFB and MANC, NT prediction for 8 transmitters. Lamina and ocellar ganglion missing; ~10% of
non-optic neurons not cross-matched. Access: Codex `codex.flywire.ai/banc` (Google sign-in + token), Harvard Dataverse
(paper: 10.7910/DVN/8TFGGB; VFB lists 10.7910/DVN/7WTH1N, CC BY 4.0; the two DOIs disagree, check before citing),
GCS `gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/` with `banc_888_edgelist_simple_v3.feather`
(~336 MB) and `banc_888_meta.feather` (~48 MB). Prior flow work: "adjusted influence" (linear dynamical model) and
local sensory-effector feedback loops, not an MFAS.
Fit 4: whole CNS with body-part effectors (great for "depth" from sensors to effectors), but sparse proofreading and
version churn; edges to unproofread fragments must be dropped.
Sources: https://www.biorxiv.org/content/10.1101/2025.07.31.667571v3.full , https://www.virtualflybrain.org/docs/data/download-connectomes/ ,
https://github.com/sjcabs/fly_connectome_data_tutorial , https://codex.flywire.ai/faq

**3. FlyWire FAFB.** Dorkenwald et al. 2024 Nature 634:124 (10.1038/s41586-024-07558-y): 139,255 neurons, 5x10^7 chemical
synapses. Schlegel et al. 2024 Nature (10.1038/s41586-024-07686-5): 8,453 cell types, ~15.1 M weighted edges. Lin et al. 2024
Nature (10.1038/s41586-024-07968-y): 2,701,601 thresholded connections. Versions: v630 (first public), v783 (Oct 2023,
latest; confirmed by Codex FAQ and FlyWire guidelines, no newer FAFB snapshot as of 2026). Annotations keep updating on
top of v783 (flywire_annotations v3.0.0 aligns to MaleCNS types). Zenodo 10676866 (CC BY 4.0):
`proofread_connections_783.feather` (852 MB, with NT probabilities). Already used by us as the MFAS-challenge graph.
Prior ordering work on this graph: Borst 2024 PLoS CB 10.1371/journal.pcbi.1011904 ("smooth-index" relaxation; applied to
a 65-neuron single-column optic lobe matrix, 13-16% recurrent connections); arXiv 2506.13799 "Feedforward Ordering in Neural
Connectomes via Feedback Arc Minimization" (greedy, forward ratio 0.7524 on the FlyWire challenge graph, per search snippet);
Bader et al. 2025 (our baseline).

**4. MANC.** Takemura et al. 2024 eLife 10.7554/eLife.97769; Marin et al. 2024 eLife (reviewed preprint 97766); Cheong et al.
eLife 96084. ~23,000 neurons; ~10 M presynaptic sites and ~74 M postsynaptic densities (Janelia MANC page). Marin: ~15.8k
central + ~6.5k sensory (5,927 SN + 535 sensory ascending). NT prediction (ACh/GABA/Glu) for all neurons. Versions v1.0
(2023-06-06), v1.2 (2024-03-11), neuPrint `manc:v1.2.1`. Flat files `gs://flyem-manc-exports`; sjcabs mirror
`manc_121_simple_edgelist.feather` (~83 MB). CC BY 4.0.
Fit 4: mid-size, dense, well annotated (hemilineage, side, serial homology); good second "real" graph with VNC-type
architecture (descending in, motor out).
Sources: https://www.janelia.org/project-team/flyem/manc-connectome , https://elifesciences.org/reviewed-preprints/97766

**5. FANC.** Azevedo et al. 2024 Nature 631:360 (10.1038/s41586-024-07389-x): ~45 M synapses, ~14,600 neuronal cell bodies.
Automated segmentation with community proofreading (sparse). Access via CAVE (FANC_auto_recon wiki, fanc-fly PyPI),
`gs://lee-lab_female-adult-nerve-cord`, CATMAID skeletons on VFB. License not stated.
Fit 2: partly proofread and harder access; MANC and BANC cover the same tissue better.
Source: https://github.com/htem/FANC_auto_recon/wiki/

**6. Hemibrain v1.2.1.** Scheffer et al. 2020 eLife 9:e57443. ~25,000 traced neurons, ~20 M synapses, ~3 M edges; neurons
truncated at volume boundary (~39% of synaptic connectivity recovered per neuron on average). Compact export
`exported-traced-adjacencies-v1.2.tar.gz` (44 MB). License not stated on VFB page (UNVERIFIED; Janelia FlyEM usually CC BY).
Prior flow work: Schlegel et al. 2021 eLife 10.7554/eLife.66018, probabilistic traversal "information flow" layers
(implemented in navis; details UNVERIFIED).
Fit 3: truncation biases any feedforward score; good only as a robustness check.

**7. Male optic lobe v1.1.** Nern et al. 2025 Nature 641:1225 (10.1038/s41586-025-08746-0): right optic lobe, ~53,000
neurons, 732 types; ~49 M connections (search snippet). neuPrint `optic-lobe:v1.1`, `gs://flyem-optic-lobe`, CC BY 4.0.
Superseded by MaleCNS (same fly).
Fit 3.

### Drosophila larva

**8. Larval brain (L1EM).** Winding, Pedigo et al. 2023 Science 379:eadd9330 (10.1126/science.add9330; PMC7614541). 3,016
neurons (480 input + 2,536 brain), ~548,000 synaptic sites; 93% of brain neurons have a L/R homologue (graph matching +
manual review); 93 connectivity clusters; four edge types (a-d, a-a, d-a, d-d). Data S1/S2 contain the adjacency matrices;
CATMAID `https://l1em.catmaid.virtualflybrain.org` (pymaid). Netzschleuder `fly_larva`: 2,956 nodes, 116,922 directed
edges (multigraph, one edge per type), edge props `count`, `etype`; node props hemisphere, homologue, cell_type,
annotations, cluster; csv 495 KiB. Code: Zenodo 10.5281/zenodo.7473710, 7473718, 7474133.
Prior work: they sort neurons by a "signal flow" algorithm on the summed graph and classify each synapse as feedforward
or feedback; a-d mostly feedforward, d-a mostly feedback; d-a sorting correlates -0.59 with a-d sorting; "signal
cascade" multi-hop propagation; "highly recurrent architecture, abundant feedback from descending neurons".
Fit 5: small enough for an exact or near-exact MFAS (ILP) to measure Rocket's optimality gap; annotations (pairs,
hemisphere, classes) are exactly what our biology analyses need; edge types give a natural "axo-dendritic only" variant.
Note: no NT field in the Netzschleuder node props.
**Larval VNC.** No complete VNC/whole-CNS larval connectome has been published (searched to Nov 2026). The same EM volume
holds the whole CNS; only circuit pieces in the nerve cord exist (earlier CATMAID papers, not surveyed in detail).

### Nematodes

**9. Cook et al. 2019.** Nature 571:63 (10.1038/s41586-019-1352-7; PMC6889226). Whole-animal adult hermaphrodite and male,
chemical and gap junctions, plus muscles and end organs. WormWiring files: `SI 5 Connectome adjacency matrices, corrected
July 2020.xlsx`, `SI 2 Synapse adjacency matrices.xlsx`, `SI 3 Synapse lists.xlsx`, `SI 4 Cell lists.xlsx`, `SI 6 Cell class
lists.xlsx`, `SI 7 Cell class ... corrected July 2020.xlsx`. Netzschleuder `celegans_2019` has 12 sub-networks, e.g.
hermaphrodite_chemical 453 nodes / 4,879 directed edges; male_chemical 575 / 5,306; hermaphrodite_gap_junction 468 / 1,447
(undirected); "synapse" variants with synapse counts (herm. chemical_synapse 468 / 3,701). Edge weight "connectivity" in
SI 5 is, to my knowledge, size in serial EM sections rather than synapse count (UNVERIFIED; check the SI 5 readme).
Gap junctions are undirected: either drop them or add as two opposite arcs (which then cancel in MFAS); report separately.
Side is encoded in names (…L/…R), giving L/R pairing for free. NT identity: Wang et al. 2024 NT atlas (listed in OpenWorm
toolbox; citation details UNVERIFIED).
Fit 5: canonical small graph, exact MFAS optimum computable, two sexes.
Sources: https://wormwiring.org/pages/adjacency.html , https://networks.skewed.de/net/celegans_2019 , https://openworm.org/ConnectomeToolbox/

**10. Witvliet et al. 2021.** Nature 596:257 (10.1038/s41586-021-03778-8; PMC8756380). Eight isogenic hermaphrodites:
datasets 1-4 L1, 5 L2, 6 L3, 7-8 adult. Brain = 162 of 218 neurons at birth, 180 of 300 in adults, + 10 glia + 32 muscles;
~1,300 synapses in ~800 connections at birth to ~8,000 synapses in adults (bioRxiv v2 text). Gap junctions included on
NemaNode but "by no means exhaustive". NemaNode annotations: cell type, NT, connection class. Zenodo 5637219 (3D models,
CC BY 4.0). Prior hierarchy work: Fig. 3 "developmental increase in feedforward signalling"; feedforward defined by
sensory -> inter -> motor layers (type-based, not an ordering); share of synapses in feedforward connections rises with age.
Fit 5: direct test of whether an MFAS-optimal feedforward share also rises over development, 8 graphs with same node set.

**11. Varshney et al. 2011.** PLoS Comput Biol 7:e1001066 (10.1371/journal.pcbi.1001066). 279 somatic neurons; 6,393 chemical
synapses, 890 electrical junctions, 1,410 NMJs; 2,194 directed chemical connections; 514 gap-junction pairs. WormAtlas
`NeuronConnect.xls` (each chemical synapse listed twice, S/R rows; 2,993 distinct directed links after dedup per a third-party
note). Prior work: "signal flow view" with a closed-form hierarchy layout (Carmel-Harel-Koren), processing depth 2-3.
Later trophic analyses (Rodgers, Tino, Johnson arXiv 2202.00350: trophic coherence halfway between max coherent and
incoherent) and hourglass (PLoS CB 2019) use this graph.
Fit 4: superseded by Cook 2019 but is the reference graph most hierarchy papers used, so good for comparison.
Source: https://www.wormatlas.org/neuronalwiring.html

**12. White 1986 / Durbin 1987.** Original series N2U (adult), JSH (L4), JSE (adult tail) and the whole-animal compilation;
on NemaNode and OpenWorm `cect`. Durbin's thesis analysis of N2U/JSH is cited by OpenWorm. Counts not re-verified.
Fit 3 (historical; Cook 2019 re-scored the same series).

**13. Brittin et al. 2021.** Nature 591:105 (10.1038/s41586-021-03284-x). Volumetric contactome of the L4 (JSH) and adult (N2U)
nerve ring, combined with Cook 2019 synapses; "core" reference connectome. Zenodo 4763083, GitHub cabrittin/elegansbrainmap.
Fit 3: useful mainly to test "contact-matched" nulls (adjacency as a null constraint).

**14. Yim et al. 2024 (dauer).** Nat Commun 15:1546 (10.1038/s41467-024-45943-3). Dauer nerve ring, 364 TEM sections at 50 nm,
CNN synapse detection with proofreading; postsynaptic weights from a diffusion model (so weights may be fractional).
Gap junctions "available upon request". Supplementary `41467_2024_45943_MOESM6_ESM.xlsx` (non-normalized), BossDB
`yim_choe_bae2023`, GitHub jabae/Yim-Choe-Bae-et_al-2024. OpenWorm lists 220 cells.
Fit 4: an alternative developmental stage with the same naming; weights not plain counts.

**15. Pristionchus pacificus, Cook et al. 2025.** Science (10.1126/science.adx2143; bioRxiv 10.1101/2024.06.13.598904;
PMC12330220). Two adult heads from nose tip to retrovesicular ganglion; contactome (Data S1, S2) and chemical connectome
(Data S3, S4); gap junctions annotated but excluded from analysis as ambiguous. Code GitHub
stevenjcook/cook_et_al_2025_pristionchus; EM at BossDB bumbarger2013. Homology to all C. elegans head neurons, so the same
annotations transfer. Neuron/edge counts not found in the text read.
Fit 4: rare cross-species comparison at fixed homologous node set.
Also: Hong et al. 2019 eLife 8:e47155 (amphid circuit only, WormWiring) fit 2; Bumbarger et al. 2013 pharynx (20 neurons)
fit 1.

### Other invertebrates and small chordates

**17. Platynereis dumerilii.** Verasztó, Jasek et al. 2025 eLife 13:RP97964 (10.7554/eLife.97964; PMC12387772). 9,162 cells
with soma in the volume; 966 neurons in 202 types (+92 non-neuronal types). Final synaptic graph: 2,675 nodes (incl. 467
fragments) and 14,066 directed edges formed by 26,881 synapses (nodes with <3 connections removed); density 0.00197. No gap
junctions identified. Annotations: class (sensory/inter/motor, effector), cell type, side (soma position), segment 0-3,
germ layer, NT/neuropeptide for 53 (26%) neuronal types. Data: CATMAID `https://catmaid.jekelylab.ex.ac.uk` (API via
catmaid/natverse), Figure 2 source data = full graph as R `tbl_graph` binary; code Zenodo 10.5281/zenodo.15830426.
Prior work: describe overall organisation as feed-forward from sensors through interneurons to effectors, shallow
sensory-motor paths, mushroom bodies "multilayer, parallel, feed-forward" with weak recurrence.
Fit 4: a third phylum with whole-body sensors-to-effectors; size fine for exact MFAS; note it includes non-neuronal
effector nodes.

**18. Ciona intestinalis larva.** Ryan, Lu, Meinertzhagen 2016 eLife 5:e16962 (10.7554/eLife.16962; PMC5140270). 177 CNS neurons,
6,618 synapses incl. 1,772 NMJ and 1,206 gap junctions; 922 polyadic. Annotations: cell type, side (left/right asymmetry is
a main result), position (rostro-caudal order). Full connectivity matrix as `elife-16962-fig16-data1.xlsx` (Figure 16
source data 1); neuron summary `elife-16962-fig3-data1.xlsx`. Prior work: shortest sensory-to-motor paths are three-synapse
arcs; feedforward triplet motifs; reciprocal connection share 0.39.
Fit 4: only chordate in scope, tiny, matrix in one xlsx; a later PNS paper (J Comp Neurol, 10.1002/cne.24353) adds peripheral
neurons. No second individual published (searched).

**19. Octopus vulgaris vertical lobe.** Bidel et al. 2023 eLife 12:e84257 (10.7554/eLife.84257). Small volume of the VL (memory
centre); network described as feedforward from SFL inputs to outputs. Follow-up bioRxiv 10.1101/2025.01.29.635406. Neuron and
edge counts n/f; a fragment with truncated cells. Fit 1.

**20. Hydra vulgaris.** Zhang et al. 2025 Curr Biol 35:5490 (endodermal nerve net, ~half of it): 20 complete neurons, no
chemical synapses found ("handshake" contacts). Data Mendeley 10.17632/5p4ry5pdk2.1, BossDB 10.60533/BOSS-2025-08G4. Fit 1 (no
directed graph). Nematostella, Trichoplax, sponges: no synaptic connectome found.

**21. In progress, not available.** Aedes aegypti female whole brain (Wellcome grant, Jefferis/Marin with Lee, Younger);
clonal raider ant Ooceraea biroi whole-brain TEM (Rockefeller thesis 2025, connectome ongoing); Cambridge group also working
on "another fly species"; a 2026 bioRxiv (10.64898/2026.02.25.708065) imaged the central complex of six insect species with
local synapse-level connectomes (species and data UNVERIFIED). No Daphnia, honeybee, bumblebee, Megaphragma connectome
graph found.

## Prior work on ordering / hierarchy on these graphs (quick index)
- Durbin 1987 (thesis), Varshney 2011 (signal-flow layout): C. elegans.
- Witvliet 2021: type-layered feedforward share rises over development.
- Rodgers, Tino, Johnson 2022 (arXiv 2202.00350); "Strong connectivity in real directed networks" PNAS 2023
  (10.1073/pnas.2215752120): trophic levels/coherence of C. elegans.
- "Hierarchy index" arXiv 2308.08898 (C. elegans layers); hourglass organisation PLoS CB 2019.
- Winding 2023: signal-flow sorting + feedforward/feedback synapse fractions + signal cascades (larva).
- Schlegel 2021: information-flow layers (hemibrain); extended in Cell 2026 visual pathways (MaleCNS).
- Borst 2024 PLoS CB: smooth-index relaxation for matrix seriation (closest to Rocket's idea).
- arXiv 2506.13799: MFAS on the FlyWire challenge graph.
- Verasztó 2025: qualitative feed-forward organisation (Platynereis).

## Not verified / open
- Edge counts for MaleCNS, BANC, MANC flat files (need a download to count).
- Exact meaning of Cook 2019 SI 5 weights.
- BANC Dataverse DOI (8TFGGB vs 7WTH1N) and neuron count per version.
- Witvliet per-dataset neuron/edge counts (get from NemaNode or `cect`).
- Licenses for hemibrain, FANC, L1EM, WormWiring, NemaNode.

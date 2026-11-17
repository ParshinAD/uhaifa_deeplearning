# E. Prior and concurrent work: feedforward ordering, hierarchy and feedback in connectomes (1987-2026)

Scope: papers that order neurons or areas, compute a feedforward/feedback share, signal-flow depth, trophic levels,
ranks, or recurrence statistics on connectomes. For each: dataset, method, key numbers, availability, relation to our work
(FlyWire ordering, 84.26% forward weight, feedback local and enriched in inhibitory/modulatory transmitters, scale-dependent
difference vs degree-matched nulls, type-dcSBM null overshoots the real brain).

Status: "verified" means I read the number in the paper text (PDF/HTML) during this search. "snippet" means it came from a search
summary of the paper and was not read in the source. UNVERIFIED means not confirmed. No number below is my own estimate unless
explicitly marked "our arithmetic".

Search done 2026-11-17 with WebSearch/WebFetch; PDFs of Durbin 1987 and Varshney 2011 extracted locally with pdftotext.

---

## 1. Directly comparable: global ordering that maximizes forward synapses

### 1.1 Durbin 1987, PhD thesis (C. elegans nerve ring) - the earliest direct precedent
- Citation: Durbin RM (1987). Studies on the development and organisation of the nervous system of Caenorhabditis elegans. PhD thesis, Univ. Cambridge. PDF: https://www.sanger.ac.uk/theses/durbin-thesis.pdf (also wormatlas.org/Durbin/durbinthesis.pdf).
- Data: chemical synapses of the nerve ring, reconstructed from two animals (series H and U), pooled into **neuron classes** (not single cells). Gap junctions ignored for the ordering.
- Method: "vertically order the neurons so that as many as possible of the synapses point downwards" with a sorting algorithm (appendix). This is exactly weighted MFAS, at class level.
- Numbers (verified, chapter 8):
  - stringent synapse set (seen in at least 3 of 4 places): >95% downward, **140 of 2890 (4.8%) upward**;
  - of the 140 upward, 116 are in reciprocal connections; only 24 persistently seen synapses in non-reciprocal upward connections; "no case in the whole nerve ring of strong, consistently seen, indirect synaptic feedback";
  - averaged (all) data: 3898 synapses/animal; 386 (9.9%) upward with the stringent-data order; **328 (8.4%) upward** when optimized for the averaged data (i.e. ~91.6% forward, our arithmetic);
  - processing depth: on average 3.5 chemical synapses from sensory neurons to head muscles (range 1 to 16; class averages up to 5.9);
  - feedback "largely restricted to affecting small groups of neuronal classes, or modules".
- Available: thesis PDF; the ordering is a figure (8.1), not a file. Data reused in later C. elegans connectomes.
- Relation: **same question, same objective, 1987.** Durbin's "feedback is mostly reciprocal and local to modules" is a precursor of our "feedback is local" claim. Must be cited. His numbers (4.8% / 8.4% feedback, class level, weak synapses removed) are the only published FAS-type C. elegans numbers I found. Note the effect of thresholding: weak synapses raise the feedback share (4.8% -> 8.4%), relevant to how we threshold FlyWire.

### 1.2 FlyWire Minimum Feedback Arc Set Challenge (2024) and its leaderboard
- URL: https://codex.flywire.ai/app/mfas_challenge?dataset=fafb
- Data: FlyWire FAFB graph, 136,648 neurons, 5,657,719 weighted edges, total weight 41,912,141 (same file we use).
- Numbers (snippet, from leaderboard via search and Vahidi 2025):
  - greedy baseline ~29 M;
  - official winner Ellis-Joyce, Both, Turaga (TuragaLab), 2024-10-08: 35,435,948 (85%); later 35,436,406;
  - Bader/Du NJIT team: 35,231,406 at deadline, then 35,459,212 (2024-11-16); Rocket-Crane paper reports 35,459,266 (84.60%) (verified in our local copy of the paper);
  - Zheng, Tang, Okubo (CIBR): 35,374,656 (84%); intermediate 30,136,267 (71.9%) from in/out-degree sort, 33,037,112 (78.8%) after RASstar;
  - Vahidi & Koutis 2025-06-02: 35,462,925 (84.61%); 2026-01-09: **35,463,823 (~84.61%)** = current top (snippet).
- Code: github.com/TuragaLab/connectome_reordering; github.com/CIBR-Okubo-Lab/min_feedback_final; github.com/SoroushVahidi/minimum-feedback-challenge.
- Relation: our 84.26% is ~0.35 pp below the best published score on the **same graph**. Any COSYNE number must say which order it uses and that it is not the record. The challenge is purely an optimization contest; none of the entries do biology or nulls (as far as found).

### 1.3 Bader et al. 2025, Rocket-crane
- Citation: Bader DA, Ellis-Joyce J, Both G, Turaga S, Asoori Sriram H, Chinthalapudi S, Du Z (2025). Rocket-crane algorithm for the Feedback Arc Set problem. Soc Netw Anal Min 15:68. DOI 10.1007/s13278-025-01491-2.
- Data: FlyWire challenge graph only (verified in local PDF; no other connectome).
- Numbers: Rocket 80.05% fast; 84.60% (35,459,266) after Crane MIP ~20 days.
- Relation: algorithmic baseline; no biology.

### 1.4 Vahidi 2025 and follow-ups
- Vahidi S (2025). Feedforward Ordering in Neural Connectomes via Feedback Arc Minimization. arXiv:2506.13799. https://arxiv.org/abs/2506.13799
  - Data: FlyWire challenge graph. Largest SCC 126,840 nodes (92.8%); 9,626 SCCs, 9,503 singletons (verified via HTML).
  - Numbers: adaptive greedy 0.7524; full pipeline 35,462,925 / 41,912,141 = 0.8461; backward weight 6,449,216; backward edges 1,156,812.
  - Biology: none beyond saying back edges are mostly short with heavy tail; claims "biological plausibility" without neuron-level analysis.
- Vahidi S, Koutis I (2026). Interval-Based Refinement Algorithms for Large-Scale Weighted Feedback Arc Set. SSRN 6221201 / arXiv:2512.22221. FlyWire only (snippet). Claims best score to date.
- Vahidi S (2026). Incumbent-Protected SCC-Neighborhood Search (IPSNS). SSRN 6281222. 33 standard FAS benchmarks, not connectomes (snippet).
- "An Audited Benchmark for Minimum Weighted Feedback Arc Set" (Research Square rs-11436856, ~Oct-Nov 2026). Content not readable; whether it includes connectomes is UNVERIFIED.
- Relation: no novelty risk for biology; the title "Feedforward ordering in neural connectomes" is close to our framing, so cite it.

### 1.5 Borst 2024, smooth-index seriation
- Citation: Borst A (2024). Connectivity Matrix Seriation via Relaxation. PLoS Comput Biol 20(2): e1011904. DOI 10.1371/journal.pcbi.1011904. Code+data: https://github.com/axelborst/Matrix_Reordering
- Data: one optic-lobe column of Drosophila, 65 neurons (from Shinomiya et al.-type column data), binarized at >=4 synapses; synthetic matrices up to 10,000 nodes.
- Method: continuous relaxation of positions + L-BFGS-B, minimizing recurrent (upper-triangle) connections; compares to FAS heuristic, out-degree sort, reverse Cuthill-McKee.
- Numbers (verified via page): 187 connections, 50 reciprocal (=25 pairs, minimum 25 recurrent); best order leaves 30 recurrent; "at least 13% and at most 16% of the connections are recurrent" (=> ~84% feedforward, our arithmetic, unweighted, thresholded). Some lamina-medulla links (e.g. L5 -> Mi1) are recurrent with high probability.
- Relation: method is a sibling of Rocket (continuous positions). The 84% for a 65-neuron column is a coincidence-level match to our 84.26% but not comparable (unweighted, thresholded, tiny). Scales only to ~20k neurons on a desktop.

### 1.6 Borst & Helmstaedter? / Borst 2023 TechSights
- Borst A, (co-author) (2023). Connecting Connectomes to Physiology. J Neurosci 43(20):3599. Notes the nilpotence test and that thresholded optic-lobe connectivity is largely feedforward; cites Carmel 2004, Seung 2009, Varshney 2011 as prior ordering methods (snippet). Co-author name UNVERIFIED.

### 1.7 Seung 2009
- Seung HS (2009). Reading the book of memory: sparse sampling versus dense mapping of connectomes. Neuron 62:17-29. DOI 10.1016/j.neuron.2009.03.020.
- Quadratic (spectral) ordering of neurons to reveal chains; conceptual precedent, toy data. No real-connectome feedback share.

---

## 2. Continuous hierarchy / depth measures on C. elegans

### 2.1 Varshney, Chen, Paniagua, Hall, Chklovskii 2011
- Citation: PLoS Comput Biol 7(2): e1001066. DOI 10.1371/journal.pcbi.1001066. arXiv:0907.2373. Data/code: https://github.com/lrvarshney/elegans ; also OpenWorm ConnectomeToolbox.
- Data: hermaphrodite, 279 neurons, 2194 directed chemical connections (verified); gap-junction network separately.
- Method: vertical coordinate = "position in the signal flow hierarchy", closed-form minimizer of a frustration energy (Koren 2005; Carmel, Harel, Koren 2004). Not an FAS.
- Numbers (verified): giant SCC 237 neurons, one SCC of 2, 40 not strongly connected; "A strongly connected giant component as small as in the chemical network is not likely in a random network"; "signal flow depth ... typically 2-3" citing Durbin 1987; between-category connections mostly sensory->inter->motor (Table S4). No feedforward percentage reported in text.
- Relation: standard dataset and the origin of the "signal flow" layout used later (graspologic). Their SCC-vs-random observation is a degree-null comparison at small scale.

### 2.2 Carmel, Harel, Koren 2004
- Citation: IEEE TVCG 10(1):46-57. DOI 10.1109/TVCG.2004.1260757.
- Hierarchy energy (graph-drawing y-coordinates) for any digraph. No connectome in this paper (search found none). Used by Varshney 2011 and (probably, UNVERIFIED) by Winding 2023 "signal flow".

### 2.3 Johnson & Jones 2017 (trophic coherence, includes C. elegans)
- Citation: Johnson S, Jones NS (2017). Looplessness in networks is linked to trophic coherence. PNAS 114:5618. DOI 10.1073/pnas.1613786114. arXiv:1505.07332.
- C. elegans neuronal row (snippet from table): N=297, B=3 basal, <k>=7.90, q=1.49, q/q~=0.42, tau=2.17 (high-feedback regime). Coherence measured against a degree-matched expectation.
- Relation: trophic coherence vs random expectation on C. elegans; a continuous counterpart of our real-vs-null comparison.

### 2.4 MacKay, Johnson, Sansom 2020 and Rodgers/Tino/Johnson
- MacKay RS, Johnson S, Sansom B (2020). How directed is a directed network? R Soc Open Sci 7:201138. DOI 10.1098/rsos.201138. Defines F0 (improved trophic levels). **No connectome analysed** (verified: food web, economies, translations, yeast).
- Rodgers N, Tino P, Johnson S (2022). Network hierarchy and pattern recovery in directed sparse Hopfield networks. Phys Rev E 105:064304. States C. elegans trophic coherence lies "exactly half way" between fully feedforward and fully incoherent (snippet; exact F0 UNVERIFIED).
- Rodgers, Tino, Johnson (2023). Strong connectivity in real directed networks. PNAS 120:e2215752120. Backward edges vs trophic order drive SCC size. Whether a connectome is in the dataset: UNVERIFIED.
- Kissack, Drysdale, Johnson (2026). Trophic structure predicts seizure propagation in brain network models. arXiv:2608.12382. Brain network models, not neuron-level connectomes (snippet).
- Relation: our repo already measures trophic levels on FlyWire (77.78% forward with trophic order, F0 = 0.71, per layering/RESEARCH_NOTES.md and biology/studies/untapped.md). **I found no published trophic analysis of FlyWire or hemibrain** (searched; none returned). Low novelty risk there.

### 2.5 Pathak, Menon, Sinha 2024
- Citation: A hierarchy index for networks in the brain reveals a complex entangled organizational structure. PNAS 121(27): e2314291121. DOI 10.1073/pnas.2314291121. arXiv:2308.08898.
- Data (verified via HTML): macaque CoCoMac (266 areas, 2,602 directed links), human DTI (188 regions, undirected), C. elegans (279 neurons).
- Method: hierarchy index H = observed minus degree-expected links between adjacent levels; optimized partition into levels.
- Numbers: macaque 16 levels, human 12 levels, C. elegans 12 levels (diameter 7, 3 modules). No numeric H or feedback share reported in text; no separate random-null test (H has a degree expectation built in).
- Relation: "hierarchy + modularity entangled" claim overlaps loosely with our "feedback is local (modular)". Not Drosophila.

### 2.6 Sabrin, Wei, van den Heuvel, Dovrolis 2020 (hourglass)
- Citation: PLoS Comput Biol 16(2): e1007526. DOI 10.1371/journal.pcbi.1007526. Follow-up weighted version: Batta et al. 2021 PLoS ONE 16:e0249846.
- Data: C. elegans 279 neurons; ~90 sensory, ~80 inter, ~120 motor.
- Method: separate feedforward sensory->motor paths from feedback (several routing schemes), find tau-core.
- Numbers (snippet): waist of 10-15 interneurons; randomized networks do not show hourglass; ~8x lower processing cost.
- Relation: real vs randomized comparison of feedforward structure in C. elegans; complementary.

### 2.7 Witvliet et al. 2021 (development)
- Citation: Connectomes across development reveal principles of brain maturation. Nature 596:257-261. DOI 10.1038/s41586-021-03778-8.
- Data: 8 isogenic C. elegans brains (180 brain neurons) from L1 to adult. Downloadable (nemanode.org; also OpenWorm toolbox).
- Method: feedforward/feedback/recurrent defined by **cell-type layers** (sensory -> inter -> motor), not by an optimized order.
- Finding (snippet): brain becomes progressively more feedforward and modular with age; stable feedforward connections strengthen more than feedback/recurrent (p=0.0015, 0.0032). Exact shares UNVERIFIED.
- Follow-up: bioRxiv 10.64898/2026.06.12.730308 "Developmental Continuity of Brain Network Core Organization in C. elegans": giant SCC grows with development (snippet).
- Relation: **ideal extra datasets** for our pipeline (8 graphs, same species, development axis). Applying MFAS would turn their categorical claim into an optimized number.

### 2.8 Cook et al. 2019
- Citation: Whole-animal connectomes of both C. elegans sexes. Nature 571:63-71. DOI 10.1038/s41586-019-1352-7. Data downloadable (WormWiring; Netzschleuder "celegans_2019"; OpenWorm toolbox).
- No feedforward/hierarchy numbers found in snippets (UNVERIFIED whether the paper reports any). Standard dataset to run on.

### 2.9 Other C. elegans recurrence work
- "Sensory recurrence and convergent integration in the C. elegans connectome", bioRxiv 10.64898/2026.10.29.762790 (posted 2026-11-03; authors not seen). Feedback coefficient from KMS framework; **~3x more recurrence than strength-matched random networks**, excess concentrated on sensory neurons (random puts it on hubs); replicates across 5 reconstructions (snippet). **Concurrent real-vs-random recurrence claim** in a different species and different measure. Read before COSYNE.
- Signed motif analysis of the C. elegans neuronal network (bioRxiv 10.1101/2025.01.09.632090; PMC13374236): positive feedforward and negative feedback loops overrepresented (D2 negative feedback z=5.14). Relevant to "feedback enriched in inhibition" (snippet).
- Chatterjee & Sinha 2007 (hierarchical structure of the worm network) - cited by Varshney; not checked.
- Milo et al. 2002 Science 298:824: feed-forward loop overrepresented in C. elegans neurons vs randomized nets (snippet).

---

## 3. Drosophila larva

### 3.1 Winding, Pedigo et al. 2023
- Citation: The connectome of an insect brain. Science 379:eadd9330. DOI 10.1126/science.add9330. bioRxiv 10.1101/2022.11.28.516756. Data: https://github.com/brain-networks/larval-drosophila-connectome (and supplementary).
- Data: 3016 neurons (480 input + 2536 brain), ~548,000 synapses; four edge types (a-d 66.4%, a-a 26.4%, d-d 5.4%, d-a) (verified in preprint text).
- Methods: "walk-sort" (median order in random walks from sensory to output), "signal flow" ordering (citation numbers only; probably Varshney/Carmel, UNVERIFIED), signal cascades.
- Numbers (verified in preprint text): **41% of brain neurons are recurrent** (receive multihop signal back from downstream partners); PNs and KCs ~1.2% and 0.1% recurrent partners; DANs 57% (most recurrent). a-d graph "most feedforward", a-a and d-d "roughly equal feedforward and feedback", d-a biased to feedback; exact fractions are only in Fig. 2F.
- **The "66% feedforward" figure in our brief is UNVERIFIED and probably a mix-up**: the 66% values in the paper are a-d share of edges (66.4%) and DNs-SEZ synapsing on brain neurons (66%). I did not find a feedforward share of 66%.
- Relation: **comparable dataset of the same genus** (3k neurons; MFAS solvable much closer to optimal). Their "modulatory DANs are the most recurrent" overlaps our "feedback enriched in modulatory transmitters". Novelty risk moderate for the qualitative claim; our edge is the whole-adult-brain optimized order + null ladder.
- Related: Eschbach et al. 2020 Nat Neurosci 23:544 (multilevel feedback onto DANs, larva); Eichler et al. 2017 Nature 548:175 (MB connectome, recurrence).
- Betzel lab: Hierarchical communities in the larval Drosophila connectome, PNAS 2024, DOI 10.1073/pnas.2320177121 (rich club, communities; no ordering).

---

## 4. Adult Drosophila (FlyWire, hemibrain, male CNS, BANC)

### 4.1 Dorkenwald et al. 2024 and Schlegel et al. 2024 (FlyWire flagship papers)
- Dorkenwald S et al. Neuronal wiring diagram of an adult brain. Nature 634:124-138. DOI 10.1038/s41586-024-07558-y.
- Schlegel P et al. Whole-brain annotation and multi-connectome cell typing of Drosophila. Nature 634:139-152. DOI 10.1038/s41586-024-07686-5. Annotations: github.com/flyconnectome/flywire_annotations.
- Method: probabilistic "information flow" traversal from seed (afferent) neurons (navis implementation; origin Schlegel et al. 2021 eLife 10:e66018, olfactory layers in hemibrain), ranks as percentiles per modality.
- Numbers: percentile ranks per modality (Extended Data); no global feedforward share. Schlegel 2024 Fig. 1f: superclass-level afferent->intrinsic->efferent flow chart.
- Relation: their ranks are the obvious external comparator for our order (we already recover the sensory->motor axis). Downloadable via Codex.

### 4.2 Lin et al. 2024
- Citation: Network statistics of the whole-brain connectome of Drosophila. Nature 634:153-165. DOI 10.1038/s41586-024-07968-y. Code/data Zenodo 10.5281/zenodo.10676865, 10.5281/zenodo.12572929.
- Data: 127,978 neurons, 2,613,129 thresholded connections.
- Numbers (snippet): reciprocity 0.138, above ER, configuration and spatial nulls; 77,607 neurons in reciprocal pairs; rich club 30% (40,218 neurons, degree > 37); 676 broadcasters, 638 integrators; rich-club mean percentile rank 44% relative to sensory. Three-node **feedforward motifs (#1-3) under-represented vs ER and configuration nulls, recurrent motifs over-represented**; feedforward loops mostly cholinergic; 3-cycles tend to contain an inhibitory edge; over-represented reciprocal pairs are E-I.
- Relation: **closest published FlyWire null-model work.** Their "highly recurrent vs ER/CFG" at motif level is consistent with our neuron-level result; their "cycles carry inhibitory edges" overlaps our "feedback enriched in inhibitory transmitters". Not an ordering, so our feedback share and the type-SBM null are new relative to them. Must cite.

### 4.3 Hoeller, Zhao, Nern, Rogers, Romani, Reiser 2025/2026 - visual pathways
- Citation: The organization of visual pathways in the Drosophila brain. bioRxiv 10.64898/2025.12.22.696097; Cell 2026 (DOI 10.1016/j.cell.2026.08.014 per search snippet, page 403 to me).
- Data: male CNS connectome (optic lobe ~52k neurons, ~700 types) into central brain.
- Method: Schlegel information-flow layers from photoreceptors (threshold 10%), connection = feedforward if > +0.5 layer, lateral within +/-0.5, feedback if < -0.5.
- Numbers (snippet, confirmed in two sources): optic-lobe connections **45% feedforward, 43% lateral, 12% feedback**; visual inputs 97% feedforward; VCN connections 53% feedback; **feedforward 81% excitatory, feedback 69% inhibitory**; shallow ~3-layer hierarchy; compared to mouse visual cortex (~0.43, 0.36, 0.21; likely citing Ding et al. 2025 Nature, UNVERIFIED).
- Relation: **highest novelty risk for the "feedback is inhibitory" claim**, though restricted to optic lobe, uses flow layers not an optimized order, and male CNS. Our claim is whole-brain and with an optimal order; frame it as consistent and extended.

### 4.4 Grindrod, Lambiotte, Sahasrabuddhe 2024
- Citation: Modularity, Hierarchical Flows and Symmetry of the Drosophila Connectome. arXiv:2412.13202.
- Data: FlyWire Codex (accessed 2024-11-11), central superclass, edges < 5 synapses removed: 32,272 nodes, 849,981 unweighted edges.
- Method: Infomap modules, then SpringRank (equivalent to Hodge / trophic) on the module graph; null = randomize edge directions keeping counts between module pairs.
- Result: significant module-level hierarchy (broadcaster -> integrator modules); left-right symmetric ranks. No p-values or flow fractions reported.
- Relation: module-level hierarchy vs a direction-randomized null. Overlaps our "scale matters (neuron vs area)" theme. Cite.

### 4.5 Other FlyWire/hemibrain network papers (no ordering share)
- Kunin AB, Guo J, Bassler KE, Pitkow X, Josic K (2023). Hierarchical Modular Structure of the Drosophila Connectome. J Neurosci 43:6384. DOI 10.1523/JNEUROSCI.0134-23.2023. Hemibrain, modularity hierarchy (nested modules), not directed hierarchy.
- Pospisil DA et al. (2024). The fly connectome reveals a path to the effectome. Nature 634:201. DOI 10.1038/s41586-024-07982-0. Eigen-decomposition of FlyWire; many small independent circuits. No feedforward share.
- Betzel, Puxeddu, Seguin, Misic (2024-25). Cascades and convergence: dynamic signal flow in a synapse-level brain network. bioRxiv 10.1101/2024.12.04.624189. Spreading model on FlyWire (snippet). Earlier mesoscale version: Worrell et al. 2017 Netw Neurosci 1:415.
- Sulyok, Balogh, Palla (2026). Network geometry of the Drosophila brain. arXiv:2602.16417. Hyperbolic embedding; not about feedback.
- Bates, Phelps, Kim, Yang et al. (2026). Distributed control circuits across a brain-and-cord connectome (BANC). Nature, DOI 10.1038/s41586-026-10735-w. 158,262 neurons, brain+VNC; local sensory-effector feedback loops, "influence" metric, median 3 hops sensory to AN/DN. **BANC is a natural second whole-CNS dataset** (code: github.com/htem/BANC-project; Dataverse doi:10.7910/DVN/7WTH1N).
- Cheong, Eichler, Sturner et al. 2024 eLife (MANC): DN->MN feedforward flow, triad classification into feedforward/feedback/recurrent. Takemura et al. 2024 eLife: MANC dataset with predicted transmitters. Possible dataset.
- Matsliah et al. 2024 Nature (FlyWire optic lobe cell types) and Nern et al. 2025 (optic lobe): not checked for ordering numbers.

---

## 5. Other species

- **Ciona larva**: Ryan, Lu, Meinertzhagen 2016 eLife 5:e16962, DOI 10.7554/eLife.16962. 177 CNS neurons. No feedforward share found. Small dataset, suitable for exact MFAS.
- **Platynereis larva**: Verasztó et al. 2025 eLife 13:97964 (whole body, >9000 cells, 202 neuron types; code Zenodo 10.5281/zenodo.11811680; CATMAID). No ordering analysis found. Usable dataset.
- **Zebrafish**: Svara et al. 2022 Nat Methods 19:1357 (tectum SIN circuit "overall feedforward"); larval zebrafish hindbrain resource bioRxiv 10.1101/2025.06.10.658982 (915 neurons, recurrent excitatory network). No global ordering.
- **Mouse cortex, MICrONS 2025**: MICrONS Consortium, Nature 640:435 (DOI 10.1038/s41586-025-08790-w; >200k cells, 523 M synapses); Ding et al. Nature 640:459 (like-to-like rule incl. feedforward/feedback between areas); Schneider-Mizell et al. 2025 Nature (inhibitory targeting). No neuron-level MFAS found. Mouse visual cortex ff/lateral/fb ~0.43/0.36/0.21 quoted by Hoeller et al. (source UNVERIFIED).
- **Human H01**: Shapson-Coe et al. 2024 Science 384:eadk4858. No recurrence/ordering analysis found.
- **Mesoscale cortex hierarchies**:
  - Markov et al. 2014 J Comp Neurol 522:225 (DOI 10.1002/cne.23458): macaque, SLN-based hierarchy; FB pathways about twice as numerous as FF; FF fewer, stronger, span fewer levels.
  - Harris, Mihalas et al. 2019 Nature 575:195 (DOI 10.1038/s41586-019-1716-z): mouse Allen data, layer-termination patterns classified FF/FB, 37 cortical areas + 24 thalamic nuclei, "shallow hierarchy".
  - These define FF/FB by laminar anatomy, not by an order; not directly comparable to a synapse-weight share. Our mouse file (data/table_mouse.txt) is a different object; I found no FAS ordering of Oh 2014 Allen mesoscale data.
- **Kale, Zalesky, Gollo 2018** Netw Neurosci 2:259: effect of ignoring directionality on 6 directed brain networks (cat, mouse, C. elegans, ...). No hierarchy/feedback share.
- Song et al. 2005 PLoS Biol 3:e68: reciprocal connections over-represented in rat L5 pairs (well known; not verified in this search).
- Sporns & Kotter 2004 PLoS Biol 2:e369: motifs in brain networks.
- Zhao & Zhou 2017 Chin Phys B 26:078901 (arXiv:1612.05347): optimal node hierarchy = minimum FAS, BP + annealing; theory, maybe tested on generic networks.
- arXiv:2409.16443: asymptotic lower bounds for FAS in random graphs. Useful as analytic reference for our random nulls.

---

## 6. Summary tables

### 6.1 Comparable published "forward share" numbers
| Dataset | Unit | Method | Forward share | Source |
|---|---|---|---|---|
| C. elegans nerve ring, stringent | neuron classes, synapse counts | sort to maximize downward synapses | 95.2% (4.8% up) | Durbin 1987 (verified) |
| C. elegans nerve ring, all synapses | classes | same, re-optimized | 91.6% (8.4% up) | Durbin 1987 (verified, our arithmetic for 91.6) |
| Fly optic column, 65 neurons, >=4 syn | neurons, binary | relaxation seriation | ~84-87% (13-16% recurrent) | Borst 2024 (verified) |
| FlyWire FAFB, 136,648 neurons | neurons, weighted | MFAS heuristics | 84.60-84.61% (record 35,463,823) | Challenge / Bader 2025 / Vahidi 2025-26 |
| Fly optic lobe (male CNS) | synapses | flow layers +/-0.5 | 45% FF, 43% lateral, 12% FB | Hoeller et al. 2025/26 (snippet) |
| Mouse visual cortex | connections | (unknown) | ~0.43 / 0.36 / 0.21 | quoted in Hoeller et al.; UNVERIFIED |
| Larval fly brain | neurons | multihop cascades | 41% of neurons recurrent | Winding 2023 (verified) |
| C. elegans | trophic | q/q~ = 0.42 (more coherent than random) | n/a | Johnson & Jones 2017 (snippet) |

Not comparable to each other: unit (class vs neuron), weighting (binary vs synapse counts), thresholds, and definition (optimized order vs flow layers with a lateral band). Our 84.26% should be compared only with the FlyWire challenge row.

### 6.2 Datasets the field uses for this question (candidates to run our pipeline on)
1. C. elegans: White 1986 / Varshney 2011 (279 neurons), Cook 2019 (both sexes, whole animal), Witvliet 2021 (8 developmental stages). Small; exact or near-exact MFAS feasible.
2. Larval Drosophila: Winding 2023 (3016 neurons), GitHub repo.
3. Adult Drosophila: FlyWire (ours), hemibrain v1.2.1, male CNS (Janelia, 2025), BANC (2026), MANC (VNC only).
4. Ciona larva (177 neurons), Platynereis larva (whole body).
5. Mesoscale: macaque (Markov 2014), mouse Allen (Oh 2014, Harris 2019), CoCoMac.
6. Mouse cortex MICrONS (synapse-level, partial volume; truncation will inflate feedback ambiguity).

### 6.3 Novelty risks for the COSYNE abstract
- **Feedback local / modular**: Durbin 1987 already says nerve-ring feedback is mostly reciprocal and confined to small modules. Cite; our novelty is whole-brain scale, weighted MFAS, and the null ladder.
- **Feedback enriched in inhibitory**: Hoeller et al. 2025/26 (optic lobe, 69% of feedback synapses inhibitory) and Lin et al. 2024 (3-cycles contain inhibitory edges; E-I reciprocal pairs over-represented). Our claim must be stated as whole-brain, optimal-order, and consistent with these.
- **Feedback enriched in modulatory**: Winding 2023 (DANs most recurrent, 57%), Eschbach 2020. Same caution.
- **Real vs random recurrence**: Lin 2024 (reciprocity and motifs vs ER/CFG/spatial nulls), concurrent C. elegans preprint (3x recurrence vs strength-matched random, Nov 2026), Grindrod et al. 2024 (module hierarchy vs direction-shuffled null), Sabrin 2020 (hourglass absent in randomized). **I found no paper that compares the optimal feedback share (MFAS) of a connectome with degree-matched or type-matched SBM nulls**, and no paper reporting that a cell-type SBM null exceeds the real brain's feedforwardness. That remains the clearest novel element (absence of evidence from this search, not proof).
- **Sensory->motor axis recovered by the order**: expected and shown in spirit by Durbin, Varshney, Dorkenwald/Schlegel info-flow ranks. Not novel by itself; useful as validation.
- **The forward score itself**: ours (84.26%) is below the public record (84.61%). Do not present it as state of the art.

### 6.4 Items I could not verify
- Winding 2023 "66% feedforward": not found (see 3.1).
- Exact feedforward shares in Witvliet 2021 Fig. 3c.
- Source of mouse ~0.43/0.36/0.21 split.
- Rodgers et al. 2022 exact F0 for C. elegans.
- Whether Cook 2019, Matsliah 2024, Nern 2025, male CNS papers report any ordering or feedback share.
- Contents of the Research Square "Audited Benchmark for MWFAS" (whether connectomes are included).
- Cell 2026 DOI of Hoeller et al. (from search snippet; publisher page returned 403).

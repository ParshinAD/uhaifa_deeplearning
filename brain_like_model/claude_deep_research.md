1# Under-Investigated Deep-Learning Thesis Topics at the Intersection of Initialization, Connectomes, and Bio-Plausible Learning
## A decision-ready research-topic scoping report for a senior data scientist's master's thesis (9–12 months)

## TL;DR

- **The single highest-value topic for this student is "Connectome-Statistics-Derived Initialization for Biologically Plausible Learning Algorithms"**: deriving weight-init distributions from the Allen mouse mesoscale connectome (degree, edge-weight, hub statistics) and testing them on forward-forward, predictive coding, and dendritic localized learning. A targeted literature scan confirmed this exact combination is unoccupied in 2024–2026 literature; it is empirical, falsifiable, and rich enough for a 4-chapter thesis even if results are negative.
- **Two strong runner-up topics**: (a) measuring Initial Guessing Bias (Francazi et al., ICML 2024) inside non-backprop learners (FF, PC, DLL) — a genuine "nobody has checked this" empirical gap; and (b) using the MICrONS mouse-cortex directed graph (its feedback loops and rich-club hubs) as a structural prior for a small recurrent network trained with Equilibrium Propagation or Predictive Coding. Avoid pure "connectome-as-CNN-skeleton" projects: MouseNet (2022), BioNIC (arXiv:2601.20876, posted late 2025/early 2026), and BT-SNN (Wang Y. et al., Frontiers in Neuroscience 18:1325062, 2024) have largely staked that territory.
- **Pitfalls to plan around now**: do not chase theory-heavy initialization work (NTK/μP, mean-field GIH proofs) — leave theory to the supervisor and lean on your engineering strengths. Lock the empirical protocol (datasets, baselines, seeds, metrics) by month 3, because a first-time scientific writer's biggest risk is a sprawling experiment matrix with no clean comparison to publish.

---

## Key Findings (synthesis of the literature scan)

### A. Initialization is a renewed hot area in 2024–2026, but with three under-served sub-niches

Standard initialization (He, Xavier, LeCun, μP, T-Fixup) is saturated for vanilla feed-forward and transformer settings. However, three sub-areas are demonstrably alive and under-investigated:

1. **Geometric/data-dependent views of initialization.** The Geometric Invariance Hypothesis (Kazemi et al., ICLR 2025 Spotlight, arXiv:2410.12025) shows that input-space curvature evolves only along an architecture-dependent subspace determined at initialization, and that data covariance projected onto the model's "average geometry" governs early training dynamics. Linear-discriminant initialization (Masden & Sinha, arXiv:2007.12782) and the Generalized Discrimination Value framework (Schilling et al., Neural Networks 2021) provide ready-made empirical tools for measuring class-separability evolution from initialization onward.
2. **Initial Guessing Bias (IGB) family.** Francazi, Lucchi & Baity-Jesi (ICML 2024, arXiv:2306.00809) showed untrained networks systematically favor a subset of classes due to architecture/preprocessing choices; this has produced a small follow-up cluster ("Where you place the norm matters," arXiv:2505.11312; "When the left foot leads to the right path," arXiv:2505.12096; Pellegrino et al., arXiv:2511.20826). IGB has only been studied for backprop-trained networks.
3. **Transformer/biological-method initialization.** "Initialization is Critical to Whether Transformers Fit Composite Functions by Reasoning or Memorizing" (Zhang et al., arXiv:2405.05409), "Structured Initialization for ViTs" (arXiv:2505.19985), "Two failure modes of deep transformers" (arXiv:2505.24333), and a critical-initialization theory for biological neural networks (bioRxiv 2025.01.10.632397) are all 2024–2025 papers, but none of them adapt to non-backprop training.

### B. Connectome-inspired architectures: the *mouse* connectome is largely untouched

The dominant lineage is C. elegans (ElegansNet, arXiv:2304.13538; Elegans-AI, Neurocomputing 2024; Roberts et al., bioRxiv 2022.09.30.510374) and Drosophila (Lappalainen JK et al., Nature 634(8036):1132–1140, 2024, doi:10.1038/s41586-024-07939-3, which uses "experimentally determined connectivity for 64 cell types in the motion pathways of the fruit fly optic lobe"; the related flyvis codebase). The mouse cortex has been used by:

- **MouseNet** (Shi et al., PLOS Comp Biol 2022): macro-scale Allen Mouse Brain Connectivity Atlas → feedforward CNN topology.
- **BioNIC** (arXiv:2601.20876, posted late 2025/early 2026): single MICrONS V1 cortical column → adjacency-masked emotion classifier.
- **BT-SNN** (Wang Y. et al., Front. Neurosci. 18:1325062, 16 Apr 2024): generates spiking-RL topologies via Tanimoto hierarchical clustering from the Allen mesoscale connectome, then tests the best topology (NET-46) on four MuJoCo continuous-control tasks (MountainCar-v2, HalfCheetah-v2, Humanoid-v2, HumanoidStandup-v2).
- **Connectome-Guided Automatic Learning Rates** (Songdechakraiwut et al., arXiv:2510.23781, Oct 2025) and **Data-Efficient Neural Training with Dynamic Connectomes** (arXiv:2508.06817, Aug 2025): construct a *functional* connectome of the ANN itself during training and use persistent homology to steer learning rate / early stopping — this is "ANN-as-connectome," not "biological-connectome-as-ANN."
- **MICrONS 2025 data papers** (MICrONS Consortium, Nature 640:435–447, 2025: "dense calcium imaging of around 75,000 neurons … co-registered with an electron microscopy reconstruction containing more than 200,000 cells and 0.5 billion synapses"; Ding et al., Nature 640:459–469, 2025, on the like-to-like wiring rule including feedback; Schneider-Mizell et al., Nature 640:448–458, 2025, doi:10.1038/s41586-024-07780-8, where consensus clustering identified **18 inhibitory "motif groups"** of cells with similar output connectivity; Tumma et al., Neural Networks 190:107688, Oct 2025, on V1/RL/AL border regions).

**Gap identified**: nobody in 2024–2026 has used **mouse connectome statistics (degree distribution, edge-weight distribution, hub structure) as an initialization distribution** for ANN weights, as opposed to using the connectome as an architectural skeleton. This is the cleanest unoccupied niche.

### C. Biologically plausible learning is mid-maturity but with combinatorial gaps

Recent state-of-the-art:

- **Dendritic Localized Learning** (Lv et al., 42nd International Conference on Machine Learning, ICML 2025 — held 13–19 July 2025 in Vancouver — arXiv:2501.09976; PMLR 267:41682–41700) — first method to satisfy weight symmetry, local error, and non-two-phase criteria simultaneously, with code on GitHub.
- **Bidirectional Spike-based Distillation** (Lv et al., arXiv:2509.20284, Sep 2025) — extends DLL framework with two more bio-plausibility criteria.
- **Forward-Forward** extensions: convolutional FF (Scodellaro et al., Scientific Reports 2025), Self-Contrastive FF (Lee et al., Nature Communications 2025, s41467-025-61037-0), Predictive FF (Ororbia & Mali 2023), Distance-Forward Learning (2024), Cascaded Forward (Pattern Recognition 161:111292, 2025), FF-INT8 for edge devices.
- **Predictive coding**: "Benchmarking Predictive Coding Networks – Made Simple" (Innocenti et al., ICLR 2025 Spotlight, arXiv:2407.01163) provides the PCX library and standard benchmarks; μPC (Innocenti et al., 2025) trains very deep PC models; "Towards the Training of Deeper Predictive Coding Neural Networks" (arXiv:2506.23800, 2025) reaches ResNet18 with iPC.
- **Equilibrium Propagation**: brain-inspired FRE-RNN (arXiv:2508.11659, Aug 2025), scalable EP for deep convolutional CRNNs (arXiv:2508.15989, Aug 2025), stochastic EP for SNNs (Lin et al., Nov 2025).
- **Feedback alignment family**: counter-current learning (Sep 2024), DFA for RNNs (ISC High Performance 2025), curl-descent (NeurIPS-style).

**Confirmed gaps**: (i) no IGB analysis in any non-backprop learner; (ii) no FF/PC/DLL with a connectome-derived recurrent topology; (iii) no application of GIH (Kazemi et al. 2025) to non-backprop training; (iv) initial-weight magnitude effects on bio-plausible RNNs (arXiv:2410.11164, 2024) studied **gain** but not connectome-informed distributions.

---

## Details: Recommended Thesis Topics

I rate each topic on (a) novelty as of late 2025 / early 2026, (b) difficulty for an engineering-strong but academically-new student, (c) publishability ceiling, and (d) safety against a negative result.

### Area 1: Neural Network Initialization

#### Topic 1.1 — "Connectome-Statistics-Derived Initialization Distributions" ★ TOP PICK

**Pitch.** Replace He/Xavier initialization with weight distributions that match summary statistics of the Allen mouse mesoscale connectome (heavy-tailed log-normal edge weights, scale-free in/out-degree, rich-club hub overrepresentation). Test whether such initializations improve early training dynamics, lazy-vs-rich regime transitions, class separability evolution (GDV master-curve framework), and Initial Guessing Bias in ResNets, MLPs, and ViTs on CIFAR-10/100 and TinyImageNet.

**Research gap.** Verified: no 2024–2026 paper derives a weight-init scheme from biological-connectome distributions. The closest neighbors — Liu et al. (arXiv:2410.11164, 2024) on initial gain in bio-plausible RNNs, and the bioRxiv 2025 "critical initialization for biological neural networks" — only use random matrices, not measured connectome statistics. Meanwhile, GIH (ICLR 2025) gives you a ready-made measurement framework for evaluating geometric consequences.

**Thesis arc (4 chapters).**
1. **Observation.** Catalog statistics of the Allen mouse mesoscale connectome (directed weighted graph, 213/426 regions): in-degree, out-degree, edge-weight distribution, reciprocity, rich-club coefficient, motif spectrum. Compare against MICrONS cortical micro-scale statistics where possible. (Roughly two months of work; produces a reusable dataset + figures.)
2. **Modeling.** Propose three init families: (a) match the marginal edge-weight distribution; (b) match degree distribution at the layer level (preserve in/out connectivity statistics in fully-connected layers); (c) full structural prior (use the actual connectome as a sparse mask + matched non-zero weight distribution).
3. **Experiment.** On MLP-784-512-256-10, ResNet18, and a small ViT, train on CIFAR-10, CIFAR-100, and TinyImageNet for 100 epochs with 5 seeds each. Measure: (i) GDV class-separability curves (Schilling et al. 2021), (ii) average-geometry rank and curvature (GIH, Kazemi 2025), (iii) Initial Guessing Bias magnitude (Francazi 2024), (iv) lazy-vs-rich indicator (NTK kernel velocity), and (v) final accuracy. Baselines: He, Xavier, orthogonal, LSUV.
4. **Interpretation.** Argue whether connectome-derived inits change *geometry-at-initialization* in ways that translate to training-dynamics benefits, or whether they are statistically indistinguishable from He/Xavier — both outcomes are publishable.

**Methodology.** PyTorch; allensdk (Python) to fetch the connectome and NetworkX/graph-tool for statistics; the PCX library for any later PC comparison; standard image-classification eval; plotting via seaborn/matplotlib.

**Risks & mitigation.**
- *Risk: connectome-init underperforms.* Mitigation: thesis is framed as "what features of init matter" — a negative result is informative; you also have the geometry/IGB measurements as primary contributions.
- *Risk: too many factors to disentangle.* Mitigation: lock to a strict factorial design (3 architectures × 3 init families × 3 datasets × 5 seeds = 135 runs, ~2–3 weeks on a single A100).
- *Risk: connectome is "just another structured random matrix."* Mitigation: include scale-free random graphs (Watts-Strogatz, Barabási-Albert) as null models — this becomes the framing.

**Difficulty.** 5/10 — heavy engineering but conceptually clear; minimal theory required.

**Publishability.** Workshop paper (NeurIPS UniReps, ICLR Re-Align, NeurIPS NeurReps, ICML bio-plausible workshops) is realistic; possible conference paper if the connectome init beats He on at least one architecture/dataset; bioRxiv companion paper covers the connectome-statistics analysis.

**Resources.** Allen Mouse Connectivity Atlas (free, allensdk); 1 A100 or even a single 24GB GPU suffices; ~$0 in data costs.

**12-month timeline.** M1–M2 literature + connectome stats; M3 init-scheme prototypes on MNIST/CIFAR-10 MLP; M4–M6 main experiment matrix; M7 deep dive on geometry/IGB metrics; M8 writing; M9 workshop submission deadline (NeurIPS workshops typically Oct, ICLR workshops typically Feb); M10–M12 thesis polish.

---

#### Topic 1.2 — "Initial Guessing Bias in Non-Backprop Learners"

**Pitch.** Replicate Francazi et al.'s IGB measurements on networks trained by Forward-Forward, Predictive Coding (PCX library), Dendritic Localized Learning, and Direct Feedback Alignment. Does IGB exist there? Is it ameliorated, exacerbated, or transformed by local learning rules?

**Gap.** Verified: no paper studies IGB in any bio-plausible learner. This is a clean, single-question empirical study.

**Arc.**
1. **Observation.** Reproduce IGB for backprop networks on MNIST, CIFAR-10, and a 100-class imbalanced toy set (matches the original IGB paper).
2. **Modeling.** Define IGB measurement protocol for FF (use linear-probe accuracy on positive/negative goodness signals), PC (use the energy-based output distribution at t=0), DLL (use the final-layer pre-activation distribution), DFA (output-layer logits at init).
3. **Experiment.** For each learner × architecture × dataset, measure IGB strength at initialization, IGB decay rate during training, and final-class confusion. Use the public code repos (DLL: github.com/Lvchangze/Dendritic-Localized-Learning; PCX; loeweX/Forward-Forward).
4. **Interpretation.** Map IGB to the architectural properties (activation, normalization placement, depth) that the original paper identified.

**Risks.** Low. The original IGB code is open; the bio-plausible code is open. Worst case: IGB is identical → still a publishable null result with a clear "architecture, not algorithm, drives IGB" message.

**Difficulty.** 4/10.

**Publishability.** Strong workshop fit (NeurIPS SciForDL, ICLR Bridging Neuroscience & ML, ICML Bio-Plausible Learning, COSYNE if you frame neuroscientifically). Conference-paper potential at TMLR.

**Resources.** Single GPU; only public datasets and code; no professor-specific data required.

**Timeline.** Faster (6–8 months); could pair with a second topic.

---

#### Topic 1.3 — "Geometric-Invariance Diagnostics for Non-Backprop Training"

**Pitch.** Apply the GIH framework (Kazemi et al. ICLR 2025) — average-geometry eigenspectra, geometric velocity, decision-boundary curvature — to networks trained by FF, PC, EqProp, DLL. Does the GIH hold for non-backprop training? If not, what changes?

**Gap.** Verified: zero papers combine GIH with non-backprop training.

**Arc.** Same four-chapter pattern: catalog GIH metrics for BP networks (replicate Kazemi); apply to non-BP; quantify divergence; interpret what local learning does to geometry.

**Risks.** Moderate — GIH involves Jacobian and second-derivative computations that, while tractable, scale poorly to large networks. Stay at MLP / small ResNet scale.

**Difficulty.** 6/10 (some math required to implement metrics correctly).

**Publishability.** Workshop strong; conference if you also propose an init scheme that re-establishes invariance for non-BP.

---

### Area 2: Connectome / Brain-Inspired Architectures

#### Topic 2.1 — "Feedback-Loop Structural Prior from the Mouse Mesoscale Connectome for Equilibrium-Propagation RNNs"

**Pitch.** The Allen mouse mesoscale connectome contains thousands of directed loops (Knoblauch et al., arXiv:1811.04698, identified loops crossing isocortex–striatum–thalamus). Use this loop structure as a sparse-mask + edge-weight prior for an RNN trained with Equilibrium Propagation or Predictive Coding. Compare to scale-free/small-world null models on standard sequential tasks (sequential MNIST, PennTreeBank-tiny, MIT-BIH ECG).

**Gap.** EqProp/PC have been demonstrated on dense feedback architectures (FRE-RNN, arXiv:2508.11659, 2025) and physics-inspired residuals, but never with a directed-graph prior taken from a real mammalian brain. Mouse-specific work (BT-SNN, BioNIC, MouseNet) has been feedforward only.

**Arc.**
1. Observation: catalog loops, hubs, and rich-club nodes in the Allen connectome at 100-µm resolution.
2. Modeling: project the directed graph to RNN connectivity with a chosen size budget (e.g., 426 regions = 426 hidden units).
3. Experiment: train with EqProp (using the Lin/Bal/Sengupta scalable code) and PCX; compare to random / Watts-Strogatz / Barabási-Albert / fully connected.
4. Interpretation: do feedback loops in the connectome provide useful inductive bias for sequence tasks?

**Difficulty.** 7/10 — EqProp is finicky.

**Publishability.** Conference-paper potential (NeurIPS Bio-Plausible workshop, ICLR Re-Align, or main-track at AAAI/IJCAI).

**Risks.** EqProp convergence in heterogeneously-connected networks is fragile; mitigate by also testing PC (more robust) and a smaller-scale version.

---

#### Topic 2.2 — "Whole-MICrONS Hub & Motif Spectrum vs Random Null Models"

**Pitch.** The MICrONS V1 dataset (Nature, April 2025) reconstructs more than 200,000 cells (about 120,000 neurons + glia) and 523 million synapses via electron microscopy, with calcium-imaging in ~75,000 of those neurons; Schneider-Mizell et al. (Nature 640:448–458, 2025) identified 18 inhibitory motif groups via consensus clustering of cell-type-specific output connectivity. Conduct a comprehensive cell-resolution graph analysis (rich-club coefficient, motif spectrum, feedback fraction) and use it to design a "MICrONS-prior" sparsity pattern for ViTs, comparing against random sparse training (RigL, SET).

**Gap.** Tumma et al. (Neural Networks 190:107688, Oct 2025) analyzed V1/RL/AL border regions; nobody has yet done a comprehensive cortex-wide motif/hub analysis combined with an ANN architectural translation.

**Difficulty.** 7/10 — MICrONS data is 1.6 PB raw; even the cleaned graph is large. Use the MICrONS Explorer cell-graph CSVs (the proofread cell-cell graph is a ~120k-neuron sparse adjacency, not 75k as some sources state — 75k is the calcium-imaged subset).

**Publishability.** Strong (Nature Methods-style methods venues; NeurIPS workshops). Risk that the "MICrONS-prior sparsity helps ViT" claim won't bear out — pivot to characterization-only paper.

**Resources.** MICrONS Explorer; CAVEclient Python tooling; cloud storage for graph (~10–50 GB).

---

#### Topic 2.3 — "Hub-Aware Pruning at Initialization Using Connectome Priors"

**Pitch.** Modern pruning-at-initialization methods (SynFlow, GraSP, SNIP, sensitivity-based pruning) are agnostic to biological priors. Test whether hub-aware sparsity patterns (high-degree-preserved, rich-club-preserved) inspired by the mouse connectome lead to better lottery-ticket networks at high sparsity (95–99%).

**Gap.** Topographical Sparse Mapping (Sciencedirect 2025, S0925231225024129) is the closest neighbor but uses generic neuro-inspired sparsity; nobody has tested explicitly hub-derived priors against SynFlow/GraSP.

**Difficulty.** 5/10.

**Publishability.** Workshop (NeurIPS Sparsity, ICLR Sparsity) easily; conference possible.

---

### Area 3: Biologically Plausible Learning

#### Topic 3.1 — "Stress-Testing Dendritic Localized Learning Beyond ICML 2025"

**Pitch.** DLL (Lv et al., 42nd ICML 2025) is brand-new and reported state-of-the-art among algorithms satisfying all three bio-plausibility criteria. The paper benchmarks MLP/CNN/RNN on MNIST/CIFAR/Harry-Potter. Extend with: (i) ViT under DLL; (ii) DLL on continual-learning benchmarks (Split-CIFAR, Permuted-MNIST); (iii) DLL with the Initial Guessing Bias measurement; (iv) DLL with weight quantization for edge devices (FF-INT8 style).

**Gap.** DLL has essentially no follow-up work as of early 2026; first-mover advantage.

**Arc.** Replicate → extend to new architecture / task → diagnose failure modes → propose fix.

**Difficulty.** 6/10.

**Publishability.** Strong — being the first follow-up to a popular ICML paper is publishable. Workshop guaranteed; conference plausible.

**Resources.** Single GPU; DLL code is open.

---

#### Topic 3.2 — "Forward-Forward with Recurrent Connectome Topology"

**Pitch.** Predictive Forward-Forward (Ororbia & Mali 2023) uses a generic recurrent circuit; Self-Contrastive FF (Nature Communications 2025) adds sequential capability. Neither uses a biologically-realistic recurrent topology. Use a small mouse cortex sub-circuit (e.g., V1+higher areas + thalamic loop) as the FF graph and test on MNIST/CIFAR + sequence tasks.

**Gap.** Verified: nobody has combined FF with a connectome-derived recurrent topology.

**Difficulty.** 7/10 — FF tuning is sensitive; recurrent FF is even more so.

**Publishability.** Conference plausible; workshop almost certain.

---

#### Topic 3.3 — "Benchmarking Bio-Plausible Learners on the Same Initialization Schemes"

**Pitch.** A controlled study: pick five learners (BP, DFA, DRTP, PC, FF, DLL) and five initialization schemes (He, Xavier, orthogonal, LSUV, connectome-derived). Run a full 5×5 grid on three datasets; produce the "biological-learner initialization sensitivity benchmark" paper.

**Gap.** Innocenti et al. (ICLR 2025) benchmarked PCNs but not against this cross-product. This is "the missing benchmark."

**Difficulty.** 5/10 (mostly engineering).

**Publishability.** Strong workshop fit. Easy to write up.

---

### Cross-cutting topics (combine 2–3 directions) — *most novel, most publishable*

These score highest because each pairwise intersection was verified empty in 2024–2026 literature:

| ID | Title | Combines |
|---|---|---|
| **X1** | Connectome-Statistics-Derived Init for Bio-Plausible Learners | Init + Connectome + Bio-plausible |
| X2 | IGB in DLL/FF/PC with Connectome Init vs Random Init | Init + Connectome + Bio-plausible |
| X3 | Mouse Feedback-Loop Prior for Equilibrium Propagation RNNs | Connectome + Bio-plausible |
| X4 | GIH-Style Geometry Diagnostics Across Bio-Plausible Algorithms | Init + Bio-plausible |
| X5 | Connectome-Statistics Sparsity Prior for Forward-Forward | Connectome + Bio-plausible + Init |

**X1 is essentially Topic 1.1 above, deliberately framed at the three-way intersection.** It is my top recommendation because (a) it is the most defensible empirical question, (b) you can guarantee a "something to write" outcome since the connectome-statistics chapter alone is a contribution, and (c) it leverages your supervisor's three specific interests directly.

---

## Recommendations (decision-ready, with thresholds for switching)

### Rank-ordered top 3 topics for this student

**#1 — Topic X1 / 1.1: Connectome-Statistics-Derived Initialization for Bio-Plausible Learners** (combined three-way)

*Why this student.* You have strong PyTorch + data-analysis skills; this topic is 80% engineering (data pipelines, training matrices, statistical comparison). It hits all three of your supervisor's interests directly. It has a guaranteed-publishable "Chapter 1" deliverable (a clean characterization of mouse mesoscale connectome statistics relevant to weight init) regardless of whether the init scheme helps downstream.

*First 30 days.* Week 1: install allensdk, pull the 213-region mesoscale connectome, reproduce Knoblauch et al. (arXiv:1811.04698) loop-counting. Week 2: compute degree, weight, motif, rich-club statistics; produce reusable Python notebook. Week 3: implement three init samplers (marginal, degree-matched, full-structural) and verify they preserve target statistics. Week 4: smallest sanity-check experiment — MLP on MNIST with connectome-derived init vs He init, single seed; aim for a clean first plot of GDV class-separability curves at initialization.

*Switch-away threshold.* If by month 4 no init variant produces a measurable difference in *any* of {GDV, IGB, kernel velocity, final accuracy} across all architectures, pivot to Topic 1.2 (IGB study).

---

**#2 — Topic 1.2: IGB in Non-Backprop Learners**

*Why this student.* Self-contained, low-risk, all open-source code. Designed for someone who needs a guaranteed publishable result and has limited academic experience. It is essentially a careful empirical study with strong narrative.

*First 30 days.* Week 1: reproduce IGB on standard CNN/MLP per Francazi et al. arXiv:2306.00809. Week 2: install PCX, DLL, FF, DFA code repos; verify all four train on MNIST. Week 3: implement an IGB measurement protocol per learner; produce four "IGB at init" histograms. Week 4: write the protocol document — that is your method chapter.

*Switch-away threshold.* If IGB is identical across all four learners after month 3, that is still a result; do not switch.

---

**#3 — Topic 2.1: Mouse Feedback-Loop Prior for EqProp/PC RNNs**

*Why this student.* Highest novelty ceiling and the most aligned with the supervisor's interest in feedback loops and hub structure. But it is technically harder (EqProp convergence issues) and more risky.

*First 30 days.* Week 1: read Scellier–Bengio 2017 and Bal/Sengupta 2023; reproduce a small EqProp run from public code. Week 2: read Lin/Bal/Sengupta 2024 (deep CRNN EqProp). Week 3: extract a small directed sub-graph from the Allen connectome (e.g., 256 regions including thalamocortical loops). Week 4: train a tiny EqProp RNN with this connectivity mask on sequential MNIST.

*Switch-away threshold.* If EqProp fails to converge with the connectome mask after 6 weeks of tuning, drop to PC (more robust) or pivot to Topic 2.3 (hub-aware pruning).

---

### What to AVOID

- **Pure NTK / mean-field theory** of initialization. You will be outclassed by theory-trained students and you will not finish in 12 months.
- **Trying to "improve backprop" on ImageNet.** Saturated; you cannot compete with industrial compute.
- **Yet another C. elegans connectome paper.** Saturated by ElegansNet / Elegans-AI / Roberts et al.; the field knows the answer.
- **Pure architecture search.** Cannot outcompute industry-scale NAS.

---

## Practical Considerations

### Mouse connectome data access

- **Allen Mouse Brain Connectivity Atlas (mesoscale)** — public, free, well-documented; access via `allensdk` Python package; the directed weighted ~426-region (or 213 with symmetric pairs) graph is downloadable as a numpy array via `mouse_connectivity_models` (GitHub: AllenInstitute/mouse_connectivity_models). NeuroCarta MATLAB toolbox provides ready connectivity graphs. This is the dataset your supervisor most likely has and the cleanest for thesis use.
- **MICrONS (V1 cube)** — public via MICrONS Explorer (microns-explorer.org) and CAVEclient Python tools. Cellular-resolution reconstruction of more than 200,000 cells (~120,000 neurons + glia) and ~523 million synapses; the calcium-imaged subset is ~75,000 neurons. Use only if needed at the cellular scale; otherwise overkill for initialization studies.
- **FlyWire / Hemibrain** — Drosophila; out of scope but useful as reference null.

### Computational resources

- Topics 1.1, 1.2, 1.3, 2.3, 3.1, 3.3 are **single-GPU** feasible (24GB A6000 or rented A100). Total cost <$500 on Lambda/RunPod.
- Topics 2.1, 2.2, 3.2 may need **2× A100** for the larger experiments. Budget $1–2k.
- All topics work with **CIFAR-10/100 + TinyImageNet** — no ImageNet-scale training required.

### Public-data vs supervisor-specific data

- All topics can be executed with public Allen / MICrONS data.
- Your supervisor's "full mouse connectome dataset" likely gives access advantages (e.g., a preprocessed graph, cell-type annotations, expert guidance on what's biologically meaningful). Use that for Chapter 1 (data characterization), then your method generalizes.

---

## Best Practices for a First-Time Scientific Thesis Writer

### Structuring a deep-learning thesis

The 4-part arc I've used for every topic above (Observation → Modeling → Experiment → Interpretation) maps cleanly to:
- **Chapter 1 (Background & Observation)**: literature review + dataset/phenomenon characterization. This chapter is your insurance policy: it must contain new descriptive content (e.g., "we measure X on dataset Y for the first time") so that even if the rest fails, you have a contribution.
- **Chapter 2 (Method)**: the model/algorithm proposal. Keep it small: one clearly-named method, one falsifiable hypothesis.
- **Chapter 3 (Experiments)**: a factorial table. Pre-register your experimental matrix in month 3; if results push you outside it, that becomes Chapter 4 narrative, not method drift.
- **Chapter 4 (Discussion/Interpretation)**: what the results mean for the broader question. This is where industry practitioners typically under-write; allocate at least a month.

### "Small enough to complete but big enough to matter"

Use the **two-axis test**: a thesis topic should be (i) reducible to a single declarative claim ("Connectome-derived initialization changes IGB on ResNet-18/CIFAR-10 by X%"); and (ii) generalizable to one neighboring claim ("...and this pattern holds across init families A, B, C"). If you cannot fit your topic in that template, it is too vague.

### Literature review section

- 60–80 references is plenty for a master's thesis.
- Organize **thematically** (initialization theory → connectome-inspired architectures → bio-plausible learning), not chronologically.
- For each theme, devote 2–3 paragraphs: (a) what the canonical references established; (b) what 2024–2026 has added; (c) what the gap is.
- Use a citation manager (Zotero or BibTeX) from day 1.

### Experiment-to-thesis pipeline

- Write the *thesis figures' captions first*, then run experiments to fill them. This forces you to know what story each figure tells.
- One Jupyter notebook = one experiment. Save the .ipynb in your thesis repo alongside a YAML config.
- Use Weights & Biases or MLflow from day 1; you will not be able to reconstruct hyperparameters in month 9 otherwise.
- Generate every figure programmatically; no Excel/Powerpoint figures in a thesis you want to publish.

### Common pitfalls for industry practitioners

1. **Treating thesis like a product roadmap.** Scientific writing requires explicit hypotheses and falsifiable claims, not feature releases. State the hypothesis in chapter 2's first sentence.
2. **Over-engineering the codebase.** You do not need a clean library; you need reproducible experiments. Resist refactoring.
3. **Under-citing.** A claim without a citation reads as bragging.
4. **Skipping ablations.** A result without an ablation is not publishable.
5. **Writing the introduction last.** Write a 1-page intro in month 1; revise monthly. It crystallizes thinking.
6. **Pivoting too late.** Set explicit checkpoints (month 3, 6, 9); each checkpoint has a "switch-away threshold."

### Positioning practical work as scientific contribution

A practical contribution becomes scientific when paired with: (a) a measurable comparison to baselines; (b) an ablation isolating the active ingredient; (c) a generalization claim across at least two domains; (d) a negative-result section. If your engineering produces (a–d), it is a paper.

---

## Caveats

- **Topic 2.2 (MICrONS whole-graph analysis) has the largest data-engineering overhead** (CAVE client, proofread cell graphs at scale). If your supervisor's "full mouse connectome dataset" is actually MICrONS-derived, this risk is reduced.
- **BioNIC (arXiv:2601.20876)** is dated late 2025 / early 2026 and may steal partial novelty from any "MICrONS-as-architecture" topic. Verify its exact scope before committing to Topic 2.2.
- **DLL follow-up window is closing.** ICML 2025 papers typically attract follow-up by NeurIPS/ICLR within 9 months. Topic 3.1 must be started immediately to be first.
- **Equilibrium Propagation is hard.** If you pick Topic 2.1, build in 2 months of dead-end tuning as part of the plan.
- **Literature scan is current to ~Nov 2025.** Submission deadlines in the next 6 months may yield competing work. Re-run a focused literature scan at months 3, 6, and 9; allocate a half-day each time.
- **Negative results are publishable but harder to write.** If your top pick produces clean negatives, frame it as "we measure X across architectures/inits and find no effect" — that is a contribution if the measurement protocol itself is novel.
- The **bioRxiv 2025 "critical initialization for biological neural networks"** paper (10.1101/2025.01.10.632397) is the closest theoretical neighbor to Topic 1.1 — read it carefully in week 1 to make sure your framing differentiates clearly.
- Schneider-Mizell et al. (Nature 2025) and Ding et al. (Nature 640:459–469, 2025) are the most likely to attract follow-up MICrONS-graph papers in 2026; check at month 3.

---

## Final word

The single recommendation I will defend strongest: **pursue Topic X1 (Connectome-Statistics-Derived Initialization, applied across both backprop and at least one bio-plausible learner) for the main thesis, with the IGB-in-non-backprop study as a fallback chapter you can pivot to in month 4 if needed.** This combination uses your engineering strengths, hits all three of your supervisor's interests, and gives you two independent paths to a publishable result. Aim for a NeurIPS Bio-Plausible Learning workshop submission at month 9 and a TMLR or ICLR Re-Align submission at month 12.
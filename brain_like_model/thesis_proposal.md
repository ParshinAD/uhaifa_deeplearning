# Can Mouse Brain Wiring Tell Neural Networks How to Start?

**Thesis Proposal — Master's in Deep Learning**
**Topic: Connectome-Statistics-Derived Weight Initialization for Deep Neural Networks**

---

## 1. The Big Picture

Every neural network is just a large collection of numbers called **weights**. Before training begins, those numbers must be set to some starting values — this is called **weight initialization**. The choice of those starting values turns out to matter enormously: a bad starting point can cause the network to fail to learn at all, learn very slowly, or converge to a worse solution.

Today's standard initialization methods (He, Xavier, Orthogonal) were designed in the 2010s using mathematical arguments about signal propagation in random networks. They all make the same core assumption: **draw every weight independently from the same Gaussian distribution, with the same scale for every neuron**. That is both simple and effective — but it is also arbitrary, and it says nothing about biology.

Meanwhile, neuroscience has spent decades mapping the actual wiring diagram of the mouse brain. That map is now publicly available. And when you look at it statistically, the brain's wiring looks nothing like a Gaussian: a few synaptic connections are very strong, most are very weak; a few regions connect to nearly everything, most connect to only a handful of others. This specific statistical structure has never been used to design initialization schemes for artificial networks. **That is the gap this thesis fills.**

---

## 2. The Research Gap

### What Already Exists

| Building block | What it is | Status |
|---------------|-----------|--------|
| Standard initialization (He, Xavier, Orthogonal) | Randomly sample weights from Gaussian or uniform distributions with theoretically motivated scale | Widely used, well understood |
| Allen Mouse Brain Connectivity Atlas | Publicly available, region-to-region map of the entire mouse brain, measured with viral tracers | Free dataset, rarely used in ML |
| Bio-plausible learning algorithms | Alternatives to backpropagation that match biological constraints better (Forward-Forward, Predictive Coding, DLL) | Active research, no standard init studies |

### What Is Missing

1. Nobody has derived a weight initialization scheme **from** connectome statistics
2. Nobody has systematically studied how initialization affects bio-plausible learners
3. The intersection of all three — connectome-derived init **tested on** bio-plausible learners — has zero published work as of 2026

### The Core Research Question

> *Does initializing artificial neural network weights using the statistical structure of the mouse brain's wiring diagram preserve or improve class separability and learning dynamics, compared to standard methods — and does this effect differ between backpropagation and biologically plausible learners?*

---

## 3. The Data: Allen Mouse Brain Connectivity Atlas

### What It Is

The Allen Mouse Brain Connectivity Atlas [Oh et al., *Nature* 2014] is a region-to-region map of the entire mouse brain built by injecting a fluorescent virus into one brain region and measuring where it travels. The result is a **directed weighted graph**:

- **213 nodes** = brain regions (ipsilateral hemisphere)
- **~18,500 directed edges** = measured axonal projections
- **Edge weights** = normalized projection volume (how much of the target region is covered by projections from the source)
- **Connection density** = 29% (about 1 in 3 possible connections exists)

### How to Access It

```bash
pip install allensdk   # free, official Allen Institute Python library
```

The data is downloaded automatically on first use (~200 MB). Our prototype notebook (`connectome_init_prototype.ipynb`) also includes a synthetic fallback with identical statistics if the download is unavailable.

### Three Key Statistical Facts

These three properties are what we translate into initialization schemes.

---

**Fact 1 — Edge weights follow a log-normal distribution**

Most connections are very weak; a few are extremely strong. On a regular scale this looks like a spike near zero with a long tail. On a log scale it looks like a bell curve. This is called a *log-normal distribution* — the same pattern seen in income distribution, city sizes, and many biological measurements.

```
Regular scale:
Strength:   weak ──────────────────────────────────► strong
Count:      ██████████████████████████████████░░░░░░░

Log scale:
log(strength):  low ─────────────────────────────► high
Count:          ░░░░░░██████████████████████████░░░░░

He init (log):  low ─────────────────────────────► high
Count:          ░░░░░░░░░░░░░░░██████████░░░░░░░░░░░░
                               ↑ much narrower than brain
```

**Implication**: standard He init produces weights that are clustered much more tightly than what the brain uses. A few very large weights rarely appear.

---

**Fact 2 — Degree distribution is scale-free (power law)**

Most brain regions connect to only a few others. A small number of "hub" regions (thalamus, motor cortex) connect to nearly everything. This is characteristic of a *scale-free network*, found in social networks, the internet, and biological systems.

```
# connections:   2   5   10   20   50   80
# brain regions: ████████  ████  ███   ██   █    █
                                            ↑ hub

vs. a random (Erdős-Rényi) graph:
# connections:   30  32  35  38  40  42
# brain regions: ██████████████████████████
                  ↑ everyone has similar connectivity
```

**Implication**: some neurons should "matter more" than others from initialization — but standard init assigns every neuron the same weight scale.

---

**Fact 3 — Rich-club structure (hubs connect to hubs)**

In the brain, hub regions preferentially connect to *each other*, not just randomly to everyone else. This creates a dense core of mutually interconnected hubs.

```
Random graph:         Brain graph (rich-club):
  ○ ─ ○ ─ ●             ○       ○
      │   │              └──► ●═══●
  ○ ─ ○   ●                   ║   ║
          │              ○    ●═══●
      ○ ─ ○                   ↑ hubs form a dense core
  (● = hub, ─ = random)       (= = hub-to-hub connections)
```

**Implication**: the structure of the brain's weight matrix is not just about individual weights — it's about which neurons are central.

---

## 4. The Method: Three New Initialization Schemes

Each scheme translates one of the three facts above into an initialization rule. All three are designed to match the variance of He initialization (`E[w²] = 2/fan_in`), so any differences we observe are due to **shape or heterogeneity**, not scale.

| Connectome property | Init scheme | Core idea | What it tests |
|--------------------|-------------|-----------|---------------|
| Log-normal weights (Fact 1) | `connectome_marginal` | Sample weight magnitudes from the actual measured edge weight distribution; apply random ± signs | Does the heavy tail shape matter? |
| Log-normal fit (Fact 1) | `connectome_lognormal` | Fit a log-normal model (μ, σ) to the connectome weights; sample from that model | Can two numbers capture the relevant structure? |
| Scale-free degrees (Fact 2) | `connectome_degree` | Each output neuron is assigned a weight scale sampled from the connectome's in-degree distribution. Hub neurons get wider distributions. | Does neuron-level heterogeneity matter? |

**Compared against:** He normal, Xavier uniform, Orthogonal (the three standard baselines).

### Variance Matching (Why It Matters)

```
He init:               E[w²] = 2/fan_in   ← target variance
connectome_marginal:   E[w²] = 2/fan_in   ← same scale, different shape
connectome_lognormal:  E[w²] = 2/fan_in   ← same scale, different shape
connectome_degree:     E[w²] = 2/fan_in   ← same average, heterogeneous per neuron
```

If we see a difference in results, it cannot be explained by "the weights were just bigger or smaller." It must be a structural effect.

---

## 5. What We Measure

### Phase 1 — Geometry at Initialization (no training)

We pass data through the network **without training** and measure whether class information survives.

| Metric | Plain-language meaning | Good value |
|--------|----------------------|------------|
| **Silhouette score** | How well separated are the two classes in the network's representation? | Closer to 1 |
| **Linear probe accuracy** | Can a simple linear classifier use the network's output to label the data correctly? | Above 50% (chance) |
| **Effective rank** | How many dimensions does the representation actually use? (rank = 1 means everything collapsed to a line) | High |
| **Dead neuron fraction** | What fraction of neurons output exactly 0 for all inputs? | Close to 0 |
| **Initial Guessing Bias (IGB)** | Does the untrained network already predict one class more often than the other? | Close to 0 |

**Key insight from preliminary experiments** (`init_geometry_experiment_v2.ipynb`): at depth 12, `random_normal` completely collapses (silhouette = NaN, rank = 1). `orthogonal` preserves the most information. He and Xavier are in between. The question is where the three connectome schemes land.

### Phase 2 — Training Dynamics (100 epochs)

| Metric | What it tells us |
|--------|-----------------|
| Final accuracy (CIFAR-10/100) | Does the init advantage at step 0 translate to a better endpoint? |
| Convergence speed | Does better initialization mean we need fewer epochs? |
| Accuracy gap vs. He baseline | Is the connectome init consistently better, worse, or the same? |

---

## 6. Experimental Plan

### Full Experiment Matrix

| Dimension | Values | Count |
|-----------|--------|-------|
| Init schemes | He, Xavier, Orthogonal + 3 connectome | 6 |
| Architectures | MLP (3-layer), ResNet-18, ViT-tiny | 3 |
| Datasets | CIFAR-10, CIFAR-100 | 2 |
| Random seeds | 5 per configuration | 5 |

**Phase 1 (geometry, no training):** 6 × 3 × 2 × 5 = **180 runs** — estimated ~1 week on a single GPU

**Phase 2 (training, 100 epochs):** same matrix × training = **~2–3 weeks on A100**

**Learners added in Phase 3:**
- Standard backpropagation (baseline)
- Forward-Forward (Hinton 2022 — network trains on "positive" and "negative" data)
- Predictive Coding (network minimizes prediction errors at every layer)
- Dendritic Localized Learning (DLL, ICML 2025 — satisfies the strongest biological constraints)

All datasets and models are standard and publicly available. Estimated compute cost: < $500 on rented GPU.

---

## 7. Why This Thesis Is Safe

A good thesis topic should have a publishable outcome whether results are positive or negative. This one does.

| Outcome | What it means scientifically | Publication target |
|---------|-----------------------------|--------------------|
| Connectome init outperforms He/Xavier | Biological connectivity statistics encode useful inductive bias for initialization | Conference paper (NeurIPS / ICLR workshop) |
| Connectome init performs the same | Weight variance is all that matters — shape is irrelevant | Null result paper ("what matters in init") |
| Effect differs between backprop and bio-plausible learners | The training algorithm interacts with initialization in a biologically meaningful way | Strong workshop paper |

**Chapter 1 is always a contribution.** The statistical analysis of the Allen connectome from an initialization perspective — degree distribution, edge weight distribution, rich-club coefficient, comparison to null models — does not exist in the literature. This chapter stands alone regardless of downstream results.

### Fallback Plan (Month 4)

If all three connectome schemes show no geometry effect at all: pivot to **Topic 1.2 — Initial Guessing Bias in non-backpropagation learners** (Francazi et al. 2024 extended to FF/PC/DLL). This is a self-contained 6-month study with all code open-source and a clear gap in the literature.

---

## 8. Timeline

| Month | Milestone | Deliverable |
|-------|-----------|-------------|
| 1–2 | Connectome statistics analysis + init scheme prototypes | ✅ Done: `connectome_init_prototype.ipynb` |
| 3 | Phase 1 geometry experiment — MLP architecture | First silhouette comparison plot |
| 4 | Phase 1 — ResNet-18 + ViT-tiny | Go/no-go decision on connectome init |
| 5 | Phase 2 — Training dynamics (backprop) | Accuracy + convergence curves |
| 6 | IGB measurement across all schemes | IGB comparison table |
| 7–8 | Phase 3 — Bio-plausible learners (FF + PC + DLL) | Cross-learner comparison |
| 9 | Workshop submission | NeurIPS Bio-Plausible Learning / UniReps |
| 10–12 | Full thesis writing + revisions | Submitted thesis |

---

## 9. Resources Needed

| Resource | Availability | Cost |
|----------|-------------|------|
| Allen Mouse Brain Connectivity Atlas | Free, via `allensdk` Python library | $0 |
| CIFAR-10, CIFAR-100 | Free, via `torchvision` | $0 |
| DLL code (ICML 2025) | Open-source, GitHub | $0 |
| PCX library (Predictive Coding) | Open-source, GitHub | $0 |
| Forward-Forward code | Open-source, GitHub | $0 |
| GPU compute | Single A100 (24GB) sufficient | < $500 rented |

---

*This document was prepared as a thesis proposal overview. Supporting code and experiments are in `connectome_init_prototype.ipynb` (connectome data + init schemes) and `init_geometry_experiment_v2.ipynb` (geometry baselines with standard inits).*

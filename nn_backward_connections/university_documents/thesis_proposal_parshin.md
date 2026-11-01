---
title: |
  Thesis Proposal:\
  Statistical Analysis of Recurrent Connectivity in Biological Neural Networks
author: Aleksei Parshin, Ido Nachum
date: 05-10-2026
---

# 1 Introduction

Artificial neural networks are usually drawn as layers: information enters on one side and flows forward, layer by layer, until a result comes out on the other side. Real brains are not built like that. A neuron can also send its signal "backward", to neurons that sit earlier in the processing chain. This backward wiring (feedback, or recurrence) is believed to matter for memory, prediction and activity the brain generates on its own. But before we ask what feedback does, there is a simpler question that is already hard: how much feedback does a brain have, and where exactly does it sit?

There is a natural way to measure this. Place all neurons on a line so that as much synaptic weight as possible points forward. Whatever still points backward under the *best possible* ordering is feedback the network cannot avoid. This is the Minimum Feedback Arc Set problem (we use the complementary maximization form; Section 2 states it formally). Finding the best ordering is computationally hard [1], and at this scale we can only approximate [2]. This makes the measurement itself an algorithmic problem, and a good part of the work below is about doing it well.

A measurement alone does not say whether the number is large or small. We will see below that about 16% of the fly's wiring points backward. Is that a lot? One way to judge is to compare against random networks that share the brain's basic statistics (size, density, degrees); in networks inferred from recordings in behaving mice, the amount of feedback relative to such controls differs strongly from one brain region to another [10]. We keep this kind of comparison in mind as one direction for the analysis.

The main graph we work with is the FlyWire connectome of an adult fruit fly, currently the only publicly available complete synapse-level connectome at this scale [3, 4]; the version we use, prepared for a public ordering challenge, has 136,648 neurons and 5,657,719 weighted directed connections. Alongside it we use a graph extracted from the MICrONS reconstruction of a cubic-millimeter patch of mouse visual cortex (67,534 neurons, about 10.4 million connections) [9]. Our starting point on the algorithmic side is the recent paper by Bader et al. [5], which introduced the "Rocket-Crane" algorithm for this problem and ran it on the same fly connectome.

The planned thesis has two connected goals:

1. Algorithmic: improve the ordering algorithm (reach a higher feedforward weight with less computation) and understand how far from the true optimum we are, so the measurement comes with a quality estimate.
2. Understanding: use the resulting orderings to describe how feedback is organized in real brains: check more carefully the observations we already have for the fly connectome, and repeat the measurement on other connectomes - mouse cortex, *C. elegans* [11], and several other publicly available connectomes listed in Section 4.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 2 Methodology

## 2.1 The Discrete Problem

The input is a directed graph $G=(V,E)$: the vertices $V$ are the $n$ neurons, and an edge $(u,v) \in E$ is a connection from neuron $u$ to neuron $v$, with a positive weight $w_{uv}$ equal to its synapse count. An ordering is a bijection $\pi\colon V \to \{1,\dots,n\}$ - every neuron gets its own position on a line - and its feedforward weight is

$$F(\pi) \;=\; \sum_{(u,v)\in E,\ \pi(u)<\pi(v)} w_{uv}.$$

We look for an ordering maximizing $F$ and report the ratio $F(\pi)/W$, where $W=\sum_{(u,v)\in E} w_{uv}$ is the total weight of all connections; the remainder $W-F(\pi)$ is the feedback under $\pi$. This is the weighted Minimum Feedback Arc Set problem in complementary form; it is NP-hard [1], so at this scale we solve it approximately.

## 2.2 The Continuous Relaxation

$F$ is defined on orderings and has no gradient, so Rocket, the gradient part of the Rocket-Crane algorithm [5], replaces the ordering with real positions $x \in \mathbb{R}^{n}$, one per neuron (sorting $x$ recovers an ordering), and the hard question "is $x_v - x_u$ positive" with a smooth one:

$$\tilde F_\beta(x) \;=\; \sum_{(u,v)\in E} \hat w_{uv}\, g\big(\beta\,(x_v - x_u)\big), \qquad \hat w_{uv} = w_{uv}/\max_{e} w_e,$$

where $g$ is a non-decreasing step-like function (the logistic sigmoid in the paper) and $\beta>0$ controls how sharp the step is: small $\beta$ gives a smooth landscape, large $\beta$ approaches the exact objective. We maximize $\tilde F_\beta$ with Adam for 20,000 steps, cycling $\beta$ over $[0.05, 1.05]$ by the paper's cosine schedule; every 100 steps the positions are scored exactly and the best are kept.

Our one change here is the shape of $g$: instead of the sigmoid, a one-sided function

$$g(z) = \begin{cases} 1, & z \ge M,\\[2pt] 1 + \tanh\big((z - M)/T\big), & z < M, \end{cases}$$

with margin $M = 0.75$ and width $T = 1.5$. Edges that already point forward with some margin get exactly zero gradient, so the gradient is spent on edges that are still backward or barely forward. The two constants were chosen on other, smaller graphs, never on the fly connectome.

## 2.3 The Full Pipeline

Our final pipeline grew out of Rocket but by now differs from it quite a bit. It has four stages; the constants below are the ones used on the fly graph. The smooth objective is only a guide: one exact scorer produces every reported number, and the final answer is the best ordering seen at any stage.

**Stage 1: warm start.** We initialize with the weighted greedy peeling heuristic of Eades, Lin and Smyth [12], which repeatedly removes vertices from the remaining graph: sinks go to the back, sources to the front, and otherwise the vertex with the largest remaining out-weight minus in-weight goes to the front. The ranks become evenly spaced positions in $[-1,1]$.

**Stage 2: gradient phase.** As in Section 2.2, starting from the warm-start positions.

**Stage 3: under-relaxed sift.** It starts from the ranks of the best stage-2 positions. Each sift sweep computes, for every neuron with all others held fixed, the exact insertion point that maximizes the neuron's own feedforward weight, and moves all improving neurons toward their targets at once (one at a time would be far too slow at this scale). A move by a fraction $\alpha$ means the neuron's position becomes fractional and the whole list is then re-ranked. With full steps ($\alpha = 1$) the iteration oscillates - neurons keep leapfrogging each other - so after the first 6 sweeps each mover goes only $\alpha = 0.7$ of the way; we run at most 40 sweeps, stopping early if no neuron can improve.

**Stage 4: block reordering.** A contiguous stretch of the order can be reordered internally without changing the direction of any edge leaving it, so each such block is an independent subproblem. Starting from the stage-3 order taken as one block, we split its induced subgraph into strongly connected components, order the components topologically (keeping each one's internal order), and recurse inside each; a block that is a single component is cut in two by position, down to blocks of 32 neurons. None of these moves can decrease the score. We run 77 cycles, each one such pass over the whole order plus two full-step ($\alpha = 1$) sift sweeps, varying the cut position between cycles; alternating helps because each kind of move opens new improvements for the other.

# 3 Preliminary Results

## 3.1 Reproducing and Improving the Optimizer

The first step was to reproduce the results of Bader et al. [5] on the fly connectome. Our reproduction reached the same plateau as the paper (about 82.9% of the total weight pointing forward; Table 1), which gave us a trusted base to build on.

| Algorithm | Paper [5] | Our runs |
|---|---|---|
| Simple baseline | 50.17% | 50.05% |
| Greedy | 69.25% | 68.07% |
| GreedyAbs | 73.12% | 72.04% |
| Rocket | 80.05% (long-run plateau 82.87%) | 82.92% |
| Rocket + Crane (integer programming, ~20 days) | 84.60% | not reproduced (needs a commercial solver) |
| **Our improved pipeline** | - | **84.26%** (~20 min) |

Table: Table 1: Share of total synaptic weight pointing forward on the FlyWire connectome.

The improved pipeline reaches a feedforward share of 84.26% in about 20 minutes on a single laptop. For comparison, the paper reaches 84.60% only after the "Crane" phase, which needs a commercial integer-programming solver (Gurobi) and about 20 days of computation. The best solution we know of scores 84.61% (a publicly shared reference solution), and a separate paper by Vahidi [6] reports the same level. That puts us within about 0.35 percentage points of the best known result at a small fraction of the computational cost, and pure heuristic ordering turned out to go further than the original paper suggested. So far most of our gain over plain Rocket has come from the discrete refinement; inside the gradient phase we only changed the starting point and the smoothing function, and there is probably more to try there.

## 3.2 A Picture of the Fly Brain as a Layered Network

Using the best ordering as a depth axis, we drew the whole fly brain as a layered network (Figure 1). Each row is a stream - a group of neurons sharing a published FlyWire annotation [4] - and each column holds one sixteenth of the ordering. A bundle between two boxes is the summed weight of the connections between those groups of neurons; bundles that point backward are drawn in red. The picture shows more structure than we expected. The main things we see in it and in the numbers behind it:

![Figure 1: The fly connectome drawn as a layered network: 17 annotated streams (rows; row height grows with the stream's neuron count) crossed with 16 slices of the optimized order (columns). Forward bundles take the color of their source stream; backward bundles are red and drawn opaque so they stand out. The red numbers at the right give each stream's share of outgoing weight that points backward, and the bar at the top gives the brain-wide forward/backward split of the weight. The mass runs diagonally from photoreceptors (top left) to descending and motor neurons (bottom right).](fig1_fly_streams.png){width=6.5in}

1. The order matches the known sensory-to-motor pathway. The optimizer is given nothing but the connection matrix, and still photoreceptors sit at median position ~7,600 out of 136,648, then lamina, medulla and lobula (the successive stages of the fly's visual system), then visual projection neurons, then descending neurons, and at the very end the motor neurons (median ~136,500); the last neuron of the entire order is a neck motor neuron. As a simple check, neurons annotated as input neurons sit mostly early in the order and annotated output neurons mostly late.
2. About 15.7% of the weight still points backward under our best ordering (an upper bound on the truly unavoidable feedback), and it is not spread evenly: the medulla sends about 20% of its outgoing weight backward, the large central-brain population about 17%, and the photoreceptors almost nothing (3%). Most of the backward weight forms short arcs within the medulla and between adjacent visual stages, plus a few long paths from the central brain back into the visual system.
3. Backward and forward connections have a different transmitter mix. If we join per-neuron transmitter predictions [7] to the ordering, acetylcholine (the main excitatory transmitter) turns out to send only 11.4% of its weight backward, below the brain-wide 15.7%, while serotonin (28.8%), glutamate (22.2%), GABA (20.9%) and dopamine (20.1%) all send more than average backward. Under this ordering, the backward connections lean toward inhibitory and modulatory transmitters.
4. The left-eye and right-eye halves of the visual system barely talk to each other: only about 3% of the connection weight inside the optic streams crosses the midline, and the two sides mix mostly in the central brain.
5. Neurons of the same cell type sit close together: in every type we have looked at, most of its neurons are tightly bunched (for example, the 2,189 Kenyon cells of the KCg-m subtype, a type from the fly's memory center, form one solid block).

All of these are, for now, observations made under a single ordering; beyond a few first sanity checks, none of them has been properly tested. Checking them is a large part of the plan below.

# 4 Future Directions

**1. Improve the algorithm and estimate its quality.** We will try to close the remaining gap of about 0.35 percentage points to the best known solution (and ideally go past it). More importantly for a thesis, we want to say something about optimality: compare against exact solutions on tractable subgraphs and derive upper bounds, so that "84.26%" comes with an estimate of how much could still be missing. We also plan a more serious attempt at the gradient phase, since so far we have mostly worked on the discrete side.

**2. Look more closely at what we already see.** The observations above need to be turned from a picture into numbers: which streams exchange feedback with which, how far backward connections reach, how the transmitter composition of feedback changes with distance, how tightly each cell type is packed in the order. Turning these into proper tests is the statistical part of the thesis: for each observation we need a statistic and some reference to compare it against. For the cell-type observation, for example, one could compare how tightly each type is packed against the packing after shuffling the type labels.

One idea for such a reference is randomized networks that keep some of the brain's statistics fixed (size, density, degrees). For the fly, local statistics such as reciprocal pairs and small motifs are already known to be above chance [8]; whether the same holds for the overall amount and shape of feedback is open. We have not fixed a specific design yet; for now it is one of the directions we are considering.

**3. More connectomes.** We also have a first run on the MICrONS mouse cortex volume, made with a simpler variant of the pipeline (the one-sided smoothing function, though tuned on the smaller graphs, so far helps only on the fly connectome), which we have only begun to analyze. Beyond that, several suitable connectomes are already public, and we want to repeat the measurement on them: *C. elegans* [11], where the graph is small enough for exact or near-exact answers; the *Drosophila* larva, a complete brain of about 3,000 neurons [13]; and the male fly's ventral nerve cord, a motor structure of about 23,000 neurons [14]. There is also a series of eight *C. elegans* connectomes taken at different ages [15], where the same measurement would show how the amount of feedback changes as the animal grows.

If the plan works out, the thesis will contain a better algorithm with an estimate of its quality, and a set of tested statements about how feedback is organized in a real brain.

# References

[1] Karp, R. M. (1972). Reducibility among combinatorial problems. In Miller, R. E. & Thatcher, J. W. (eds.), *Complexity of Computer Computations*, 85–103. Plenum Press, New York.

[2] Even, G., Naor, J., Schieber, B. & Sudan, M. (1998). Approximating minimum feedback sets and multicuts in directed graphs. *Algorithmica* 20, 151–174.

[3] Dorkenwald, S. et al. (2024). Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138.

[4] Schlegel, P. et al. (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634, 139–152.

[5] Bader, D. A. et al. (2025). Rocket-crane algorithm for the Feedback Arc Set problem. *Social Network Analysis and Mining* 15, 68.

[6] Vahidi, S. (2025). Feedforward ordering in neural connectomes via feedback arc minimization. *arXiv preprint* arXiv:2506.13799.

[7] Eckstein, N. et al. (2024). Neurotransmitter classification from electron microscopy images at synaptic sites in *Drosophila melanogaster*. *Cell* 187, 2574–2594.e23.

[8] Lin, A. et al. (2024). Network statistics of the whole-brain connectome of *Drosophila*. *Nature* 634, 153–165.

[9] The MICrONS Consortium (2025). Functional connectomics spanning multiple areas of mouse visual cortex. *Nature* 640, 435–447.

[10] Nachum, I. & Stark, E. (2026). Optimized graph layouts reveal region-specific patterns of recurrent connectivity in cortical networks. *Computational and Systems Neuroscience (COSYNE)*.

[11] Cook, S. J. et al. (2019). Whole-animal connectomes of both *Caenorhabditis elegans* sexes. *Nature* 571, 63–71.

[12] Eades, P., Lin, X. & Smyth, W. F. (1993). A fast and effective heuristic for the feedback arc set problem. *Information Processing Letters* 47, 319–323.

[13] Winding, M. et al. (2023). The connectome of an insect brain. *Science* 379, eadd9330.

[14] Takemura, S. et al. (2024). A connectome of the male *Drosophila* ventral nerve cord. *eLife* 13, RP97769.

[15] Witvliet, D. et al. (2021). Connectomes across development reveal principles of brain maturation. *Nature* 596, 257–261.

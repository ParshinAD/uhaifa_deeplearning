# Literature review: provable lower bounds on weighted MFAS for the FlyWire connectome

Status: phase 0 literature pass, 2026-11-25. Scope: what is known about certified lower bounds on
minimum feedback weight (equivalently upper bounds on achievable feedforward %), and which
methods can tighten our current ceiling on this graph with the local machine.

Conventions. FB = feedback (backward) weight, FF% = 100 (W - FB) / W, W = 41,912,141.
LB = certified lower bound on min FB. "Verified" = I opened the paper / page or a publisher /
dblp / arXiv record and checked the bibliographic data. "From memory" = I did not re-check it
in this pass; treat it as unverified. "Guess" = my own estimate, not from a source.

Local calibration numbers quoted below come from our own scout logs in `ceiling/logs/`
(`scout_structure.log`, `scout_tri_full.log`, `scout_primitives_graph.log`), not from the literature.

---

## 0. Headline findings

1. **No published certified lower bound for this instance.** Neither the Rocket-Crane paper
   (Bader et al., SNAM 2025) nor Vahidi's arXiv:2506.13799 reports any lower bound, LP value or
   optimality gap. The FlyWire challenge page lists scores only. A NeuroTrailblazers "open problems"
   page states the challenge has "no known optimality bound" (fetched 2026-11-25).
2. **The leaderboard moved past our comparator.** The live leaderboard
   (https://codex.flywire.ai/app/mfas_challenge, fetched 2026-11-25) now has
   35,544,233 (Caleb Westfall, 2026-11-08) = **84.8065 %**, FB = 6,367,908. Next entries:
   35,543,204 (G. G. de Polavieja, 2026-11-07), 35,540,505 (J.-R. Denoual, 2026-10-28),
   35,535,717 (A. Hassan, 2026-10-19), 35,505,475 (W. Burton, 2026-10-11),
   35,463,823 (Vahidi and Koutis, 2026-01-09) = 84.6147 % (this is the file we use as comparator),
   35,459,266 (Bader et al., 2024-11-16) = 84.6038 %. The charter's "best known 84.6147 %" is out of date
   (percentages computed by me from the listed integers).
3. **Someone is building exactly our certificate.** A GitHub repo `mdindoost/mfascert`
   ("certified parallel bounds for Minimum Feedback Arc Set (FlyWire challenge)") is indexed by web
   search: C++17/OpenMP parallel directed-cycle packing, certificate = list of simple directed
   cycles with integer multiplicities y_C respecting capacities w_e, LB = sum y_C, a Python
   checker `check_cert.py`, and a "Lemma 2" decomposition of feedback(order) - LB into
   "unsaturated backward capacity" and "multi-backward waste". On 2026-11-25 the repo returns
   HTTP 404 and is not in the user's public repo list; no paper or arXiv record found (arXiv API
   author search "Dindoost" shows no FAS paper). **No bound value could be verified.** Author
   M. Dindoost is at NJIT (same institution as Bader). Expect a paper from that group; re-check
   before we publish anything.
4. Theory says the cycle LP (fractional cycle packing) can be far below the integer optimum on
   expander-like or random-like parts (integrality gap up to ~2 for MAS-type relaxations, all known
   polyhedral inequality classes have "strength" at most slightly above 3/2). So a pure cycle-packing
   ceiling will probably not reach 84.8 %. Guess: the best certifiable ceiling within hours lands
   around 87-88.5 %, not lower.

---

## 1. LP relaxations for FAS / MAS / linear ordering

### 1.1 Dicycle (cycle covering) LP and its dual

Covering LP (fractional FAS): min sum_e w_e x_e, s.t. x(C) >= 1 for every directed cycle C, x >= 0.
Dual (fractional cycle packing): max sum_C y_C, s.t. sum_{C contains e} y_C <= w_e, y >= 0.
Weak duality: any feasible y gives LB = sum y_C <= tau* <= OPT. Since OPT is an integer for
integer weights, ceil(sum y_C) is also valid. This is exactly our certificate kind K1.

The equivalent maximization form is the acyclic subgraph problem with dicycle inequalities
x(C) <= |C| - 1. References:

- Grötschel, Jünger, Reinelt (1985). On the acyclic subgraph polytope. Math. Programming 33(1):28-42.
  doi:10.1007/BF01582009 (verified via Springer record and the 2022 comment below). Dicycle inequalities
  are facet-defining; their separation is polynomial (shortest path); consequently the problem is
  polynomial on weakly acyclic digraphs (which include planar digraphs). Also introduces Möbius-ladder
  inequalities for this polytope.
- Grötschel, Jünger, Reinelt (1984). A cutting plane algorithm for the linear ordering problem.
  Operations Research 32(6):1195-1220. doi:10.1287/opre.32.6.1195 (verified, cited in the 2022 comment).
- Grötschel, Jünger, Reinelt (1985). Facets of the linear ordering polytope. Math. Programming
  33(1):43-60. doi:10.1007/BF01582010 (verified). 3-dicycle, simple k-fence and Möbius-ladder
  inequalities are facets of the LOP polytope.
- Grötschel, Jünger, Reinelt (1985). Acyclic subdigraphs and linear orderings: polytopes, facets, and a
  cutting plane algorithm. In Graphs and Order, NATO ASI Series C 147. Springer,
  https://link.springer.com/chapter/10.1007/978-94-009-5315-4_7 (record found, not read).
- Martí, Reinelt (2011). The Linear Ordering Problem: Exact and Heuristic Methods in Combinatorial
  Optimization. Springer, Applied Mathematical Sciences 175. doi:10.1007/978-3-642-16729-4
  (verified record). Standard reference for LOP polytope, cuts, exact methods.

### 1.2 Linear ordering (dense) formulation

Variables x_ij for every ordered pair, x_ij + x_ji = 1, 3-dicycle (triangle) inequalities
x_ij + x_jk + x_ki <= 2. On the full graph: n(n-1)/2 = 9.3e9 variables and ~4e14 triangle constraints:
**not usable globally.** Baharev et al. (2021) call it "TI" and show it is the weakest practical choice
even on 100-node sparse graphs. It is usable only on small pieces (section 3.5, method R5).

Its value restricted to a node set H is at least the sparse dicycle LP on G[H] (triangle inequalities
on the complete digraph imply all dicycle inequalities via chords). Whether it is strictly stronger on
sparse pieces: guess "yes, sometimes", needs the calibration test proposed in R5.

### 1.3 Stronger inequality classes and integrality gaps

- Möbius ladders, fences: facets (GJR 1985, above). Their dual meaning for us: a Möbius ladder
  subgraph M with k dicycles needs at least (k+1)/2 arcs removed (unweighted rank inequality);
  with uniform capacity t on its arcs it certifies t (k+1)/2, more than the fractional cycle packing
  of M (k/2 for the ladder). In our framework this is a K2 piece with a known exact value. [rank
  inequality form: Goemans-Hall 1996 describe k-Möbius ladder inequalities as rank inequalities]
- Goemans, Hall (1996). The strongest facets of the acyclic subgraph polytope are unknown. IPCO 1996,
  LNCS 1084:415-429. doi:10.1007/3-540-61310-2_31 (verified record and abstract). Defines "strength" of a
  relaxation (improvement over the trivial one) and shows that (almost) all known valid inequalities
  have strength at most slightly more than 3/2; new facets from a pseudo-random (quadratic residue)
  construction.
- Newman, Vempala (2001). Fences are futile: on relaxations for the linear ordering problem.
  IPCO 2001, LNCS 2081:333-347. doi:10.1007/3-540-45535-3_26 (verified record). The standard LP
  relaxation has integrality gap 2 for MAS; adding fence / Möbius-type inequalities does not help on
  random-like instances; no polytime relaxation has gap below 66/65 unless P = NP.
- Newman (2004). Cuts and orderings: on semidefinite relaxations for the linear ordering problem.
  APPROX 2004, LNCS 3122:195-206 (verified record). SDP relaxation with gap at most 1.64 w.h.p. on
  random graphs G(n, d/n).
- Even, Naor, Schieber, Sudan (1998). Approximating minimum feedback sets and multicuts in directed
  graphs. Algorithmica 20(2):151-174. doi:10.1007/PL00009191 (verified via Wikidata / citing papers).
  Rounding of the fractional FAS: OPT <= O(tau* log tau* log log tau*) (weighted). So the cycle LP is
  never worse than a polylog factor, but this does not make it tight.
- Seymour (1995). Packing directed circuits fractionally. Combinatorica 15(2):281-288.
  doi:10.1007/BF01200760 (verified). Vertex version of the same tau vs tau* bound.
- A matching Omega(log n) style lower bound on the dicycle-LP integrality gap is often attributed to
  expander constructions; I did not verify a specific reference (from memory).

What this means for us (interpretation, not measured): MAS-type LP relaxations lose most on locally
tree-like / expander-like structure, where every short cycle is "cheap" fractionally but any ordering
must still cut about half the arcs. The fly connectome has a large rich club and high reciprocity
(Lin et al. 2024), so short cycles are abundant, which favours the cycle LP. But the gap between our
current LB and the best order (2.2 % of W, see section 6) is unlikely to close fully.

### 1.4 Practicality on 5.6M edges

| Relaxation | Size on FlyWire | Practical? |
|---|---|---|
| Dicycle LP, row/column generation | rows <= 5.0M arcs (residual giant SCC), columns = generated cycles | yes, the only global LP that fits |
| Dicycle LP restricted to 3-cycles | 2,765,514 directed 3-cycles in residual graph (our scout) | yes, solvable as one LP |
| Node-potential (Younger 1963) MIP, LP relaxation | n + m | LP bound is trivial (about 0); useless for bounds |
| Full LOP with triangles | 9.3e9 vars | no, pieces only |
| Möbius / fence cuts | separation heuristic only | only as small pieces |
| SDP (Newman 2004, Charikar-Makarychev-Makarychev 2007) | n vectors, m constraints | no at n = 1.25e5 without new tooling; certificate checking hard |

---

## 2. Scalable solvers for fractional packing LPs

### 2.1 Algorithms

- Garg, Könemann (2007). Faster and simpler algorithms for multicommodity flow and other fractional
  packing problems. SIAM J. Comput. 37(2):630-652. doi:10.1137/S0097539704446232 (verified); FOCS 1998
  version doi:10.1109/SFCS.1998.743463. Packing with an oracle that returns a minimum-length column.
  For cycle packing: lengths l_e = delta / w_e, repeatedly route along the shortest cycle (bottleneck
  amount), multiply l_e by (1 + eps c / w_e), stop at shortest cycle length >= 1, scale by
  log_{1+eps}((1+eps)/delta). (1 - eps)^-3 approximation; textbook iteration bound O(m eps^-2 log m)
  oracle calls.
- Fleischer (2000). Approximating fractional multicommodity flow independent of the number of
  commodities. SIAM J. Discrete Math. 13(4):505-520 (verified via citing papers). Key practical trick:
  phases with a length threshold, approximate oracle; an alpha-approximate shortest column only
  degrades the guarantee by alpha. Lets one reuse one Dijkstra tree for many routings.
- Plotkin, Shmoys, Tardos (1995), Math. Oper. Res. 20(2):257-301; Arora, Hazan, Kale (2012),
  Theory of Computing 8:121-164 (multiplicative weights survey); Young (1995) SODA "Randomized rounding
  without solving the linear program". All from memory, not re-verified.
- PDLP / first-order LP: HiGHS 1.7.0 (March 2024) added cuPDLP-C as `solver=pdlp`
  (https://github.com/ERGO-Code/HiGHS/releases/tag/v1.7.0, verified). Serial CPU, no basis, tolerances on
  the scaled problem only. scipy 1.10 bundles an older HiGHS without PDLP, so this needs `highspy`
  (new dependency, owner approval).

### 2.2 Realistic cost on this graph

Measured in `scout_primitives_graph.log` (our run): residual giant SCC after the 2-cycle reduction has
124,984 nodes and 5,000,481 arcs; one full scipy Dijkstra = 0.115 s; 100 sources = 11.0 s; BFS = 0.014 s.

- Textbook GK with one exact shortest-cycle oracle call per routing: m eps^-2 log m = 5e6 x 400 x 15
  = 3e10 calls for eps = 0.05. **Infeasible** as written (my arithmetic).
- One exact pricing round = shortest cycle through every node = 125k single-source Dijkstras:
  about 4 h on one core, about 20 min on 14 processes (extrapolated from 0.115 s; with `limit=1` the
  search is pruned and should be several times faster: guess).
- Therefore: use an LP solver on a restricted cycle pool (column generation), and use Dijkstra only
  for pricing rounds. Expect 5-20 pricing rounds (guess) = a few hours total.
- MWU on a fixed pool in numpy/torch: each iteration is a sparse mat-vec over nnz = 3 x 2.77M (3-cycles)
  to ~20M (with 4/5-cycles); 0.05-0.3 s per iteration; 1e3-1e4 iterations for eps about 0.01-0.05.
  Minutes to an hour (guess). Useful when HiGHS is too slow or memory-bound.

### 2.3 Pricing pitfalls (engineering, from scipy behaviour)

- scipy.sparse.csgraph drops explicit zeros, so arcs with dual x_e = 0 vanish from Dijkstra. Find
  zero-length cycles separately: SCCs of the subgraph {e : x_e = 0} with more than one node contain
  violated cycles. Then run Dijkstra on x_e + tiny or on integer-scaled lengths.
- Shortest cycle through s = min over in-arcs (u, s) of d(s, u) + x_us. One Dijkstra from s gives it.
- Pricing finds the most violated cycle per source; add many per round (one per source plus
  near-ties), as in GJR-style cutting planes (the 2022 comment says Dijkstra separation suits sparse
  graphs better than Floyd-Warshall).

### 2.4 Rounding a float packing into an integer certificate

1. From LP / MWU get float y_C >= 0. Compute load_e = sum_{C contains e} y_C.
2. Scale s = min(1, min_e w_e / load_e). Pick a denominator D (for example 2^20).
3. Y_C = floor(s y_C D) (Python int). Then sum_C Y_C [e in C] <= D w_e holds exactly; recheck in
   integers, and if any edge still fails (float noise), decrement Y on the cycles through it.
4. LB = ceil(sum Y_C / D). Loss <= (#cycles) / D + (1 - s) sum y, negligible for D = 2^20.
5. Optional purely integral variant (K1 with D = 1): floor y, then greedily re-pack residual capacity
   with integer cycles. Loses up to 1 per cycle before re-packing.

The covering dual x from the same LP is also useful: if x passes a full pricing round
(all cycles have x(C) >= 1), then sum w_e x_e is an upper bound on what ANY cycle packing can certify.
This is not a bound on FF%, but it tells us when to stop working on cycle packing (diagnostic).

### 2.5 Gap decomposition (useful diagnostic)

For any packing y and any order with backward set B (identity I derived myself; it seems to be the
"Lemma 2" of mfascert as described by the search snippet):

    FB(order) - sum_C y_C = sum_{e in B} (w_e - load_e) + sum_C y_C (|C intersect B| - 1)

The first term is backward capacity not used by the packing, the second is cycles cut more than once.
Both are >= 0. Where the first term is large, the packing can grow; where the second is large, the
order is probably improvable or the LP is fractional there. This localizes the gap by region / cell
type, which the charter asks for.

---

## 3. Exact and bounding methods for FAS in practice

### 3.1 Lazy cycle generation and branch and cut

- Baharev, Schichl, Neumaier, Achterberg (2021). An exact method for the minimum feedback arc set problem.
  ACM J. Exp. Algorithmics 26, Art. 1.4, 1-28. doi:10.1145/3446429 (verified, PDF read). Set-cover
  formulation over cycles, cycle matrix extended lazily (Gurobi lazy constraints in a callback), a
  heuristic FAS of the relaxation's solution is used to find new cycles. Test set 4,468 graphs; largest
  cycle matrix 2,282 rows. Practical limit stated by the authors: with 1 CPU-hour, 16 of 101 random
  graphs with n = 75, c = 9 and 28 of 101 with n = 100, c = 7 stay unsolved. So exact methods reach about
  100 nodes / several hundred arcs, not 1e5.
- Grötschel, Jünger, Reinelt (2022). Comments on "An exact method for the minimum feedback arc set
  problem". ACM JEA 27, Art. 1.3. doi:10.1145/3545001 (verified, PDF read). A plain 1985-style branch and
  cut with Dijkstra dicycle separation (C + Gurobi) is competitive and mostly faster on the same
  de Bruijn and Imase-Itoh graphs (100-120 nodes, 300-760 arcs).
- Pho, Lapidus (1973). AIChE J. 19(6):1170-1181. doi:10.1002/aic.690190614 (set-cover / tearing origin,
  cited in the comment).

For us: B&C on the full graph is out of reach without Gurobi and probably even with it (Rocket-Crane used
Gurobi only as a primal heuristic and reports no dual bound). B&C is useful on pieces of up to about
100 nodes with HiGHS (scipy.optimize.milp), with the caveat that a HiGHS optimum is not independently
checkable unless we re-verify it (section 3.6).

### 3.2 Lagrangian relaxation and decomposition

Relaxing all cycle constraints with multipliers gives back the cycle LP (no gain). The useful
Lagrangian idea is **weight splitting** (Lagrangian decomposition), which is our K2:

    if w = w_0 + w_1 + ... + w_k with all w_i >= 0, then
    OPT_w(G) >= LB_cycles(w_0) + sum_i OPT_{w_i}(H_i)

because for any FAS F, w(F) = sum_i w_i(F) and F intersect E(H_i) is a FAS of H_i. Any valid lower
bound may replace OPT_{w_i}(H_i). The best split is an LP whose dual adds, for each piece H, the
constraint "x restricted to H dominates a convex combination of FAS incidence vectors of H", i.e. the
full acyclic-subgraph polytope of H. This is strictly stronger than the cycle LP only when pieces are
solved integrally (exact DP / B&C); piece-wise cycle LPs add nothing over the global cycle LP.
(Standard reasoning; I did not find a paper that does this for MFAS specifically.)

Important consequence: after a maximal cycle packing, the residual capacity graph is acyclic, so
pieces cut from the residual give zero. Pieces must get their capacity BEFORE (or jointly with) the
cycle packing, and the split must be optimized.

### 3.3 SDP relaxations

- Charikar, Makarychev, Makarychev (2007). On the advantage over random for maximum acyclic subgraph.
  FOCS 2007:625-633. doi:10.1109/FOCS.2007.65 (verified). SDP relates advantage over 1/2 to the cut norm.
- Guruswami, Manokaran, Raghavendra (2008). Beating the random ordering is hard: inapproximability of
  maximum acyclic subgraph. FOCS 2008:573-582 (verified via citing text only). Under UGC no polytime
  algorithm, and so no polytime certifiable bound in general, separates MAS >= 1 - eps from <= 1/2 + eps.
- Newman (2004), see 1.3.
Not practical here: n = 1.25e5 vectors, and certifying an SDP bound with integer arithmetic needs an
exact PSD certificate. Skip.

### 3.4 Reductions / kernelization relevant to bounds

Exact (no loss) and already used:
- SCC split: OPT adds over SCCs. Backward weight outside the giant SCC in our order is 425 (scout).
- 2-cycle reduction: for a reciprocal pair u->v (a), v->u (b), a >= b, cost = b + (a - b)[v before u].
  So replace the pair by one arc u->v of weight a - b and add constant b. Exact for OPT. Our residual
  graph (124,984 nodes, 5.0M arcs, capacity 35,392,715) is this reduction.
- Parallel arcs: merge (FlyWire edge list has none after aggregation; assumption, check loader).
Other literature rules:
- Hecht (2018). Exact localisations of feedback sets. Theory Comput. Syst. 62:1048-1084.
  doi:10.1007/s00224-017-9777-6, arXiv:1702.07612 (verified record). Isolated cycles, essential minors,
  "orthogonal splitting"; exact for "resolvable" graphs; defines lower bounds mu, upsilon (Section 8,
  not read). Polynomial but O(|E|^2) to O(|E|^4): too slow at 5M arcs.
- Hecht, Gonciarz, Horvát (2021). Tight localizations of feedback sets. ACM JEA 26:1-19,
  arXiv:2001.01440 (verified record). Heuristic, empirical ratio <= 2.
- Xiong, Zhou, Xiao, Khoussainov (2024). Finding small feedback arc sets on large graphs. Comput. Oper.
  Res. 169:106724. doi:10.1016/j.cor.2024.106724 (verified record; method not read). Reduction + divide
  and conquer heuristic (RASstar); on FlyWire 33,037,112 = 78.8 % (Okubo lab README). Heuristic only.
- Bessy et al. (2011). Kernels for feedback arc set in tournaments. JCSS 77(6):1071-1078 (cited in
  SNAM refs; tournaments only).
- Degree-1 contraction rules exist for unweighted FVS; for weighted FAS they are not exact in general
  (my own check: contracting the only in-arc of v removes the option to cut it). Not needed for
  bounds anyway.

### 3.5 Published lower bounds on large sparse instances

I found none for graphs of this size. Exact results in Baharev et al. and the 2022 comment are on
<= 120-node graphs. Simpson, Srinivasan, Thomo (2016), "Efficient computation of feedback arc set at
web-scale", PVLDB 10(3):133-144 (cited in SNAM refs) are heuristics without bounds (from memory).

### 3.6 Independent verification of piece optima

- Exact DP over subsets for a piece H with |H| = k: f(S) = min cost to place S first;
  f(S + v) = f(S) + c(arcs from v into S). 2^k states: k = 20 -> 1e6 states, trivial; k = 24 -> 1.7e7,
  fine in numpy; k = 26 -> 6.7e7 states x 8 B = 0.5 GB, still ok. This is the verifier's K2 check.
- Larger pieces (30-100 nodes) need MIP. Independent check options: VIPR certificates
  (Cheung, Gleixner, Steffy (2017). Verifying integer programming results. IPCO 2017, LNCS 10328:148-160.
  doi:10.1007/978-3-319-59250-3_13; checker https://github.com/scipopt/vipr, verified) need exact SCIP
  (new dependency, heavy). Alternative: the piece's own K1 packing plus a few verified sub-pieces.

---

## 4. Cycle packing, Erdős-Pósa, planar duality, local bounds

- Reed, Robertson, Seymour, Thomas (1996). Packing directed circuits. Combinatorica 16:535-554
  (verified record). Younger's conjecture: Erdős-Pósa property for directed cycles (vertex version).
  Quarter / half-integral packing follow-ups (Masařík et al., Combinatorica; ESA 2019) exist (records seen).
- Seymour 1995 and ENSS 1998 (section 1.3): tau <= O(tau* log tau* log log tau*).
- Krivelevich, Nutov, Salavatipour, Yuster, Yuster (2007). Approximation algorithms and hardness
  results for cycle packing problems. ACM Trans. Algorithms 3(4) (venue from memory; PDF at
  http://www.math.tau.ac.il/~krivelev/apxcycle.pdf seen in search). Integral arc-disjoint cycle packing in
  digraphs is hard to approximate; fractional is an LP. For us: fractional certificates are allowed, so
  integral packing only matters as a simple greedy LB.
- Lucchesi, Younger (1978). A minimax theorem for directed graphs. J. London Math. Soc. (2) 17(3):369-374.
  doi:10.1112/jlms/s2-17.3.369 (verified). Min dijoin = max disjoint dicuts. By planar duality: in planar
  digraphs min (weighted) FAS = max packing of dicycles, i.e. the dicycle LP is integral and tight.
  The connectome is far from planar, so no direct use; it explains why the cycle LP is exact on some
  structured pieces.
- Tournaments: number of 3-cycles c3 = C(n,3) - sum_i C(s_i, 2) from the score sequence, and each arc
  lies in at most n - 2 triangles, so FAS >= c3 / (n - 2) (uniform fractional triangle packing;
  classical, from memory). For weighted dense pieces this is dominated by the triangle LP, so it adds
  nothing over method R1. Kenyon-Mathieu, Schudy (2007) STOC (PTAS for weighted tournaments, cited in SNAM).
- Fox, Himwich, Mani (2024). Extremal results on feedback arc sets in digraphs. Random Struct.
  Algorithms 64(2):287-308 (cited in SNAM refs). Upper bounds on FAS, not lower bounds.
- arXiv:2607.20996 (2026), "The optimal constant for minimum weight feedback arc sets in oriented
  graphs": fas_w(D) <= w(D)/2 - (sqrt2/12) sum_v w_2(v) (from search snippet). Upper bound, not useful for us.
- arXiv:2501.06935 (2025), "Feedback arc sets and feedback arc set decompositions in weighted and
  unweighted oriented graphs": upper bounds (record seen, not read).
- Diamond, Kon, Raphael (2024). Asymptotic lower bounds for the feedback arc set problem in random graphs.
  arXiv:2409.16443 (verified record). W.h.p. bound about M (1/2 - sqrt(log n / d_avg)) for G(n,p). Not a
  certificate for a fixed graph; useful only as intuition: a random graph with our m and d_avg = 41 would
  have FAS near 0.5 - 0.53 = negative -> the formula is vacuous at this density/size (my arithmetic with
  n = 1.25e5: sqrt(11.7/41) = 0.53). So no intuition gain either.

"FAS of a k-subgraph adds over edge-disjoint pieces": this is the weight-splitting lemma of 3.2. The
literature gives no better local formula than exact solution of the piece; local degree-based formulas
(score-sequence style) are fractional triangle packings in disguise.

Spectral / bisection idea (my own, considered and dropped): for a circulation f <= w, any order has
backward f-weight >= the f-cut at the middle position, which is >= lambda_2(L_f) n / 8. Back of envelope:
for an expander-like f of total weight F this gives at most about F/4, while packing the same f into
3-4-cycles gives about F/3.5. Not better, and it needs a PSD certificate. Not recommended.

---

## 5. Connectome / hierarchy literature

- Bader, Ellis-Joyce, Both, Turaga, Asoori Sriram, Chinthalapudi, Du (2025). Rocket-crane algorithm for the
  feedback arc set problem. Soc. Netw. Anal. Min. 15:68. doi:10.1007/s13278-025-01491-2 (local PDF read).
  Best 35,459,266 (84.60 %). Gurobi used as MIP primal heuristic "prioritizing feasible solutions over
  optimality gaps". Cites the cycle-constraint formulation and says it is impractical due to the number of
  cycles. **No lower bound, no gap reported.** Mentions an empirical approximation ratio only in passing.
- Vahidi (2025). Feedforward ordering in neural connectomes via feedback arc minimization. arXiv:2506.13799v1
  (local PDF read). Best 35,462,925 (0.8461). Leaderboard as of 2025-06-02. Future work mentions "exploring
  approximation bounds or optimality guarantees". **No lower bound.**
- Challenge leaderboard and rules: https://codex.flywire.ai/app/mfas_challenge (fetched 2026-11-25).
  No mention of bounds. Okubo lab repo README (github.com/CIBR-Okubo-Lab/min_feedback_final): scores only.
- Borst (2024). Connectivity matrix seriation via relaxation. PLoS Comput. Biol. 20(2):e1011904
  (cited in both papers). Continuous relaxation heuristic; the winning team's method is inspired by it.
- Dorkenwald et al. (2024). Neuronal wiring diagram of an adult brain. Nature 634:124-138 (dataset).
- Lin et al. (2024). Network statistics of the whole-brain connectome of Drosophila. Nature 634:153-165.
  doi:10.1038/s41586-024-07968-y (verified record). Highly recurrent, rich club with 30 % of neurons,
  connection reciprocity 0.138 (pair-level), higher than ER / configuration nulls; high reciprocity in
  central complex and antennal lobes. No feedforwardness limit.
- Kunin et al. (2023). Hierarchical modular structure of the Drosophila connectome. J. Neurosci.
  43(37):6384-6400 (cited in arXiv:2506.13799).
- Agony: Gupte, Shankar, Li, Muthukrishnan, Iftode (2011). Finding hierarchy in directed online social
  networks. WWW 2011 (verified record). Tatti, "Faster way to agony" (2014,
  https://users.ics.aalto.fi/ntatti/papers/tatti14agony.pdf) and "Tiers for peers" (arXiv:1903.02999):
  agony is polytime (convex penalty) via LP / flow duality; its dual is an Eulerian-subgraph type object
  (from memory: min agony = max Eulerian subgraph in the unweighted case, not verified). Agony charges
  backward edges by rank difference and same-level edges by 1; it is **not** a lower bound on FAS.
- Trophic coherence: MacKay, Johnson, Sansom (2020). How directed is a directed network? R. Soc. Open Sci.
  7(9):201138. doi:10.1098/rsos.201138 (verified). Trophic levels and incoherence F_0 in [0,1] computable on
  any network; a descriptive statistic, gives no certified FAS bound. Johnson et al. (2014) PNAS "Trophic
  coherence determines food-web stability" (from memory).
- Zhao, Zhou (2017). Feedback arcs and node hierarchy in directed networks. Chinese Phys. B 26(7):078901
  (cited in arXiv:2506.13799; statistical-physics estimates, not certificates, from memory).

Conclusion of this section: I found no theoretical or certified limit on feedforwardness for any
connectome. Our bound would be the first one I know of, unless mfascert publishes first.

---

## 6. Where we stand (from our logs)

- Our order: FF 84.2582 %, FB = 6,597,734. Best leaderboard: FB = 6,367,908 (84.8065 %).
- Certified: 2-cycle floor 2,815,303 (93.2828 %). 2-cycle + greedy 3-cycle: 4,537,757 (89.1732 %, charter),
  scout full enumeration with max-mincap-first greedy: 4,600,674 (89.0231 %, `scout_tri_full.log`,
  not yet a verified certificate).
- Gap from best LB to best known order: 6,367,908 - 4,600,674 = 1,767,234 = 4.2 % of W.
- Residual graph after 2-cycles: capacity 35,392,715; our order pays 6,597,734 - 2,815,303 = 3,782,431
  there; all 2,765,514 residual 3-cycles packed greedily give 1,785,371.
- Our order's backward edges: median rank span 18,156, 90 % > 61,014 ranks (`scout_structure.log`).
  Long spans do not mean long cycles, but they suggest the remaining backward weight is spread
  globally rather than concentrated in small dense knots, which is bad news for piece methods (guess).

---

## 7. Ranked recommendation

Ordered by expected gain per compute hour on the M-series Mac (48 GB, 14 cores, Python 3.9,
scipy 1.10 HiGHS). All strengths are guesses unless stated.

### R1. Exact LP over all residual 3-cycles (then the full cycle LP by column generation)

- What: solve max sum y_C over the 2,765,514 residual 3-cycles with capacities = residual w
  (scipy.optimize.linprog, method highs-ipm or highs-ds). Rows = arcs that lie on some 3-cycle (<= 5M,
  most rows empty -> drop them), nnz = 8.3M.
- Strength: >= greedy 1,785,371 gain. Guess: LP optimum 1.9-2.2M, i.e. ceiling 88.0-88.7 %
  (+0.3-1.0 pp). The LP dual x restricted to 3-cycles also gives the cap of any 3-cycle-only packing.
- Cost: build 1 min (scout already enumerates in 13 s). HiGHS IPM on 2.8M x ~3M with 8M nnz: guess
  10-60 min, 5-15 GB RAM. If too slow: MWU in numpy/torch on the same pool (minutes, eps 0.01-0.02).
- Certificate: K1, rounded as in 2.4. Verifier unchanged.
- Risk: HiGHS IPM memory or time on this size (scipy 1.10 has no PDLP); float residuals (handled by
  scaling + floor). Low risk overall. Do this first.

### R2. Column generation with Dijkstra pricing (full dicycle LP)

- What: start from R1's pool; each round take dual x, find zero-length cycles (SCCs of {x_e = 0}),
  then for every source s compute d(s, .) with scipy dijkstra(limit=1) and add cycles with x(C) < 1
  (up to a few per source, plus all 4-cycles that close via the 2-hop matrix A^2 if cheap). Re-solve
  with warm start (HiGHS simplex) or MWU.
- Strength: this converges to tau*, the best any cycle packing can prove. Guess: +0.5-1.5 pp over R1,
  total ceiling about 87-88 %. A final pricing round that finds no violated cycle also certifies
  sum w x as the cycle-LP value (diagnostic upper limit on K1 bounds).
- Cost: one full pricing round about 20 min on 14 processes (from measured 0.115 s per Dijkstra,
  likely less with limit pruning); 5-20 rounds; master LP grows to maybe 5-10M columns (guess) ->
  HiGHS may become the bottleneck; then switch master to MWU / PDLP (PDLP needs `highspy`, approval).
- Certificate: K1. Long cycles are fine for the verifier.
- Risk: master LP size; slow tail convergence (typical for column generation); tau* itself may be far
  from OPT (integrality gap), which caps this whole family.

### R3. Garg-Könemann / Fleischer as a stand-alone alternative to R1+R2

- What: phase-based GK: threshold alpha, for each source run Dijkstra with current lengths and route
  every found cycle shorter than alpha(1+eps) (re-checking its length after each update), raise alpha
  by (1+eps) when none found. Output scaled flow; round as in 2.4.
- Strength: (1 - eps)^3 of tau* in theory; same ceiling as R2.
- Cost: number of phases about log_{1+eps}(1/delta) = (1/eps) ln m / ln(1+eps): for eps = 0.1 about
  1,600 phases x 20 min pricing = far too slow if every phase needs a full round. Only practical in
  C++/numba with incremental Dijkstra (numba = new dependency). The mfascert description (C++/OpenMP)
  suggests this route is what that group did.
- Certificate: K1. Risk: cost; I rank it below R1+R2 for Python.

### R4. Weight splitting with exact small pieces (K2), placed BEFORE the cycle packing

- What: choose node sets H of 18-24 nodes where the LP is loose: high second term in the gap
  decomposition (2.5), fractional dual x, dense reciprocal / rich-club knots. Give each piece capacity
  c_H <= w on its induced arcs, solve OPT_{c_H}(H) by subset DP, pack cycles on w - sum c_H. Optimize the
  split by trying c_H = w on H vs the R2 loads on H and keeping the piece only if
  OPT_{c_H}(H) > (R2 packing value lost on arcs of H).
- Strength: gains come only from the piece's integrality gap. For dense near-tournament pieces the
  triangle LP is usually close to integral (guess), so guess: +0.05-0.3 pp total. Möbius-ladder
  subgraphs are a special case with a closed-form value (k+1)/2.
- Cost: DP on 24 nodes about 1-5 s in numpy; thousands of pieces = hours; selection heuristic is the
  hard part.
- Certificate: K2, already in the verifier (DP recomputation).
- Risk: net gain near zero after accounting for capacity taken from cycles; pieces must be chosen with
  the global LP in mind (residual after maximal packing is acyclic, so pieces from the residual give 0).

### R5. Calibration arm: how loose is the cycle LP here? (cheap, decides R4 vs stop)

- What: on 30-80 node connectome subgraphs (dense lanes, mouse meso 148 if solvable) compare: cycle LP,
  LOP-LP with all 3-dicycle inequalities on the complete digraph (n(n-1)/2 vars, lazy triangles,
  HiGHS), exact optimum (HiGHS milp with lazy cycles, or DP for <= 24 nodes). Also run R1/R2 on
  random graphs with the same degree sequence.
- Strength: no bound by itself. Tells whether LOP-LP on pieces beats the cycle LP (if yes, add a K3
  "LOP-LP dual on a piece" certificate: rational multipliers on triangle inequalities, checkable in
  integers) and how big the integrality gap is per piece.
- Cost: minutes to an hour. Risk: small pieces may not represent the global gap.

### R6. Heavy-layer split (cheap variant of R4)

- What: w = min(w, t) + max(w - t, 0). The heavy part lives on arcs with w > t (backward weight
  percentiles in our order: 99.9 % below 105 syn per edge, so for t around 50-100 the heavy subgraph is
  small: guess). Solve or tightly bound the heavy subgraph (SCC split, then pieces small enough for
  DP / B&C), cycle-pack the light part.
- Strength: guess small (+0-0.2 pp); heavy edges are rarely backward in good orders, so the heavy
  subgraph is probably nearly acyclic. Try only if R5 shows large per-piece gaps on heavy knots.
- Certificate: K1 + K2. Risk: heavy subgraph too large for exact methods, or acyclic (gain 0).

Not recommended: SDP (size and certification), full LOP triangle LP (size), spectral / bisection bound
(weaker than packing, section 4), Younger node-potential LP (trivial bound), random-graph asymptotics
(not certificates), agony / trophic measures (not FAS bounds).

### Dependencies per method

| Method | Needs beyond current env |
|---|---|
| R1 | none (scipy 1.10 linprog/HiGHS); optional torch for MWU |
| R2 | none; optional `highspy` (HiGHS >= 1.7, PDLP) if the master LP gets too big |
| R3 | practical only with `numba` or a C/C++ extension |
| R4, R6 | none (numpy DP; scipy milp for pieces above DP size, but then not independently verified) |
| R5 | none; K3 certificate needs a verifier extension first (charter rule) |
| Verified MIP pieces > 26 nodes | exact SCIP + VIPR (heavy, not advised) |

### Expected outcome (guess)

R1 then R2 should move the certified ceiling from 89.02-89.17 % to about 87-88.5 %. R4/R6 may add a few
tenths. A certified ceiling near 85 % looks out of reach with these tools because of the integrality
gap of all known polyhedral relaxations on random-like structure (Goemans-Hall 1996; Newman-Vempala 2001).
The final report should state the cycle-LP value tau* (from R2's dual) separately from the certified LB,
so the reader can see how much of the remaining gap is "LP gap" versus "heuristic gap".

---

## 8. Sources fetched in this pass

- Local PDFs: `s13278-025-01491-2 (2).pdf` (Rocket-Crane, SNAM 15:68), `2506.13799v1.pdf` (Vahidi).
- https://codex.flywire.ai/app/mfas_challenge (leaderboard, 2026-11-25)
- https://www.neurotrailblazers.org/content-library/connectomics/open-problems-undergrad/
- https://github.com/mdindoost/mfascert (search index only; 404 on fetch)
- https://arnold-neumaier.at/ms/feedback.pdf (Baharev et al. 2021, author copy)
- https://www.zib.de/userpage/groetschel/pubnew/Juenger-Groetschel-FeedbackArcSet.pdf (GJR 2022 comment)
- https://github.com/CIBR-Okubo-Lab/min_feedback_final (README)
- https://github.com/ERGO-Code/HiGHS/releases/tag/v1.7.0
- Publisher / dblp / arXiv records for the other verified items, as listed inline.

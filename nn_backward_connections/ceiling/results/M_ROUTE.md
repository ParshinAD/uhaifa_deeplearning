# Method R: order-guided routing (H2) and where the unexplained feedback sits (H5)

Date 2026-11-25. Code: `ceiling/src/m_route_core.py` (C kernel compiled at run time with the
system `cc` and called through ctypes; no new package), `m_route_greedy.py`,
`m_route_improve.py`, `m_route_rrr.py`, `m_route_h5.py`. All runs from the repo root with the
`allen` env. The ceiling/ files are not committed; `git_commit` in the JSONs is the base commit
657786f, so a rerun needs these files as they are now.

## Best verified result

**c21_route_improve_r1: LB = 5,843,849 syn, ceiling 86.0569 %, verifier PASS.**
That is 88.57 % of our H64 feedback (6,597,734). Remaining gap to our order 753,885 syn
(1.80 pp of W), to the leaderboard top (6,367,908) 524,059 syn (1.25 pp).
Pure order-guided routing (H2 in the strict sense, every cycle = one H64 backward edge + a
forward path) explains 5,604,533 syn = 84.95 % of the H64 feedback (c20_route_greedy_g1, PASS).

## What was done

Every backward edge b = (u -> v) of the guide order is a commodity with demand w(b); the
kernel searches a path v ~> u over edges with residual capacity > 0 (bidirectional level BFS,
hop limit H, rank window pruning on the forward DAG), closes it with b into a simple cycle and
pushes the bottleneck. Integer amounts, so D = 1 in every certificate. Phases with growing hop
limit H = 1, 2, 3, ... route all short cycles first. H = 1 is exactly the 2-cycle packing
(2,815,303, the c00 floor).

1. Greedy (`m_route_greedy.py`): commodities ordered by rank span (then weight), phases
   1,2,3,4,5,6,8,10,14,20.
2. Improvement attempts: commodity order (weight, random), capacity rounds alpha = 1/4, 1/2,
   3/4, 1 (in round alpha a forward edge offers only floor(alpha w) - load), rip-up and reroute
   (`m_route_rrr.py`: for a batch of unrouted b, find the saturated edges on a shortest
   uncapacitated forward path, rip up all cycles through them, reroute batch then ripped,
   keep the round only if the total grew).
3. Beyond one backward edge per cycle: after the DAG routing, route the leftover H64 backward
   demand on the FULL residual graph (no rank window), so a cycle may contain several H64
   backward edges (`--residual-phases`). Every cycle of the residual graph has at least one
   H64 backward edge, so these commodities cover all cycles. Control r2: only 2-cycles on the
   DAG, then full-graph packing (no guide). Variant r3: per H, DAG(H) then full graph(H).
   Other guide orders: H64 with rank jitter (uniform noise of 50 ranks) and H64 reversed
   inside blocks of 20 ranks.
4. H5 table from the g1 state (`m_route_h5.py`), plus an isolation test: each unrouted b alone
   on the original forward DAG with unlimited capacities.

## Runs (all certificates checked by `ceiling/src/verify_cert.py`)

Wall times were measured with 3 to 6 of these jobs running at once on the shared machine, so
they are indicative. Feedback F = backward weight of the guide order. "DAG routed" = demand
routed by one-backward-edge cycles of that guide.

| run | what | DAG routed | LB | ceiling % | verifier | wall s |
|---|---|---|---|---|---|---|
| g1 | greedy, span order, H up to 20 | 5,604,533 of 6,597,734 (84.95 %) | 5,604,533 | 86.6279 | PASS | 365 |
| g2 | greedy, weight order, H up to 10 | 5,572,493 | 5,572,493 | 86.7043 | PASS | 414 |
| g3 | greedy, random order (seed 0), H up to 10 | 5,565,784 | 5,565,784 | 86.7204 | PASS | 416 |
| g4 | guide = H64 + rank jitter 50 (F 6,989,008), span order | 5,487,738 (78.52 % of its F) | 5,487,738 | 86.9066 | PASS | 454 |
| g5 | guide = H64 reversed in blocks of 20 (F 7,268,303) | 5,400,404 (74.30 %) | 5,400,404 | 87.1149 | PASS | 448 |
| i1 | capacity rounds 1/4,1/2,3/4,1, H up to 10 | 5,551,943 | 5,551,943 | 86.7534 | PASS | 1579 |
| smoke | rip-up and reroute from g1, batch 500, 3 min | 5,604,533 (+0, 115 rounds, none kept) | 5,604,533 | 86.6279 | PASS | 226 |
| smoke20 | rip-up and reroute from g1, batch 20, 3 min | 5,604,572 (+39, 19 of 146 rounds kept) | 5,604,572 | 86.6278 | PASS | 218 |
| **r1** | DAG greedy H up to 10, then full residual graph H 2..8 | 5,604,161 | **5,843,849** | **86.0569** | PASS | 1519 |
| r2 | control: 2-cycles only, then full graph H 2..8 (no guide) | 2,815,303 | 5,680,500 | 86.4466 | PASS | 638 |
| r3 | interleaved per H: DAG(H) then full graph(H), H up to 8 | 5,471,485 | 5,835,041 | 86.0779 | PASS | 1033 |

Files: `certs/c20_route_greedy_g{1..5}.npz`, `certs/c21_route_improve_{i1,r1,r2,r3}.npz`,
`results/m_route_*.json` (command, commit, seed, phase log, verifier line), `logs/m_route_*.log`.
The two rip-up smoke certificates (c22_route_rrr_smoke*.npz) were deleted after the run since
they equal g1 within 39 syn; their JSON and log stay. The routing state files
(`results/state/`) were deleted at the end; `m_route_h5.py --state g1` needs
`m_route_greedy.py --tag g1` rerun first (6 min).

Per hop phase in g1 (routed increment): H1 2,815,303; H2 1,696,149; H3 777,694; H4 216,956;
H5 65,950; H6 20,033; H8 10,363; H10 1,713; H14 351; H20 21. Paths beyond 6 hops add almost
nothing. The longest cycle in g1 has 19 edges.

## What did not help

- Capacity rounds (i1): -52,590 against plain greedy. Holding capacity back made early phases
  take longer paths in later rounds instead of saving room.
- Rip-up and reroute: batch 500 never improved (each round lost 26 to 110 syn); batch 20 gained
  39 syn in 3 min. Greedy with short-first phases is a strong local optimum for this move.
- Commodity order: span-first is best; weight-first -32,040, random -38,749 (one seed, so the
  random number has unknown spread; the gap between orders is small relative to the 1 M
  unrouted).
- Worse guide orders route less: jitter 50 (-0.93 pp FF of the guide) loses 116,795 of DAG
  routing, block reversal (-1.60 pp) loses 204,129. So with this router the routable demand
  follows the quality of the guide; a better guide order would likely route more (guess, not
  tested; the leaderboard order is not available to us and `data/best_solution/` is off limits).

## H2 verdict (prediction in HYPOTHESES.md)

- "Greedy order-guided routing beats H1's LB": yes. Pure H2 greedy 5,604,533 (certified)
  against 4,600,674 (greedy 2+3-cycles, scout, uncertified). Kill line not hit.
- "LP optimum of the restricted flow lands in [5.3, 6.2] M": not tested; no LP was solved.
  The greedy integer value 5.60 M is a lower bound on that LP and lies in the interval.
- Most unrouted demand is NOT for lack of a path: of 993,201 unrouted syn only 7,196 (on 2,259
  edges) have no forward path v ~> u even with unlimited capacity. The rest lost the competition
  for capacity, so the restricted flow is capacity-bound, and the LP could still be well above
  greedy.
- Allowing cycles with several H64 backward edges adds 239,688 on top of the DAG routing (r1).
  The guide helps: the same full-graph packing without the DAG stage (r2) gets 163,349 less.
  Interleaving DAG and full-graph per hop limit (r3) gives 8,808 less than r1 (one run each).
  Verdict: supported for the greedy part; LP part open.

## H5 data (from g1, `results/m_route_unrouted.npz`, `results/m_route_h5_g1.json`)

npz fields: back_eid, src, tgt, demand, routed, unrouted, span, super_class_src,
super_class_tgt, no_forward_path_alone, super_class_values. Total unrouted 993,201 on 221,476
edges (128,485 fully unrouted).

By rank span (rank u - rank v):

| span | edges | demand | unrouted | unrouted / demand | share of all unrouted |
|---|---|---|---|---|---|
| 1-10 | 38,703 | 262,617 | 2,720 | 0.010 | 0.003 |
| 11-100 | 24,180 | 205,999 | 12,072 | 0.059 | 0.012 |
| 101-1000 | 60,757 | 481,675 | 72,698 | 0.151 | 0.073 |
| 1001-10000 | 264,480 | 1,577,670 | 300,027 | 0.190 | 0.302 |
| 10001-100000 | 749,642 | 3,751,842 | 563,199 | 0.150 | 0.567 |
| > 100000 | 35,483 | 317,931 | 42,485 | 0.134 | 0.043 |

By super_class (both endpoints in the class; otherwise cross-class; classes with < 100
unrouted syn omitted: motor 0, sensory 103, sensory_ascending 1, ascending 227):

| class | edges | demand | unrouted | unrouted / demand | share of all unrouted |
|---|---|---|---|---|---|
| central | 168,395 | 2,167,029 | 409,090 | 0.189 | 0.412 |
| optic | 785,379 | 3,117,142 | 354,566 | 0.114 | 0.357 |
| cross-class | 199,184 | 1,216,658 | 207,724 | 0.171 | 0.209 |
| visual_projection | 17,012 | 66,096 | 15,049 | 0.228 | 0.015 |
| descending | 1,666 | 20,598 | 5,439 | 0.264 | 0.005 |
| visual_centrifugal | 830 | 5,266 | 1,002 | 0.190 | 0.001 |

Observation: short-span feedback (< 100 ranks) is almost fully explained; 87 % of the unrouted
demand has span 1e3 to 1e5. But the unrouted rate does not grow with span past 1e4 (0.19 at
1e3-1e4, 0.15 at 1e4-1e5, 0.13 above), so long spans dominate mainly because they carry most of
the feedback. Central brain feedback is unexplained at a higher rate than optic (0.19 vs 0.11),
yet the optic lobe still holds 36 % of the unrouted syn because it has the most feedback.
H5 verdict: partly supported. "Long span" yes in absolute terms, "central brain" yes in rate,
"not in the optic lobe" no: about a third of the unexplained weight is optic. Caveat: this
splits what one greedy certificate fails to explain; it is not a statement about where OPT
pays, and a different packing would move the unrouted demand around.

## Next ideas

1. Fractional routing (Garg-Konemann / Fleischer with a bidirectional Dijkstra in the same C
   kernel) on the same commodities, to get the restricted-flow LP value and a K1 certificate
   with D > 1; its dual lengths also give an upper bound on what any H64-guided packing can do.
2. Run the full-graph residual stage on top of the best packing of method P (short-cycle LP),
   or the other way round; the two methods use the same capacity.
3. A better guide order (our next champion) should route more; rerun g1 + r1 when one exists.
4. Targeted rip-up with an exact small max-flow per blocked edge instead of random batches.

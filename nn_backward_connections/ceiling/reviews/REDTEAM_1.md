# Red-team review 1: feedforward ceiling certificates

Reviewer did not write any of the reviewed code. Scope: `ceiling/src/CERT_FORMAT.md`,
`ceiling/src/verify_cert.py`, `ceiling/src/cert_io.py`, `ceiling/tests/test_verify_cert.py`,
`src/mfas/io.py`, `src/mfas/metrics.py`, certificates `ceiling/certs/c1r6_cg_r6.npz` and
`ceiling/certs/c21_route_improve_r1.npz`. Scratch code (not in the repo):
`/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad/redteam/`
(`mycheck.py`, `attacks.py`, outputs `mycheck_out.txt`, `attacks_out.txt`, `verify_out.txt`).

**Overall verdict: PASS.** No way found to make an invalid certificate pass. Both certificates
re-check exactly with an independent checker. Small remarks at the end, none affects the bound.

## 1. Mathematics: PASS

Scorer (`metrics.py::_ff_weight`): edge e is feedforward iff `arr[tgt] > arr[src]`, strict.
Feedback = W - ff, W = sum of all weights = 41,912,141 (`g.total_weight`, same value as the
verifier's `W`, see section 3).

Argument, for any position vector p (ties allowed). Let B = edges with p[tgt] <= p[src].
- Every directed closed walk has an edge in B: if all edges were strictly increasing, p would
  increase around the walk and return to its start. So simplicity is not even needed for
  soundness; the verifier's simplicity check is extra strictness. Ties and self loops only put
  more edges into B, so they can only raise feedback.
- Feedback(p) = sum_{e in B} w_e >= (1/D) sum_{e in B} load(e), with
  load(e) = sum_{C contains e} y_C + sum_{pieces containing e} cap. Splitting the sum:
  the cycle part is >= sum_C y_C (each cycle has at least one edge in B, y_C >= 1 > 0), the piece
  part is sum_j (cap-weighted backward edges of piece j under p restricted to its nodes)
  >= sum_j claim_j, if claim_j <= min over strict orders of the piece. A tied restriction
  costs at least as much as some strict tie-break, so the min over strict orders is the right one.
- Feedback is an integer, so feedback >= ceil((sum y + sum claim)/D) >= floor(...) = LB.
  Using floor is conservative and correct. Hence ff(p) <= W - LB for every p.

K2 rule: the DP in `piece_min_feedback` computes best[S] = min_v best[S\v] + sum_{u in S\v}
C[v,u] (v placed last, edges v->u with u earlier are backward), which is the exact min over
orders. Diagonal set to 0, so self loops are under-counted (safe). Parallel edges are summed
into C with `np.add.at` (correct). Overflow: all DP values <= sum(caps), checked <= int64 max.
The tests compare the DP with brute force on random graphs (`test_dp_matches_brute_force`).

Graph facts relevant here (own count, `mycheck.py`): 0 self loops, 0 duplicate (u,v) pairs,
min weight 2. So the self-loop and multi-edge corner cases do not arise on the connectome anyway.

## 2. Verifier bypass attempts: PASS (0 of 22 bypassed)

Read every check in `_verify`. Order of checks matters and is right: edge ids are range-checked
(`_check_ids`, line 165) before they are used as fancy indices, so negative ids cannot wrap.
ptr: `ptr[0]==0`, `ptr[-1]==len`, every length >= 2 (cycles) / >= 1 (pieces), which also forces
strict monotonicity, so overlapping or non-monotone slices are impossible. Closure check
compares `tgt[e_i]` with `src[e_{i+1}]` including wrap-around. Simplicity key `cid*n + src`:
node ids from the loader are in [0, n), so no collisions; a collision could only cause a false
reject anyway. Load uses int64 only when `max(vals)*max(count)` fits, otherwise Python ints;
`D*w` the same. Sums via `_exact_sum` (Python int when the bound fails). The header checksum
is computed from the loaded graph arrays (src, tgt, weight), and meta is used only for binding,
never as input to the bound.

Crafted invalid certificates (`attacks.py`, run on toy graphs against brute-force optimum):

| id | attack | result |
|---|---|---|
| a01 | y stored big-endian `>i8` | FAIL (dtype) |
| a02 | 4 copies of a cycle, y = 2^62, int64 sum wraps | FAIL (overloaded, exact load 2^64) |
| a03 | D = -1, y = -100 | FAIL (D < 1) |
| a04 | closed walk reusing edge and node | FAIL (node twice) |
| a05 | negative edge id -1 (python wrap) | FAIL (out of range) |
| a06 | edge id = m | FAIL (out of range) |
| a07 | non-monotone ptr [0,4,2,4] | FAIL (length -2) |
| a08 | zero-length cycle with y = 100 | FAIL (length 0) |
| a09 | D as shape (1,) | FAIL (ndim) |
| a10 | y as shape (1,1) | FAIL (ndim) |
| a11 | sum y = 3k-1 with D = 3 (floor vs ceil) | PASS with LB = 0, floor correct, not a bypass |
| a12 | zip with two `y.npy` members (valid + huge) | FAIL (reads last one, overloaded) |
| a13 | piece with self loop, claim counts the loop | FAIL (claim 7 > exact 2) |
| a14 | two pieces both using full edge capacity | FAIL (overloaded) |
| a15 | cycle + piece share an edge over capacity | FAIL (overloaded) |
| a16 | meta dataset name swapped | FAIL |
| a17 | meta numbers as float / string | FAIL (string total_weight); float n_nodes alone would match, harmless |
| a18 | uint64 y = 2^64-1 | FAIL (dtype) |
| a19 | cyc_edges as Fortran 2-d | FAIL (ndim) |
| a20 | D = 2^62, load one unit over D*w | FAIL (exact object path) |
| a21 | multigraph, parallel edge over-used | FAIL (overloaded) |
| a22 | single-edge piece with claim 1 | FAIL (exact min 0) |

Existing suite: `python -m pytest -q ceiling/tests/test_verify_cert.py` -> `106 passed in 16.52s`.

## 3. Graph binding: PASS

Parsed `data/connectome_graph.csv.gz` with pandas from scratch (strings -> Python int, own
`np.unique` + `searchsorted` remap, own round-trip assert `ids[idx] == raw`), not via `io.py`.

```
columns ['Source Node  ID', 'Target Node ID', 'Edge Weight'] rows 5657719
same src True same tgt True same w True n 136648 136648
my checksum 4603e90178daf8f28eb9a399f96ed5636ec831ab19c2cea737a18817dd20271e
scorer total_weight 41912141.0
```

The checksum equals `edge_checksum` in both certificate headers. The verifier CLI uses
`load_dataset("connectome")`, the same loader the scorer and the champion use. No dropped,
merged or re-indexed edges; m = number of CSV rows.

## 4. Independent re-check of the two certificates: PASS, exact match

`mycheck.py` (no import of verify_cert / cert_io / io.py; closure via own next-index, simplicity
via lexsort on both src and tgt per cycle, load with int64 after bounding sum(y) < 2^62, capacity
compared in Python int):

```
c1r6_cg_r6:           D=10000 C=3734535 entries=12846791 maxlen=15 closure_bad=0 repeat_src=0
                      repeat_tgt=0 overloaded=0 sum_y=58900452725 LB=5890045 tight=1425169
c21_route_improve_r1: D=1 C=1760774 entries=5790408 maxlen=11 closure_bad=0 repeat_src=0
                      repeat_tgt=0 overloaded=0 sum_y=5843849 LB=5843849 tight=2291938
```

Official verifier (`python ceiling/src/verify_cert.py <cert>`): both PASS with
LB = 5,890,045 and 5,843,849, same sum_y and same tight-edge counts. Both certificates are pure
K1 (no pieces), so the K2 code path is not used by the claimed bound.

## 5. Champion sanity check: PASS

Champion `results/champions/H64_connectome_s42_cuda.npz` (same rank as
`load_graph_and_champion_order("connectome")`), backward = `rank[tgt] <= rank[src]`:

```
champion ff (frozen scorer) 35314407  fb 6597734  pct 84.25817950936937
c1r6:  cycles with no champion-backward edge: 0   FB >= LB: True  gap 707689
c21:   cycles with no champion-backward edge: 0   FB >= LB: True  gap 753885
```

So with the best certificate: champion ff = 35,314,407 <= W - LB = 36,022,096
(ceiling 85.9467 %, verifier Fraction 3602209600/41912141).

## Remarks (no effect on soundness)

1. The K2 piece path is only tested on toy graphs; neither real certificate uses it. If a future
   certificate uses pieces, that path deserves its own independent re-check.
2. `meta.notes` of c1r6 points to inputs in a session scratch folder (`.../scratchpad/mshort/...`);
   those generator inputs will be lost. The certificate is self-contained, so validity is not
   affected, only reproducibility of how it was built.
3. A zip with duplicate member names is accepted and the last member wins (numpy behaviour).
   The verifier decides on what it reads, so not a bypass, but a different reader could see other
   arrays. Rejecting duplicate members would remove the ambiguity.
4. Meta fields are compared with `==`, so `136648.0` equals `136648`. Harmless, the checksum binds
   the graph.
5. The bound is on the min feedback of this weighted graph under the frozen scorer; it says
   nothing about the mouse graph (float weights are rejected by design).

# Review of method D (upper bound on the cycle LP value tau*)

Reviewer: independent fresh-context agent, 2026-11-25. Worktree commit b01ab22. Read-only except this file.
Scratch: `/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad/review_dual/`
(`rcheck.c`, `rcheck.py`, `bf_test.py`, `full_check.py`, `full_check.log`, `scipy_sample.py`).

Claim: tau* <= 3096300477189/500000 = 6,192,600.954378.

**Verdict: PROVEN (PASS-with-minor-issues).** My own checker, which does not use the project's kernels,
reproduces L = S = 10^6 on all 136,648 heads with 0 violating edges and the same sum w X = 6,192,600,954,378.
The issues below are about provenance and doc wording. None affects soundness.

## Where the certificate is

`scratchpad/mdual/repair_cg2_final_heads.npz`, key `X`, int64, 5,657,719 entries, min 0, max 1,000,004,
3,261,604 zeros. Written by `m_dual_repair.py ... heads` (np.savez of X1, then `chk.check(G, w, X1)` on the
same array). It can be regenerated deterministically from `scratchpad/cg2/duals_final.npz` (key x): X0 =
ceil(S x / w) on SCC edges, then one stale raise per head. I recomputed X0 myself: X1 >= X0 everywhere,
576,529 edges changed, X1 = 0 outside SCCs, sum w X0 / S = 6,139,728.814759, raise cost 52,872.139619. All
of this matches the JSON.

## 1. Math of the shortest-cycle check (`min_cycle` in `m_dual_core.py`): PASS

- With X >= 0, the minimum over closed walks equals the minimum over simple cycles. Every simple cycle has a
  unique vertex s of minimum rank (rank is a permutation: argsort of argsort), and all its other vertices have
  larger rank. So the Dijkstra from s, restricted to rank > rank(s) and closed by any edge into s, finds a
  cycle of length <= the shortest cycle whose minimum-rank vertex is s. Every closed walk it reports is a
  real cycle, so it never underestimates. Hence L equals the exact minimum.
- Pruning: `d >= B` break and `nd >= B` skip only drop walks that cannot beat the current best, which is
  safe because only strictly shorter cycles matter. `best` is read without the lock (volatile int64, aligned).
  That is benign because it only decreases, and it is written under the mutex.
- Overflow: cutoff = INT64_MAX/4, X <= ~1e6 (asserted < 2^52 in `to_int_lengths`), path sums <= n * max X
  ~ 1.4e11. No overflow.
- Zero lengths and zero cycles: handled (a zero cycle gives L = 0, and then UB = inf). Self-loops: found when
  s = u (the dataset has 0 self-loops, and its 0 duplicate pairs I checked). Edges between SCCs carry no
  cycle and are set to 0. The check does not rely on this anyway, since it runs on the whole graph.
- Sum: `wsum` uses int64 with a guard (2405 * 1,000,004 * m ~ 1.4e16 < 2^62). It is computed on exactly the
  checked X1. I recomputed it in Python ints and got 6,192,600,954,378. 6192600954378 / 10^6 =
  3096300477189/500000.

## 2. Rounding: PASS

X0 uses ceil, but soundness does not depend on that. The exact check certifies L >= S on the stored integers,
and the sum is taken over the same integers. The raise also uses `max(X_b, S - dist)` with stale distances,
so it is also safe on its own.

## 3. Independent check on the full graph: PASS

- My own C (`rcheck.c`, 4 pthreads). For every head v, an unrestricted Dijkstra from v with cap
  S + 1 - min X over the in-edges of v. Then for every in-edge e = (u, v) it computes X_e + dist(v, u). This
  covers all heads and all 5,657,719 edges, in 1,989 s wall.
  Result: min = 1,000,000 = S exactly, 0 edges below S, 124,527 heads tight at S. sum w X =
  6,192,600,954,378, UB = 6,192,600.954378.
- Second implementation: scipy `csgraph.dijkstra` (float64, epsilon 1/(4n) added so zero edges are not
  dropped, then floor; exact because all values are < 2^53). Run on the witness head (tgt of edge 3,577,400)
  plus 2,000 random heads: min = 1,000,000, 0 violations. The witness head gives exactly S, which matches
  the JSON witness.

## 4. Kernels vs brute force on adversarial graphs: PASS

`bf_test.py`: 8 hand-made cases plus 1,500 random multigraphs (2-9 nodes). The hand-made cases cover a zero
cycle, a zero cycle that avoids vertex 0, parallel edges, a self-loop as the minimum, a star where the hub has
the lowest rank and a cheaper zero/one cycle lies among the leaves, lengths of 1e12, and an acyclic graph.
The random cases use lengths in {0,1,2}, sparse lengths up to 1e6 and lengths in 1-49. The project's
`min_cycle` was compared against brute force (networkx simple_cycles with the cheapest parallel edge plus
self-loops) with 1 and 4 threads, with random rank permutations instead of degree order, and with
cutoff = true L + 1. My checker was compared too. Result: 0 mismatches for both. (A first run showed
mismatches that came from my own test harness, which set S = BIG + 1 on acyclic graphs. The test was fixed
and the code was not changed.)

## 5. Wording in M_DUAL.md: PASS-with-issues

- Correct: the text explicitly says the result is NOT a proven integrality gap. It says that leaderboard -
  tau* >= 175,307 is proven and that the split between suboptimality and gap is open. The numbers check out:
  85.2248 %, 54,187, 0.13 pp, 175,307, 0.42 pp, 2.83 %, 405,133.
- Issue (minor): the header says "ceiling/ files of this method are not committed" at deecf6b. They are
  committed in b01ab22, so the header is stale. The JSON `git_commit` deecf6b refers to uncommitted code.
- Issue (minor, provenance): the certificate X and the input duals exist only in the scratchpad. There is
  no sha256 in the JSON. I suggest storing a hash of X (and of duals_final.npz) in the result JSON, or
  pinning X, because a scratchpad cleanup would make the bound non-replayable without rerunning CG2.
- Not verified by me: the lower bound 6,138,414 (CG2), and therefore the bracket width and the sentence
  "5.92-6.0 M is wrong". The calibration statements and the window LP numbers were also not checked.

# Origin TODO — the researcher's original 7-item list, and where each item went

**Why this file exists.** The original list of research ideas was given verbally and was never
committed as a file. It survived only as `(TODO N)` annotations inside the *idea* column of
`experiments/roadmap.md`. That column is rewritten whenever an idea is promoted — when `A-SCC`
became `H36` (commit `92f49a7`, 2026-08-09) the row was replaced and the `(TODO 6)` marker was
lost with it. The list is therefore recorded here **once**, immutably; the roadmap keeps
priorities, the queue files keep status, and this file keeps provenance.

Rule: never renumber or reword a row below. If an item is promoted, add the new ID to the
"lives as" column — do not overwrite the original wording.

| TODO | original idea (as given) | lives as | queue owning status | status at 2026-08-15 |
|---|---|---|---|---|
| **1** | seed-to-seed distance / ordering variability without the greedy start | `Q02` | `questions.md` | **answered** (2026-08-03) — score stable (82.8845 ± 0.0225 pp), ordering not (Spearman 0.958); sinks stabler than sources but the effect closes by k=10000 |
| **2** | partial / stochastic sift — "SGD on part of the neurons" | `A-SUB` | `backlog.md` | open (low) — must be disambiguated first: as *continuous* block-coordinate GD it repeats the killed H13; only the *discrete* stochastic-sift recast is novel |
| **3** | alternate discrete ↔ gradient refinement | `A-ALT` | `backlog.md` | open (med) — scaffold exists in `dr_tmp/kick_gate.py`; Q01 gives the design constraint (a gradient phase only ranks a better order higher above β·std ≈ 470; Rocket runs at 148) |
| **4** | why does starting Rocket from the best solution drift the score DOWN? | `Q01` | `questions.md` | **answered** (2026-08-17) — at the achievable scale the surrogate ranks the better order LOWER (net −174.88); it is a relaxation-scale property, not an optimizer weakness |
| **5** | very-close / tight-position initialization | `A-INIT` | `backlog.md` | **prototype gate passed** (2026-08-15) — see `backlog.md` § A-INIT and `log.md`; the theory is derived and the prior ("doubly discouraged") was falsified on the proxies |
| **6** | SCC decomposition + SCC-structured insertion | `A-SCC` → **`H36`** | `backlog.md` | sizing gate passed (2026-08-09); variant cycle **not yet run**. NOTE: the `auto/campaign` queue uses the id `H36` for a *different* item (recursive within-giant-SCC insertion) — see the ID-collision note in `roadmap.md` |
| **7** | alternate surrogate (flat top/bottom, slower-decaying tanh) | `A-SURR` | `backlog.md` | open (low), near-dead — H11 (−0.039 pp) and H34 (−0.64 pp) closed the continuous-surrogate family |

**Not from this list** (added later, tracked in the same queues): `Q03` (gap structure vs H35 +
degeneracy of the 84.6% level), `A-CLU` (cluster-decomposed ordering), `A-SCALE` (≡ the killed
H03), `G01`/`G02` (Track C — random graphs vs brains, thesis goal #2), and the whole H01–H35
improvement series.

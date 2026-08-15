# `autoresearch/lit/` — literature scan index

Owner: **scout** (queue item **L01**). This directory did not exist before 2026-08-16; the campaign
had run its entire life with zero literature input (see `queue.json` L01 `outcome`).

**No compute was run for this scan.** A GPU/CPU experiment (`dr_tmp/night_run.sh`, the P07/S01
epoch grid) was running unattended throughout. Nothing here rests on a run made by the scout;
every external claim carries a retrieved source, and every internal claim cites a file already in
the repo.

---

## Scan L01 — 2026-08-16

### Files

| file | contents |
|---|---|
| `vahidi-2025.md` | **written by a second scout running concurrently**, not by this scan. Deeper on arXiv:2506.13799 itself — it retrieved the paper's GitHub code repository, the back-edge length table (mean 20,536 / max 126,650), the executed Algorithm-4 parameters, and a critique of meta-rule M4's own evidence. Read it together with the next row; the two were written independently and agree on every shared fact. |
| `notes-vahidi-flywire.md` | the two Vahidi papers, the FlyWire leaderboard, and the mechanism that produced our target number |
| `notes-lop-and-ocm.md` | Linear Ordering Problem metaheuristics (2026 state of the art) and PACE 2024 one-sided crossing minimization |
| `notes-scalable-fas-and-negatives.md` | scalable FAS heuristics, multilevel ordering, and the honest negatives |
| `hypotheses.md` | the ranked shortlist of new falsifiable hypotheses (queue ids **H45 / H46 / H47** + two gated ones). Note the id history: a concurrent cycle allocated `H44` and independently queued its own `H45`/`H46` for the same two mechanisms while this scan was writing; the duplicates were merged, not overwritten. See the banner at the top of `hypotheses.md`. |

### Queries actually issued (WebSearch), 2026-08-16

1. `FlyWire connectome challenge feedback arc set maximum feedforward edge weight solution`
2. `minimum feedback arc set heuristic large graphs 2025 scalable algorithm millions of edges`
3. `linear ordering problem large scale heuristic insertion neighborhood ejection chain 2024 2025`
4. `PACE 2024 one-sided crossing minimization heuristic track solver description local search large instances`
5. `connectome hierarchy ordering directed graph flow hierarchy fly brain 2025 feedforward ranking`
6. `OCMu64 PACE 2024 heuristic solver one-sided crossing minimization sifting local search description`
7. `"Finding small feedback arc sets on large graphs" Computers Operations Research divide and conquer reduction rules`
8. `multilevel coarsening refinement minimum linear arrangement directed graph ordering million nodes`
9. `Vahidi Koutis 2026 feedback arc set connectome FlyWire 35,463,823 improved ranking algorithm`
10. `"Interval-Based Refinement" weighted feedback arc set Vahidi Koutis non-overlapping backward-edge intervals dynamic program`
11. `ejection chain variable depth search Lin-Kernighan permutation linear ordering feedback arc set large graph`

### Pages actually fetched (WebFetch)

- `arxiv.org/abs/2506.13799`, `arxiv.org/html/2506.13799v1` (×3, different sections), `ar5iv.labs.arxiv.org/html/2506.13799`
- `codex.flywire.ai/app/mfas_challenge` (the leaderboard)
- `arxiv.org/abs/2605.31051`, `arxiv.org/html/2605.31051v1`
- `arxiv.org/html/2411.17596` (Arcee)
- `arxiv.org/abs/2412.06361` (weberknecht — abstract only, PDF text not extractable)
- `drops.dagstuhl.de/.../LIPIcs.IPEC.2024.26` (PACE 2024 report)
- `arxiv.org/html/2412.16181v2`
- `arxiv.org/a/vahidi_s_1` (author listing)

### Retrieval failures — recorded, not worked around

| source | what happened |
|---|---|
| `2506.13799v1.pdf` (local, repo root) | **could not be read.** The `Read` tool needs `pdftoppm` (poppler), which is not installed, and Bash is disabled in this session, so no text extractor could be invoked. **Everything this scan says about Vahidi 2025 comes from the arXiv HTML rendering, not from the local PDF.** A future cycle with Bash should re-derive Algorithm 2 from the PDF and check it against `notes-vahidi-flywire.md`. |
| Vahidi & Koutis 2026, SSRN 6221201 | **403 Forbidden** on both `papers.ssrn.com` and `ssrn.com`. Only the abstract was obtained, and only through search-engine snippets (two independent queries returning consistent text). **The full text was not read.** Every statement attributed to it is flagged as abstract-only. |
| `S0305054824001965` (Comput. Oper. Res. 2024) | ScienceDirect 403. Abstract only, via search snippets. |
| `arxiv.org/pdf/2412.06361` (weberknecht) | WebFetch returned undecoded PDF stream; the small model could not extract text. Only the arXiv abstract page was usable, and it contains no algorithmic detail. |
| Springer `10.1007/s13278-025-01491-2` (Bader, Rocket-Crane) | 303 redirect to an IdP login. Not pursued — the local PDF `s13278-025-01491-2 (2).pdf` is the repo's own reference and the Crane phase is already documented in `CLAUDE.md` as Gurobi-MIP + ~20 days, i.e. out of scope by construction. |

### Coverage gaps — what a follow-up scan should do

1. **Read the local `2506.13799v1.pdf`** with a working extractor and verify the Algorithm 2 gain
   derivation in `notes-vahidi-flywire.md` line by line. That derivation is currently *mine*,
   checked against the paper's stated formula; it is not a quotation of the paper's proof.
2. **Obtain Vahidi & Koutis 2026 (SSRN 6221201)** through an institutional route. It is the only
   document that describes how the exact number in `data/best_solution` was produced.
3. Not covered at all: spin-glass / Potts annealing for MFAS; GPU-parallel permutation local
   search; DAG structure-learning orderings (NOTEARS-family). Judged low expected value against
   meta-rule **M1** (continuous family exhausted) and **M7** (expensive global embeddings lose),
   but they are genuinely un-searched, so this is a gap and not a finding.
4. Not covered: whether any of the other leaderboard entrants (Hashorva; Ellis-Joyce et al.;
   Zheng et al.) published a method. The leaderboard lists no methods.

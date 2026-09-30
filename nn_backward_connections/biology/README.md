# Biology — biological discoveries from MFAS-ordered connectomes

This folder collects **biological findings and interesting facts** extracted from
the MFAS/layering work: what ordering whole connectomes by feedforward optimality
reveals about real brains. Target venue: **COSYNE abstract** (findings ranked by
expected interest to the computational-neuroscience community).

## Structure

| path | contents |
|---|---|
| `FINDINGS.md` | The master list: every finding, ranked by priority for the scientific community, with evidence, provenance and review status. |
| `studies/` | One markdown report per dedicated study (question, method, numbers, caveats, review verdict). |
| `analysis/` | Scripts that produced the numbers (each runnable via the `allen` env `$PY`). |
| `results/` | Raw outputs of runs: JSON/CSV/NPY. Every quoted number traces to a file here or to an existing artifact elsewhere in the repo. |
| `outputs/` | Figures (PNG, dpi≈120). |

## Rules (inherited from the repo, non-negotiable)

1. **Frozen oracle only.** All feedforward scores come from `src/mfas/metrics.py`
   (`score_from_order`, discrete score on CPU). We evaluate orderings; we never
   edit the scorer.
2. **Every number is logged.** A claim quotes the script + output file that
   produced it. No estimated, remembered or hand-edited numbers.
3. **Review status is explicit.** Each finding carries one of:
   `unreviewed` → `reviewed` (independent re-check passed) → `retracted`.
   Nothing enters the COSYNE shortlist unreviewed.
4. **Caveats travel with the claim.** Attenuation direction, seed counts,
   device provenance, annotation coverage — stated next to the number,
   not in a footnote elsewhere.
5. All content in English (repo rule); Russian summaries live in chat, not here.

## Relationship to `layering/`

`layering/FLY_FINDINGS.md`, `layering/MICRONS_FINDINGS.md` and
`layering/QA_ORDER_AND_ROBUSTNESS.md` are the source studies for many findings
here; this folder does not duplicate their full text — `FINDINGS.md` states each
claim in COSYNE-ready form and points back. New studies live here directly.

# Project memory

Facts and decisions that are not derivable from the code or git history. One entry per
fact, newest first. Each entry links to where the evidence lives.

## 2026-11-24 — A7 inversion replicated on the Male CNS

`male_cns/pipeline/run_nulls_l4.py`, write-up `male_cns/README.md` section 11. Same L4
type-dcSBM generator and G01 protocol as FlyWire. Real male graph 80.83 % FF (2 seeds) is
below all 12 type-matched draws (81.228 +- 0.013), gap -0.40 pp (FlyWire -0.57); at 5x budget
-0.47 pp for one draw (FlyWire -0.65). 2-cycle floor real 7.64 % vs null 2.82 % of W.
Fallback-label control (untyped neurons as singleton types) 81.267, still above real.
Independent review: PASS-with-corrections, all applied.

- Not done: W0 and L2R rungs on the male, more than one probe draw, a simple-graph variant
  of the null (draws have 4.19 % parallel slots), A7 on MICrONS.

## 2026-11-17 — Male CNS v1.0 analysed (second fly, other sex), first pass

All in `male_cns/` (README = findings, BUILD = provenance). Graph: traced-only pairs, weight >= 2
(the FlyWire graph's floor), n = 164,425, m = 15.27M. Feather files need pyarrow, which is
in a separate gitignored venv `male_cns/.venv`; `allen` was not changed.

- H64 with connectome constants, one seed on MPS: 81.81 % FF (FlyWire 84.26 %). Same G01
  protocol: male CNS 80.83, male brain only 80.68, female 83.66.
- The ~3 pp gap needs no extra male feedback structure: male random nulls lose as much order
  or more (male graph 2.2x denser; density alone not isolated); real graph beats the best null
  by ~1.8-2.0 pp in both. The 2-cycle floor explains about half of the gap.
- Replicated in the male: A3 chemistry (DA most backward, GABA/Glu backward, ACh forward), B7
  inhibitory hubs (CT1, APL), A4 axis and type contiguity, B1 giant SCC, 2-cycle floor ~42 % of FB.
- New: FlyWire positions of ascending, some descending and JO neurons are artefacts of the
  missing nerve cord; do not compare those lanes across datasets. Cross-sex type-position
  Spearman 0.77 vs same-brain ceiling 0.92-0.95; dimorphic types do not move more (p = 0.81).
  Left/right: both orders have a side bias of opposite sign (likely optimizer choice).
- Interactive page `male_cns/outputs/streams/malecns_streams.html` (same as FlyWire "Fly
  Connectome Streams"); biology write-up `male_cns/FINDINGS.md` / `.ru.md`. Extra reciprocity in
  the male sits in mushroom body, olfaction and central complex, not in the optic lobe.
- Not done: more seeds (CUDA), B6, A5 re-tune, B4 (A7 L4 rung done 2026-11-24, entry above). Page not published as an artifact
  (publishing needs the user's approval).

## 2026-11-17 — FlyWire best known score moved to 84.807 %

Live leaderboard (codex.flywire.ai/app/mfas_challenge, read 2026-11-17): top is Caleb
Westfall, 2026-11-08, 35,544,233 / 41,912,141 = 84.807 %; places 2-5 (Oct-Nov 2026) at
84.71-84.80 %. Vahidi (84.615 %) and Bader et al. (84.604 %) are now below them. Every
row has a downloadable order. Codex has used a new synapse table (Yu et al. 2025) since
July 2025, so external scores may target different weights than our v783 graph.

- Why it matters: `reports/STATUS.md`, `autoresearch/sota.json` target, `DASHBOARD.md`
  and `layering/FLY_FINDINGS*.md` quote 84.61 % as the best/reference. The gap from H64
  (84.258 %) is ~0.55 pp, not 0.36 pp.
- How to apply: quote the leaderboard value with its date; re-score a downloaded order
  with the frozen oracle and confirm its synapse table before comparing.
- Evidence: `docs/datasets_survey/V_verification.md` (claim 1), `D_nonneural_benchmarks.md`.

## 2026-11-17 — dataset survey done

A web survey ranked other datasets for the pipeline: `docs/datasets_survey/README.md`
(ranked picks, corrections, limits). Finished in a second session (gap files G, H;
fact-check V2; mouse-table check M). Do not redo it.

Corrections found by the survey:
- `layering/FLY_FINDINGS.md` / `.ru.md` §11 quote Winding 2023 as "66 % forward"; the
  paper has no such number (its recurrence figure is 41 % of brain neurons). Not yet fixed.
- Prior work: Durbin 1987 (PhD thesis, ch. 8) sorted C. elegans nerve-ring neuron classes
  max-forward: 4.8 % of chemical synapses upward (8.4 % on averaged data), feedback
  "almost all" reciprocal, in small modules. He does not say "local". Cite it.
- Hoeller et al. (Male CNS optic lobe): 45 % FF / 43 % lateral / 12 % FB by flow layers,
  69 % of FB inhibitory. Different method; not comparable with an MFAS feedback share.
- `data/table_mouse.txt` provenance still unknown; Oh 2014 (raw, p-thresholded, 4 simple
  normalizations), SEU-net, Allen-net, BRICseq-net and Knox 2019 ruled out with a positive
  control; Harris 2019 and Calabrese 2015 ruled out earlier on size.
- Evidence: `docs/datasets_survey/V2_verification.md`, `M_mouse_table.md`.

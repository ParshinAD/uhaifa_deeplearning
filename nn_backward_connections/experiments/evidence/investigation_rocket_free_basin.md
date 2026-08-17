# Forensic investigation — the "Rocket-free arm beats the mouse champion" observation

**Date:** 2026-08-16
**Trigger:** `autoresearch/HANDOFF.md` § "H41 — the live scientific question":
> "The prototype's mouse row (+0.0589 pp) came from a greedy+sift start that skipped Rocket
> and sits in a different, higher basin (93.083 vs 92.902). It does not transfer."

**Constraint honoured:** no compute was run. `dr_tmp/night_run.sh` was in flight (connectome
epoch grid, arm `epochs=2500`) throughout. Everything below is read off disk.

---

## VERDICT

**REAL on mouse — and the campaign missed it — but it does NOT generalize, and the
generalization is already falsified by artifacts on this disk.**

Three separable claims:

| claim | verdict |
|---|---|
| The comparison is like-for-like (same graph, same denominator, same frozen scorer, same stage-3/stage-4 constants) | **REAL — confirmed on every axis** |
| A Rocket-free pipeline beats the **mouse** champion by **+0.1659 pp** at ~1/1000 the wall-clock | **REAL — and never followed up** |
| The gain is an artefact of the prototype giving stage 4 a huge cycle budget | **FALSE — disproved twice on disk** |
| "Rocket is unnecessary" / this transfers to the primaries | **FALSE — connectome −0.2743 pp, microns −0.1293 pp, both measured** |

So the headline the task hypothesised ("a pipeline with no gradient phase beats the champion")
is true **only on the 148-node tripwire graph**. On both primaries the same intervention loses,
by ~2× and ~1× the mouse gain respectively. HANDOFF's one-clause dismissal was *directionally*
right but *understated the mouse fact*: it compared 93.083 against the Rocket-start stage-3
fixed point (92.902) and never against the **champion** (92.917014).

---

## 1. Is the comparison like-for-like? — YES, on every axis

### 1.1 The fixture is the production `mouse` dataset

`experiments/proto_H41.py:124-125`:
```python
if name == "mouse":
    return load_dataset("mouse"), None
```
No synthetic, no sub-sample. The `hard400` / `hard6000` fixtures in the same file are synthetic;
`mouse` is not.

| quantity | prototype (`experiments/outputs/proto_H41.json` → `mouse@10s/_setup`) | production (`results/20260810T213945Z-H42-mouse-s42-verify-ea0edb.json`) |
|---|---|---|
| `total_weight` | `9.159036176272162` | `9.159036176272162` |
| `n_nodes` | `148` | (not recorded in run JSON; `148` in `experiments/outputs/proto_sift_fullrange.json` → `mouse[0].n`) |
| `n_edges` | `583` | `583` (same source) |

Denominator identical to 16 significant figures. Same graph.

Independent chain-of-custody check: the prototype's `greedy_pct = 90.12629794323948`
(`proto_H41.json` → `mouse@10s/_setup`) is bit-identical to `greedy_pct` in
`experiments/outputs/proto_sift_fullrange.json` → `mouse[*]` and to `greedy_raw` in
`experiments/outputs/proto_h32_trophic.json` → `fixtures/mouse/rows[*]`. And production's
Rocket output `pure_best_pct = 92.47934218373807` equals `h02_pct` in
`proto_sift_fullrange.json` → `mouse[*]`. The whole mouse fixture lineage is one graph.

### 1.2 The scorer is the frozen one

`experiments/proto_H41.py:73` imports `pct, score_from_order` from `mfas.metrics` — the frozen
oracle (`src/mfas/metrics.py`, governance banner at lines 19-21). `sift_underrelaxed` and
`alternate_scc_sift` both take their accept/reject from `mfas.metrics.score_from_order`
(`src/mfas/refine/underrelax.py:38-40, 44`). Mouse weights are float, so `_as_sum_dtype`
upcasts to float64 — same accumulation as production. No float32 path anywhere.

### 1.3 The stages are the production stages with the production constants

`experiments/proto_H41.py:88-96` copies the constants **from `src/mfas/experiments/H42.py`**:

| constant | proto_H41.py | H42.py (production) |
|---|---|---|
| stage-3 `k_full` | `K_FULL = 6` | `_K_FULL = 6` |
| stage-3 `alpha` | `ALPHA = 0.7` | `_ALPHA = 0.7` |
| stage-3 `max_sweeps` | `STAGE3_SWEEPS = 40` | `_MAX_SWEEPS["mouse"] = 40` |
| stage-4 `sift_sweeps` | `ALT_SIFT_SWEEPS = 2` | `_ALT_SIFT_SWEEPS["mouse"] = 2` |
| stage-4 `k_full` | `ALT_K_FULL = 2` | `_ALT_K_FULL = 2` |
| stage-4 `min_block` | `MIN_BLOCK = 32` | `_MIN_BLOCK = 32` |
| stage-4 `n_cycles` | `BIG = 100_000`, wall-bound at 10 s | `_ALT_CYCLES["mouse"] = 32` |

The **only** two differences from the production mouse pipeline:
1. **Rocket (stage 2) is skipped** — `experiments/proto_H41.py:161-178` (`stage123`, docstring:
   *"Rocket (stage 2) is skipped on purpose"*).
2. Stage 4 is stopped by a 10 s wall-clock instead of 32 cycles.

Difference (2) is shown in §2 to be worth exactly **0.000000 pp**.

### 1.4 Pipeline stages, budget, cycle counts actually run

`proto_H41.json` → `mouse@10s`:

| stage | value |
|---|---|
| greedy-FAS | `90.12629794323948 %` |
| + stage-3 `sift_underrelaxed` (14 of 40 sweeps → converged early, i.e. a true fixed point) | `stage3_pct = 93.08288021668459 %` |
| wall for greedy + stage 3 | `stage123_wall_s = 0.00799417495727539 s` |
| arm A: + stage-4 `alternate_scc_sift`, **7385 cycles**, 10.0 s | `A_champion.final_pct = 93.08288021668459 %`, `delta_pp = 0.0` |
| arm B: + stage 4 + segment moves, 3099 cycles, 10.0 s | `B_champion_plus_segment.final_pct = 93.14175369790512 %` |

`sift_underrelaxed` stops early "only at a true fixed point (no movers) or when `time_budget_s`
is exceeded" (`src/mfas/refine/underrelax.py`, docstring). The prototype passes **no** time
budget to stage 3, and 14 < 40, so the greedy-start stage-3 order is a genuine fixed point.
Production's Rocket-start stage 3 is also a fixed point: `sift_converged: true`,
`n_sift_sweeps: 5`, `sift_sweeps_requested: 40`. **Fixed point vs fixed point of the same
operator.** Not a truncation comparison.

---

## 2. Quantification — and the crux (basin, not cycle budget)

### 2.1 The numbers

| arm | pct | source |
|---|---|---|
| **champion H42 mouse** | **92.91701410211007** | `results/20260810T213945Z-H42-mouse-s42-verify-ea0edb.json` → `pct`; `autoresearch/sota.json` → `datasets.mouse.pct_mean = 92.917`, n=20 seeds, std 0 |
| champion, stage-3 only (Rocket-start) | 92.90180243040469 | same file → `variant_attrs.sift_best_pct` |
| champion, Rocket only | 92.47934218373807 | same file → `variant_attrs.pure_best_pct` |
| **Rocket-free, stage-3 only (greedy-start)** | **93.08288021668459** | `proto_H41.json` → `mouse@10s/_setup.stage3_pct` |
| Rocket-free + stage 4 (7385 cycles) | 93.08288021668459 | `proto_H41.json` → `mouse@10s/A_champion.final_pct` |
| Rocket-free + stage 4 + segment | 93.14175369790512 | `proto_H41.json` → `mouse@10s/B_champion_plus_segment.final_pct` |

Deltas against the champion (arithmetic on the figures above):

* Rocket-free (greedy → sift, no stage 4 needed): **+0.16586611457452 pp**
* Rocket-free + H41 segment moves: **+0.22473959579505 pp**

For scale: the largest mouse move this campaign has ever shipped is H36's **+0.0152 pp**
(`sota.json` → `datasets.mouse.caveats[1]`: 92.90180243 → 92.91701410); H42's mouse delta is
exactly **0.0000 pp** (same caveat block, "TIE, NOT A WIN"). The Rocket-free arm is ~11×
H36's move.

### 2.2 Wall-clock cost

| | wall |
|---|---|
| champion mouse run (s42 verify) | `wall_clock_s = 8.159990787506104 s` |
| of which stage 3 + stage 4 | `sift_time_s = 0.0025932788848876953` + `alt_time_s = 0.03835177421569824` = **0.0410 s** |
| ⇒ Rocket's share | ≈ 8.12 s (≈ 99.5 %) |
| Rocket-free arm (greedy + stage 3) | `stage123_wall_s = 0.00799417495727539 s` |

The Rocket-free arm is **~1020× cheaper** and scores higher. It is strictly dominant on mouse.

### 2.3 THE CRUX: is it the cycle budget? — **No.** Disproved twice.

**(a) The prototype's own arm A.** Stage 4 was given **7385 cycles** — 231× production's 32 —
and returned `delta_pp = 0.0`, i.e. `A_champion.final_pct == _setup.stage3_pct` to all 16
digits. Extra stage-4 compute bought nothing from the greedy-start order. `alternate_scc_sift`
is deterministic (the `split_frac` schedule cycles through a fixed
0.5/0.382/0.618/0.25/0.75 pattern, visible in both `alt_log` and the prototype `log_tail`), so
production's first 32 cycles are a prefix of the same sequence and must also return
93.08288021668459. *(This last step is an inference from determinism, not a measurement — it is
exactly what the settling run in §5 nails down.)*

**(b) `experiments/outputs/proto_H42_mouse.json`.** Nine stage-4 allocations from
`(sift_sweeps=8, n_cycles=8)` through `(1, 128)` — a 16× spread in cycles — all return
**exactly** `92.91701410211007` from the Rocket-start stage-3 order of `92.90180243040469`.
Mouse stage 4 is saturated in *both* basins; it cannot manufacture 0.166 pp.

**Where the difference actually lives.** Entirely at the stage-3 boundary:

```
Rocket-start stage-3 fixed point   92.90180243040469
greedy-start stage-3 fixed point   93.08288021668459
                        difference  +0.18107778627990 pp
```

Same function, same `k_full=6`, `alpha=0.7`, `max_sweeps=40`, both converged. The 5000-epoch
gradient phase moves the order from greedy (90.126) to 92.479 — but into a basin whose
under-relaxed-sift fixed point is **0.181 pp worse** than the one reachable from raw greedy.
On mouse, Rocket is not neutral; it is **actively harmful**.

### 2.4 Two independent corroborations, pre-dating H41

* **`dr_tmp/size_global_discrete.json`** (Phase 6, 2026-06-22 — a *different* sift, written
  standalone in that script, still scored by `mfas.metrics`):
  `mouse[*].sift_fullrange_on_greedy_noRocket` = **93.0471786135207 / 93.01179337907796 /
  93.00438972100531** (seeds 42/123/999), against `sift_fullrange_on_h02` = 92.89539478121993 /
  92.90180243040469 / 92.89539478121993. Rocket-free wins in all three, and all three are above
  the champion's 92.917014.
* **`experiments/outputs/proto_h32_trophic.json` / `proto_h33_magnetic.json`**, control arm
  `greedy_sift` (this one uses the *production* Jacobi sift, `mfas.refine.insertion.sift`):
  mouse = **92.91629587453781**, std 0 over 3 seeds — i.e. **0.0007 pp BELOW** the champion.

That third row is the important qualifier: **skipping Rocket alone is not enough.** The win
needs the *under-relaxed two-phase* sift (H35's `sift_underrelaxed`, α=0.7, k_full=6). Plain
Jacobi from greedy lands 0.0007 pp short of the champion; under-relaxed from greedy lands
0.166 pp ahead. The mechanism is `greedy start × under-relaxation`, not `no Rocket` on its own.

---

## 3. Does it generalize? — **No. Already answered on disk, in the negative, on both primaries.**

Both epoch grids run the **production H42 pipeline with only `epochs` varied** —
`experiments/proto_P07_sizing.py:73-106` reads `H42._K_FULL`, `H42._ALPHA`,
`H42._MAX_SWEEPS`, `H42._ALT_CYCLES`, `H42._ALT_SIFT_SWEEPS`, `H42._ALT_K_FULL`,
`H42._MIN_BLOCK` straight out of the shipped module. The `epochs=0` arm is therefore
*exactly* "production minus Rocket".

### microns — `experiments/outputs/proto_P07.json`, `arms[*]`

| epochs | pure_pct | sift_pct | final_pct | est_run_wall_s |
|---|---|---|---|---|
| **0** | 79.3686163429024 | 82.70498268341854 | **83.11151992749353** | 218.73 |
| 2500 | 82.34563853761912 | 82.90382614083374 | 83.12094166464239 | 362.85 |
| 5000 | 82.59187638473075 | 82.97747932104014 | 83.13557749891774 | 391.72 |
| 10000 | 82.81725135006481 | 83.06121655210264 | 83.15791435335748 | 578.56 |
| 20000 | 83.03422402189739 | 83.17129698620641 | 83.22166529431371 | 951.79 |
| 80000 (champion) | — | — | **83.24085291200831** | 3258–3398 |

Champion source: `results/20260810T074449Z-H42-microns-s42-confirm-f91b66.json` → `pct`.
**Rocket-free − champion = −0.12933298451478 pp.** Monotone increasing in epochs; no
non-monotonicity, no basin inversion.

### connectome — `experiments/outputs/proto_S01_connectome.json`, `arms[0]` (untracked; being written by the live night run)

| epochs | pure_pct | sift_pct | final_pct | est_run_wall_s |
|---|---|---|---|---|
| **0** | 68.91342773446004 | 83.44720686065644 | **83.87974978419737** | 760.13 |
| 20000 (champion) | — | — | **84.15409511053134** | 1233.59 |

Champion source: `results/20260810T064240Z-H42-connectome-s42-confirm-f41d7e.json` → `pct`
(`score = 35270783.0` / `total_weight = 41912141.0`).
**Rocket-free − champion = −0.27434532633397 pp.** Rocket is worth more on connectome than on
microns, and it is the dataset that carries the mission target.

### The mouse-only picture

| dataset | n | Rocket-free final | champion | Δ (Rocket-free − champion) |
|---|---|---|---|---|
| mouse | 148 | 93.08288021668459 | 92.91701410211007 | **+0.16587 pp** |
| microns | 67,534 | 83.11151992749353 | 83.24085291200831 | −0.12933 pp |
| connectome | 136,648 | 83.87974978419737 | 84.15409511053134 | −0.27435 pp |

Sign flips with graph size. This is the signature of mouse being a 148-node, 583-edge,
near-saturated instance, exactly as `experiments/findings.md` and `sota.json`'s mouse caveats
already say.

### An honest asymmetry that partially favours Rocket in this table

Both primary Rocket-free arms had their **stage-3 sift capped**: connectome `n_sift_sweeps = 40`
= `H42._MAX_SWEEPS["connectome"]`, microns `n_sift_sweeps = 12` = `H42._MAX_SWEEPS["microns"]`.
Those caps were sized against a *Rocket* start. On mouse the greedy start needed 14 sweeps where
the Rocket start needed 5 — nearly 3×. So the connectome/microns `epochs=0` arms were almost
certainly stopped before their fixed point, and **−0.274 / −0.129 pp are lower bounds on how well
greedy + sift could do** with an uncapped stage 3. This does not rescue the "Rocket is
unnecessary" claim (the gaps are large and the epoch curves are cleanly monotone), but it is a
real, cheap, untested follow-up — see §5b.

---

## 4. Cross-check against the campaign's own beliefs

**M2 — "Only the starting BASIN (H02) and DISCRETE refinement (H30/H35) ever moved the metric."**
→ **CONFIRMED and sharpened.** This is a starting-basin effect, squarely on M2's live axis. What
it adds: M2 was written about gradient *dynamics* being **null**. On mouse the gradient phase is
not null, it is **negative** — it destroys 0.181 pp of reachable stage-3 quality. M2 should be
read as "dynamics interventions are null; the gradient phase itself has a *sign* that depends on
the graph."

**M6 — "The init → plateau curve is flat on connectome (82.87–82.93 % across inits spanning
36–69 % quality)."** → **CONFIRMED but its scope must be narrowed.** M6 is a statement about the
*Rocket plateau*, and the P07/S01 grids do not touch it. What they show is that the **refined**
score is *not* flat in Rocket epochs: connectome 83.880 → 84.154 across 0 → 20,000 epochs,
microns 83.112 → 83.241 across 0 → 80,000. So "init barely matters for the plateau" must never be
paraphrased as "the gradient phase is ceremonial." The `HANDOFF.md` agenda item 2 already caught
this on microns ("the early read was wrong"); the same correction applies to the whole framing.

**The belief that is actually WRONG.** `experiments/findings.md`, Phase-6 summary,
"Cross-cutting insight (extends #2)":
> "The **discrete refiner does the work; the basin it starts from barely matters.** Full-range
> exact-gain sift recovers the gap from a greedy-FAS order about as well as from a Rocket order
> … so **greedy+sift is the design**; no smarter init earns its cost."

That sentence is now measurably false in **both** directions and was never revised:
* on the primaries the basin matters a lot and Rocket is worth +0.27 / +0.13 pp (P07 / S01);
* on mouse the basin matters a lot in the *other* direction and the campaign never noticed that
  its own control arm was beating its own champion.

It is also self-inconsistent with the shipped design: findings.md says "greedy+sift is the
design", yet H30 → H35 → H36 → H42 all keep Rocket. That inconsistency is the reason this
observation could sit in three separate artifacts for eight weeks without anyone comparing it to
`sota.json`.

**No `killed.json` entry covers this.** The nearest, M7, is about *expensive* warm-starts losing
to greedy — orthogonal. There is no killed item "remove the gradient phase". `queue.json` does
not contain one either.

---

## 5. What would settle it

### (a) Primary — the mouse Rocket-free arm on the exact production pipeline (≈ 10 s total)

Everything in §2 is production-shaped except that the prototype stopped stage 4 on wall-clock
instead of at 32 cycles. `experiments/proto_P07_sizing.py` is the tool that removes that last
gap: it runs the shipped H42 pipeline verbatim with only `epochs` varied. It has no `mouse`
entry, so add exactly one line to `PLAN` (`experiments/proto_P07_sizing.py:64-70`):

```python
    "mouse": ("proto_P07_mouse.json", [0, 5000]),
```

then run:

```bash
cd /d/1/bot/UHaifa/deep_learning/mfas_autoresearch/nn_backward_connections && \
PYTHONPATH=src /c/ProgramData/anaconda3/envs/allen/python.exe \
  experiments/proto_P07_sizing.py mouse
```

**DO NOT RUN WHILE `dr_tmp/night_run.sh` IS ALIVE** — it touches the GPU (`select_device("auto")`)
and would poison the connectome grid's `wall_clock_s`. Wait for `night run COMPLETE` in
`dr_tmp/night_run.log`.

Cost: the `epochs=5000` control is the champion's own 8.2 s run; the `epochs=0` arm is ~0.05 s.
Writes `experiments/outputs/proto_P07_mouse.json`.

**Pass/fail, stated in advance:**
* `arms[epochs=0].final_pct == 93.08288021668459` and
  `arms[epochs=5000].final_pct == 92.91701410211007` → the observation is **confirmed at
  production fidelity**; file it as a queue item (an `H43` = "H42 with `_EPOCHS['mouse'] = 0`"),
  and note it must go through `confirm` at ≥5 seeds and `audit.py --gate promotion` like anything
  else. It will still be a **mouse-only** win, so under `CAMPAIGN.md` it cannot promote as a
  general win — but it *does* mean the mouse row is beatable by 0.166 pp by a strictly cheaper
  pipeline, which is worth recording however it is labelled.
* `arms[epochs=0].final_pct != 93.08288021668459` → the prototype's wall-clock-bound stage 4
  differed materially from the 32-cycle one; re-open as an artefact.

If editing `PLAN` is undesirable, the identical measurement with no source change (same imports,
same constants, byte-equivalent to `one_arm`):

```bash
cd /d/1/bot/UHaifa/deep_learning/mfas_autoresearch/nn_backward_connections && \
PYTHONPATH=src /c/ProgramData/anaconda3/envs/allen/python.exe - <<'PY'
import numpy as np
from mfas import io
from mfas.baseline.rocket import RocketConfig, run_rocket
from mfas.metrics import pct
from mfas.refine import alternate_scc_sift, sift_underrelaxed
from mfas.experiments.H02 import _init_positions_from_order, greedy_fas_order
from mfas.experiments import H42
from mfas.utils.seeding import select_device
device, _ = select_device("auto")
g = io.load_dataset("mouse")
order = greedy_fas_order(g)
for epochs in (0, 5000):
    r = run_rocket(g, RocketConfig(epochs=epochs), seed=42, device=device,
                   init_positions=_init_positions_from_order(order, device), time_limit=None)
    rank0 = np.argsort(np.argsort(r.best_positions, kind="stable"), kind="stable").astype(np.int64)
    sr, ss, slog = sift_underrelaxed(g, rank0, k_full=H42._K_FULL, alpha=H42._ALPHA,
                                     max_sweeps=H42._MAX_SWEEPS["mouse"], time_budget_s=None)
    _, asc, alog = alternate_scc_sift(g, sr, n_cycles=H42._ALT_CYCLES["mouse"],
                                      sift_sweeps=H42._ALT_SIFT_SWEEPS["mouse"],
                                      k_full=H42._ALT_K_FULL, alpha=H42._ALPHA,
                                      min_block=H42._MIN_BLOCK, time_budget_s=None)
    print(epochs, pct(r.best_score, g.total_weight), pct(ss, g.total_weight),
          pct(asc, g.total_weight), len(slog), len(alog))
PY
```

### (b) Secondary, and the one with real upside — uncap stage 3 on the Rocket-free primaries

§3's caveat: the `epochs=0` arms on connectome and microns both hit their stage-3 sweep cap
(40 and 12), which was sized for a Rocket start. Re-run those two arms with
`max_sweeps` raised (e.g. 200 / 60) and stage 4 unchanged. If greedy + an *uncapped* under-relaxed
sift closes a meaningful part of the 0.274 pp on connectome, then ~1200 s of GPU per run is
buying something a cheap CPU refiner could buy instead — which is precisely where
`HANDOFF.md` § "Where the score is most likely to come from" already points (Vahidi 2025:
greedy + bounded-span insertion + SCC, no gradient phase at all, 84.61 %).
Cost: connectome ~15–25 min, microns ~10–20 min. Not cheap; do it after (a) and after the
night run finishes.

---

## Evidence trail, file by file

| file | key | value | what it establishes |
|---|---|---|---|
| `autoresearch/HANDOFF.md` | § "H41 — the live scientific question" | "93.083 vs 92.902 … does not transfer" | the dismissed observation; compares against stage 3, not the champion |
| `experiments/proto_H41.py` | `:124-125`, `:88-96`, `:161-178` | `load_dataset("mouse")`; H42 constants; "Rocket (stage 2) is skipped on purpose" | fixture identity + constant parity + what was skipped |
| `experiments/outputs/proto_H41.json` | `mouse@10s/_setup` | `total_weight 9.159036176272162`, `n_nodes 148`, `greedy_pct 90.12629794323948`, `stage3_pct 93.08288021668459`, `stage3_sweeps 14`, `stage123_wall_s 0.00799417495727539` | the Rocket-free score and its cost |
| ″ | `mouse@10s/A_champion` | `final_pct 93.08288021668459`, `delta_pp 0.0`, `n_cycles 7385` | 231× the production cycle budget buys 0.000000 pp |
| ″ | `mouse@10s/B_champion_plus_segment` | `final_pct 93.14175369790512`, `n_segment_moves 4` | segment moves add +0.0589 pp on top |
| `results/20260810T213945Z-H42-mouse-s42-verify-ea0edb.json` | `pct` / `variant_attrs` | `92.91701410211007`; `pure_best_pct 92.47934218373807`; `sift_best_pct 92.90180243040469`; `sift_converged true`, `n_sift_sweeps 5`; `n_alt_cycles 32`; `wall_clock_s 8.159990787506104`; `sift_time_s 0.0026`, `alt_time_s 0.0384` | the champion, stage by stage, and that ~99.5 % of its wall is Rocket |
| `autoresearch/sota.json` | `datasets.mouse` | `pct_mean 92.917`, `pct_std 0.0`, `n_seeds 20`; caveats "TIE, NOT A WIN", H36's +0.0152 pp | the comparator, and the scale of every mouse move ever shipped |
| `experiments/outputs/proto_H42_mouse.json` | `arms[*]` | 9 allocations, `(8,8)`→`(1,128)`, all `final_pct 92.91701410211007` | mouse stage 4 is saturated; cycle budget cannot explain 0.166 pp |
| `dr_tmp/size_global_discrete.json` | `mouse[*].sift_fullrange_on_greedy_noRocket` | `93.0471786135207 / 93.01179337907796 / 93.00438972100531` | independent Phase-6 corroboration, different sift, 3 seeds |
| `experiments/outputs/proto_h32_trophic.json`, `proto_h33_magnetic.json` | `fixtures/mouse/…/greedy_sift`, `quality_rows/mouse[*].greedy_sift_pct` | `92.91629587453781`, std 0 | the qualifier: **plain** Jacobi sift from greedy does *not* beat the champion |
| `experiments/proto_P07_sizing.py` | `:73-106`, `:64-70` | reads `H42._*` constants; `PLAN` has microns + connectome, **no mouse** | the `epochs=0` arms are production-minus-Rocket; and why (a) needs one added line |
| `experiments/outputs/proto_P07.json` | `arms[0].final_pct` | `83.11151992749353` (microns, epochs 0) | Rocket-free loses 0.129 pp on microns |
| `results/20260810T074449Z-H42-microns-s42-confirm-f91b66.json` | `pct` | `83.24085291200831` | microns comparator |
| `experiments/outputs/proto_S01_connectome.json` (untracked, live) | `arms[0].final_pct` | `83.87974978419737` (connectome, epochs 0), `n_sift_sweeps 40` (capped) | Rocket-free loses 0.274 pp on connectome; stage 3 was truncated |
| `results/20260810T064240Z-H42-connectome-s42-confirm-f41d7e.json` | `pct`, `score`, `total_weight` | `84.15409511053134`, `35270783.0`, `41912141.0` | connectome comparator |
| `src/mfas/metrics.py` | `:19-21`, `_as_sum_dtype` | governance banner; float64 upcast | one frozen scorer, used by prototype and production alike |
| `src/mfas/refine/underrelax.py` | `sift_underrelaxed` docstring | "Stops early only at a true fixed point … or when `time_budget_s` is exceeded" | both mouse stage-3 orders are fixed points, not truncations |
| `autoresearch/killed.json` | `meta_rules` M2, M6, M7 | — | no entry covers "remove the gradient phase"; M2 confirmed, M6 scope-narrowed |
| `experiments/findings.md` | Phase-6 summary, "Cross-cutting insight" | "the basin it starts from barely matters … greedy+sift is the design" | the belief this investigation falsifies in both directions |

### Not determinable from artifacts

* Whether the production mouse pipeline with `_ALT_CYCLES = 32` (rather than a 10 s wall-clock
  budget) reaches exactly `93.08288021668459` from the greedy start. Strongly implied by
  determinism + arm A's zero delta, but **not measured**. Settled by §5(a).
* Whether an uncapped stage-3 sift closes part of the 0.274 pp / 0.129 pp on the primaries.
  Settled by §5(b).
* Whether the Rocket-free mouse result survives ≥5 confirm seeds. Almost certainly yes — every
  stage in it is deterministic and `make_init_positions` is never called (the P02 seed-class
  argument in `CLAUDE.md`) — but it has never been run through `eval.run_variant`.

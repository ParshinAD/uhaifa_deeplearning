"""
Generates h30_sift_sweeps.ipynb — run the winning H30 algorithm end-to-end and
experiment with the sift pass limit (`max_sweeps`).

H30 (findings.md #4) = H02 greedy-FAS warm-start -> unchanged Rocket (PURE) ->
full-range exact-gain Jacobi "sift" post-phase. The production variant fixes the sift
pass limit at 12 (connectome/microns) / 30 (mouse). On the connectome it was still
improving at sweep 12 (capped, not converged), so this notebook makes it easy to raise
the cap and SEE whether more sweeps keep paying off.

The notebook reuses the repo's frozen modules so its numbers match the eval harness
(`mfas.metrics` is the same int64/float64 oracle used everywhere).

Run with:
  /opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python create_h30_sift_notebook.py
"""
import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.23"},
}


def md(source):
    return nbformat.v4.new_markdown_cell(source)


def code(source):
    return nbformat.v4.new_code_cell(source)


cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md("""\
# H30 — full-range exact-gain "sift": run it & sweep the pass limit

This notebook runs the **winning H30 algorithm** (findings.md #4) end-to-end and lets you
**experiment with the number of sift sweeps** (`max_sweeps`) — the one knob the production
variant fixes (12 for connectome/microns, 30 for mouse).

**Pipeline** (identical to `src/mfas/experiments/H30.py`):
1. **Greedy-FAS warm start** (H02): `greedy_fas_order(g)` → evenly-spaced positions.
2. **Unchanged Rocket** at the baseline epoch budget → the *pure* Rocket order.
3. **Full-range exact-gain Jacobi sift** → the refined order, kept only if the **frozen
   oracle** says it strictly beats the pure order (best-by-oracle).

Everything reuses the repo's frozen modules (`mfas.io`, `mfas.baseline.rocket`,
`mfas.experiments.H02`, `mfas.refine.sift`, `mfas.metrics`), so the scorer is the same
int64/float64 oracle used by the eval harness — the numbers here are directly comparable.

> **Why raising the cap is interesting.** On the connectome the production sift hit its
> 12-sweep cap *while every sweep was still improving* (`experiments/log.md`, H30 entry:
> "12/12 sweeps all improving") — i.e. it stopped on the budget, not at a fixed point.
> Mouse, by contrast, reaches a true fixed point in a few sweeps. This notebook shows both.

> **Leakage-safe / equal-compute (CLAUDE.md).** The sift adds **0 gradient steps**; every
> move is chosen from input edge weights + current ranks; the frozen oracle only
> accepts/rejects whole rank vectors. The *pure* Rocket score is reported separately from
> the refinement.
"""))

# ── 1. Setup ──────────────────────────────────────────────────────────────────
cells.append(md("""\
## 1. Setup — imports, device, CONFIG

All hyperparameters live in `CONFIG` (no magic numbers). We add the repo's `src/` to the
path so the frozen `mfas.*` modules import cleanly regardless of where the kernel started.

Optimization (Rocket) runs on MPS/CUDA if available; the **discrete score and the sift are
always computed on CPU in NumPy** (MPS float32 index ops can corrupt large-graph scores —
see CLAUDE.md), so results are deterministic given the seed (modulo MPS training jitter).
"""))

cells.append(code("""\
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def _find_repo_root(start: Path) -> Path:
    # Walk up until we find the frozen oracle; works from notebooks/ or the repo root.
    for p in [start, *start.parents]:
        if (p / 'src' / 'mfas' / 'metrics.py').exists():
            return p
    raise FileNotFoundError(f'Could not locate src/mfas/metrics.py above {start}')


REPO = _find_repo_root(Path.cwd())
sys.path.insert(0, str(REPO / 'src'))

from mfas.io import load_dataset
from mfas.baseline.rocket import RocketConfig, run_rocket
from mfas.experiments.H02 import greedy_fas_order, _init_positions_from_order
from mfas.refine import sift
from mfas.metrics import score_from_order, pct

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

CONFIG = dict(
    # 'mouse' (~seconds) | 'connectome' (~2-5 min) | 'microns' (~minutes; needs the .npz cache)
    DATASET             = 'mouse',
    SEED                = GLOBAL_SEED,
    # Baseline gradient budget per dataset (= the H02/H30 production budget).
    ROCKET_EPOCHS       = {'mouse': 5_000, 'connectome': 20_000, 'microns': 80_000},
    # EXTENDED sift cap for the experiment (production H30 fixes 12 / 30 below).
    SIFT_MAX_SWEEPS     = 60,
    # The cap H30 actually ships, per dataset (for the reference line / comparison).
    SIFT_PRODUCTION_CAP = {'connectome': 12, 'microns': 12, 'mouse': 30},
    OUTPUT_DIR          = REPO / 'notebooks' / 'outputs',
)
CONFIG['OUTPUT_DIR'].mkdir(parents=True, exist_ok=True)

DATASET  = CONFIG['DATASET']
SEED     = CONFIG['SEED']
PROD_CAP = CONFIG['SIFT_PRODUCTION_CAP'][DATASET]
print(f'Repo: {REPO}')
print(f'Device: {DEVICE}  |  dataset: {DATASET}  |  seed: {SEED}')
print(f'Production sift cap for {DATASET}: {PROD_CAP}  |  extended cap (this run): {CONFIG[\"SIFT_MAX_SWEEPS\"]}')
"""))

# ── 2. Load data ──────────────────────────────────────────────────────────────
cells.append(md("""\
## 2. Load a dataset

`mfas.io.load_dataset` resolves the data files relative to the repo and validates basic
facts (node/edge counts, total weight). Start with **`mouse`** for instant iteration; switch
`CONFIG['DATASET']` to `connectome` to see the interesting pass-limit behaviour (the fly
connectome keeps improving past the production cap; mouse converges almost immediately).
"""))

cells.append(code("""\
g = load_dataset(DATASET)
TOTAL = g.total_weight
print(g)
"""))

# ── 3. Pure Rocket ────────────────────────────────────────────────────────────
cells.append(md("""\
## 3. Stage 1+2 — greedy-FAS warm start + (unchanged) Rocket = *pure* Rocket

This is exactly H02: a leakage-safe **Eades–Lin–Smyth / GreedyAbs** greedy-FAS ordering
(graph structure + weights only, never the oracle) mapped to evenly-spaced positions in
`[-1, 1]`, then handed to the **unchanged** `run_rocket` at the baseline epoch budget. The
result is the *pure* Rocket order — the number reported separately from any refinement.
"""))

cells.append(code("""\
epochs = CONFIG['ROCKET_EPOCHS'][DATASET]
cfg = RocketConfig(epochs=epochs)

# Greedy-FAS order alone (context): the discrete heuristic Rocket is warm-started from.
order = greedy_fas_order(g)
greedy_pct = pct(score_from_order(order, g.src, g.tgt, g.weight), TOTAL)

t0 = time.time()
init_positions = _init_positions_from_order(order, DEVICE)
rocket = run_rocket(g, cfg, seed=SEED, device=DEVICE, init_positions=init_positions)
rocket_wall = time.time() - t0

pure_pct = rocket.best_pct
print(f'greedy-FAS order alone     : {greedy_pct:8.4f}%')
print(f'pure Rocket (H02 warmstart): {pure_pct:8.4f}%   '
      f'({rocket.best_score:,.0f} / {TOTAL:,.0f})')
print(f'  epochs={rocket.n_epochs_done:,}   wall={rocket_wall:.1f}s   device={DEVICE}')
"""))

# ── 4. Sift (extended) ────────────────────────────────────────────────────────
cells.append(md("""\
## 4. Stage 3 — full-range exact-gain sift (extended sweep cap)

We run the **same** `mfas.refine.sift` the production H30 uses, but with the **extended**
`max_sweeps` from CONFIG. Each Jacobi sweep moves every node toward its *exact*
feedforward-maximising rank on the whole line at once; the frozen oracle accepts/rejects the
whole resulting vector (best-by-oracle), so the returned order can never regress.

`sift` returns `(best_rank, best_score, sweep_log)`, where each `sweep_log` row records:
`sweep`, `candidate_pct` (the working Jacobi iterate — **may transiently dip** on dense
cyclic cores), `accepted` (did the oracle keep it as the new best?), `n_movers` (how many
nodes still want to move; `0` = a true fixed point → early stop), and `wall` (seconds).
"""))

cells.append(code("""\
# Integer ranks of the pure-Rocket positions (rank[node] = position on the line).
rank0 = np.argsort(np.argsort(rocket.best_positions, kind='stable'),
                   kind='stable').astype(np.int64)
init_pct = pct(score_from_order(rank0, g.src, g.tgt, g.weight), TOTAL)
assert abs(init_pct - pure_pct) < 1e-9, 'ranks of the best positions must score == pure Rocket'

t0 = time.time()
best_rank, refined_score, sweep_log = sift(g, rank0, max_sweeps=CONFIG['SIFT_MAX_SWEEPS'])
sift_wall = time.time() - t0

refined_pct = pct(refined_score, TOTAL)
hit_fixed_point = sweep_log[-1]['n_movers'] == 0
print(f'pure Rocket            : {pure_pct:8.4f}%')
print(f'sift (max_sweeps={CONFIG[\"SIFT_MAX_SWEEPS\"]:>3}) : {refined_pct:8.4f}%   '
      f'(+{refined_pct - pure_pct:.4f} pp over pure Rocket)')
print(f'  sweeps run={len(sweep_log)}   accepted={sum(r[\"accepted\"] for r in sweep_log)}   '
      f'reached fixed point={hit_fixed_point}   sift wall={sift_wall:.1f}s')
"""))

# ── 4b. Trajectory DataFrame ──────────────────────────────────────────────────
cells.append(md("""\
### Per-sweep trajectory

`best_pct_after` is the **best-by-oracle** score after sweep *k* — the running maximum of
`candidate_pct`, floored by the initial (pure Rocket) order. It is monotone non-decreasing;
the `candidate` iterate can dip below it.
"""))

cells.append(code("""\
df = pd.DataFrame(sweep_log)
# best-by-oracle after j sweeps = running max of candidate_pct, floored by the initial order.
best_curve = np.maximum.accumulate(
    np.concatenate([[init_pct], df['candidate_pct'].to_numpy()]))   # length = len(df)+1
df['best_pct_after'] = best_curve[1:]
df[['sweep', 'n_movers', 'candidate_pct', 'accepted', 'best_pct_after', 'wall']]
"""))

# ── 4c. Plot ──────────────────────────────────────────────────────────────────
cells.append(md("""\
### Plot — does raising the cap beyond the production limit keep helping?

The green (best-by-oracle) curve is what H30 would return at each cap. If it is still rising
at the red production-cap line, a larger `max_sweeps` would score higher (at extra
wall-clock); if it has flattened, the production cap was already enough.
"""))

cells.append(code("""\
fig, ax = plt.subplots(figsize=(10, 5.5))
sweeps = df['sweep'].to_numpy()
ax.plot(sweeps, df['candidate_pct'], 'o-', ms=4, color='tab:gray', alpha=0.7,
        label='Jacobi working iterate (candidate)')
ax.plot(sweeps, df['best_pct_after'], 's-', ms=4, color='tab:green',
        label='best-by-oracle (what H30 returns)')
ax.axhline(pure_pct, color='tab:blue', ls='--',
           label=f'pure Rocket = {pure_pct:.4f}%')
if PROD_CAP <= sweeps.max() + 1:
    ax.axvline(PROD_CAP - 1, color='tab:red', ls=':',
               label=f'production cap = {PROD_CAP} sweeps')
if hit_fixed_point:
    ax.axvline(sweeps.max(), color='black', ls=':', alpha=0.5,
               label='fixed point (no movers)')
ax.set_xlabel('sift sweep')
ax.set_ylabel('feedforward weight (%)')
ax.set_title(f'H30 sift trajectory — {DATASET} (seed {SEED})\\n'
             f'does raising the sweep cap beyond {PROD_CAP} keep improving?')
ax.legend(loc='lower right')
plt.tight_layout()
out_png = CONFIG['OUTPUT_DIR'] / f'h30_sift_trajectory_{DATASET}_s{SEED}.png'
plt.savefig(out_png, dpi=120, bbox_inches='tight')
plt.show()
print('saved', out_png)
"""))

# ── 5. Pass-limit experiment ──────────────────────────────────────────────────
cells.append(md("""\
## 5. The pass-limit experiment — best score achievable at each `max_sweeps`

Because best-by-oracle is a **running maximum** over sweeps, the score H30 would return with
`max_sweeps = c` equals the running max over the first `c` sweeps of this single long run —
running `sift` again with a smaller cap produces an identical trajectory prefix. So one long
run answers *"how many sweeps are actually worth it?"* without re-running anything.

The table reports, per candidate cap: the best feedforward %, the gain over pure Rocket, and
the **marginal gain over the production cap** (what raising the limit buys you).
"""))

cells.append(code("""\
prod_best = best_curve[min(PROD_CAP, len(df))]
caps = sorted({1, 3, 6, PROD_CAP, 20, 30, 40, CONFIG['SIFT_MAX_SWEEPS']})
rows = []
for c in caps:
    best_at_c = best_curve[min(c, len(df))]          # best-by-oracle after c sweeps
    rows.append(dict(
        max_sweeps=c,
        best_pct=round(best_at_c, 5),
        gain_vs_pure_pp=round(best_at_c - pure_pct, 5),
        gain_vs_prod_cap_pp=round(best_at_c - prod_best, 5),
    ))
cap_table = pd.DataFrame(rows)
ext_best_preview = best_curve[min(CONFIG['SIFT_MAX_SWEEPS'], len(df))]
print(f'production cap = {PROD_CAP} sweeps -> {prod_best:.4f}%   '
      f'(extended {CONFIG[\"SIFT_MAX_SWEEPS\"]} sweeps -> {ext_best_preview:.4f}%)')
cap_table
"""))

# ── 6. Headline ───────────────────────────────────────────────────────────────
cells.append(md("""\
## 6. Headline summary

`pure Rocket` → `sift @ production cap` → `sift @ extended cap`, with the marginal pp the
extra sweeps recovered. For the connectome you should see the extended cap add a little more
on top of the +0.85 pp the production sift already recovers; for mouse the extended cap
typically adds **nothing** (the sift converged to a fixed point well before the cap).
"""))

cells.append(code("""\
ext_cap = CONFIG['SIFT_MAX_SWEEPS']
ext_best = best_curve[min(ext_cap, len(df))]
print(f'dataset            : {DATASET}  (seed {SEED}, {g.n_nodes:,} nodes, {g.n_edges:,} edges)')
print(f'pure Rocket        : {pure_pct:8.4f}%')
print(f'sift @ cap={PROD_CAP:<3}     : {prod_best:8.4f}%   (+{prod_best - pure_pct:.4f} pp over pure)')
print(f'sift @ cap={ext_cap:<3}     : {ext_best:8.4f}%   (+{ext_best - pure_pct:.4f} pp over pure)')
print(f'raising cap {PROD_CAP} -> {ext_cap} : {ext_best - prod_best:+.4f} pp   '
      f'(reached fixed point: {hit_fixed_point})')
print()
print('NOTE: the sift adds 0 gradient steps; the extra cost of more sweeps is pure wall-clock.')
print('      Pure Rocket is reported separately from the refinement (CLAUDE.md).')
"""))

# ── Notes ─────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Notes & honest caveats

- **This is an exploration notebook, not a logged experiment.** A real "raise the cap"
  result would go through the unchanged `eval/run_variant.py` harness on **both** primary
  datasets, **≥3 seeds**, with mean ± std and a CI lower bound (CLAUDE.md invariants).
  Treat the single-seed numbers here as a quick look, not a confirmed win.
- **Cost of more sweeps.** Each connectome sweep is ~3 s (a full pass over 5.66M edges), so
  raising `max_sweeps` to 60 adds ~3 min of wall-clock. The sift still adds **0 gradient
  steps**, so the equal-compute-vs-Rocket framing is unchanged — only wall-clock grows.
- **MPS jitter.** Only the *pure* Rocket order feeding the sift is MPS-nondeterministic; the
  sift's best-by-oracle guarantees the refined order is ≥ pure regardless.
- **To test the connectome:** set `CONFIG['DATASET'] = 'connectome'` and re-run from cell 2.
  That is the case where the production cap was binding (still improving at sweep 12).
- **Frozen oracle.** Scores come from `mfas.metrics` (int64/float64), the same scorer the
  harness uses — do not re-implement scoring here.
"""))

nb['cells'] = cells
with open('h30_sift_sweeps.ipynb', 'w') as f:
    nbformat.write(nb, f)
print('Notebook written: h30_sift_sweeps.ipynb')

# Critic evidence — H36 sizing gate red-team (2026-08-09)

Verbatim preservation of the scripts and outputs the **critic** produced while red-teaming
`experiments/size_collective_moves.py`. They were written into gitignored `dr_tmp/critic_collective/`;
because the 2026-08-09 entry in [`../log.md`](../log.md) **cites their numbers**, they are copied here
so every cited figure traces to a committed artifact (invariant #3 + `PROTOCOL.md` § "Shared
reproducibility contract").

> **Status: preserved as-is, NOT productionised.** These are the critic's working scripts. They carry
> a hardcoded absolute `ROOT` path and were run from `dr_tmp/`, so they will not execute unmodified
> from this directory on another machine. They are committed as **evidence of what was run**, not as
> re-runnable tooling. The re-runnable equivalents are:
> - `experiments/diagnostics/verify_collective_moves.py` — promoted, path-portable version of `bf_check.py`.
> - `experiments/diagnostics/control_1opt_movers.py` — promoted, path-portable version of
>   `oneopt_control.py` + `oneopt_realized.py`. **Written but NOT yet run** — it has no committed
>   output JSON; the numbers below come from the critic's originals in this folder.
> - `experiments/size_collective_moves.py --seed N` / `--top-k N` — replaces `seed_sweep.py` and
>   `repro_connALL.json` (seed 123 and full-K have since been re-run as first-class artifacts in
>   `../outputs/`; **seed 999 has not** — see the gap note below).

## What each file backs

| file | backs which cited claim |
|---|---|
| `bf_check.py` | exact-gain algebra vs brute-force full-graph oracle rescoring (S1 max err 1.6e-14 over 17,441 pairs; small-SCC DP 1.4e-14; S2 1.5e-13 over 2,345 passes) |
| `mutation_test.py`, `mutation2.py` | the probe's `assert gain == oracle delta` checks are **non-vacuous** (injected bugs fire them) |
| `oneopt_control.py` | single-node movers left on the H35 orders: **163 / 608 / 0** (connectome / microns / mouse, seed 42) |
| `oneopt_realized.py`, `oneopt_realized.log` | realizable single-node recovery by restarting the sift: **+0.00328 / +0.00555 / +0.00000 pp** — the number that must be subtracted before any "collective" claim |
| `seed_sweep.py`, `seed_sweep.json`, `seed_sweep.log` | uncontrolled 3-seed connectome iteration **+0.06196 / +0.06628 / +0.06272 = +0.0637 ± 0.0023 pp** (CI_lo +0.0610); mouse identical on all 3 seeds |
| `repro_connALL.json`, `connALL.log` | **full-K** single S1 pass = **+0.04293 pp** vs +0.01736 at K=200k (2.5×) → the sizing number is a lower bound |
| `repro_conn500.json`, `repro_mouse.json` | bit-exact reproduction of the logged round-1 JSON at `--top-k 500 --rounds 1` and on the full mouse run |

## Known gap (do not paper over)

Re-running the critic's numbers as first-class artifacts was **started and then stopped on request**:

- ✅ connectome seed **123** → `../outputs/collective_moves_sizing_connectome_s123.json`
  (reproduced +0.06628 pp exactly)
- ✅ connectome **full-K** → `../outputs/collective_moves_fullk_connectome_s42.json`
- ❌ connectome seed **999** — run was killed before it wrote its JSON. The `+0.06272` figure for that
  seed rests **only** on `seed_sweep.json` in this folder.
- ❌ `control_1opt_movers.py` — never executed; no `../outputs/control_1opt_movers.json` exists.

To close the gap:

```
python experiments/size_collective_moves.py --datasets connectome --seed 999 \
    --top-k 200000 --rounds 6 --out collective_moves_sizing_connectome_s999.json
python experiments/diagnostics/control_1opt_movers.py
```
(env `/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`; ~8 min and ~10 min respectively,
and both need the gitignored H35 `results/*_positions.npy` present.)

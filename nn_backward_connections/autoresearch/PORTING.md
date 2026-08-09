# Porting the campaign to another machine

Written for the agent that sets the campaign up on a new box. Follow it top to bottom. Do not
start the driver until every gate below is green — a campaign running on an unverified
environment produces numbers that have to be thrown away.

The reference port is **Windows laptop + NVIDIA RTX 4060**, from the original **Apple Silicon /
MPS** MacBook. The instructions are written for that case and hold for any device change.

---

## 0. What actually has to travel

| item | in git? | how |
|---|---|---|
| the whole repo + branch `auto/campaign` | yes | `git clone` (or a bundle) |
| `data/connectome_graph.csv.gz`, `data/table_mouse.txt`, `data/best_solution/` | yes | comes with the clone |
| `results/rocket_best_positions.npy` (parity anchor) | yes | comes with the clone |
| all `results/*.json` run records | yes | comes with the clone |
| **`data/processed/microns.npz`** (~55 MB) | **no — gitignored** | **copy by hand** |
| **`data/processed/microns_proofread.npz`** (~10 KB) | **no — gitignored** | **copy by hand** |
| `dr_tmp/` (~180 KB, prior evidence the queue cites) | no — gitignored | copy by hand (recommended) |
| `.claude/settings.local.json` (auth token) | no — gitignored | **do not copy — run `claude` and `/login` on the new machine** |
| `data/raw/` (48 GB MICrONS source) | no | **do not copy** — only needed to rebuild `microns.npz`, which you already have |

---

## 1. Get the code across

**Preferred — via the existing GitHub remote:**

```bash
# on the OLD machine, once
cd <old>/mfas_autoresearch && git push -u origin auto/campaign

# on the NEW machine
git clone git@github.com:ParshinAD/uhaifa_deeplearning.git mfas_autoresearch
cd mfas_autoresearch && git checkout auto/campaign
```

**Offline alternative — a git bundle** (one file, full history, no network):

```bash
# OLD machine
cd <old>/mfas_autoresearch
git bundle create ~/mfas-campaign.bundle auto/campaign
# copy the bundle + the two .npz + dr_tmp/ across (USB, scp, whatever)

# NEW machine
git clone -b auto/campaign mfas-campaign.bundle mfas_autoresearch
cd mfas_autoresearch
git remote set-url origin git@github.com:ParshinAD/uhaifa_deeplearning.git   # optional
```

Then drop the hand-copied payload into place:

```
nn_backward_connections/data/processed/microns.npz
nn_backward_connections/data/processed/microns_proofread.npz
nn_backward_connections/dr_tmp/
```

Note: on the original machine this checkout was a **git worktree** of a bigger repository. On the
new machine it is an ordinary clone — that is fine and in fact simpler. Nothing in the campaign
depends on being a worktree.

---

## 2. Choose the shell environment — use WSL2

The campaign infrastructure is bash: `autoresearch/driver.sh`, the PreToolUse guard hook
(`.claude/hooks/protect-frozen-files.sh`), `scripts/reproduce_baseline.sh`. On native Windows
those need a POSIX shell that Claude Code will reliably invoke for hooks — and the guard hook is a
safety mechanism, not a nicety. **Install WSL2 (Ubuntu) and run everything inside it.**

CUDA works in WSL2 through the Windows NVIDIA driver: install the normal Windows driver, then
inside WSL `nvidia-smi` must show the 4060. Do **not** install a Linux NVIDIA driver inside WSL.

Keep the repo on the **Linux filesystem** (`~/mfas_autoresearch`), never under `/mnt/c/...` —
cross-filesystem I/O is slow enough to distort every wall-clock measurement, and wall-clock is a
hard constraint in this campaign.

If you insist on native Windows instead, you must port `driver.sh` to PowerShell and verify the
hook fires — do not assume it does; test it with a deliberate write to `src/mfas/metrics.py` and
confirm the write is blocked.

---

## 3. Build the Python environment

The pinned stack is Python 3.9.23 / torch 2.8.0 / numpy 1.23.5 / scipy 1.10.1 / pandas 1.5.3
(`environment.yml`, `requirements.txt`). Recreate it with a **CUDA** torch build:

```bash
conda create -n allen python=3.9 -y && conda activate allen
conda install -c conda-forge numpy=1.23.5 scipy=1.10.1 pandas=1.5.3 networkx=3.2.1 \
    pyyaml=6.0.2 matplotlib=3.9.4 nbformat=5.10.4 "pytest>=7,<9" -y
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

If no `cp39` CUDA wheel exists for torch 2.8 on your CUDA version, use Python 3.11 and the newest
stable torch instead, and relax the numpy/scipy/pandas pins to whatever that torch supports. That
is **acceptable**: the reproduction claim of this project rests on the deterministic scorer, not
on matching a training trajectory (which is impossible across devices anyway). But it makes the
hardware re-baseline in step 5 mandatory rather than merely important — and record the exact
versions you ended up with.

Verify the GPU is actually visible to torch:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`src/mfas/utils/seeding.select_device("auto")` prefers MPS → CUDA → CPU, so on this machine
`auto` resolves to CUDA with no code change. The discrete score is always computed on CPU in
int64/float64 — leave that alone.

---

## 4. Verification gate — all of it must pass

```bash
cd nn_backward_connections
PYTHONPATH=src python -m pytest tests/ -q          # expect 25 passed
```

`tests/test_metrics.py::test_scorer_parity_connectome` is the one that matters: it scores the
committed `results/rocket_best_positions.npy` and must return exactly **34,751,902** on total
weight **41,912,141** (82.9161%). This is the portability anchor. **If it fails, stop** — the
environment is wrong (wrong data file, wrong dtype handling, corrupted transfer) and nothing
measured on it is admissible.

Then check the guard hook actually blocks (it is your safety net for unattended running):

```bash
export CLAUDE_PROJECT_DIR=$PWD
echo '{"tool_input":{"file_path":"'$PWD'/src/mfas/metrics.py"}}' | bash .claude/hooks/protect-frozen-files.sh; echo "rc=$? (expect 2)"
echo '{"tool_input":{"file_path":"/etc/passwd"}}' | bash .claude/hooks/protect-frozen-files.sh; echo "rc=$? (expect 2)"
echo '{"tool_input":{"file_path":"'$PWD'/autoresearch/queue.json"}}' | bash .claude/hooks/protect-frozen-files.sh; echo "rc=$? (expect 0)"
```

And that headless Claude runs and is logged in:

```bash
claude -p "reply with exactly: HEADLESS_OK" --dangerously-skip-permissions < /dev/null
```

If that says "Not logged in", run `claude` interactively once and `/login`.

---

## 5. Re-point the machine-specific config

Edit **`autoresearch/campaign.yaml`**:

- `campaign.sandbox_root` → the new absolute path
- `environment.machine` → e.g. `"Windows/WSL2 laptop, RTX 4060 (CUDA)"`
- `environment.python` → output of `which python` in the `allen` env
- `environment.claude_config_dir` → the new home (or delete if you use the default)
- `environment.torch` → the version you actually installed

Edit **`autoresearch/state.json`**: set `"machine"` to the new machine and
`"hardware_rebaseline_done": false`.

Commit this as one commit: `phase7 port: re-point config for <machine>`.

---

## 6. P01 — the hardware re-baseline (mandatory, before any research cycle)

Every number in `autoresearch/sota.json` was measured on Apple MPS. The Phase-7 verdict rule
compares each variant to the champion, so judging a CUDA-measured variant against an MPS-measured
champion is a moving-comparator error — and one the auditor **cannot** catch, because the device
is not part of `config_hash`.

Run the current champions at the screen seeds on the new hardware:

```bash
PY=python   # the allen env
for S in 42 123 999; do
  $PY -m eval.run_variant --exp H35 --dataset connectome --seed $S --out results/ --role implement --device auto
done
for S in 42 123 999; do
  $PY -m eval.run_variant --exp H30 --dataset microns    --seed $S --out results/ --role implement --device auto
done
for S in 42 123 999; do
  $PY -m eval.run_variant --exp H30 --dataset mouse      --seed $S --out results/ --role implement --device auto
done
$PY autoresearch/audit.py --variant H35 --comparator H30 --role implement --comparator-role implement
```

Then:

- If the new means/σ differ materially from `sota.json`, update it with
  `update_sota.py ... --force` and record the hardware change as the justification in
  `experiments/log.md`.
- Set `campaign.yaml` `screen_delta_pp` per dataset to **2 × the new σ** (on MPS these were
  connectome 0.012, microns 0.002).
- Set `state.json.hardware_rebaseline_done = true`.
- Log the device name, torch version, and per-dataset wall-clock. A discrete GPU should be
  markedly faster than MPS — if the connectome run is not comfortably inside the 3600 s budget,
  something is wrong (check that the repo is not on `/mnt/c`).

Expect the *scores* to land close to the MPS numbers (same algorithm, same seeds) but not
identical, and the *σ* to be genuinely different. Both outcomes are normal; an identical score to
four decimals across devices would be the surprising result.

### 6b. Device-determinism check (P02) — required before the 1-seed screen is allowed

The screen may run **1 seed** for variants the classifier calls deterministic (PROTOCOL.md
§ Phase-7.4), and that permission rests on a property of the *device*, not of the code: that
repeating a seed reproduces bit-identically. Establish it on the new machine, once, by re-running
one seed per primary dataset in a **fresh process** and comparing the position vectors:

```bash
$PY -m eval.run_variant --exp H35 --dataset connectome --seed 42 --out results/ --role verify --device auto
$PY -m eval.run_variant --exp H30 --dataset microns    --seed 42 --out results/ --role verify --device auto
$PY experiments/analyze_p02.py       # compares role=implement vs role=verify, bit-for-bit
```

Use `--role verify` so these diagnostic runs never pool into the champion's `implement`
evidence. If the vectors are **not** bit-identical, the accelerator injects run-to-run noise
here: keep the screen at 3 seeds, and treat the observed spread as device noise rather than seed
variance (that is what it was on MPS — see `experiments/log.md` P01/P02).

Also re-run the cheap classifier corroboration, which covers every variant in ~5 minutes:

```bash
$PY autoresearch/seed_class.py --all --out autoresearch/seed_class.json
$PY experiments/proto_p02_determinism.py      # 16 variants x 3 runs on mouse
PYTHONPATH=src $PY -m pytest tests/test_seed_class.py tests/test_seed_plan.py -q
```

**Then record the answer in `campaign.yaml`, per dataset — the check does not apply itself.**
`seed_policy.device_determinism_verified.<ds>` starts at `false` on a new machine, and a dataset
may only appear with a reduced seed list under `seed_policy.screen_seeds_by_class.deterministic`
once its flag is `true`. `tests/test_seed_plan.py` enforces exactly that coupling, so a config
claiming a 1-seed screen without the evidence fails the suite rather than quietly saving an hour
it has not earned. Both keys move together, or neither moves:

```yaml
seed_policy:
  screen_seeds_by_class:
    deterministic:
      connectome: [42]            # <- reduce only after the flag below is true
      microns: [42, 123, 999]
  device_determinism_verified:
    connectome: true
    microns: false                # <- until its repeat run lands
```

Verify the result with `$PY autoresearch/seed_plan.py --variant H35 --role implement`, which
prints the plan the sweep will actually execute.

---

## 7. Start

```bash
cd nn_backward_connections
nohup bash autoresearch/driver.sh > /dev/null 2>&1 &
tail -f autoresearch/logs/driver.log
```

Stop with `touch autoresearch/STOP` (after the current cycle) or
`bash autoresearch/driver.sh --abort` (now).

---

## 8. One machine at a time

Two machines running `auto/campaign` will diverge: every cycle writes `sota.json`, `state.json`,
`queue.json` and `experiments/log.md`, and merging those is not mechanical. Either run the
campaign on exactly one machine, or give each its own branch (`auto/campaign-cuda`,
`auto/campaign-mps`) and treat them as independent replications — never auto-merge their
champion registries.

If you do run them as replications, that is scientifically valuable: a win that confirms on two
different devices with different non-determinism is much stronger evidence than one that does not.

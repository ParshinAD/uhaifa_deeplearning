# Promoted evidence — the files the record CITES, reproducible by checkout

`.gitignore` excludes `dr_tmp/` and `results/*_positions.npy`. Its own comment
states both the rule and the remedy: *"Keepers ... must be promoted, not left
here, so they stay reproducible-by-checkout."* That had not been done.

`autoresearch/queue.json` and `experiments/log.md` cite 17 files under `dr_tmp/`
as evidence — including the full trail behind a champion promotion (H44) and the
audit record behind a champion REFUSAL (H52 on connectome). None of them were in
the repository, so a fresh `git checkout` could not open a single one.

This directory is that promotion. The originals are left where they are and the
historical citations are **not** rewritten — they record what was true when they
were written. This index is what makes them resolve. Cite the copy in new work.

Created 2026-08-17, from an audit of what was and was not committed.
Related: queue item **P08** (champion evidence not reproducible by checkout).

## Cited scratch files

| cited in the record as | promoted copy | size |
|---|---|---|
| `dr_tmp/FINDINGS_underrelaxation.md` | `experiments/evidence/FINDINGS_underrelaxation.md` | 6.4 KB |
| `dr_tmp/audit_H44_mouse.json` | `experiments/evidence/audit_H44_mouse.json` | 10.0 KB |
| `dr_tmp/audit_H52_connectome.json` | `experiments/evidence/audit_H52_connectome.json` | 7.4 KB |
| `dr_tmp/audit_P05_regression.json` | `experiments/evidence/audit_P05_regression.json` | 16.5 KB |
| `dr_tmp/exp_sweeps.json` | `experiments/evidence/exp_sweeps.json` | 7.4 KB |
| `dr_tmp/guardtest/` | — | directory - not promoted |
| `dr_tmp/investigation_rocket_free_basin.md` | `experiments/evidence/investigation_rocket_free_basin.md` | 24.7 KB |
| `dr_tmp/kick_gate.py` | `experiments/evidence/kick_gate.py` | 6.5 KB |
| `dr_tmp/night_run.sh` | `experiments/evidence/night_run.sh` | 5.7 KB |
| `dr_tmp/probe_scc.json` | `experiments/evidence/probe_scc.json` | 0.3 KB |
| `dr_tmp/proto_H36.py` | `experiments/evidence/proto_H36.py` | 14.4 KB |
| `dr_tmp/proto_H36_out.json` | `experiments/evidence/proto_H36_out.json` | 1.0 KB |
| `dr_tmp/proto_H36c.py` | `experiments/evidence/proto_H36c.py` | 4.5 KB |
| `dr_tmp/proto_H36d.py` | `experiments/evidence/proto_H36d.py` | 3.3 KB |
| `dr_tmp/proto_H42.py` | `experiments/evidence/proto_H42.py` | 5.4 KB |
| `dr_tmp/proto_H42_mouse.py` | `experiments/evidence/proto_H42_mouse.py` | 4.0 KB |
| `dr_tmp/proto_P05.py` | `experiments/evidence/proto_P05.py` | 4.2 KB |
| `dr_tmp/size_global_discrete.json` | `experiments/evidence/size_global_discrete.json` | 3.1 KB |

## Champion position vectors

`results/*_positions.npy` is excluded by size — 158 files, 21 MB. But the audit's
`rescore` check re-scores those vectors with the frozen oracle, and every parity
claim in the log ("0 of 136,648 positions differ") is a statement about them, so a
checkout containing none of them can verify none of it.

The three CURRENT champions are promoted here. The other 155 stay excluded: they
are regenerable from a logged command, and 21 MB of superseded vectors is not
worth the repository weight.

| dataset | champion | file | size | pct |
|---|---|---|---|---|
| connectome | H42 | `experiments/evidence/20260810T105922Z-H42-connectome-s31415-confirm-f41d7e_positions.npy` | 0.5 MB | `84.15409511053134` |
| microns | H42 | `experiments/evidence/20260810T121650Z-H42-microns-s31415-confirm-f91b66_positions.npy` | 0.3 MB | `83.24085291200831` |
| mouse | H52 | `experiments/evidence/20260816T211722Z-H52-mouse-s1234-confirm-b458f1_positions.npy` | 0.0 MB | `93.10282596057698` |


# Datasets

Two directed, weighted graphs (edge lists of `(source, target, weight)`). Files are kept where
they already live; `src/mfas/io.py` resolves `data/raw/<file>` first, then legacy `data/<file>`.
Node IDs are remapped to contiguous `[0, n)` indices on load (see `io.remap_node_ids`).

## 1. `connectome_graph.csv.gz` — FlyWire fly connectome (MFAS challenge)

- **Provenance:** FlyWire fly-connectome MFAS challenge
  (`codex.flywire.ai/app/mfas_challenge`). Vertices = neurons, edges = synaptic connections,
  weight = synapse count.
- **Format:** gzipped CSV **with header**. Columns (note the **double space** in the first
  header): `Source Node  ID`, `Target Node ID`, `Edge Weight`. Comma-delimited.
- **Node IDs:** 18-digit sparse FlyWire identifiers (non-contiguous).
- **Weights:** integers (synapse counts).

| stat | value |
|---|---|
| nodes (n) | 136,648 |
| edges (m) | 5,657,719 |
| density | 3.03 × 10⁻⁴ |
| total weight | 41,912,141 |
| weight min / median / mean / max / p99 | 2 / 4 / 7.41 / 2,405 / 54 |
| self-loops / duplicate edges | 0 / 0 |
| bidirectional edges | 1,039,790 (18.4%) |
| #SCCs / giant SCC | 9,626 / 126,840 (92.8%) |

## 2. `table_mouse.txt` — mouse connectivity graph

- **Provenance:** researcher-provided mouse connectivity table.
- **Format:** **tab-separated, NO header**, three columns `source target weight`. The file may
  lack a trailing newline — rows are counted by parsing, not `wc -l`.
- **Node IDs:** small (near-)contiguous integers (1..149).
- **Weights:** floats (normalized connectivity strengths).

| stat | value |
|---|---|
| nodes (n) | 148 |
| edges (m) | 583 |
| density | 2.68 × 10⁻² |
| total weight | computed from data (≈ 9.17) |
| weight min / median / mean / max / p99 | 1.84e-4 / 6.36e-3 / 1.57e-2 / 0.614 / 0.109 |
| self-loops / duplicate edges | 0 / 0 |
| bidirectional edges | 166 (28.5%) |
| #SCCs / giant SCC | 46 / 102 (68.9%) |

## Reproduce these statistics

```python
from mfas import io, graph
for name in ("connectome", "mouse"):
    g = io.load_dataset(name)
    print(name, graph.degree_stats(g))
```

## Notes
- `data/raw/` is the documented canonical location; `data/processed/` holds cached `.npz`
  (gitignored).
- The exact scorer sums integer weights in int64 and float weights in float64 — never float32 —
  to keep totals exact.

# Male CNS v1.0 graph build (token-free, pinned)

Source: Janelia FlyEM Male CNS v1.0 (Berg et al., preprint 10.1101/2025.10.09.680999; Cell 2026),
public bucket `gs://flyem-male-cns`, CC BY. Downloaded 2026-11-17 with `curl` from
`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/<file>`
into `data/raw/male_cns/` (gitignored).

| file | bytes | sha256 |
|---|---|---|
| connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather | 508,025,642 | 9b3beab17bad5f618be3f2c02d3139a8d07b822565919c013f1e5506d93e604b |
| body-annotations-male-cns-v1.0-minconf-0.5.feather | 14,483,314 | 2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2 |
| body-neurotransmitters-male-cns-v1.0.feather | 43,282,834 | 95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621 |
| connectome-weights-male-cns-v1.0-minconf-0.5.feather (all segments, not used) | 1,051,241,946 | e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1 |

## Graph definition
- edges: the traced-only weights table (both partners traced bodies), 25,563,197 rows,
  which matches the preprint's "25.6M edges". weight = synapse count (minconf 0.5).
- 101 self-loops dropped.
- edges with weight 1 dropped (10,292,892 edges, 8.3 % of synapses after self-loop removal): the FlyWire MFAS graph
  `data/connectome_graph.csv.gz` has minimum weight 2 (`data/README.md`), so this keeps the
  two graphs like-for-like. Same convention for every number in `male_cns/`.
- nodes: bodies with at least one kept edge.

## Canonical result (`male_cns/results/build_stats.json`)
- `data/processed/malecns.npz`: n = 164,425, m = 15,270,204, total weight = 113,731,681
- `data/processed/malecns_nodes.csv.gz`: annotations + NT predictions per node
- `data/processed/malecns_ann.csv.gz`: lanes / flow / region / NT in the FlyWire vocabulary
  (`male_cns/ann_prep.py`)

## Reader environment
The `allen` env has no feather reader. `build_graph.py` runs in a separate venv,
`male_cns/.venv` (uv; Python 3.11, pandas 3.1.1, pyarrow 26.0.1), gitignored. The `allen`
env was not changed. Recreate it with:

    uv venv male_cns/.venv --python 3.11 && uv pip install --python male_cns/.venv/bin/python pandas pyarrow numpy
    male_cns/.venv/bin/python -I male_cns/build_graph.py

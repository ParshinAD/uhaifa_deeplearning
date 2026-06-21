# MICrONS `microns` build recipe (token-free, pinned)

Segmentation version: **v117** (root_ids pinned to this static release).
Built by `experiments/build_microns.py`. No CAVE token required.

## Source files (public BossDB/GCS, no account)
| file | URL | bytes | sha256 |
|---|---|---|---|
| synapses | https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/synapse_graph/synapses_pni_2.csv | 51,020,159,848 | (first 1GB) 88255ed2529013661a6ee65ef03cb7c420ca86660a45a3eef930482b15b4a7fb |
| nucleus  | https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/nucleus_neuron_classification/nucleus_neuron_svm.csv | 13,439,975 | e54596dd4f346094f36a684578dd4b1b375330c1845042cc3726873033aeaa1a |
| proofread| https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/proofreading_status/proofreading_status_public_release.csv  | 57,324 | 25fc84d9cae5222e744d87be79f8d354684eb680a20497b293cbfb00d413bacf |

## Graph definition
- nodes = root_ids the nucleus SVM calls `neuron` (any nucleus), root_id 0 dropped.
- edges = synapses with BOTH endpoints in the neuron set; self-loops dropped;
  multi-synapse pairs aggregated. **weight = synapse count** (default).
  Summed synapse `size` also computed (alternative weighting, stored in microns_stats.json).
- proofread subgraph = edges with both endpoints having a proofread axon (status_axon
  in {clean, extended}), as a sensitivity check.

## Canonical result
- microns.npz: n=67,534, m=10,436,569, total_weight=15,400,557
- microns_proofread.npz: n=245, m=2,424

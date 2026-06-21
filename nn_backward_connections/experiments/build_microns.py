"""Build the MICrONS (minnie65) connectome as a canonical GraphData dataset.

Token-free, reproducible build from the **pinned v117 static release** on the public
BossDB/GCS buckets (no CAVE account required). All three input files are pinned to the
SAME v117 segmentation, so their ``pt_root_id`` values join consistently.

Inputs (downloaded to ``data/raw/microns/`` by Phase-5 Part A; NO header rows):

* ``synapses_pni_2.csv`` (~51 GB) — automated synapse detections. 16 columns:
  ``id, valid, pre_x, pre_y, pre_z, pre_supervoxel_id, pre_pt_root_id,
    post_x, post_y, post_z, post_supervoxel_id, post_pt_root_id,
    ctr_x, ctr_y, ctr_z, size`` → we use cols [6]=pre_root, [11]=post_root, [15]=size.
* ``nucleus_neuron_svm.csv`` — per-nucleus neuron/non-neuron SVM call. 9 columns:
  ``id, valid, pt_supervoxel_id, pt_root_id, x, y, z, classification_system, cell_type``
  → neuron filter = ``cell_type == 'neuron'`` and ``pt_root_id != 0``.
* ``proofreading_status_public_release.csv`` — proofread cells. 10 columns;
  ``pt_root_id`` = col [6], ``status_axon`` = col [9] (clean/extended = proofread axon).

Graph definition (canonical = LARGE neuron graph):
  nodes  = root_ids classified neuron by the nucleus SVM (a root_id is a neuron if ANY
           of its nuclei is called neuron); root_id 0 dropped (invalid).
  edges  = synapses whose pre AND post root_id are both in the neuron set;
           self-loops (pre == post) dropped; multi-synapse pairs aggregated.
  weight = synapse count for the (pre, post) pair  [DEFAULT, documented choice].
           The summed synapse ``size`` is computed too and stored as an alternative.

Outputs:
  ``data/processed/microns.npz``             — canonical GraphData (weight = synapse count).
  ``data/processed/microns_proofread.npz``   — proofread-axon subgraph (sensitivity check).
  ``data/processed/microns_BUILD.md``        — build recipe (URLs, sizes, sha256, v117 pin).
  ``data/processed/microns_stats.json``      — full stats for both graphs.

Run (env `allen`):
  python -m experiments.build_microns
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# repo root = parent of this file's parent (experiments/ -> repo)
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "microns"
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT))

from src.mfas import io as mio  # noqa: E402  (GraphData, save_processed)

SYN = RAW / "synapses_pni_2.csv"
NUC = RAW / "nucleus_neuron_svm.csv"
PR = RAW / "proofreading_status_public_release.csv"

SEG_VERSION = "v117"  # static-release segmentation pin (root_ids tied to this version)
SYN_URL = ("https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/"
           "synapse_graph/synapses_pni_2.csv")
NUC_URL = ("https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/"
           "nucleus_neuron_classification/nucleus_neuron_svm.csv")
PR_URL = ("https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/"
          "proofreading_status/proofreading_status_public_release.csv")

CHUNK = 10_000_000  # synapse rows per chunk


def sha256(path: Path, limit_bytes: int | None = None) -> str:
    """SHA-256 of a file (optionally only the first ``limit_bytes`` for huge files)."""
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as f:
        while True:
            n = 1 << 20
            if limit_bytes is not None:
                n = min(n, limit_bytes - read)
                if n <= 0:
                    break
            b = f.read(n)
            if not b:
                break
            h.update(b)
            read += len(b)
    return h.hexdigest()


def load_neuron_root_ids() -> np.ndarray:
    """Return the sorted unique set of root_ids the nucleus SVM calls 'neuron'."""
    nuc = pd.read_csv(NUC, header=None,
                      usecols=[3, 8], names=["root_id", "cell_type"],
                      dtype={3: np.int64, 8: str})
    neuron = nuc.loc[(nuc.cell_type == "neuron") & (nuc.root_id != 0), "root_id"]
    return np.unique(neuron.to_numpy(dtype=np.int64))


def load_proofread_root_ids() -> np.ndarray:
    """Return root_ids with a PROOFREAD AXON (status_axon in {clean, extended})."""
    pr = pd.read_csv(PR, header=None, usecols=[6, 9],
                     names=["root_id", "status_axon"], dtype={6: np.int64, 9: str})
    keep = pr.loc[pr.status_axon.isin(["clean", "extended"]), "root_id"]
    return np.unique(keep.to_numpy(dtype=np.int64))


def aggregate_edges(neuron_sorted: np.ndarray):
    """Stream the 51 GB synapse CSV in chunks; aggregate per-(pre,post) count & size-sum.

    Returns a DataFrame with columns ``pre_idx, post_idx, count, size_sum`` where the
    indices are into ``neuron_sorted`` (compact contiguous neuron ids). Self-loops and
    edges touching a non-neuron are excluded.
    """
    n = neuron_sorted.shape[0]
    running = None  # DataFrame[key, cnt, ssum]; key = pre_idx * n + post_idx
    total_rows = 0
    kept_syn = 0
    for ci, chunk in enumerate(pd.read_csv(
            SYN, header=None, usecols=[6, 11, 15],
            names=["pre", "post", "size"],
            dtype={6: np.int64, 11: np.int64, 15: np.int64}, chunksize=CHUNK)):
        total_rows += len(chunk)
        pre = chunk["pre"].to_numpy()
        post = chunk["post"].to_numpy()
        size = chunk["size"].to_numpy()
        # membership via searchsorted (neuron_sorted is sorted unique)
        pi = np.searchsorted(neuron_sorted, pre)
        pi_ok = (pi < n) & (neuron_sorted[np.clip(pi, 0, n - 1)] == pre)
        qi = np.searchsorted(neuron_sorted, post)
        qi_ok = (qi < n) & (neuron_sorted[np.clip(qi, 0, n - 1)] == post)
        mask = pi_ok & qi_ok & (pre != post)
        pi, qi, size = pi[mask], qi[mask], size[mask]
        kept_syn += int(mask.sum())
        if pi.size == 0:
            continue
        key = pi.astype(np.int64) * n + qi.astype(np.int64)
        df = pd.DataFrame({"key": key, "cnt": 1, "ssum": size.astype(np.int64)})
        g = df.groupby("key", sort=False).agg(cnt=("cnt", "sum"),
                                              ssum=("ssum", "sum")).reset_index()
        running = g if running is None else (
            pd.concat([running, g], ignore_index=True)
            .groupby("key", sort=False)
            .agg(cnt=("cnt", "sum"), ssum=("ssum", "sum")).reset_index())
        print(f"  chunk {ci}: rows={total_rows:,} kept_syn={kept_syn:,} "
              f"distinct_pairs={len(running):,}", flush=True)
    running["pre_idx"] = (running["key"] // n).astype(np.int64)
    running["post_idx"] = (running["key"] % n).astype(np.int64)
    return running, total_rows, kept_syn


def build_graphdata(edges: pd.DataFrame, neuron_sorted: np.ndarray, weight_col: str,
                    name: str) -> "mio.GraphData":
    """Make a GraphData from aggregated edges, RE-remapping to used nodes only.

    Only nodes that appear in at least one edge become graph nodes (via the canonical
    ``remap_node_ids`` on the original root_ids, so node_ids stay the on-disk root_ids).
    """
    pre_root = neuron_sorted[edges["pre_idx"].to_numpy()]
    post_root = neuron_sorted[edges["post_idx"].to_numpy()]
    weight = edges[weight_col].to_numpy(dtype=np.int64)
    src_idx, tgt_idx, node_ids = mio.remap_node_ids(pre_root, post_root)
    return mio.GraphData(src=src_idx.astype(np.int32), tgt=tgt_idx.astype(np.int32),
                         weight=weight, node_ids=node_ids, name=name)


def graph_stats(g: "mio.GraphData") -> dict:
    """Compute connectome-style stats: n, m, density, SCCs, degree & weight summaries."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    n, m = g.n_nodes, g.n_edges
    A = sp.csr_matrix((np.ones(m), (g.src, g.tgt)), shape=(n, n))
    n_scc, labels = connected_components(A, directed=True, connection="strong")
    _, counts = np.unique(labels, return_counts=True)
    giant = int(counts.max())
    out_deg = np.bincount(g.src, minlength=n)
    in_deg = np.bincount(g.tgt, minlength=n)
    w = g.weight
    def q(a):
        return [float(np.min(a)), float(np.percentile(a, 50)),
                float(np.percentile(a, 99)), float(np.max(a)), float(np.mean(a))]
    return dict(
        name=g.name, n_nodes=n, n_edges=m,
        density=float(m / (n * (n - 1))) if n > 1 else 0.0,
        total_weight=float(g.total_weight), weight_dtype=g.weight_dtype,
        n_scc=int(n_scc), giant_scc=giant, giant_scc_frac=float(giant / n),
        out_degree_min_med_p99_max_mean=q(out_deg),
        in_degree_min_med_p99_max_mean=q(in_deg),
        weight_min_med_p99_max_mean=q(w),
        n_isolated_sinks=int((out_deg == 0).sum()),
        n_isolated_sources=int((in_deg == 0).sum()),
    )


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    for f in (SYN, NUC, PR):
        if not f.exists():
            raise FileNotFoundError(f"missing input {f} — download it first")
    syn_size = SYN.stat().st_size
    print(f"synapse file: {syn_size:,} bytes (expected 51,020,159,848)")
    if syn_size != 51_020_159_848:
        raise RuntimeError("synapse download incomplete — re-run the curl resume first")

    print("loading neuron + proofread root_ids ...")
    neuron_sorted = load_neuron_root_ids()
    proofread = load_proofread_root_ids()
    print(f"  neuron root_ids: {neuron_sorted.size:,}; proofread-axon: {proofread.size:,}")

    print("aggregating synapses (chunked) ...")
    edges, total_rows, kept_syn = aggregate_edges(neuron_sorted)
    print(f"done: {total_rows:,} synapse rows; {kept_syn:,} neuron-neuron non-self synapses; "
          f"{len(edges):,} directed edges")

    # canonical: weight = synapse count
    g = build_graphdata(edges, neuron_sorted, "cnt", "microns")
    mio.save_processed(g, PROC / "microns.npz")
    print("saved", PROC / "microns.npz", "->", repr(g))

    # proofread-axon subgraph (sensitivity): keep edges with both endpoints proofread
    pr_set = np.intersect1d(proofread, neuron_sorted)
    pre_root = neuron_sorted[edges["pre_idx"].to_numpy()]
    post_root = neuron_sorted[edges["post_idx"].to_numpy()]
    in_pr = np.isin(pre_root, pr_set) & np.isin(post_root, pr_set)
    edges_pr = edges.loc[in_pr].reset_index(drop=True)
    g_pr = build_graphdata(edges_pr, neuron_sorted, "cnt", "microns_proofread")
    mio.save_processed(g_pr, PROC / "microns_proofread.npz")
    print("saved", PROC / "microns_proofread.npz", "->", repr(g_pr))

    print("computing stats ...")
    stats = dict(canonical=graph_stats(g), proofread_subset=graph_stats(g_pr),
                 build=dict(seg_version=SEG_VERSION, total_synapse_rows=total_rows,
                            kept_neuron_neuron_synapses=kept_syn,
                            neuron_root_ids=int(neuron_sorted.size),
                            proofread_axon_root_ids=int(proofread.size)))
    (PROC / "microns_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    # build recipe with sha256 (synapse: hash first 1 GB only — full hash of 51 GB is slow)
    recipe = f"""# MICrONS `microns` build recipe (token-free, pinned)

Segmentation version: **{SEG_VERSION}** (root_ids pinned to this static release).
Built by `experiments/build_microns.py`. No CAVE token required.

## Source files (public BossDB/GCS, no account)
| file | URL | bytes | sha256 |
|---|---|---|---|
| synapses | {SYN_URL} | {SYN.stat().st_size:,} | (first 1GB) {sha256(SYN, 1<<30)} |
| nucleus  | {NUC_URL} | {NUC.stat().st_size:,} | {sha256(NUC)} |
| proofread| {PR_URL}  | {PR.stat().st_size:,} | {sha256(PR)} |

## Graph definition
- nodes = root_ids the nucleus SVM calls `neuron` (any nucleus), root_id 0 dropped.
- edges = synapses with BOTH endpoints in the neuron set; self-loops dropped;
  multi-synapse pairs aggregated. **weight = synapse count** (default).
  Summed synapse `size` also computed (alternative weighting, stored in microns_stats.json).
- proofread subgraph = edges with both endpoints having a proofread axon (status_axon
  in {{clean, extended}}), as a sensitivity check.

## Canonical result
- microns.npz: n={g.n_nodes:,}, m={g.n_edges:,}, total_weight={g.total_weight:,.0f}
- microns_proofread.npz: n={g_pr.n_nodes:,}, m={g_pr.n_edges:,}
"""
    (PROC / "microns_BUILD.md").write_text(recipe)
    print("wrote", PROC / "microns_BUILD.md")
    print("\nNEXT: paste these into src/mfas/io.py DATASETS['microns'] expected_* :")
    print(f"  expected_n={g.n_nodes}, expected_m={g.n_edges}, "
          f"expected_total_weight={int(g.total_weight)}")


if __name__ == "__main__":
    main()

"""Dataset I/O — load directed weighted edge lists into a canonical form.

Two datasets with different on-disk schemas are supported (see ``DATASETS``):

* ``connectome`` — FlyWire fly-connectome MFAS-challenge graph. Gzipped CSV WITH a
  header. NOTE the source column header contains a DOUBLE space:
  ``"Source Node  ID"``. Node IDs are 18-digit sparse integers; weights are
  integers (synapse counts).
* ``mouse`` — a smaller mouse connectivity graph. Tab-separated, NO header, three
  columns ``source target weight``. Node IDs are small contiguous integers;
  weights are floats.

Both are remapped to contiguous ``[0, n)`` indices via ``np.searchsorted`` on the
sorted unique node IDs. This reproduces, exactly, the index assignment used by the
original notebook's ``{n: i for i, n in enumerate(np.unique(...))}`` mapping, so the
saved artifact ``results/rocket_best_positions.npy`` stays consistent with the
loaded graph (validated transitively by the scorer-parity test).
"""
from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "GraphData",
    "DATASETS",
    "load_dataset",
    "load_connectome",
    "load_mouse",
    "remap_node_ids",
    "save_processed",
    "load_processed",
    "data_dir",
]


# ──────────────────────────────────────────────────────────────────────────────
# Canonical in-memory representation
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GraphData:
    """A directed weighted graph with contiguous node indices.

    Attributes
    ----------
    src, tgt : int32 arrays, shape (n_edges,)
        Contiguous source / target node indices in ``[0, n_nodes)``.
    weight : int64 or float64 array, shape (n_edges,)
        Edge weights in their native dtype (never float32).
    node_ids : array, shape (n_nodes,)
        ``node_ids[i]`` is the original on-disk ID of contiguous index ``i``
        (sorted ascending, the inverse of the remap).
    name : str
        Dataset name (``"connectome"`` / ``"mouse"``).
    """

    src: np.ndarray
    tgt: np.ndarray
    weight: np.ndarray
    node_ids: np.ndarray
    name: str

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.weight.shape[0])

    @property
    def total_weight(self) -> float:
        """Sum of all edge weights, computed from data in the native dtype."""
        return float(self.weight.sum())

    @property
    def weight_dtype(self) -> str:
        return str(self.weight.dtype)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"GraphData(name={self.name!r}, n_nodes={self.n_nodes:,}, "
                f"n_edges={self.n_edges:,}, total_weight={self.total_weight:,.0f}, "
                f"weight_dtype={self.weight_dtype})")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ──────────────────────────────────────────────────────────────────────────────
# expected_* are sanity-check values; None means "no reference value, don't assert".
DATASETS = {
    "connectome": dict(
        filename="connectome_graph.csv.gz",
        loader="load_connectome",
        weight_dtype="int64",
        expected_n=136_648,
        expected_m=5_657_719,
        expected_total_weight=41_912_141,
    ),
    "mouse": dict(
        filename="table_mouse.txt",
        loader="load_mouse",
        weight_dtype="float64",
        expected_n=148,
        expected_m=583,
        expected_total_weight=None,  # no paper reference for the mouse graph
    ),
}


def data_dir() -> Path:
    """Return the repository ``data/`` directory (resolved relative to this file)."""
    return Path(__file__).resolve().parent.parent.parent / "data"


def _resolve_path(filename: str, path: Optional[Path | str]) -> Path:
    """Resolve a dataset file, preferring ``data/raw/`` then legacy ``data/``.

    We do not move the existing 39 MB datasets; this resolver finds them wherever
    they live.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {p}")
        return p
    base = data_dir()
    candidates = [base / "raw" / filename, base / filename]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find {filename!r} in any of: "
        + ", ".join(str(c) for c in candidates)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Node-ID remapping
# ──────────────────────────────────────────────────────────────────────────────
def remap_node_ids(src_raw: np.ndarray, tgt_raw: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remap arbitrary node IDs to contiguous ``[0, n)`` indices.

    Uses ``np.searchsorted`` on ``np.unique`` (sorted) node IDs — the result is
    identical to a ``{id: i for i, id in enumerate(np.unique(...))}`` dict but
    vectorized (O(m log n)).

    Returns ``(src_idx int32, tgt_idx int32, node_ids)`` where
    ``node_ids[i]`` is the original ID of index ``i``.
    """
    src_raw = np.asarray(src_raw)
    tgt_raw = np.asarray(tgt_raw)
    node_ids = np.unique(np.concatenate([src_raw, tgt_raw]))
    src_idx = np.searchsorted(node_ids, src_raw).astype(np.int32)
    tgt_idx = np.searchsorted(node_ids, tgt_raw).astype(np.int32)
    return src_idx, tgt_idx, node_ids


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────
def load_connectome(path: Optional[Path | str] = None, validate: bool = True
                    ) -> GraphData:
    """Load the FlyWire fly-connectome graph (gzipped CSV with header).

    Header columns: ``"Source Node  ID"`` (DOUBLE space), ``"Target Node ID"``,
    ``"Edge Weight"``. Weights are integers.
    """
    fpath = _resolve_path(DATASETS["connectome"]["filename"], path)
    src_raw, tgt_raw, w_raw = [], [], []
    with gzip.open(fpath, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_raw.append(int(row["Source Node  ID"]))   # double space, intentional
            tgt_raw.append(int(row["Target Node ID"]))
            w_raw.append(int(row["Edge Weight"]))

    src_raw = np.array(src_raw, dtype=np.int64)
    tgt_raw = np.array(tgt_raw, dtype=np.int64)
    weight = np.array(w_raw, dtype=np.int64)

    src_idx, tgt_idx, node_ids = remap_node_ids(src_raw, tgt_raw)
    g = GraphData(src=src_idx, tgt=tgt_idx, weight=weight,
                  node_ids=node_ids, name="connectome")
    if validate:
        _validate(g, "connectome")
    return g


def load_mouse(path: Optional[Path | str] = None, validate: bool = True
               ) -> GraphData:
    """Load the mouse connectivity graph (TSV, no header, float weights).

    The file may lack a trailing newline; rows are counted by parsing, not by
    ``wc -l``.
    """
    fpath = _resolve_path(DATASETS["mouse"]["filename"], path)
    src_raw, tgt_raw, w_raw = [], [], []
    with open(fpath, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()  # whitespace/tab tolerant
            src_raw.append(int(parts[0]))
            tgt_raw.append(int(parts[1]))
            w_raw.append(float(parts[2]))

    src_raw = np.array(src_raw, dtype=np.int64)
    tgt_raw = np.array(tgt_raw, dtype=np.int64)
    weight = np.array(w_raw, dtype=np.float64)

    src_idx, tgt_idx, node_ids = remap_node_ids(src_raw, tgt_raw)
    g = GraphData(src=src_idx, tgt=tgt_idx, weight=weight,
                  node_ids=node_ids, name="mouse")
    if validate:
        _validate(g, "mouse")
    return g


def load_dataset(name: str, path: Optional[Path | str] = None,
                 validate: bool = True) -> GraphData:
    """Dispatch to the appropriate loader by dataset ``name``."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASETS)}")
    loader = globals()[DATASETS[name]["loader"]]
    return loader(path=path, validate=validate)


def _validate(g: GraphData, name: str) -> None:
    """Assert basic structural facts against the registry's expected values."""
    spec = DATASETS[name]
    if spec["expected_n"] is not None:
        assert g.n_nodes == spec["expected_n"], (
            f"{name}: expected {spec['expected_n']:,} nodes, got {g.n_nodes:,}")
    if spec["expected_m"] is not None:
        assert g.n_edges == spec["expected_m"], (
            f"{name}: expected {spec['expected_m']:,} edges, got {g.n_edges:,}")
    if spec["expected_total_weight"] is not None:
        assert g.total_weight == spec["expected_total_weight"], (
            f"{name}: expected total weight {spec['expected_total_weight']:,}, "
            f"got {g.total_weight:,.0f}")


# ──────────────────────────────────────────────────────────────────────────────
# Processed cache (npz)
# ──────────────────────────────────────────────────────────────────────────────
def save_processed(g: GraphData, path: Path | str) -> None:
    """Persist a :class:`GraphData` to a compressed ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, src=g.src, tgt=g.tgt, weight=g.weight,
                        node_ids=g.node_ids, name=np.array(g.name))


def load_processed(path: Path | str) -> GraphData:
    """Load a :class:`GraphData` from a ``.npz`` file written by :func:`save_processed`."""
    d = np.load(path, allow_pickle=False)
    return GraphData(src=d["src"], tgt=d["tgt"], weight=d["weight"],
                     node_ids=d["node_ids"], name=str(d["name"]))

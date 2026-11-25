"""Read and write lower-bound certificates (format: ceiling/src/CERT_FORMAT.md).

Only type checks happen here. Validity is decided by verify_cert.py alone.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

INT64_MAX = np.iinfo(np.int64).max
ARRAY_KEYS = ("D", "cyc_ptr", "cyc_edges", "y", "piece_ptr", "piece_edges", "piece_cap", "piece_claim")


def edge_checksum(src, tgt, weight) -> str:
    h = hashlib.sha256()
    for a in (src, tgt, weight):
        a = np.asarray(a)
        if a.dtype.kind not in "iu":
            raise TypeError(f"edge_checksum needs integer arrays, got {a.dtype}")
        h.update(np.ascontiguousarray(a, dtype="<i8").tobytes())
    return h.hexdigest()


def _int_array(x, name):
    if isinstance(x, (bool, np.bool_)):
        raise TypeError(f"{name}: bool is not an integer value")
    if isinstance(x, (int, np.integer)):
        x = [x]
    if isinstance(x, (list, tuple)):
        for v in x:
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
                raise TypeError(f"{name}: non-integer element {v!r}")
            if not (-INT64_MAX - 1 <= int(v) <= INT64_MAX):
                raise OverflowError(f"{name}: value {v} does not fit int64")
        return np.array([int(v) for v in x], dtype=np.int64)
    a = np.asarray(x)
    if a.dtype.kind not in "iu":
        raise TypeError(f"{name}: integer array required, got dtype {a.dtype}")
    if a.dtype.kind == "u" and a.size and int(a.max()) > INT64_MAX:
        raise OverflowError(f"{name}: value does not fit int64")
    return a.astype(np.int64).ravel()


def _ragged(items, name):
    """List of int arrays or a (ptr, flat) pair -> (ptr, flat) int64."""
    if isinstance(items, tuple) and len(items) == 2 and not np.isscalar(items[0]):
        return _int_array(items[0], name + " ptr"), _int_array(items[1], name)
    parts = [_int_array(c, name) for c in items]
    lens = np.array([len(p) for p in parts], dtype=np.int64)
    ptr = np.zeros(len(parts) + 1, dtype=np.int64)
    np.cumsum(lens, out=ptr[1:])
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    return ptr, flat


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_cert(path, *, graph, D, cycles=None, y=None, pieces=None, piece_caps=None,
               piece_claims=None, method, notes):
    if np.asarray(graph.weight).dtype.kind not in "iu":
        raise TypeError("graph weights must be integers (format v1)")
    D = _int_array(D, "D")
    if D.size != 1:
        raise ValueError("D must be a scalar")
    arrays = {"D": D.reshape(())}
    if cycles is not None:
        if y is None:
            raise ValueError("cycles given without y")
        arrays["cyc_ptr"], arrays["cyc_edges"] = _ragged(cycles, "cycles")
        arrays["y"] = _int_array(y, "y")
    elif y is not None:
        raise ValueError("y given without cycles")
    if pieces is not None:
        if piece_caps is None or piece_claims is None:
            raise ValueError("pieces need piece_caps and piece_claims")
        arrays["piece_ptr"], arrays["piece_edges"] = _ragged(pieces, "pieces")
        if isinstance(piece_caps, (list, tuple)) and len(piece_caps) and not np.isscalar(piece_caps[0]):
            piece_caps = _ragged(piece_caps, "piece_caps")[1]
        arrays["piece_cap"] = _int_array(piece_caps, "piece_caps")
        arrays["piece_claim"] = _int_array(piece_claims, "piece_claims")
    elif piece_caps is not None or piece_claims is not None:
        raise ValueError("piece_caps / piece_claims given without pieces")
    meta = dict(dataset=graph.name, n_nodes=int(graph.n_nodes), n_edges=int(graph.n_edges),
                total_weight=int(np.asarray(graph.weight).astype(object).sum()),
                edge_checksum=edge_checksum(graph.src, graph.tgt, graph.weight),
                method=str(method), git_commit=_git_commit(),
                created=datetime.datetime.now().isoformat(timespec="seconds"), notes=str(notes))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez_compressed(f, meta=np.array(json.dumps(meta)), **arrays)
    return path


def read_cert(path):
    """Return (meta dict, dict of arrays). Never unpickles."""
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if "meta" not in data:
        raise ValueError("certificate has no meta")
    meta_arr = data.pop("meta")
    if meta_arr.dtype.kind != "U" or meta_arr.shape != ():
        raise ValueError("meta must be a scalar unicode string")
    return json.loads(str(meta_arr)), data

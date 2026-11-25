"""Independent checker for lower-bound certificates (format: ceiling/src/CERT_FORMAT.md).

Usage (from the repo root):
    python ceiling/src/verify_cert.py <cert.npz> [--dataset connectome|mouse|toy:<path>]
                                      [--max-piece-nodes 20]

Everything in the decision path is int64 with explicit overflow guards, or Python int
where a guard would fail. No floats. The float ceiling_pct is only a display value of
the exact Fraction.

Mouse dataset: its weights are float64, which the integer format v1 cannot express.
verify() rejects any graph with non-integer weight dtype with a clear FAIL reason
instead of trying a rational conversion.

toy:<path> loads a small graph either from an .npz with integer arrays src, tgt, weight
(optional n_nodes) or from a text file with three integer columns "src tgt weight".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from cert_io import ARRAY_KEYS, edge_checksum, read_cert  # noqa: E402

INT64_MAX = int(np.iinfo(np.int64).max)
DEFAULT_MAX_PIECE_NODES = 20
META_KEYS = ("dataset", "n_nodes", "n_edges", "total_weight", "edge_checksum")


class CertError(Exception):
    pass


def _need(cond, msg):
    if not cond:
        raise CertError(msg)


def _int64_array(arrays, key, ndim=1):
    a = arrays[key]
    _need(a.dtype == np.dtype("<i8") or a.dtype == np.dtype("=i8"),
          f"{key}: dtype must be int64, got {a.dtype}")
    _need(a.ndim == ndim, f"{key}: expected {ndim}-d array, got shape {a.shape}")
    return a


def _exact_sum(a):
    """Exact Python-int sum of an int64 array."""
    if a.size == 0:
        return 0
    if max(abs(int(a.max())), abs(int(a.min()))) * a.size <= INT64_MAX:
        return int(a.sum())
    return int(sum(int(x) for x in a.tolist()))


def _check_ptr(ptr, flat_len, min_len, name):
    _need(ptr.size >= 1, f"{name}_ptr is empty (needs at least [0])")
    _need(int(ptr[0]) == 0, f"{name}_ptr[0] must be 0")
    _need(int(ptr[-1]) == flat_len, f"{name}_ptr[-1] = {int(ptr[-1])} != len of edges {flat_len}")
    lens = np.diff(ptr)
    if lens.size:
        bad = np.flatnonzero(lens < min_len)
        _need(bad.size == 0, f"{name} {int(bad[0]) if bad.size else -1} has length "
                             f"{int(lens[bad[0]]) if bad.size else 0} < {min_len}")
    return lens


def _check_ids(ids, m, name):
    if ids.size:
        bad = np.flatnonzero((ids < 0) | (ids >= m))
        _need(bad.size == 0, f"{name}[{int(bad[0]) if bad.size else 0}] = "
                             f"{int(ids[bad[0]]) if bad.size else 0} out of range [0, {m})")


def piece_min_feedback(ps, pt, caps, max_nodes=DEFAULT_MAX_PIECE_NODES):
    """Exact min over node orders of the cap-weighted backward edges of one piece.

    best[S] = min cost of placing node set S first. Appending v after S makes every
    edge v->u with u in S backward. Self-loops cost 0 here (the metric counts them as
    always backward, so 0 can only under-estimate the true minimum).
    """
    nodes = np.unique(np.concatenate([ps, pt]))
    k = int(nodes.size)
    _need(k <= max_nodes, f"piece has {k} nodes > max {max_nodes}")
    _need(_exact_sum(caps) <= INT64_MAX, "piece capacity sum exceeds int64")
    a = np.searchsorted(nodes, ps)
    b = np.searchsorted(nodes, pt)
    C = np.zeros((k, k), dtype=np.int64)
    np.add.at(C, (a, b), caps)
    np.fill_diagonal(C, 0)
    full = 1 << k
    cost = np.zeros((k, full), dtype=np.int64)
    for j in range(k):
        cost[:, 1 << j:1 << (j + 1)] = cost[:, :1 << j] + C[:, j:j + 1]
    idx = np.arange(full, dtype=np.int64)
    pop = np.zeros(full, dtype=np.int64)
    for j in range(k):
        pop += (idx >> j) & 1
    best = np.full(full, INT64_MAX, dtype=np.int64)
    best[0] = 0
    for p in range(1, k + 1):
        layer = idx[pop == p]
        for v in range(k):
            sel = layer[((layer >> v) & 1) == 1]
            prev = sel ^ (1 << v)
            best[sel] = np.minimum(best[sel], best[prev] + cost[v, prev])
    return int(best[full - 1])


def _verify(cert_path, graph, max_piece_nodes, out):
    w = np.asarray(graph.weight)
    _need(w.dtype.kind in "iu",
          f"graph weights have dtype {w.dtype}; format v1 needs integer weights "
          "(float-weight datasets such as mouse are not supported)")
    src = np.asarray(graph.src).astype(np.int64)
    tgt = np.asarray(graph.tgt).astype(np.int64)
    w = w.astype(np.int64)
    n, m = int(graph.n_nodes), int(w.size)
    _need(src.size == m and tgt.size == m, "graph arrays have different lengths")
    W = _exact_sum(w)
    _need(m == 0 or int(w.min()) >= 0, "graph has negative weights")

    try:
        meta, arrays = read_cert(cert_path)
    except Exception as e:  # malformed file, pickles, bad json
        raise CertError(f"cannot read certificate: {type(e).__name__}: {e}")
    _need(isinstance(meta, dict), "meta is not a JSON object")
    for k in META_KEYS:
        _need(k in meta, f"meta lacks {k}")
    _need(meta["dataset"] == graph.name, f"meta dataset {meta['dataset']!r} != graph {graph.name!r}")
    _need(meta["n_nodes"] == n, f"meta n_nodes {meta['n_nodes']} != {n}")
    _need(meta["n_edges"] == m, f"meta n_edges {meta['n_edges']} != {m}")
    _need(meta["total_weight"] == W, f"meta total_weight {meta['total_weight']} != {W}")
    _need(meta["edge_checksum"] == edge_checksum(src, tgt, w), "edge_checksum mismatch")
    unknown = set(arrays) - set(ARRAY_KEYS)
    _need(not unknown, f"unknown arrays in certificate: {sorted(unknown)}")

    _need("D" in arrays, "D missing")
    Darr = _int64_array(arrays, "D", ndim=0)
    D = int(Darr)
    _need(D >= 1, f"D = {D} < 1")
    out["D"] = D

    k1 = [k for k in ("cyc_ptr", "cyc_edges", "y") if k in arrays]
    _need(len(k1) in (0, 3), f"K1 needs cyc_ptr, cyc_edges and y together, got {k1}")
    if k1:
        ptr = _int64_array(arrays, "cyc_ptr")
        ce = _int64_array(arrays, "cyc_edges")
        y = _int64_array(arrays, "y")
    else:
        ptr, ce, y = np.zeros(1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    clens = _check_ptr(ptr, ce.size, 2, "cycle")
    C = int(clens.size)
    _need(y.size == C, f"len(y) = {y.size} != number of cycles {C}")
    _check_ids(ce, m, "cyc_edges")
    if C:
        bad = np.flatnonzero(y < 1)
        _need(bad.size == 0, f"y[{int(bad[0]) if bad.size else 0}] = "
                             f"{int(y[bad[0]]) if bad.size else 0} < 1")
        cid = np.repeat(np.arange(C, dtype=np.int64), clens)
        nxt = np.arange(1, ce.size + 1, dtype=np.int64)
        nxt[ptr[1:] - 1] = ptr[:-1]
        bad = np.flatnonzero(tgt[ce] != src[ce[nxt]])
        _need(bad.size == 0, f"cycle {int(cid[bad[0]]) if bad.size else 0} is not a closed "
                             "walk in traversal order (tgt of an edge != src of the next)")
        _need(C * n <= INT64_MAX, "too many cycles for the simplicity key")
        key = np.sort(cid * n + src[ce])
        dup = np.flatnonzero(key[1:] == key[:-1])
        _need(dup.size == 0, f"cycle {int(key[dup[0]] // n) if dup.size else 0} visits a node twice")
        del key, cid, nxt

    k2 = [k for k in ("piece_ptr", "piece_edges", "piece_cap", "piece_claim") if k in arrays]
    _need(len(k2) in (0, 4), f"K2 needs piece_ptr, piece_edges, piece_cap, piece_claim, got {k2}")
    if k2:
        pptr = _int64_array(arrays, "piece_ptr")
        pe = _int64_array(arrays, "piece_edges")
        pcap = _int64_array(arrays, "piece_cap")
        claim = _int64_array(arrays, "piece_claim")
    else:
        pptr, pe = np.zeros(1, np.int64), np.zeros(0, np.int64)
        pcap, claim = np.zeros(0, np.int64), np.zeros(0, np.int64)
    plens = _check_ptr(pptr, pe.size, 1, "piece")
    P = int(plens.size)
    _need(pcap.size == pe.size, f"len(piece_cap) {pcap.size} != len(piece_edges) {pe.size}")
    _need(claim.size == P, f"len(piece_claim) {claim.size} != number of pieces {P}")
    _check_ids(pe, m, "piece_edges")
    if P:
        bad = np.flatnonzero(pcap < 1)
        _need(bad.size == 0, f"piece_cap[{int(bad[0]) if bad.size else 0}] < 1")
        bad = np.flatnonzero(claim < 0)
        _need(bad.size == 0, f"piece_claim[{int(bad[0]) if bad.size else 0}] < 0")
        pid = np.repeat(np.arange(P, dtype=np.int64), plens)
        _need(P * m <= INT64_MAX, "too many pieces for the duplicate key")
        key = np.sort(pid * m + pe)
        dup = np.flatnonzero(key[1:] == key[:-1])
        _need(dup.size == 0, f"piece {int(key[dup[0]] // m) if dup.size else 0} has a duplicate edge")
        del key, pid

    # per-edge load <= D * w(e)
    ids = np.concatenate([ce, pe])
    vals = np.concatenate([np.repeat(y, clens), pcap])
    if ids.size:
        cnt = np.bincount(ids, minlength=m)
        if int(vals.max()) * int(cnt.max()) <= INT64_MAX:
            load = np.zeros(m, dtype=np.int64)
            np.add.at(load, ids, vals)
        else:
            load = np.zeros(m, dtype=object)
            load[:] = 0
            np.add.at(load, ids, vals.astype(object))
        del cnt
    else:
        load = np.zeros(m, dtype=np.int64)
    del ids, vals
    if m and D * int(w.max()) <= INT64_MAX:
        capw = w * D
    else:
        capw = w.astype(object) * D
    over = np.flatnonzero(load > capw)
    _need(over.size == 0, f"edge {int(over[0]) if over.size else 0} overloaded: load "
                          f"{load[over[0]] if over.size else 0} > D*w = {capw[over[0]] if over.size else 0}")
    slack = capw - load
    out["slack_units"] = _exact_sum(slack) if slack.dtype != object else int(slack.sum())
    out["n_edges_tight"] = int(np.count_nonzero(slack == 0)) if m else 0
    del load, capw, slack

    # pieces: claim <= exact minimum
    max_k = 0
    if P:
        ps_all, pt_all = src[pe], tgt[pe]
        for j in range(P):
            a, b = int(pptr[j]), int(pptr[j + 1])
            try:
                opt = piece_min_feedback(ps_all[a:b], pt_all[a:b], pcap[a:b], max_piece_nodes)
            except CertError as e:
                raise CertError(f"piece {j}: {e}")
            max_k = max(max_k, int(np.unique(np.concatenate([ps_all[a:b], pt_all[a:b]])).size))
            _need(int(claim[j]) <= opt, f"piece {j}: claim {int(claim[j])} > exact minimum {opt}")

    total = _exact_sum(y) + _exact_sum(claim)
    LB = total // D
    out.update(n_cycles=C, n_cycle_edge_entries=int(ce.size),
               max_cycle_len=int(clens.max()) if C else 0,
               n_pieces=P, n_piece_edge_entries=int(pe.size), max_piece_nodes=max_k,
               sum_y=_exact_sum(y), sum_claim=_exact_sum(claim),
               LB=LB, W=W, ceiling_pct_fraction=Fraction(100 * (W - LB), W) if W else None)
    out["ceiling_pct"] = round(float(out["ceiling_pct_fraction"]), 6) if W else None


def verify(cert_path, graph, max_piece_nodes=DEFAULT_MAX_PIECE_NODES) -> dict:
    out = {"cert": str(cert_path), "status": "FAIL", "reason": None}
    t0 = time.time()
    try:
        _verify(cert_path, graph, max_piece_nodes, out)
        out["status"] = "PASS"
    except CertError as e:
        out["reason"] = str(e)
    out["seconds"] = round(time.time() - t0, 2)
    return out


def load_graph(spec):
    if spec.startswith("toy:"):
        from src.mfas.io import GraphData
        p = Path(spec[4:])
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as z:
                s, t, w = z["src"], z["tgt"], z["weight"]
                nn = int(z["n_nodes"]) if "n_nodes" in z.files else None
        else:
            arr = np.loadtxt(p, dtype=np.int64, ndmin=2)
            s, t, w, nn = arr[:, 0], arr[:, 1], arr[:, 2], None
        if nn is None:
            nn = int(max(s.max(), t.max())) + 1 if s.size else 0
        return GraphData(src=s.astype(np.int32), tgt=t.astype(np.int32), weight=w,
                         node_ids=np.arange(nn), name="toy")
    from src.mfas.io import load_dataset
    return load_dataset(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cert")
    ap.add_argument("--dataset", default="connectome")
    ap.add_argument("--max-piece-nodes", type=int, default=DEFAULT_MAX_PIECE_NODES)
    args = ap.parse_args()
    graph = load_graph(args.dataset)
    res = verify(args.cert, graph, args.max_piece_nodes)
    show = {k: (str(v) if isinstance(v, Fraction) else v) for k, v in res.items()}
    print(json.dumps(show, indent=1))
    print(res["status"] + ("" if res["status"] == "PASS" else ": " + res["reason"]))
    sys.exit(0 if res["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

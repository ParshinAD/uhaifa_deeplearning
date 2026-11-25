"""Tests for ceiling/src/verify_cert.py: brute-force agreement on toy graphs and adversarial certificates."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "ceiling" / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cert_io import edge_checksum, write_cert  # noqa: E402
from verify_cert import piece_min_feedback, verify  # noqa: E402
from src.mfas.io import GraphData  # noqa: E402
from src.mfas.metrics import score_from_positions  # noqa: E402

BIG = 2 ** 62


def make_graph(edges, n=None, name="toy"):
    e = np.array(edges, dtype=np.int64).reshape(-1, 3)
    n = int(e[:, :2].max()) + 1 if n is None else n
    return GraphData(src=e[:, 0].astype(np.int32), tgt=e[:, 1].astype(np.int32),
                     weight=e[:, 2].astype(np.int64), node_ids=np.arange(n), name=name)


def brute_min_feedback(src, tgt, w, nodes):
    nodes = list(nodes)
    best = None
    for perm in itertools.permutations(range(len(nodes))):
        pos = np.zeros(max(nodes) + 1 if nodes else 1, dtype=np.int64)
        pos[nodes] = perm
        fb = int(w.sum()) - int(score_from_positions(pos, src, tgt, w))
        best = fb if best is None else min(best, fb)
    return best


def simple_cycles(g):
    adj = {}
    for e, (u, v) in enumerate(zip(g.src.tolist(), g.tgt.tolist())):
        adj.setdefault(u, []).append((v, e))
    out = []

    def dfs(start, u, seen, path):
        for v, e in adj.get(u, []):
            if v == start:
                out.append(path + [e])
            elif v > start and v not in seen:
                dfs(start, v, seen | {v}, path + [e])

    for s in range(g.n_nodes):
        dfs(s, s, {s}, [])
    return out


def meta_for(g, **over):
    meta = dict(dataset=g.name, n_nodes=g.n_nodes, n_edges=g.n_edges,
                total_weight=int(g.weight.sum()), edge_checksum=edge_checksum(g.src, g.tgt, g.weight),
                method="test", git_commit="x", created="x", notes="")
    meta.update(over)
    return meta


def save_raw(path, meta, **arrays):
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def cyc_arrays(cycles, y, D=1):
    lens = [len(c) for c in cycles]
    ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    edges = np.array([e for c in cycles for e in c], dtype=np.int64)
    return dict(D=np.array(D, dtype=np.int64), cyc_ptr=ptr, cyc_edges=edges, y=np.array(y, dtype=np.int64))


def piece_arrays(pieces, caps, claims):
    lens = [len(p) for p in pieces]
    return dict(piece_ptr=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                piece_edges=np.array([e for p in pieces for e in p], dtype=np.int64),
                piece_cap=np.array([c for cs in caps for c in cs], dtype=np.int64),
                piece_claim=np.array(claims, dtype=np.int64))


# e0 0->1 (3), e1 1->2 (2), e2 2->0 (4), e3 1->0 (1), e4 2->3 (5), e5 3->4 (2), e6 4->2 (1), e7 0->3 (2)
G = make_graph([(0, 1, 3), (1, 2, 2), (2, 0, 4), (1, 0, 1), (2, 3, 5), (3, 4, 2), (4, 2, 1), (0, 3, 2)])
G_OPT = 4
GOOD_CYC = [[0, 1, 2], [0, 3], [4, 5, 6]]
GOOD_Y = [2, 1, 1]


def run(tmp_path, meta=None, graph=G, max_piece_nodes=20, **arrays):
    p = save_raw(tmp_path / "c.npz", meta_for(graph) if meta is None else meta, **arrays)
    return verify(p, graph, max_piece_nodes=max_piece_nodes)


def assert_fail(res, word):
    assert res["status"] == "FAIL", res
    assert word in res["reason"], res["reason"]


def test_toy_brute_optimum():
    assert brute_min_feedback(G.src, G.tgt, G.weight, range(5)) == G_OPT


def test_valid_k1(tmp_path):
    res = run(tmp_path, **cyc_arrays(GOOD_CYC, GOOD_Y))
    assert res["status"] == "PASS", res
    assert res["LB"] == 4 <= G_OPT
    assert res["slack_units"] == int(G.weight.sum()) - (2 * 3 + 1 * 2 + 1 * 3)


def test_rotation_is_fine(tmp_path):
    res = run(tmp_path, **cyc_arrays([[1, 2, 0], [3, 0], [5, 6, 4]], GOOD_Y))
    assert res["status"] == "PASS", res


def test_valid_k1_plus_k2(tmp_path):
    arrays = cyc_arrays([[4, 5, 6]], [1])
    arrays.update(piece_arrays([[0, 1, 2, 3]], [[3, 2, 4, 1]], [3]))
    res = run(tmp_path, **arrays)
    assert res["status"] == "PASS", res
    assert res["LB"] == 4


def test_empty_cert(tmp_path):
    res = run(tmp_path, D=np.array(1, dtype=np.int64))
    assert res["status"] == "PASS" and res["LB"] == 0


def test_writer_roundtrip(tmp_path):
    p = write_cert(tmp_path / "w.npz", graph=G, D=3, cycles=[np.array(c) for c in GOOD_CYC],
                   y=[6, 3, 3], pieces=[[7]], piece_caps=[[1]], piece_claims=[0], method="t", notes="")
    res = verify(p, G)
    assert res["status"] == "PASS", res
    assert res["LB"] == 4


def test_writer_refuses_floats(tmp_path):
    with pytest.raises(TypeError):
        write_cert(tmp_path / "w.npz", graph=G, D=1, cycles=GOOD_CYC, y=[2.0, 1.0, 1.0], method="t", notes="")
    with pytest.raises(TypeError):
        write_cert(tmp_path / "w.npz", graph=G, D=1.5, cycles=GOOD_CYC, y=GOOD_Y, method="t", notes="")
    with pytest.raises(TypeError):
        write_cert(tmp_path / "w.npz", graph=G, D=1, cycles=[[0, 1, 2.0]], y=[1], method="t", notes="")


def random_graph(rng, n):
    edges = [(u, v, int(rng.integers(1, 6))) for u in range(n) for v in range(n)
             if u != v and rng.random() < 0.45]
    if not edges:
        edges = [(0, 1, 1)]
    return make_graph(edges, n=n)


@pytest.mark.parametrize("seed", range(25))
def test_random_valid_certs_bound_brute_force(tmp_path, seed):
    rng = np.random.default_rng(seed)
    g = random_graph(rng, int(rng.integers(3, 8)))
    D = int(rng.integers(1, 4))
    res_cap = g.weight.astype(np.int64) * D
    cycles, ys = [], []
    cyc = simple_cycles(g)
    rng.shuffle(cyc)
    for c in cyc:
        yv = int(res_cap[c].min())
        if yv > 0:
            yv = int(rng.integers(1, yv + 1))
            res_cap[c] -= yv
            cycles.append(c)
            ys.append(yv)
    arrays = cyc_arrays(cycles, ys, D) if cycles else dict(D=np.array(D, dtype=np.int64))
    pe = np.flatnonzero(res_cap > 0)
    if pe.size:
        caps = res_cap[pe]
        nodes = np.unique(np.concatenate([g.src[pe], g.tgt[pe]]))
        claim = brute_min_feedback(g.src[pe], g.tgt[pe], caps, nodes)
        assert piece_min_feedback(g.src[pe].astype(np.int64), g.tgt[pe].astype(np.int64), caps) == claim
        arrays.update(piece_arrays([pe.tolist()], [caps.tolist()], [claim]))
    opt = brute_min_feedback(g.src, g.tgt, g.weight, range(g.n_nodes))
    res = run(tmp_path, graph=g, **arrays)
    assert res["status"] == "PASS", res
    assert res["LB"] <= opt
    if pe.size:
        arrays["piece_claim"] = arrays["piece_claim"] + 1
        assert_fail(run(tmp_path, graph=g, **arrays), "exact minimum")


@pytest.mark.parametrize("seed", range(40))
def test_dp_matches_brute_force(seed):
    rng = np.random.default_rng(100 + seed)
    g = random_graph(rng, int(rng.integers(2, 8)))
    caps = rng.integers(1, 10, size=g.n_edges).astype(np.int64)
    nodes = np.unique(np.concatenate([g.src, g.tgt]))
    assert piece_min_feedback(g.src.astype(np.int64), g.tgt.astype(np.int64), caps) == \
        brute_min_feedback(g.src, g.tgt, caps, nodes)


# ---- adversarial certificates ----

def test_open_walk(tmp_path):
    assert_fail(run(tmp_path, **cyc_arrays([[0, 1]], [1])), "closed walk")


def test_non_simple_closed_walk(tmp_path):
    assert_fail(run(tmp_path, **cyc_arrays([[0, 3, 7, 5, 6, 2]], [1])), "visits a node twice")


def test_reversed_order(tmp_path):
    assert_fail(run(tmp_path, **cyc_arrays([[2, 1, 0]], [1])), "closed walk")


def test_edge_id_out_of_range(tmp_path):
    assert_fail(run(tmp_path, **cyc_arrays([[0, 1, 8]], [1])), "out of range")
    assert_fail(run(tmp_path, **cyc_arrays([[0, 1, -1]], [1])), "out of range")
    a = piece_arrays([[0, 99]], [[1, 1]], [0])
    assert_fail(run(tmp_path, D=np.array(1, dtype=np.int64), **a), "out of range")


def test_length_one_cycle(tmp_path):
    assert_fail(run(tmp_path, **cyc_arrays([[0]], [1])), "length")


def test_overload_by_one_unit(tmp_path):
    D = 3
    ok = run(tmp_path, **cyc_arrays(GOOD_CYC, [6, 3, 3], D))
    assert ok["status"] == "PASS" and ok["LB"] == 4
    assert_fail(run(tmp_path, **cyc_arrays(GOOD_CYC, [6, 4, 3], D)), "overloaded")


def test_overload_cycle_plus_piece(tmp_path):
    a = cyc_arrays([[4, 5, 6]], [1])
    a.update(piece_arrays([[4, 5, 6]], [[5, 2, 1]], [1]))
    assert_fail(run(tmp_path, **a), "overloaded")


@pytest.mark.parametrize("bad", [0, -1, -BIG])
def test_y_nonpositive(tmp_path, bad):
    assert_fail(run(tmp_path, **cyc_arrays(GOOD_CYC, [2, bad, 1])), "< 1")


def test_overflow_wraparound_load(tmp_path):
    # four copies of the same cycle with y = 2^62: an int64 sum wraps to 0
    assert_fail(run(tmp_path, **cyc_arrays([[4, 5, 6]] * 4, [BIG] * 4)), "overloaded")


def test_overflow_capacity_and_sum_exact(tmp_path):
    # D * w(e) does not fit int64 and sum(y) does not fit int64; both must be exact
    D = BIG
    a = cyc_arrays([[0, 1, 2], [4, 5, 6]], [2 ** 63 - 1, BIG], D)
    res = run(tmp_path, **a)
    assert res["status"] == "PASS", res
    assert res["LB"] == (2 ** 63 - 1 + BIG) // BIG == 2 <= G_OPT
    a = cyc_arrays([[4, 5, 6]], [BIG + 1], D)
    assert_fail(run(tmp_path, **a), "overloaded")


def test_piece_claim_one_above(tmp_path):
    base = piece_arrays([[0, 1, 2, 3]], [[3, 2, 4, 1]], [3])
    assert run(tmp_path, D=np.array(1, dtype=np.int64), **base)["status"] == "PASS"
    bad = piece_arrays([[0, 1, 2, 3]], [[3, 2, 4, 1]], [4])
    assert_fail(run(tmp_path, D=np.array(1, dtype=np.int64), **bad), "exact minimum")


def test_piece_duplicate_edge(tmp_path):
    a = piece_arrays([[0, 0, 1, 2]], [[1, 1, 2, 4]], [1])
    assert_fail(run(tmp_path, D=np.array(1, dtype=np.int64), **a), "duplicate")


def test_piece_cap_zero_and_negative_claim(tmp_path):
    a = piece_arrays([[0, 1, 2]], [[0, 2, 4]], [0])
    assert_fail(run(tmp_path, D=np.array(1, dtype=np.int64), **a), "piece_cap")
    a = piece_arrays([[0, 1, 2]], [[1, 2, 4]], [-1])
    assert_fail(run(tmp_path, D=np.array(1, dtype=np.int64), **a), "piece_claim")


def test_piece_too_large(tmp_path):
    k = 21
    g = make_graph([(i, (i + 1) % k, 1) for i in range(k)])
    a = piece_arrays([list(range(k))], [[1] * k], [1])
    assert_fail(run(tmp_path, graph=g, D=np.array(1, dtype=np.int64), **a), "nodes > max")
    a = piece_arrays([[0, 1, 2, 3]], [[3, 2, 4, 1]], [3])
    assert_fail(run(tmp_path, max_piece_nodes=2, D=np.array(1, dtype=np.int64), **a), "nodes > max")


@pytest.mark.parametrize("field,value", [("n_edges", 9), ("n_nodes", 6), ("total_weight", 21),
                                         ("edge_checksum", "0" * 64), ("dataset", "connectome")])
def test_header_mismatch(tmp_path, field, value):
    assert_fail(run(tmp_path, meta=meta_for(G, **{field: value}), **cyc_arrays(GOOD_CYC, GOOD_Y)), field)


def test_header_missing_field(tmp_path):
    meta = meta_for(G)
    del meta["edge_checksum"]
    assert_fail(run(tmp_path, meta=meta, **cyc_arrays(GOOD_CYC, GOOD_Y)), "lacks")


def test_cert_against_other_graph(tmp_path):
    p = save_raw(tmp_path / "c.npz", meta_for(G), **cyc_arrays(GOOD_CYC, GOOD_Y))
    w = G.weight.copy()
    w[[4, 5]] = w[[5, 4]]  # same n, m, W; only the checksum can notice
    g2 = GraphData(src=G.src, tgt=G.tgt, weight=w, node_ids=G.node_ids, name="toy")
    assert_fail(verify(p, g2), "edge_checksum")


@pytest.mark.parametrize("key,arr", [
    ("y", np.array([2.0, 1.0, 1.0])),
    ("cyc_edges", np.array([0, 1, 2, 0, 3, 4, 5, 6], dtype=np.float64)),
    ("cyc_edges", np.array([0, 1, 2, 0, 3, 4, 5, 6], dtype=np.int32)),
    ("y", np.array([2, 1, 1], dtype=np.uint64)),
    ("y", np.array([True, True, True])),
    ("D", np.array(1.0)),
    ("D", np.array([1], dtype=np.int64)),
])
def test_smuggled_types(tmp_path, key, arr):
    a = cyc_arrays(GOOD_CYC, GOOD_Y)
    a[key] = arr
    assert_fail(run(tmp_path, **a), key)


def test_pickled_object_array(tmp_path):
    a = cyc_arrays(GOOD_CYC, GOOD_Y)
    a["y"] = np.array([2, 1, 1], dtype=object)
    assert_fail(run(tmp_path, **a), "cannot read")


def test_structure_errors(tmp_path):
    a = cyc_arrays(GOOD_CYC, GOOD_Y)
    del a["y"]
    assert_fail(run(tmp_path, **a), "K1 needs")
    a = cyc_arrays(GOOD_CYC, GOOD_Y)
    a["extra"] = np.zeros(1, dtype=np.int64)
    assert_fail(run(tmp_path, **a), "unknown")
    a = cyc_arrays(GOOD_CYC, GOOD_Y)
    a["cyc_ptr"] = np.array([0, 3, 5, 7], dtype=np.int64)
    assert_fail(run(tmp_path, **a), "cyc")
    a = cyc_arrays(GOOD_CYC, [2, 1])
    assert_fail(run(tmp_path, **a), "len(y)")
    a = cyc_arrays(GOOD_CYC, GOOD_Y, D=0)
    assert_fail(run(tmp_path, **a), "D = 0")


def test_float_weight_graph_rejected(tmp_path):
    g = GraphData(src=G.src, tgt=G.tgt, weight=G.weight.astype(np.float64), node_ids=G.node_ids, name="toy")
    p = save_raw(tmp_path / "c.npz", meta_for(G), **cyc_arrays(GOOD_CYC, GOOD_Y))
    assert_fail(verify(p, g), "integer weights")


@pytest.mark.skipif(not any((ROOT / "data" / d / "connectome_graph.csv.gz").exists() for d in ("raw", ".")),
                    reason="no connectome data")
def test_connectome_2cycle(tmp_path):
    from cert_2cycle import build
    from src.mfas.io import load_dataset
    g = load_dataset("connectome")
    ptr, edges, y = build(g)
    p = write_cert(tmp_path / "c00.npz", graph=g, D=1, cycles=(ptr, edges), y=y, method="2cycle", notes="test")
    res = verify(p, g)
    assert res["status"] == "PASS", res
    assert res["LB"] == 2_815_303
    assert round(float(res["ceiling_pct_fraction"]), 4) == 93.2828

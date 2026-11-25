import json, sys, zipfile, io, itertools
from pathlib import Path
import numpy as np

REPO = Path("/Users/abed359/IdeaProjects/university/deeplearning_thesis_ceiling/nn_backward_connections")
sys.path[:0] = [str(REPO / "ceiling" / "src"), str(REPO)]
from cert_io import edge_checksum
from verify_cert import verify
from src.mfas.io import GraphData

TMP = Path("/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad/redteam/att")
TMP.mkdir(exist_ok=True)


def mk(edges, name="toy"):
    e = np.array(edges, dtype=np.int64)
    n = int(e[:, :2].max()) + 1
    return GraphData(src=e[:, 0].astype(np.int32), tgt=e[:, 1].astype(np.int32),
                     weight=e[:, 2].astype(np.int64), node_ids=np.arange(n), name=name)


def opt(g):
    n = g.n_nodes; best = None
    for p in itertools.permutations(range(n)):
        p = np.array(p)
        fb = int(g.weight[~(p[g.tgt] > p[g.src])].sum())
        best = fb if best is None else min(best, fb)
    return best


def meta(g):
    return dict(dataset=g.name, n_nodes=g.n_nodes, n_edges=g.n_edges, total_weight=int(g.weight.sum()),
                edge_checksum=edge_checksum(g.src, g.tgt, g.weight), method="x", git_commit="x", created="x", notes="")


def cyc(cycles, y, D=1, dt=np.int64):
    lens = [len(c) for c in cycles]
    return dict(D=np.array(D, dtype=np.int64), cyc_ptr=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                cyc_edges=np.array([e for c in cycles for e in c], dtype=np.int64), y=np.array(y, dtype=dt))


def run(name, g, arrays, m=None, raw=None):
    p = TMP / (name + ".npz")
    if raw is not None:
        p.write_bytes(raw)
    else:
        np.savez(p, meta=np.array(json.dumps(m or meta(g))), **arrays)
    r = verify(p, g)
    o = opt(g)
    invalid = r["status"] == "PASS" and r["LB"] > o
    print(f"{name}: {r['status']} LB={r.get('LB')} opt={o} BYPASS={invalid} reason={r['reason']}")
    return r


# 0->1 w3, 1->2 w2, 2->0 w4, 1->0 w1 ; opt = 1 (order 0,1,2: back edges 2->0 (4), 1->0(1)... brute decides)
G = mk([(0, 1, 3), (1, 2, 2), (2, 0, 4), (1, 0, 1)])
print("opt G", opt(G))

# 1 big-endian y
a = cyc([[0, 1, 2]], [2]); a["y"] = np.array([2], dtype=">i8"); run("a01_bigendian_y", G, a)
# 2 y huge so that 4 copies wrap int64 to small
B = 2 ** 62
run("a02_wrap_load", G, cyc([[0, 1, 2]] * 4, [B] * 4, D=1))
# 3 negative D with negative y (D*w negative, load negative)
run("a03_negative_D_y", G, cyc([[0, 1, 2]], [-100], D=-1))
# 4 figure-eight closed walk 0->1->0->1? use edges 0(0->1),3(1->0),0,3 : repeats edge & node
run("a04_repeat_edge_walk", G, cyc([[0, 3, 0, 3]], [1]))
# 5 negative edge id (python wraparound indexing) -1 = edge 3
run("a05_negative_edge_id", G, cyc([[0, -1]], [1]))
# 6 edge id == m
run("a06_edge_id_m", G, cyc([[0, 4]], [1]))
# 7 non-monotone ptr with total length right: [0,4,2,6] over edges
a = dict(D=np.array(1, dtype=np.int64), cyc_ptr=np.array([0, 4, 2, 4], dtype=np.int64),
         cyc_edges=np.array([0, 1, 2, 0], dtype=np.int64), y=np.array([1, 1, 1], dtype=np.int64))
run("a07_nonmonotone_ptr", G, a)
# 8 ptr with cycle of length 0 (duplicate pointer) to give a free y
a = dict(D=np.array(1, dtype=np.int64), cyc_ptr=np.array([0, 3, 3], dtype=np.int64),
         cyc_edges=np.array([0, 1, 2], dtype=np.int64), y=np.array([1, 100], dtype=np.int64))
run("a08_empty_cycle", G, a)
# 9 D as 1-element 1-d array
a = cyc([[0, 1, 2]], [1]); a["D"] = np.array([1], dtype=np.int64); run("a09_D_1d", G, a)
# 10 y given as 2-d (C,1)
a = cyc([[0, 1, 2]], [5]); a["y"] = a["y"].reshape(1, 1); run("a10_y_2d", G, a)
# 11 floor vs ceil: sum y = D*k - 1
r = run("a11_floor", G, cyc([[0, 3]], [1 * 3 - 1], D=3))
# 12 duplicate zip member: first y.npy valid, second y.npy huge
buf = io.BytesIO()
good = cyc([[0, 3]], [1]); bad = cyc([[0, 3]], [10 ** 6])
with zipfile.ZipFile(buf, "w") as z:
    for k, v in [("meta", np.array(json.dumps(meta(G))))] + list(good.items()):
        b = io.BytesIO(); np.save(b, v); z.writestr(k + ".npy", b.getvalue())
    b = io.BytesIO(); np.save(b, bad["y"]); z.writestr("y.npy", b.getvalue())
run("a12_dup_zip_member", G, None, raw=buf.getvalue())
# 13 piece with claim > opt using self loop edges: graph with self loop
GS = mk([(0, 1, 3), (1, 0, 2), (0, 0, 5)])
a = dict(D=np.array(1, dtype=np.int64), piece_ptr=np.array([0, 3], dtype=np.int64),
         piece_edges=np.array([0, 1, 2], dtype=np.int64), piece_cap=np.array([3, 2, 5], dtype=np.int64),
         piece_claim=np.array([7], dtype=np.int64))
run("a13_piece_selfloop_claim", GS, a)
# 14 two overlapping pieces both claiming full edge weight (cap sum over D*w)
a = dict(D=np.array(1, dtype=np.int64), piece_ptr=np.array([0, 2, 4], dtype=np.int64),
         piece_edges=np.array([0, 3, 0, 3], dtype=np.int64), piece_cap=np.array([3, 1, 3, 1], dtype=np.int64),
         piece_claim=np.array([1, 1], dtype=np.int64))
run("a14_overlapping_pieces", G, a)
# 15 cycle + piece sharing edge beyond capacity
a = cyc([[0, 3]], [1]); a.update(piece_ptr=np.array([0, 2], dtype=np.int64), piece_edges=np.array([0, 3], dtype=np.int64),
                                 piece_cap=np.array([3, 1], dtype=np.int64), piece_claim=np.array([1], dtype=np.int64))
run("a15_cycle_piece_overload", G, a)
# 16 meta with different dataset name but same arrays (graph name check)
m = meta(G); m["dataset"] = "connectome"; run("a16_meta_name", G, cyc([[0, 3]], [1]), m=m)
# 17 meta with numbers as strings / floats
m = meta(G); m["n_nodes"] = float(m["n_nodes"]); m["total_weight"] = str(m["total_weight"]); run("a17_meta_types", G, cyc([[0, 3]], [1]), m=m)
# 18 uint64 y holding 2^64-1 (== -1 as int64)
a = cyc([[0, 3]], [1]); a["y"] = np.array([2 ** 64 - 1], dtype=np.uint64); run("a18_uint64", G, a)
# 19 cyc_edges as int64 view of a non-contiguous / Fortran 2-d array
a = cyc([[0, 1, 2]], [1]); a["cyc_edges"] = np.asfortranarray(np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64))
run("a19_fortran_2d", G, a)
# 20 huge D overflow path with y just over D*w on one edge
D = 2 ** 62; run("a20_bigD_over", G, cyc([[0, 3]], [D * 1 + 1], D=D))
# 21 cycle on parallel-edge multigraph legit (sanity: should PASS with LB <= opt)
GP = mk([(0, 1, 1), (0, 1, 1), (1, 0, 5)])
run("a21_parallel_edges_overuse", GP, cyc([[0, 2], [1, 2]], [1, 5]))
# 22 piece that has nodes > 20 limit evaded by sharing? piece with 1 edge claims its cap
run("a22_single_edge_piece_claim", G, dict(D=np.array(1, dtype=np.int64), piece_ptr=np.array([0, 1], dtype=np.int64),
    piece_edges=np.array([2], dtype=np.int64), piece_cap=np.array([4], dtype=np.int64), piece_claim=np.array([1], dtype=np.int64)))

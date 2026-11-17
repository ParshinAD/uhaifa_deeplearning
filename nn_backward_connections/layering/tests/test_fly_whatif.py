"""What-if moves of the fly streams page: independent numpy references vs the
page's JS core (layering/tools/fly_whatif_core.js, run through
layering/tests/fly_whatif_check.mjs), scored by the frozen oracle.

Design: layering/VIZ_V2_DESIGN.md section 8(b). Skipped if `node` is not on
PATH or the artifact files (artifact_data.json, edges.zlib) are missing;
regenerate them with `$PY layering/tools/fly_artifact_export.py`.
Expected scores are recomputed here by the oracle, never hard-coded.
"""
import base64
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CONFIG = {
    "art_dir": REPO_ROOT / "layering" / "outputs" / "fly_artifact",
    "check_mjs": REPO_ROOT / "layering" / "tests" / "fly_whatif_check.mjs",
    "node": shutil.which("node"),
    "toy_seed": 42,
    "toy_cases": 3000,
    "toy_max_n": 11,
    "toy_delta_pad": 3,           # toy deltas range beyond saturation by this much
    "cell_bins": 16,              # case 3: largest H64 streams cell at this B
    "shift_delta": 5000,          # cases 3, 4, 5
    "far_delta": -10**6,          # case 5, clamped
    "right_delta": -2000,         # case 8
    "sep_blocks": ((0, 1000), (5000, 6000)),   # case 7, H64 rank ranges
    "moved_layouts": (("streams", 16), ("cell_class", 32)),
    "stats_tol": 1e-9,
    "swap_size_warn": 0.05,
}

pytestmark = pytest.mark.skipif(
    CONFIG["node"] is None
    or not (CONFIG["art_dir"] / "artifact_data.json").exists()
    or not (CONFIG["art_dir"] / "edges.zlib").exists(),
    reason="needs node on PATH and layering/outputs/fly_artifact/{artifact_data.json,edges.zlib}")


# --- independent references -------------------------------------------------

def shift_ref(rank, sel, delta):
    """Rigid shift, insertion form of the design (section 8(b))."""
    order = np.argsort(rank, kind="stable")
    s_ord, u_ord = order[sel[order]], order[~sel[order]]
    p = np.cumsum(~sel[order])[sel[order]]
    t = np.clip(p + delta, 0, len(u_ord))
    new_order = np.insert(u_ord, t, s_ord)
    new_rank = np.empty_like(rank)
    new_rank[new_order] = np.arange(len(rank))
    return new_rank


def shift_brute(rank, sel, delta):
    """Pure-Python brute force: walk U, emit S members at their target slot."""
    order = sorted(range(len(rank)), key=lambda v: rank[v])
    S = [v for v in order if sel[v]]
    U = [v for v in order if not sel[v]]
    target, before = {}, 0
    for v in order:
        if sel[v]:
            target[v] = min(max(before + delta, 0), len(U))
        else:
            before += 1
    out = []
    for k in range(len(U) + 1):
        out += [s for s in S if target[s] == k]
        if k < len(U):
            out.append(U[k])
    new_rank = np.empty(len(rank), dtype=np.int64)
    new_rank[out] = np.arange(len(rank))
    return new_rank


def shift_p(rank, sel):
    """Unselected nodes before each selected node (in rank order) and nU."""
    order = np.argsort(rank, kind="stable")
    return np.cumsum(~sel[order])[sel[order]], int((~sel).sum())


def swap_ref(rank, A, B):
    """Group swap of the design (section 4); A, B are node index arrays."""
    Z = np.sort(rank[np.r_[A, B]])
    first, second = (B, A) if np.median(rank[A]) <= np.median(rank[B]) else (A, B)
    seq = np.r_[first[np.argsort(rank[first])], second[np.argsort(rank[second])]]
    r = rank.copy()
    r[seq] = Z
    return r


def check_shift_props(rank, sel, delta, new_rank):
    """Properties (i), (iii), (iv), (v) of design section 8(b)."""
    p, nU = shift_p(rank, sel)
    if -p.min() <= delta <= nU - p.max():                       # (i) rigid
        assert np.all(new_rank[sel] - rank[sel] == delta)
    for m in (sel, ~sel):                                       # (iii)
        assert np.all(np.diff(new_rank[m][np.argsort(rank[m])]) > 0)
    if delta == 0:                                              # (iv)
        assert np.array_equal(new_rank, rank)
    if delta <= -p.max():                                       # (v)
        assert new_rank[sel].max() == sel.sum() - 1
        assert np.array_equal(new_rank, shift_ref(rank, sel, -int(p.max())))
    if delta >= nU - p.min():
        assert new_rank[sel].min() == len(rank) - sel.sum()
        assert np.array_equal(new_rank, shift_ref(rank, sel, nU - int(p.min())))


def sha(rank):
    return hashlib.sha256(np.asarray(rank, dtype="<i4").tobytes()).hexdigest()


def run_toys(cases, tmp_path):
    f = tmp_path / "toys.json"
    f.write_text(json.dumps(cases))
    out = subprocess.run([CONFIG["node"], str(CONFIG["check_mjs"]), "--toys", str(f)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


# --- toys -------------------------------------------------------------------

def test_shift_toy_block(tmp_path):
    rank, sel = np.arange(8), np.arange(8) < 4
    expect = [4, 0, 1, 2, 3, 5, 6, 7]
    assert np.argsort(shift_ref(rank, sel, 1)).tolist() == expect
    assert np.argsort(shift_brute(rank, sel, 1)).tolist() == expect
    js = run_toys([{"op": "shift", "rank": rank.tolist(), "sel": [0, 1, 2, 3], "delta": 1}], tmp_path)
    assert np.argsort(js[0]["rank"]).tolist() == expect


def test_shift_contiguous_block_passes_one_neighbour(tmp_path):
    rng = np.random.default_rng(CONFIG["toy_seed"])
    cases, refs = [], []
    for _ in range(200):
        n = int(rng.integers(3, CONFIG["toy_max_n"] + 1))
        rank = rng.permutation(n)
        a = int(rng.integers(0, n - 1))
        b = int(rng.integers(a + 1, n + 1))                     # block = ranks [a, b)
        if b - a == n:
            continue
        sel = (rank >= a) & (rank < b)
        for d in (1, -1):
            if (d == 1 and b == n) or (d == -1 and a == 0):
                continue
            r = shift_ref(rank, sel, d)
            moved = np.nonzero(~sel & (r != rank))[0]           # (ii)
            assert len(moved) == 1 and r[moved[0]] - rank[moved[0]] == -d * sel.sum()
            assert np.all(r[sel] - rank[sel] == d)
            cases.append({"op": "shift", "rank": rank.tolist(),
                          "sel": np.nonzero(sel)[0].tolist(), "delta": d})
            refs.append(r)
    js = run_toys(cases, tmp_path)
    assert all(np.array_equal(j["rank"], r) for j, r in zip(js, refs))


def test_shift_random_toys_reference_brute_js(tmp_path):
    rng = np.random.default_rng(CONFIG["toy_seed"])
    cases, refs = [], []
    for _ in range(CONFIG["toy_cases"]):
        n = int(rng.integers(2, CONFIG["toy_max_n"] + 1))
        rank = rng.permutation(n)
        sel = rng.random(n) < rng.random()
        if sel.all() or not sel.any():
            continue
        pad = n + CONFIG["toy_delta_pad"]
        delta = int(rng.integers(-pad, pad + 1))
        r = shift_ref(rank, sel, delta)
        assert np.array_equal(r, shift_brute(rank, sel, delta))
        check_shift_props(rank, sel, delta, r)
        cases.append({"op": "shift", "rank": rank.tolist(),
                      "sel": np.nonzero(sel)[0].tolist(), "delta": delta})
        refs.append((r, sel, delta, rank))
    js = run_toys(cases, tmp_path)
    for j, (r, sel, delta, rank) in zip(js, refs):
        assert np.array_equal(j["rank"], r) and j["ord_ok"]
        p, nU = shift_p(rank, sel)
        assert j["delta"] == int(np.clip(delta, -p.max(), nU - p.min()))
        assert j["clamped"] == (j["delta"] != delta)
        assert j["rigid"] == bool(-p.min() <= j["delta"] <= nU - p.max())
        assert j["saturated"] == (j["delta"] in (-int(p.max()), nU - int(p.min())))
        b = j["bounds"]
        assert (b["minP"], b["maxP"], b["nU"]) == (int(p.min()), int(p.max()), nU)
    assert len(js) > CONFIG["toy_cases"] // 2


def test_shift_saturates_at_upper_bound(tmp_path):
    rank, sel = np.arange(8), np.isin(np.arange(8), [1, 4])
    p, nU = shift_p(rank, sel)
    hi = nU - int(p.min())
    deltas = [hi - 1, hi, hi + 1, hi + 100]
    js = run_toys([{"op": "shift", "rank": rank.tolist(), "sel": [1, 4], "delta": d} for d in deltas], tmp_path)
    assert not js[0]["saturated"]
    for j in js[1:]:
        assert j["saturated"] and j["delta"] == hi
        assert j["rank"] == js[1]["rank"]
        assert sorted(np.asarray(j["rank"])[sel].tolist()) == [6, 7]


def test_swap_size_warning_threshold(tmp_path):
    rank = np.arange(200)
    A = np.arange(100)
    cases = [{"op": "swap", "rank": rank.tolist(), "A": A.tolist(), "B": np.arange(100, 100 + nB).tolist()}
             for nB in (94, 96)]
    js = run_toys(cases, tmp_path)
    assert js[0]["warnings"] == ["sizes differ; 6 members of the larger group stay in its own region"]
    assert js[1]["warnings"] == []
    assert 6 / 100 > CONFIG["swap_size_warn"] > 4 / 100


def test_swap_toys(tmp_path):
    ident = np.arange(12)
    A, B = np.array([0, 1, 2]), np.array([10])
    r = swap_ref(ident, A, B)
    assert r[10] == 0 and sorted(r[A].tolist()) == [1, 2, 10]  # B@0, A@{1,2,10}
    sepA, sepB = np.arange(0, 3), np.arange(6, 9)
    once = swap_ref(ident, sepA, sepB)
    assert np.array_equal(swap_ref(once, sepA, sepB), ident)
    inter = (np.array([0, 2, 7]), np.array([1, 8, 9]))
    rng = np.random.default_rng(CONFIG["toy_seed"])
    cases = [{"op": "swap", "rank": ident.tolist(), "A": A.tolist(), "B": B.tolist()},
             {"op": "swap", "rank": once.tolist(), "A": sepA.tolist(), "B": sepB.tolist()},
             {"op": "swap", "rank": ident.tolist(), "A": inter[0].tolist(), "B": inter[1].tolist()},
             {"op": "swap", "rank": ident.tolist(), "A": [0, 1], "B": [1, 2]}]
    refs = [r, ident, swap_ref(ident, *inter), None]
    for _ in range(500):
        n = int(rng.integers(2, CONFIG["toy_max_n"] + 1))
        rank = rng.permutation(n)
        lab = rng.integers(0, 3, n)                             # 0 rest, 1 A, 2 B
        if not (lab == 1).any() or not (lab == 2).any():
            continue
        a, b = np.nonzero(lab == 1)[0], np.nonzero(lab == 2)[0]
        cases.append({"op": "swap", "rank": rank.tolist(), "A": a.tolist(), "B": b.tolist()})
        refs.append(swap_ref(rank, a, b))
    js = run_toys(cases, tmp_path)
    for j, ref in zip(js, refs):
        if ref is not None:
            assert np.array_equal(j["rank"], ref) and j["ord_ok"]
    assert js[0]["warnings"][0] == "sizes differ; 2 members of the larger group stay in its own region"
    assert not js[1]["warnings"] and not js[1]["interleaved"]
    assert js[2]["interleaved"] and any("interleaved" in w for w in js[2]["warnings"])
    assert js[3] == {"error": "A and B share 1 neurons"}


# --- real data: cases 1-9 ---------------------------------------------------

@pytest.fixture(scope="module")
def real(tmp_path_factory):
    """Graph + artifact arrays, the case moves run through Node, and references."""
    from layering import io_utils
    from src.mfas.metrics import score_from_order               # frozen oracle

    g, _, rank, _ = io_utils.load_graph_and_champion_order("connectome")
    data = json.loads((CONFIG["art_dir"] / "artifact_data.json").read_text(encoding="utf-8"))
    dec = lambda s, t: np.frombuffer(base64.b64decode(s), dtype=t)
    cloud = data["cloud"]
    assert np.array_equal(dec(cloud["x_u32"], np.uint32), rank)
    side = dec(cloud["side_u8"], np.uint8)
    lanes = {k: dec(v["lane_u8"], np.uint8).astype(np.int64) for k, v in cloud["groups"].items()}
    n = g.n_nodes
    B = CONFIG["cell_bins"]
    cell = lanes["streams"] * B + np.minimum(rank * B // n, B - 1)
    c3 = cell == np.bincount(cell).argmax()
    left, right = side == 1, side == 2
    mb = lanes["streams"] == data["lane_list"].index("mushroom body")
    mbA, mbB = np.nonzero(mb & left)[0], np.nonzero(mb & right)[0]
    (a0, a1), (b0, b1) = CONFIG["sep_blocks"]
    sepA = np.nonzero((rank >= a0) & (rank < a1))[0]
    sepB = np.nonzero((rank >= b0) & (rank < b1))[0]
    p_left, _ = shift_p(rank, left)
    D = CONFIG["shift_delta"]
    node0 = int(np.argmin(rank))

    idx = lambda m: np.nonzero(m)[0]
    sh = lambda m, d, **kw: dict({"op": "shift", "sel": idx(m).tolist(), "delta": int(d)}, **kw)
    sw = lambda a, b: {"op": "swap", "A": a.tolist(), "B": b.tolist()}
    reset = {"op": "reset"}
    plan = [
        ("case1", sh(left, 0)), ("", reset),
        ("case2", sh(np.arange(n) == node0, 1)), ("", reset),
        ("case3", sh(c3, D, layouts=[list(x) for x in CONFIG["moved_layouts"]])), ("", reset),
        ("case4", sh(left, D)), ("", reset),
        ("case5_far", sh(left, CONFIG["far_delta"])), ("", reset),
        ("case5_bound", sh(left, -int(p_left.max()))), ("", reset),
        ("case5", sh(left, -D)), ("", reset),
        ("case6", sw(mbA, mbB)), ("", reset),
        ("case7_once", sw(sepA, sepB)), ("case7_twice", sw(sepA, sepB)), ("", reset),
        ("case8_a", sh(c3, D)), ("case8_b", sh(right, CONFIG["right_delta"])), ("case8", sw(mbA, mbB)),
    ]
    tmp = tmp_path_factory.mktemp("whatif")
    npy = tmp / "case9.npy"
    (tmp / "moves.json").write_text(json.dumps({"npy_out": str(npy), "moves": [m for _, m in plan]}))
    proc = subprocess.run([CONFIG["node"], str(CONFIG["check_mjs"]), str(CONFIG["art_dir"]),
                           str(tmp / "moves.json")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    js = json.loads(proc.stdout)

    # independent numpy replay of the same plan
    ref, cur = {}, rank
    for (name, m), out in zip(plan, js["moves"]):
        sel_of = lambda: np.isin(np.arange(n), m["sel"])
        if m["op"] == "reset":
            cur = rank
        elif m["op"] == "shift":
            cur = shift_ref(cur, sel_of(), m["delta"])
        else:
            cur = swap_ref(cur, np.array(m["A"]), np.array(m["B"]))
        if name:
            ref[name] = (cur, out)
    score = lambda r: int(score_from_order(r, g.src, g.tgt, g.weight))
    return {"g": g, "rank": rank, "data": data, "lanes": lanes, "js": js, "ref": ref,
            "score": score, "stderr": proc.stderr, "npy": npy,
            "sets": {"c3": c3, "left": left, "mbA": mbA, "mbB": mbB, "node0": node0,
                     "p_left": p_left}}


def test_node_h64_checks(real):
    assert real["js"]["failures"] == 0, real["stderr"][-3000:]
    assert real["js"]["h64"] == real["score"](real["rank"]) == real["data"]["edges"]["ff_h64_syn"]
    print(real["stderr"])


def test_built_html_decode(real):
    if not (CONFIG["art_dir"] / "fly_streams.html").exists():
        pytest.skip("fly_streams.html is not built yet")
    assert "SKIP built-HTML decode" not in real["stderr"], "fly_streams.html exists but has no #edges payload"
    for branch in ("fromBase64", "atob"):
        assert f"built-HTML #edges ({branch}): FF {real['js']['h64']}" in real["stderr"]


def test_cases_scores_and_ranks_match_oracle(real):
    for name, (r, out) in real["ref"].items():
        assert out["rank_sha256"] == sha(r), name
        assert out["score"] == real["score"](r), name
        print(name, out["op"], out["score"], out.get("ms"))


def test_case_properties(real):
    rank, ref, s = real["rank"], real["ref"], real["sets"]
    D = CONFIG["shift_delta"]
    assert np.array_equal(ref["case1"][0], rank)                               # 1
    assert ref["case2"][0][s["node0"]] == 1                                    # 2
    r3, out3 = ref["case3"]                                                    # 3
    assert out3["rigid"] and np.all(r3[s["c3"]] - rank[s["c3"]] == D)
    check_shift_props(rank, s["c3"], D, r3)
    r4, out4 = ref["case4"]                                                    # 4
    assert not out4["rigid"]
    check_shift_props(rank, s["left"], D, r4)
    p, nU = shift_p(rank, s["left"])
    b = out4["bounds"]
    assert (b["minP"], b["maxP"], b["nU"]) == (int(p.min()), int(p.max()), nU)
    far, outf = ref["case5_far"]                                               # 5
    assert np.array_equal(far, ref["case5_bound"][0])
    assert outf["clamped"] and outf["saturated"] and outf["delta"] == -int(p.max())
    assert far[s["left"]].max() == s["left"].sum() - 1
    check_shift_props(rank, s["left"], -D, ref["case5"][0])
    out6 = ref["case6"][1]                                                     # 6
    nA, nB = len(s["mbA"]), len(s["mbB"])
    assert out6["interleaved"]
    assert (abs(nA - nB) / max(nA, nB) > CONFIG["swap_size_warn"]) == any("sizes" in w for w in out6["warnings"])
    assert np.array_equal(ref["case7_twice"][0], rank)                         # 7
    assert not np.array_equal(ref["case7_once"][0], rank)


def test_case9_download_npy(real):
    r8, out8 = real["ref"]["case8"]
    raw = real["npy"].read_bytes()
    assert raw[:8] == b"\x93NUMPY\x01\x00"
    hdr_len = int.from_bytes(raw[8:10], "little")
    assert (10 + hdr_len) % 64 == 0 and raw[10 + hdr_len - 1:10 + hdr_len] == b"\n"
    got = np.load(real["npy"])
    assert got.dtype == np.dtype("<i4") and got.shape == (len(r8),)
    assert np.array_equal(np.sort(got), np.arange(len(got)))
    assert np.array_equal(got, r8)
    assert real["score"](got) == out8["score"]
    print("case 9 npy", real["npy"], "oracle", out8["score"])


@pytest.mark.parametrize("grouping,B", CONFIG["moved_layouts"])
def test_moved_layout_parity(real, grouping, B):
    from layering.tools.fly_streams import build_layout, layout_stats, CONFIG as SCFG
    g, data = real["g"], real["data"]
    r3, out3 = real["ref"]["case3"]
    js = out3["layouts"][f"{grouping}|B{B}"]
    ref_L = data["layouts"][grouping][f"B{B}"]
    assert (ref_L["ink_threshold_frac"], ref_L["export_floor_frac"]) == \
        (SCFG["min_bundle_frac"], SCFG["export_bundle_frac"])
    glane = real["lanes"][grouping]
    glist = [x["name"] for x in ref_L["lanes"]]
    src, tgt = g.src.astype(np.int64), g.tgt.astype(np.int64)
    w = g.weight.astype(np.float64)
    Wtot = float(w.sum())
    out_w = np.bincount(src, weights=w, minlength=g.n_nodes)
    L = build_layout(r3, glane, glist, src, tgt, w, B, out_w)
    stats = layout_stats(L, Wtot, SCFG["min_bundle_frac"] * Wtot)

    nz = np.nonzero(L["counts"])[0]
    cells = js["cells"]
    assert [c["lane"] * B + c["bin"] for c in cells] == nz.tolist()
    for key in ("count", "intra_ff", "intra_fb", "sink0"):
        src_key = "counts" if key == "count" else key
        assert [c[key] for c in cells] == L[src_key][nz].tolist(), key
    tot = {k: np.zeros(len(L["counts"])) for k in ("out_ff", "out_fb", "in_ff", "in_fb")}
    for s_, t_, wf, wb in L["bundles"]:
        tot["out_ff"][s_] += wf; tot["out_fb"][s_] += wb
        tot["in_ff"][t_] += wf; tot["in_fb"][t_] += wb
    for key in tot:
        assert [c[key] for c in cells] == tot[key][nz].tolist(), key
    floor = SCFG["export_bundle_frac"] * Wtot
    ref_b = [[s_, t_, wf, wb] for s_, t_, wf, wb in L["bundles"] if wf + wb >= floor]
    assert [[b["s"], b["t"], b["ff"], b["fb"]] for b in js["bundles"]] == ref_b
    assert set(js["stats"]) == set(stats)
    for k, v in stats.items():
        assert abs(js["stats"][k] - v) <= CONFIG["stats_tol"], k
    assert [x["med_rank"] for x in js["lanes"]] == L["med_rank_lane"]
    assert [x["n"] for x in js["lanes"]] == np.bincount(glane, minlength=len(glist)).tolist()
    print(grouping, B, "cells", len(cells), "bundles above floor", len(ref_b),
          "inter FF %", round(float(stats["inter_ff_pct_of_total"]), 4))

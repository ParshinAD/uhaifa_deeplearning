"""Non-vacuity test: inject bugs and confirm the probe's self-check assertions FIRE."""
import sys, numpy as np
from pathlib import Path
ROOT = Path("/Users/abed359/IdeaProjects/university/deeplearning_thesis/nn_backward_connections")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from mfas import io
from mfas.metrics import score_from_positions
import size_collective_moves as S

g = io.load_dataset("mouse")
src = np.asarray(g.src, np.int64); tgt = np.asarray(g.tgt, np.int64); w = np.asarray(g.weight)
rank = S.rank_of(np.load(S.latest_h35_positions("mouse", 42)))

def try_(label, fn):
    try:
        fn(); print(f"  {label:<46} -> NO ASSERTION (vacuous!)")
    except AssertionError as e:
        print(f"  {label:<46} -> AssertionError FIRED: {str(e)[:70]}")

orig_pair = S._pair_gain_curve
orig_topo = S._topo_order_condensation

print("A) S1 FINAL oracle check (verify_every=0 so only the final check can fire)")
def bug_offby(u, v, lo, hi, o, i, r):
    return orig_pair(u, v, lo, hi, o, i, r) + 1e-3          # inflate every gain
S._pair_gain_curve = bug_offby
try_("gain inflated by 1e-3", lambda: S.size_s1(g, rank, src, tgt, w, 123, [], 0))
def bug_shift(u, v, lo, hi, o, i, r):
    return np.roll(orig_pair(u, v, lo, hi, o, i, r), 1)     # off-by-one in the split index
S._pair_gain_curve = bug_shift
try_("gain curve rolled by 1 (off-by-one split)", lambda: S.size_s1(g, rank, src, tgt, w, 123, [], 0))
S._pair_gain_curve = orig_pair
try_("UNMODIFIED (control: must NOT fire)", lambda: S.size_s1(g, rank, src, tgt, w, 123, [], 0))

print("B) S1 PERIODIC in-loop check (verify_every=1)")
S._pair_gain_curve = bug_offby
try_("gain inflated, verify_every=1", lambda: S.size_s1(g, rank, src, tgt, w, 123, [], 1))
S._pair_gain_curve = orig_pair

print("C) S2 oracle check")
def bug_topo(nc, cs, ct, min_loc):
    o = orig_topo(nc, cs, ct, min_loc)
    return o[::-1] if len(o) > 1 else o                     # REVERSE the topo order
S._topo_order_condensation = bug_topo
try_("condensation order reversed", lambda: S.size_s2(g, rank, src, tgt, w, [64], 10))
S._topo_order_condensation = orig_topo
try_("UNMODIFIED (control: must NOT fire)", lambda: S.size_s2(g, rank, src, tgt, w, [64], 10))

print("D) iterated round check")
S._pair_gain_curve = bug_offby
try_("gain inflated inside size_iterated", lambda: S.size_iterated(g, rank, src, tgt, w, 2, 123, 64, 10))
S._pair_gain_curve = orig_pair

print("E) contiguous-interval lemma, adversarial: star hub outside every interval")
rng = np.random.default_rng(7)
for trial in range(200):
    n = int(rng.integers(8, 16))
    e = [(i, j) for i in range(n) for j in range(n) if i != j and rng.random() < .5]
    # add a hub wired to EVERYTHING in both directions (max boundary-crossing pressure)
    e += [(0, k) for k in range(1, n)] + [(k, n - 1) for k in range(n - 1)]
    s_ = np.array([a for a, b in e], np.int64); t_ = np.array([b for a, b in e], np.int64)
    ww = rng.integers(1, 20, s_.size)
    rk = rng.permutation(n).astype(np.int64); sq = np.argsort(rk, kind="stable")
    oa, ia = S.build_adjacency(s_, t_, ww, n)
    base = score_from_positions(rk, s_, t_, ww)
    for ei in np.flatnonzero(rk[s_] > rk[t_]):
        u, v = int(s_[ei]), int(t_[ei]); lo, hi = int(rk[v]), int(rk[u])
        gc = S._pair_gain_curve(u, v, lo, hi, oa, ia, rk); mid = sq[lo+1:hi]
        for r in range(hi - lo):
            ns = np.concatenate((mid[:r], np.array([u, v]), mid[r:]))
            nr = rk.copy(); nr[ns] = np.arange(lo, hi + 1)
            assert abs((score_from_positions(nr, s_, t_, ww) - base) - gc[r]) < 1e-9, (trial, r)
print("  dense+hub adversarial S1 check: PASSED (formula == full-graph oracle delta)")

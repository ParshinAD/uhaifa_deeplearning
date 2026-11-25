"""Method CLOSE (H9): column generation on the cycle packing LP where every round also builds an exact upper bound.

Round = [PDHG on the pool (m_cg2_core kernels), repair of the best pool duals into a feasible fractional FAS
(m_close_core.raise_cycles on H64 backward edges, exact integer check m_dual_check.check -> UB on tau*),
add the cycles that forced the raises (and optionally short-cycle pricing) as columns, prune].
Optional stabilization: repair / pricing on x_mix = alpha x + (1 - alpha) x_center, x_center = best feasible
repaired lengths so far (in-out separation); only cycles violated under the pool duals x are added.
Final: PDHG, repair + check, integer K1 certificate (as m_cg2_run: floor(D y) + repair + refill), verifier.
Usage: m_close_run.py --tag T --state S.npz [options]"""
import argparse
import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import m_close_core as cc
import m_dual_check as chk
import m_dual_core as core
from cert_io import write_cert
from m_cg2_core import hashes, lib, rotate_min, transpose
from layering.io_utils import load_graph_and_champion_order

SCR = Path("/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--state", required=True)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--time-limit", type=float, default=3300)
ap.add_argument("--pdhg-iters", type=int, default=1500)
ap.add_argument("--pdhg-final-iters", type=int, default=2000)
ap.add_argument("--pdhg-tol", type=float, default=0.004)
ap.add_argument("--check-every", type=int, default=100)
ap.add_argument("--omega", type=float, default=1.0)
ap.add_argument("--chunks", type=int, default=8, help="sequential head groups in the raise")
ap.add_argument("--add-thr", type=float, default=1e-3, help="add the raise cycle of b if its deficit > this (fraction of S)")
ap.add_argument("--alpha", type=float, default=1.0, help="x_mix = alpha x + (1 - alpha) x_center; 1 = no stabilization")
ap.add_argument("--center", default="", help="npz with integer X (scale S) of a feasible repair, initial stabilization center")
ap.add_argument("--short-H", type=int, default=0)
ap.add_argument("--short-k", type=int, default=50)
ap.add_argument("--short-floor", type=float, default=1e-3)
ap.add_argument("--margin", type=float, default=1e-4)
ap.add_argument("--prune-eps", type=float, default=0.05)
ap.add_argument("--prune-age", type=int, default=2)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--S", type=int, default=10 ** 6)
ap.add_argument("--D", type=int, default=10000)
ap.add_argument("--no-cert", action="store_true")
args = ap.parse_args()
CONFIG = vars(args)

t0 = time.time()
L = lib()
NT = args.threads
S = args.S
g, _, rank, _ = load_graph_and_champion_order("connectome")
n, m = g.n_nodes, g.n_edges
wi = g.weight.astype(np.int64)
W = int(wi.sum())
w = wi.astype(np.float64)
invw = 1.0 / w
esrc, etgt = g.src.astype(np.int32), g.tgt.astype(np.int32)
G = core.Graph(g.src, g.tgt, n)
inscc = core.scc_internal(g.src, g.tgt, n)
back = rank[g.src] >= rank[g.tgt]
out_dir = SCR / "close"
out_dir.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, "| t", round(time.time() - t0, 1), flush=True)


def subset(cptr, cedge, keep):
    lens = np.diff(cptr)[keep]
    nptr = np.zeros(lens.size + 1, np.int64)
    np.cumsum(lens, out=nptr[1:])
    mask = np.repeat(keep, np.diff(cptr))
    return nptr, np.ascontiguousarray(cedge[mask])


s = np.load(args.state)
cptr, cedge, y, x, age = s["cptr"], s["cedge"], s["y"], s["x"], s["age"]
rnd0 = int(s["round"])
cedge = rotate_min(cedge, cptr)
h = hashes(cptr, cedge)
log("pool", cptr.size - 1, "nnz", cedge.size)

oo = np.lexsort((etgt, esrc))
optr = np.zeros(n + 1, np.int64)
np.cumsum(np.bincount(esrc, minlength=n), out=optr[1:])
oadj, oeid = np.ascontiguousarray(etgt[oo]), oo.astype(np.int32)
io = np.lexsort((esrc, etgt))
iptr = np.zeros(n + 1, np.int64)
np.cumsum(np.bincount(etgt, minlength=n), out=iptr[1:])
iadj, ieid = np.ascontiguousarray(esrc[io]), io.astype(np.int32)

best_ub = dict(UB=None, X=None, name=None)
if args.center:
    Xc = np.load(args.center)["X"].astype(np.int64)
    rc = chk.check(G, wi, Xc, threads=NT)
    best_ub.update(UB=Fraction(rc["UB"]), X=Xc, name=args.center)
    log("center", args.center, "UB", round(rc["UB_float"], 1), "L", rc["L"])


def pdhg(cptr, cedge, y, x, iters, tol):
    K = cptr.size - 1
    eptr, ecol = transpose(cptr, cedge, m)
    colsum = np.zeros(K)
    L.collen(K, cptr, cedge, invw, colsum, NT)
    T = args.omega / colsum
    rs = invw * np.diff(eptr)
    Sd = np.where(rs > 0, 1.0 / np.where(rs > 0, rs, 1) / args.omega, 0.0)
    x = x.copy()
    x[rs == 0] = 0
    y = y.copy()
    ybar = np.minimum.reduceat(w[cedge], cptr[:-1])
    v, ysum, xsum, cong, yp = np.zeros(K), np.zeros(K), np.zeros(m), np.zeros(m), np.zeros(K)
    out = np.zeros(3)
    best = dict(lb=0.0, ub=np.inf, y=None, x=None)
    hist = []
    it = 0
    while it < iters:
        ysum[:] = 0
        xsum[:] = 0
        L.pdhg(K, m, cptr, cedge, eptr, ecol, invw, T, Sd, x, y, v, ysum, xsum, args.check_every, NT)
        it += args.check_every
        res = []
        for yy, xx in ((y, x), (ysum / args.check_every, xsum / args.check_every)):
            L.bounds(K, m, cptr, cedge, eptr, ecol, invw, ybar, xx, yy, cong, yp, out, NT)
            res.append((out[0], out[1]))
            if out[0] > best["lb"]:
                best["lb"], best["y"] = float(out[0]), yp.copy()
            if out[1] < best["ub"]:
                best["ub"], best["x"] = float(out[1]), xx.copy()
        if res[1][1] - res[1][0] < res[0][1] - res[0][0]:
            y, x = ysum / args.check_every, xsum / args.check_every
        gap = 100 * (best["ub"] - best["lb"]) / best["ub"]
        hist.append([it, round(time.time() - t0, 1), round(best["lb"], 1), round(best["ub"], 1)])
        log("  pdhg it", it, "best", round(best["lb"], 1), round(best["ub"], 1), "gap%", round(gap, 4))
        if gap < tol or time.time() - t0 > args.time_limit:
            break
    return y, x, best, hist


def repair_round(xd, tag):
    """Repair x_mix (stabilized pool duals) -> exact UB; returns info and the raise cycles violated under xd."""
    ts = time.time()
    xm = xd
    if args.alpha < 1 and best_ub["X"] is not None:
        xm = args.alpha * xd + (1 - args.alpha) * best_ub["X"].astype(np.float64) / S * w
    X0 = chk.to_int_lengths(xm * invw, S)
    X0[~inscc] = 0
    F = np.flatnonzero(back & inscc & (X0 < S))
    X1, cp, ce, cb = cc.raise_cycles(G, X0, F, S, int(args.add_thr * S), chunks=args.chunks, threads=NT)
    info = dict(sum_wX0=round(float((wi * X0).sum() / S), 1), raise_cost=round(float((wi * (X1 - X0)).sum() / S), 1),
                raised=int((X1 > X0).sum()), raise_cycles=int(cb.size), raise_s=round(time.time() - ts, 1))
    rc = chk.check(G, wi, X1, threads=NT)
    info.update(L=rc["L"], UB=rc["UB"], UB_float=round(rc["UB_float"], 2), check_s=rc["check_s"],
                UB_status="PASS L>=S" if rc["L"] >= S else "FAIL L<S")
    if rc["L"] >= S and (best_ub["UB"] is None or Fraction(rc["UB"]) < best_ub["UB"]):
        best_ub.update(UB=Fraction(rc["UB"]), X=X1, name=tag)
        np.savez(out_dir / ("X_" + args.tag + ".npz"), X=X1, UB=np.array(rc["UB"]))
    ce32 = ce.astype(np.int32)
    tl = np.add.reduceat(xd[ce32] * invw[ce32], cp[:-1]) if cb.size else np.zeros(0)
    keep = tl < 1 - args.margin
    info["raise_cycles_violated_pool_duals"] = int(keep.sum())
    info["raise_cycle_len_quantiles"] = [round(float(q), 4) for q in np.quantile(tl, [0.01, 0.1, 0.5])] if tl.size else None
    cp, ce32 = subset(cp, ce32, keep)
    return info, cp, ce32


def add_new(cp, ce, info):
    global cptr, cedge, h, y, age
    ce = rotate_min(ce, cp)
    nh = hashes(cp, ce)
    _, first = np.unique(nh, return_index=True)
    sel = np.zeros(nh.size, bool)
    sel[first[~np.isin(nh[first], h)]] = True
    cp, ce = subset(cp, ce, sel)
    info["added"] = int(sel.sum())
    info["added_by_len"] = {int(a): int(b) for a, b in zip(*np.unique(np.diff(cp), return_counts=True))}
    cptr = np.concatenate([cptr, cptr[-1] + cp[1:]])
    cedge = np.concatenate([cedge, ce])
    h = np.concatenate([h, nh[sel]])
    y = np.concatenate([y, np.zeros(int(sel.sum()))])
    age = np.concatenate([age, np.zeros(int(sel.sum()), np.int32)])


def short_price(xd):
    ln = np.maximum(xd * invw, 0)
    srcs = np.arange(n, dtype=np.int64)
    max_rec = srcs.size * args.short_k
    rec_ptr, rec_size = np.zeros(max_rec, np.int64), np.zeros(max_rec, np.int32)
    rec_len, rec_hash = np.zeros(max_rec), np.zeros(max_rec, np.uint64)
    edges = np.zeros(max_rec * 6, np.int32)
    st = np.zeros(4, np.int64)
    ts = time.time()
    nrec = L.price_short(n, optr, oadj, oeid, np.ascontiguousarray(ln[oeid]), iptr, iadj, ieid, np.ascontiguousarray(ln[ieid]),
                         esrc, etgt, ln, srcs, srcs.size, args.short_k, args.short_H, 1.0 - args.margin, args.short_floor,
                         max_rec, max_rec * 6, rec_ptr, rec_size, rec_len, rec_hash, edges, st, NT)
    rp, rs_ = rec_ptr[:nrec], rec_size[:nrec].astype(np.int64)
    nptr = np.zeros(nrec + 1, np.int64)
    np.cumsum(rs_, out=nptr[1:])
    ne = edges[np.repeat(rp - nptr[:-1], rs_) + np.arange(nptr[-1])]
    return dict(kind="short", raw=int(nrec), seconds=round(time.time() - ts, 1)), nptr, ne


rounds = []
r = 0
while True:
    tr = time.time()
    final = r >= args.rounds
    iters = args.pdhg_final_iters if final else args.pdhg_iters
    log("round", rnd0 + r, "pool", cptr.size - 1, "PDHG iters", iters, "final" if final else "")
    y, x, best, hist = pdhg(cptr, cedge, y, x, iters, args.pdhg_tol)
    rec = dict(round=rnd0 + r, pool=int(cptr.size - 1), primal=best["lb"], dual_estimate_pool=best["ub"], pdhg_hist=hist)
    log("round", rnd0 + r, "pool LP primal", round(best["lb"], 1), "dual est", round(best["ub"], 1))
    xd = best["x"]
    info, rcp, rce = repair_round(xd, "round" + str(rnd0 + r))
    rec["repair"] = info
    log("round", rnd0 + r, "repair", json.dumps(info))
    if final or time.time() - t0 > args.time_limit or (out_dir / ("stop_" + args.tag)).exists():
        ybest = best["y"]
        add_new(rcp, rce, info)
        rounds.append(rec)
        break
    lc = np.zeros(cptr.size - 1)
    L.collen(cptr.size - 1, cptr, cedge, xd * invw, lc, NT)
    age = np.where((y <= 1e-12) & (lc > 1 + args.prune_eps), age + 1, 0).astype(np.int32)
    keep = age < args.prune_age
    rec["pruned"] = int((~keep).sum())
    cptr, cedge = subset(cptr, cedge, keep)
    y, age, h = y[keep], age[keep], h[keep]
    add_new(rcp, rce, info)
    if args.short_H:
        sinfo, sp_, se_ = short_price(xd)
        add_new(sp_, se_, sinfo)
        rec["short"] = sinfo
        log("round", rnd0 + r, "short", json.dumps(sinfo))
    log("round", rnd0 + r, "pruned", rec["pruned"], "added raise cycles", info["added"])
    rec["seconds"] = round(time.time() - tr, 1)
    rounds.append(rec)
    r += 1

state = out_dir / ("state_" + args.tag + ".npz")
np.savez(state, cptr=cptr, cedge=cedge, y=y, x=x, age=age, round=np.int64(rnd0 + r + 1))
np.savez(out_dir / ("duals_" + args.tag + ".npz"), x=best["x"], dual_estimate_pool=best["ub"])
log("state saved", state)

out = dict(command=" ".join([Path(sys.executable).name] + sys.argv), config=CONFIG,
           git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           rounds=rounds, pool_size=int(cptr.size - 1), pool_lp_primal=best["lb"], pool_lp_dual_estimate=best["ub"],
           state=str(state), UB=str(best_ub["UB"]) if best_ub["UB"] is not None else None,
           UB_float=round(float(best_ub["UB"]), 2) if best_ub["UB"] is not None else None, UB_from=best_ub["name"],
           UB_lengths=str(out_dir / ("X_" + args.tag + ".npz")))
if not args.no_cert:
    D = args.D
    K = ybest.size
    cptr, cedge = cptr[:K + 1], cedge[:cptr[K]]
    yi = np.floor(ybest[:K] * D).astype(np.int64)
    cap = D * wi
    load = np.zeros(m, np.int64)
    L.intload(K, cptr, cedge, yi, load)
    while (load > cap).any():
        over = load > cap
        hit = np.flatnonzero(np.maximum.reduceat(over[cedge].astype(np.int8), cptr[:-1]) > 0)
        hit = hit[yi[hit] > 0]
        for c in hit:
            e = cedge[cptr[c]:cptr[c + 1]]
            yi[c] -= min(yi[c], int((load - cap)[e].max()))
        load[:] = 0
        L.intload(K, cptr, cedge, yi, load)
    lb_round = int(yi.sum()) // D
    lc = np.zeros(K)
    L.collen(K, cptr, cedge, best["x"] * invw, lc, NT)
    order = np.argsort(lc, kind="stable").astype(np.int64)
    res = cap - load
    gain = L.refill(K, order, cptr, cedge, res, yi)
    keep = yi > 0
    pp, pe = subset(cptr, cedge, keep)
    cert = HERE.parent / "certs" / ("c4" + args.tag + ".npz")
    write_cert(cert, graph=g, D=D, cycles=(pp, pe.astype(np.int64)), y=yi[keep], method="m_close",
               notes="H9 column generation with repair-cycle pricing, PDHG; run " + args.tag + "; floor*D + repair + refill")
    lb = int(yi[keep].sum()) // D
    v = subprocess.run([sys.executable, str(HERE / "verify_cert.py"), str(cert)], capture_output=True, text=True, cwd=ROOT)
    status = "PASS" if v.returncode == 0 else "FAIL"
    log("cert", cert.name, "LB", lb, "ceiling", round(100 * (W - lb) / W, 4), "lb before refill", lb_round,
        "refill gain", round(gain / D, 1), "verifier", status)
    out.update(cert=str(cert.relative_to(ROOT)), D=D, cycles_in_cert=int(keep.sum()), LB=lb,
               ceiling_pct=round(100 * (W - lb) / W, 4), lb_before_refill=lb_round, verifier_status=status,
               verifier_stdout=v.stdout[-1500:])
out["runtime_s"] = round(time.time() - t0, 1)
(HERE.parent / "results" / ("m_close_" + args.tag + ".json")).write_text(json.dumps(out, indent=1))
log("done")

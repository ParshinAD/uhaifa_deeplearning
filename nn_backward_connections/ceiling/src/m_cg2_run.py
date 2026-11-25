"""Method CG2: column generation for the cycle packing LP with C kernels (m_cg2_core.py).
One run = rounds of [PDHG on the pool, prune dead columns, price all sources, add new columns],
then a final PDHG, an integer K1 certificate (floor(D y) + repair + greedy refill over the pool),
the verifier, a run JSON and a state file to continue from.
Usage: m_cg2_run.py --tag T (--init | --state S.npz) [options]; see CONFIG / argparse below."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from cert_io import read_cert, write_cert
from m_cg2_core import hashes, lib, rotate_min, transpose
from m_route_core import Router, merge_records
from src.mfas.io import load_dataset

SCR = Path("/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--init", action="store_true", help="start from m_short pool_r6 + pdhg_r6")
ap.add_argument("--route-seed", default="", help="K1 cert whose cycles are added as columns")
ap.add_argument("--state", default="")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--time-limit", type=float, default=3300)
ap.add_argument("--pdhg-iters", type=int, default=1500)
ap.add_argument("--pdhg-final-iters", type=int, default=3000)
ap.add_argument("--pdhg-tol", type=float, default=0.02, help="stop PDHG when gap % below this")
ap.add_argument("--check-every", type=int, default=100)
ap.add_argument("--omega", type=float, default=1.0)
ap.add_argument("--n-sources", type=int, default=0, help="0 = all sources")
ap.add_argument("--pop-limit", type=int, default=0, help="per-source Dijkstra pop budget, 0 = none (exact pricing)")
ap.add_argument("--k", type=int, default=20, help="cycles per source")
ap.add_argument("--max-add", type=int, default=4_000_000)
ap.add_argument("--floor-len", type=float, default=1e-3)
ap.add_argument("--sat-weight", type=float, default=0.0, help="search length += floor_len * sat_weight * (primal load / w)")
ap.add_argument("--resid-phases", default="", help="hop limits for routing cycles in the residual graph of the primal (m_route_core Router)")
ap.add_argument("--resid-scale", type=int, default=10, help="residual capacity units per synapse for the router")
ap.add_argument("--resid-work-limit", type=int, default=3_000_000)
ap.add_argument("--short-H", type=int, default=0, help="short-cycle pricing (m_cg2_core price_short) up to this many edges (4 or 5), 0 = off")
ap.add_argument("--short-k", type=int, default=50)
ap.add_argument("--short-floor", type=float, default=1e-3, help="score = l(C) + short_floor * |C| when choosing the k best per source")
ap.add_argument("--margin", type=float, default=1e-4)
ap.add_argument("--prune-eps", type=float, default=0.05)
ap.add_argument("--prune-age", type=int, default=2)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--D", type=int, default=10000)
ap.add_argument("--no-cert", action="store_true")
args = ap.parse_args()
CONFIG = vars(args)

t0 = time.time()
L = lib()
NT = args.threads
g = load_dataset("connectome")
n, m = g.n_nodes, g.n_edges
wi = g.weight.astype(np.int64)
W = int(wi.sum())
w = wi.astype(np.float64)
invw = 1.0 / w
esrc = g.src.astype(np.int32)
etgt = g.tgt.astype(np.int32)
out_dir = SCR / "cg2"
out_dir.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, "| t", round(time.time() - t0, 1), flush=True)


def csr_from_blocks(blocks):
    lens = np.concatenate([np.full(b.shape[0], b.shape[1], np.int64) for b in blocks])
    cptr = np.zeros(lens.size + 1, np.int64)
    np.cumsum(lens, out=cptr[1:])
    return cptr, np.concatenate([b.ravel() for b in blocks]).astype(np.int32)


def subset(cptr, cedge, keep):
    lens = np.diff(cptr)[keep]
    nptr = np.zeros(lens.size + 1, np.int64)
    np.cumsum(lens, out=nptr[1:])
    mask = np.repeat(keep, np.diff(cptr))
    return nptr, np.ascontiguousarray(cedge[mask])


if args.init:
    z = np.load(SCR / "mshort" / "pool_r6.npz")
    cptr, cedge = csr_from_blocks([z[k] for k in z.files if k.startswith("c")])
    pz = np.load(SCR / "mshort" / "pdhg_r6.npz")
    y = pz["ylast"].astype(np.float64).copy()
    x = pz["x"].astype(np.float64).copy()
    age = np.zeros(y.size, np.int32)
    rnd0 = 0
    hist_all = []
else:
    s = np.load(args.state)
    cptr, cedge, y, x, age = s["cptr"], s["cedge"], s["y"], s["x"], s["age"]
    rnd0 = int(s["round"])
    hist_all = json.loads(str(s["hist"]))
cedge = rotate_min(cedge, cptr)
h = hashes(cptr, cedge)
log("pool", cptr.size - 1, "nnz", cedge.size, "unique hashes", np.unique(h).size)
if args.route_seed:
    _, arr = read_cert(ROOT / args.route_seed)
    rp, re = arr["cyc_ptr"].astype(np.int64), arr["cyc_edges"].astype(np.int32)
    re = rotate_min(re, rp)
    rh = hashes(rp, re)
    _, first = np.unique(rh, return_index=True)
    keep = np.zeros(rh.size, bool)
    keep[first] = True
    keep &= ~np.isin(rh, h)
    rp, re = subset(rp, re, keep)
    cptr = np.concatenate([cptr, cptr[-1] + rp[1:]])
    cedge = np.concatenate([cedge, re])
    h = np.concatenate([h, rh[keep]])
    y = np.concatenate([y, np.zeros(int(keep.sum()))])
    age = np.concatenate([age, np.zeros(int(keep.sum()), np.int32)])
    log("route seed added", int(keep.sum()), "of", rh.size)

# pricing graph (fixed)
oo = np.lexsort((etgt, esrc))
optr = np.zeros(n + 1, np.int64)
np.cumsum(np.bincount(esrc, minlength=n), out=optr[1:])
oadj, oeid = np.ascontiguousarray(etgt[oo]), oo.astype(np.int32)
io = np.lexsort((esrc, etgt))
iptr = np.zeros(n + 1, np.int64)
np.cumsum(np.bincount(etgt, minlength=n), out=iptr[1:])
iadj, ieid = np.ascontiguousarray(esrc[io]), io.astype(np.int32)
cand = np.flatnonzero((np.diff(optr) > 0) & (np.diff(iptr) > 0)).astype(np.int64)


def pdhg(cptr, cedge, y, x, iters, tol):
    K = cptr.size - 1
    eptr, ecol = transpose(cptr, cedge, m)
    colsum = np.zeros(K)
    L.collen(K, cptr, cedge, invw, colsum, NT)
    T = args.omega / colsum
    rs = invw * np.diff(eptr)
    S = np.where(rs > 0, 1.0 / np.where(rs > 0, rs, 1) / args.omega, 0.0)
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
        L.pdhg(K, m, cptr, cedge, eptr, ecol, invw, T, S, x, y, v, ysum, xsum, args.check_every, NT)
        it += args.check_every
        res = []
        for yy, xx in ((y, x), (ysum / args.check_every, xsum / args.check_every)):
            L.bounds(K, m, cptr, cedge, eptr, ecol, invw, ybar, xx, yy, cong, yp, out, NT)
            res.append((out[0], out[1], yp.copy(), xx))
            if out[0] > best["lb"]:
                best["lb"], best["y"] = float(out[0]), yp.copy()
            if out[1] < best["ub"]:
                best["ub"], best["x"] = float(out[1]), xx.copy()
        if res[1][1] - res[1][0] < res[0][1] - res[0][0]:
            y, x = ysum / args.check_every, xsum / args.check_every
        gap = 100 * (best["ub"] - best["lb"]) / best["ub"]
        hist.append([it, round(time.time() - t0, 1), round(best["lb"], 1), round(best["ub"], 1)])
        log("  pdhg it", it, "cur", round(res[0][0], 1), round(res[0][1], 1), "best", round(best["lb"], 1),
            round(best["ub"], 1), "gap%", round(gap, 4))
        if gap < tol or time.time() - t0 > args.time_limit:
            break
    return y, x, best, hist


def price(xd, yprim):
    ln = np.maximum(xd * invw, 0) + args.floor_len
    if args.sat_weight:
        sat = np.bincount(cedge, weights=np.repeat(yprim, np.diff(cptr)), minlength=m) * invw
        ln += args.floor_len * args.sat_weight * np.minimum(sat, 1.0)
    rng = np.random.default_rng(args.seed + rnd0 + r)
    srcs = rng.permutation(cand)
    if args.n_sources:
        srcs = srcs[:args.n_sources]
    max_rec = srcs.size * args.k
    max_edge = max_rec * 24
    rec_ptr, rec_size = np.zeros(max_rec, np.int64), np.zeros(max_rec, np.int32)
    rec_len, rec_hash = np.zeros(max_rec), np.zeros(max_rec, np.uint64)
    edges = np.zeros(max_edge, np.int32)
    st = np.zeros(6, np.int64)
    ts = time.time()
    nrec = L.price(n, optr, oadj, oeid, iptr, iadj, ieid, esrc, ln, np.ascontiguousarray(ln[oeid]), srcs, srcs.size, args.k, 1.0 - args.margin, args.pop_limit,
                   max_rec, max_edge, rec_ptr, rec_size, rec_len, rec_hash, edges, st, NT)
    info = dict(sources=int(st[5]), raw=int(nrec), pops=int(st[0]), relax=int(st[1]), too_long=int(st[3]),
                overflow=int(st[4]), seconds=round(time.time() - ts, 1))
    rp, rs_, rl, rh = rec_ptr[:nrec], rec_size[:nrec], rec_len[:nrec], rec_hash[:nrec]
    _, first = np.unique(rh, return_index=True)
    info["unique"] = int(first.size)
    first = first[~np.isin(rh[first], h)]
    info["new"] = int(first.size)
    first = first[np.argsort(rl[first], kind="stable")][:args.max_add]
    nptr = np.zeros(first.size + 1, np.int64)
    np.cumsum(rs_[first], out=nptr[1:])
    idx = np.repeat(rp[first] - nptr[:-1], rs_[first]) + np.arange(nptr[-1])
    ne = edges[idx]
    tl = np.add.reduceat(xd[ne] * invw[ne], nptr[:-1]) if first.size else np.zeros(0)
    info["violated_true_len"] = int((tl < 1 - args.margin).sum())
    info["added"] = int(first.size)
    info["mean_dual_len"] = round(float(tl.mean()), 4) if tl.size else None
    info["by_len"] = {int(a): int(b) for a, b in zip(*np.unique(rs_[first], return_counts=True))}
    return nptr, ne, rh[first], info


def finish_records(rec_ptr, rec_size, rec_len, rec_hash, edges, nrec, xd, info):
    rp, rs_, rl, rh = rec_ptr[:nrec], rec_size[:nrec], rec_len[:nrec], rec_hash[:nrec]
    _, first = np.unique(rh, return_index=True)
    info["unique"] = int(first.size)
    first = first[~np.isin(rh[first], h)]
    info["new"] = int(first.size)
    first = first[np.argsort(rl[first], kind="stable")][:args.max_add]
    nptr = np.zeros(first.size + 1, np.int64)
    np.cumsum(rs_[first], out=nptr[1:])
    idx = np.repeat(rp[first] - nptr[:-1], rs_[first]) + np.arange(nptr[-1])
    ne = edges[idx]
    tl = np.add.reduceat(xd[ne] * invw[ne], nptr[:-1]) if first.size else np.zeros(0)
    info["violated_true_len"] = int((tl < 1 - args.margin).sum())
    info["added"] = int(first.size)
    info["mean_dual_len"] = round(float(tl.mean()), 4) if tl.size else None
    info["by_len"] = {int(a): int(b) for a, b in zip(*np.unique(rs_[first], return_counts=True))}
    return nptr, ne, rh[first], info


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
    info = dict(kind="short", H=args.short_H, sources=int(srcs.size), raw=int(nrec), work=int(st[0]), candidates=int(st[1]),
                overflow=int(st[3]), seconds=round(time.time() - ts, 1))
    return finish_records(rec_ptr, rec_size, rec_len, rec_hash, edges, nrec, xd, info)


def resid_route(yprim):
    """Cycles of the residual graph of the fractional primal yprim, routed greedily by the method-R kernel."""
    ts = time.time()
    load = np.bincount(cedge, weights=np.repeat(yprim, np.diff(cptr)), minlength=m)
    capr = np.maximum(np.floor(args.resid_scale * (w - load) - 1e-6), 0).astype(np.int64)
    R = Router(g.src, g.tgt, n, capr > 0, capr, None)
    parts, rows = [], []
    for H in [int(a) for a in args.resid_phases.split(",")]:
        comm = np.flatnonzero(capr > 0)
        comm = comm[np.argsort(-capr[comm], kind="stable")]
        rec, st = R.run(comm, H, args.resid_work_limit)
        parts.append(rec)
        rows.append(dict(H=H, cycles=st["records"], routed=round(st["routed"] / args.resid_scale, 1)))
        log("  resid H", H, rows[-1])
    rc, ra, rp, re = merge_records(parts)
    re = rotate_min(re.astype(np.int32), rp)
    rh = hashes(rp, re)
    u, first, inv = np.unique(rh, return_index=True, return_inverse=True)
    amt = np.bincount(inv, weights=ra) / args.resid_scale
    keep = ~np.isin(u, h)
    first, amt = first[keep], amt[keep]
    sel = np.zeros(rh.size, bool)
    sel[first] = True
    order = np.argsort(first)
    nptr, ne = subset(rp, re, sel)
    info = dict(phases=rows, cycles=int(rh.size), unique=int(u.size), new=int(first.size),
                routed=round(float(ra.sum()) / args.resid_scale, 1), seconds=round(time.time() - ts, 1))
    return nptr, ne, rh[np.sort(first)], amt[order], info


rounds = []
r = 0
while True:
    tr = time.time()
    final = r >= args.rounds
    iters = args.pdhg_final_iters if final else args.pdhg_iters
    log("round", rnd0 + r, "pool", cptr.size - 1, "PDHG iters", iters, "final" if final else "")
    y, x, best, hist = pdhg(cptr, cedge, y, x, iters, args.pdhg_tol / (4 if final else 1))
    rec = dict(round=rnd0 + r, pool=int(cptr.size - 1), primal=best["lb"], dual_estimate_pool=best["ub"], pdhg_hist=hist)
    hist_all.append([rnd0 + r, int(cptr.size - 1), best["lb"], best["ub"]])
    log("round", rnd0 + r, "pool LP primal", round(best["lb"], 1), "dual est", round(best["ub"], 1))
    if final or time.time() - t0 > args.time_limit or (out_dir / ("stop_" + args.tag)).exists():
        rounds.append(rec)
        break
    xd = best["x"]
    lc = np.zeros(cptr.size - 1)
    L.collen(cptr.size - 1, cptr, cedge, xd * invw, lc, NT)
    age = np.where((y <= 1e-12) & (lc > 1 + args.prune_eps), age + 1, 0).astype(np.int32)
    keep = age < args.prune_age
    rec["pruned"] = int((~keep).sum())
    new_blocks = []
    if args.resid_phases:
        nptr, ne, nh, ya, info = resid_route(best["y"])
        rec["resid"] = info
        log("round", rnd0 + r, "resid", info)
        new_blocks.append((nptr, ne, nh, ya))
    if args.short_H:
        nptr, ne, nh, info = short_price(xd)
        rec["short"] = info
        log("round", rnd0 + r, "short", info)
        new_blocks.append((nptr, ne, nh, np.zeros(nh.size)))
    if args.k > 0:
        nptr, ne, nh, info = price(xd, best["y"])
        if new_blocks:
            fresh = ~np.isin(nh, np.concatenate([b[2] for b in new_blocks]))
            nptr, ne = subset(nptr, ne, fresh)
            nh = nh[fresh]
            info["added_after_resid_dedup"] = int(fresh.sum())
        rec["price"] = info
        log("round", rnd0 + r, "price", info)
        new_blocks.append((nptr, ne, nh, np.zeros(nh.size)))
    cptr, cedge = subset(cptr, cedge, keep)
    y, age, h = y[keep], age[keep], h[keep]
    log("round", rnd0 + r, "pruned", rec["pruned"])
    for nptr, ne, nh, ya in new_blocks:
        cptr = np.concatenate([cptr, cptr[-1] + nptr[1:]])
        cedge = np.concatenate([cedge, ne])
        h = np.concatenate([h, nh])
        y = np.concatenate([y, ya])
        age = np.concatenate([age, np.zeros(nh.size, np.int32)])
    rec["seconds"] = round(time.time() - tr, 1)
    rounds.append(rec)
    r += 1

xbest = best["x"]
state = out_dir / ("state_" + args.tag + ".npz")
np.savez(state, cptr=cptr, cedge=cedge, y=y, x=x, age=age, round=np.int64(rnd0 + r + 1), hist=np.array(json.dumps(hist_all)))
np.savez(out_dir / ("duals_" + args.tag + ".npz"), x=xbest, dual_estimate_pool=best["ub"])
log("state saved", state)

out = dict(command=" ".join([Path(sys.executable).name] + sys.argv), config=CONFIG,
           git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           rounds=rounds, pool_size=int(cptr.size - 1), pool_lp_primal=best["lb"], pool_lp_dual_estimate=best["ub"],
           state=str(state))
if not args.no_cert:
    D = args.D
    K = cptr.size - 1
    yi = np.floor(best["y"] * D).astype(np.int64)
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
    L.collen(K, cptr, cedge, xbest * invw, lc, NT)
    order = np.argsort(lc, kind="stable").astype(np.int64)
    res = cap - load
    gain = L.refill(K, order, cptr, cedge, res, yi)
    keep = yi > 0
    pp, pe = subset(cptr, cedge, keep)
    cert = HERE.parent / "certs" / ("c3" + args.tag + ".npz")
    write_cert(cert, graph=g, D=D, cycles=(pp, pe.astype(np.int64)), y=yi[keep], method="m_cg2",
               notes="column generation, C pricing all sources, PDHG; run " + args.tag + "; floor*D + repair + refill")
    lb = int(yi[keep].sum()) // D
    v = subprocess.run([sys.executable, str(HERE / "verify_cert.py"), str(cert)], capture_output=True, text=True, cwd=ROOT)
    status = "PASS" if v.returncode == 0 else "FAIL"
    log("cert", cert.name, "LB", lb, "ceiling", round(100 * (W - lb) / W, 4), "lb before refill", lb_round,
        "refill gain", round(gain / D, 1), "verifier", status)
    out.update(cert=str(cert.relative_to(ROOT)), D=D, cycles_in_cert=int(keep.sum()), LB=lb,
               ceiling_pct=round(100 * (W - lb) / W, 4), lb_before_refill=lb_round, verifier_status=status,
               verifier_stdout=v.stdout[-1500:])
out["runtime_s"] = round(time.time() - t0, 1)
(HERE.parent / "results" / ("m_cg2_" + args.tag + ".json")).write_text(json.dumps(out, indent=1))
log("done")

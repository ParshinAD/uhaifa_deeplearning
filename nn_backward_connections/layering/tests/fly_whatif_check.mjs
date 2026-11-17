// Node check of layering/tools/fly_whatif_core.js (the page's what-if code)
// against the Python export. Design: layering/VIZ_V2_DESIGN.md section 8(a).
//
//   node layering/tests/fly_whatif_check.mjs layering/outputs/fly_artifact [moves.json]
//   node layering/tests/fly_whatif_check.mjs --toys toys.json
//
// Without a moves file: decode edges.zlib, assert score(H64) == DATA.edges.ff_h64_syn
// == CONFIG.ffH64, layout / lane matrix / NT parity at H64 for every grouping and B,
// also through the lay()/mxData()/ntData() accessor rule, the base64 + atob paths,
// and the built fly_streams.html #edges payload when the build has inlined it.
// With a moves file ({"npy_out": path, "moves": [{op: "shift", sel, delta, layouts?},
// {op: "swap", A, B}, {op: "reset"}]}, applied in sequence from H64): the same checks
// (log on stderr), then JSON {h64, moves: [{score, rank_sha256, ...}]} on stdout and
// the ranksToNpy bytes of the last move written to npy_out.
// --toys: run shift / swap on small rank vectors only (no data), JSON on stdout.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import {createRequire} from "node:module";
import {fileURLToPath} from "node:url";

const CONFIG = {
  ffH64: 35314407,                 // frozen-oracle H64 score, design section 2.2
  groupings: ["streams", "cell_class", "super_class"],
  bins: [8, 16, 32],
  topTypes: 5,
  statsTol: 1e-9,
  edgesTag: '<script id="edges" type="application/octet-stream">',
};

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WIcore = createRequire(import.meta.url)(path.join(HERE, "..", "tools", "fly_whatif_core.js"));

let failures = 0;
let log = (...a) => console.log(...a);
function check(ok, msg) {
  if (!ok) { failures++; log("FAIL", msg); }
  return ok;
}
function same(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
function sha(R) { return crypto.createHash("sha256").update(Buffer.from(R.buffer, R.byteOffset, R.byteLength)).digest("hex"); }
function maskOf(N, idx) { const m = new Uint8Array(N); for (const i of idx) m[i] = 1; return m; }
function ordOf(R) { const o = new Int32Array(R.length); for (let i = 0; i < R.length; i++) o[R[i]] = i; return o; }
function ms(t0) { return Math.round((performance.now() - t0) * 10) / 10; }

function runToys(file) {
  const out = [];
  for (const c of JSON.parse(fs.readFileSync(file, "utf8"))) {
    const R = Int32Array.from(c.rank), ORD = ordOf(R), N = R.length;
    if (c.op === "shift") {
      const r = WIcore.shift(R, ORD, maskOf(N, c.sel), c.delta);
      out.push({rank: Array.from(r.R), ord_ok: same(Array.from(r.ORD), Array.from(ordOf(r.R))),
                delta: r.delta, clamped: r.clamped, rigid: r.rigid, saturated: r.saturated,
                bounds: WIcore.shiftBounds(R, ORD, maskOf(N, c.sel))});
    } else {
      let r;
      try { r = WIcore.swapGroups(R, ORD, maskOf(N, c.A), maskOf(N, c.B)); }
      catch (e) { out.push({error: e.message}); continue; }
      out.push({rank: Array.from(r.R), ord_ok: same(Array.from(r.ORD), Array.from(ordOf(r.R))),
                earlier: r.earlier, interleaved: r.interleaved, warnings: r.warnings});
    }
  }
  console.log(JSON.stringify(out));
}

function cmpLayout(got, ref, tag) {
  for (const k of ["n_bins", "ink_threshold_frac", "export_floor_frac"])
    check(got[k] === ref[k], `${tag} ${k} ${got[k]} vs ${ref[k]}`);
  check(same(got.lanes, ref.lanes), `${tag} lanes (name/color/n/med_rank)`);
  check(got.cells.length === ref.cells.length, `${tag} cell count ${got.cells.length} vs ${ref.cells.length}`);
  let bad = 0;
  for (let i = 0; i < Math.min(got.cells.length, ref.cells.length); i++)
    if (!same(got.cells[i], ref.cells[i])) { if (!bad) log("  first differing cell", JSON.stringify(got.cells[i]), JSON.stringify(ref.cells[i])); bad++; }
  check(bad === 0, `${tag} ${bad} cells differ`);
  check(same(got.bundles, ref.bundles), `${tag} bundles above the export floor (${got.bundles.length} vs ${ref.bundles.length})`);
  for (const k of Object.keys(ref.stats))
    check(Math.abs(got.stats[k] - ref.stats[k]) <= CONFIG.statsTol, `${tag} stats.${k} ${got.stats[k]} vs ${ref.stats[k]}`);
  check(same(Object.keys(got.stats).sort(), Object.keys(ref.stats).sort()), `${tag} stats keys`);
}

async function decodeFromZlib(z, DATA, nodeOf) {
  const raw = await WIcore.inflate(z);
  return WIcore.decodeEdges(raw, DATA.n, DATA.m, nodeOf, DATA.W_total);
}

async function main() {
  if (process.argv[2] === "--toys") return runToys(process.argv[3]);
  const dir = process.argv[2], movesFile = process.argv[3];
  if (movesFile) log = (...a) => console.error(...a);
  const DATA = JSON.parse(fs.readFileSync(path.join(dir, "artifact_data.json"), "utf8"));
  const b64 = (s, T) => { const u8 = WIcore.b64ToBytes(s); return new T(u8.buffer, u8.byteOffset, u8.byteLength / T.BYTES_PER_ELEMENT); };
  const N = DATA.n, cloud = DATA.cloud;
  const R0 = Int32Array.from(b64(cloud.x_u32, Uint32Array));
  const nodeOf = ordOf(R0);
  const sinkU8 = b64(cloud.sink_u8, Uint8Array), typeU16 = b64(cloud.type_u16, Uint16Array);
  const ntU8 = b64(cloud.nt_u8, Uint8Array);
  const lanes = Object.fromEntries(CONFIG.groupings.map((g) => [g, b64(cloud.groups[g].lane_u8, Uint8Array)]));

  const z = fs.readFileSync(path.join(dir, "edges.zlib"));
  let t0 = performance.now();
  const raw = await WIcore.inflate(z);
  const tInflate = ms(t0);
  t0 = performance.now();
  const E = WIcore.decodeEdges(raw, N, DATA.m, nodeOf, DATA.W_total);
  const tDecode = ms(t0);
  t0 = performance.now();
  const S0 = WIcore.score(R0, E);
  const tScore = ms(t0);
  log("H64 FF", S0, "| edges.ff_h64_syn", DATA.edges.ff_h64_syn, "| inflate ms", tInflate, "decode ms", tDecode, "score ms", tScore);
  check(S0 === DATA.edges.ff_h64_syn && S0 === CONFIG.ffH64, `H64 score ${S0}`);
  check(E.M === DATA.edges.m && E.N === DATA.edges.n, "edges meta n/m");

  // decode guards
  let threw = false;
  try { WIcore.decodeEdges(raw, N, DATA.m, nodeOf, DATA.W_total + 1); } catch (e) { threw = true; }
  check(threw, "decodeEdges must reject a wrong weight total");
  threw = false;
  try { WIcore.decodeEdges(raw.subarray(1), N, DATA.m, nodeOf, DATA.W_total); } catch (e) { threw = true; }
  check(threw, "decodeEdges must reject a wrong payload length");

  // base64 path, both branches (Uint8Array.fromBase64 and the atob fallback)
  const zB64 = z.toString("base64");
  const fromB64 = Uint8Array.fromBase64;
  for (const branch of ["fromBase64", "atob"]) {
    if (branch === "atob") delete Uint8Array.fromBase64;
    try {
      const s = WIcore.score(R0, await decodeFromZlib(WIcore.b64ToBytes(zB64), DATA, nodeOf));
      check(s === S0, `base64 (${branch}) of edges.zlib scores ${s}`);
      log(`base64 (${branch}) of edges.zlib: FF`, s);
    } finally { Uint8Array.fromBase64 = fromB64; }
  }

  // the page's real input: #edges text of the built HTML
  const htmlPath = path.join(dir, "fly_streams.html");
  const html = fs.existsSync(htmlPath) ? fs.readFileSync(htmlPath, "utf8") : "";
  const i0 = html.indexOf(CONFIG.edgesTag);
  const text = i0 < 0 ? "" : html.slice(i0 + CONFIG.edgesTag.length, html.indexOf("</script>", i0)).trim();
  if (!text || text === "__EDGES__") {
    log("SKIP built-HTML decode: fly_streams.html has no #edges payload yet (build step 5 not done)");
  } else {
    for (const branch of ["fromBase64", "atob"]) {
      if (branch === "atob") delete Uint8Array.fromBase64;
      try {
        const s = WIcore.score(R0, await decodeFromZlib(WIcore.b64ToBytes(text), DATA, nodeOf));
        check(s === S0, `built-HTML #edges (${branch}) scores ${s}`);
        log(`built-HTML #edges (${branch}): FF`, s);
      } finally { Uint8Array.fromBase64 = fromB64; }
    }
  }

  // parity at H64, through the accessor rule
  const cfgOf = (g, B) => ({lanes: DATA.layouts[g]["B" + B].lanes, typeNames: DATA.type_names,
                            ink_threshold_frac: DATA.layouts[g]["B" + B].ink_threshold_frac,
                            export_floor_frac: DATA.layouts[g]["B" + B].export_floor_frac,
                            W_total: DATA.W_total, topTypes: CONFIG.topTypes});
  const build = (R, g, B) => WIcore.buildLayout(R, lanes[g], cloud.groups[g].K, B, E, sinkU8, typeU16, cfgOf(g, B));
  const K = DATA.lane_list.length, NNT = DATA.nt_stats.values.length;
  const WIh = {isH64: true, version: 0, cache: {}};
  const WIw = {isH64: false, version: 1, cache: {}};
  for (const g of CONFIG.groupings) for (const B of CONFIG.bins) {
    t0 = performance.now();
    WIw.cache[g + "|B" + B] = build(R0, g, B);
    log(`buildLayout ${g} B${B} (K=${cloud.groups[g].K}) ms`, ms(t0));
  }
  t0 = performance.now();
  WIw.cache.mx = {...DATA.lane_matrix, W_ff_pct: WIcore.laneFF(R0, lanes.streams, K, E, DATA.W_total)};
  const tMx = ms(t0);
  t0 = performance.now();
  WIw.cache.nt = {...DATA.nt_stats, ...WIcore.ntSums(R0, ntU8, lanes.streams, K, NNT, E)};
  log("laneFF ms", tMx, "ntSums ms", ms(t0));
  for (const g of CONFIG.groupings) for (const B of CONFIG.bins) {
    check(WIcore.lay(WIh, DATA, g, B) === DATA.layouts[g]["B" + B], `lay() at H64 must return DATA.layouts.${g}.B${B}`);
    cmpLayout(WIcore.lay(WIw, DATA, g, B), DATA.layouts[g]["B" + B], `${g} B${B}`);
  }
  const mx = WIcore.mxData(WIw, DATA).W_ff_pct, mxRef = DATA.lane_matrix.W_ff_pct;
  let mxErr = 0;
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) mxErr = Math.max(mxErr, Math.abs(mx[i][j] - mxRef[i][j]));
  check(mxErr <= CONFIG.statsTol, `laneFF max abs diff ${mxErr}`);
  const nt = WIcore.ntData(WIw, DATA), ntRef = DATA.nt_stats;
  for (const k of ["w_ff", "w_fb", "lane_out_ff", "lane_out_fb"]) check(same(nt[k], ntRef[k]), `ntSums ${k}`);
  check(WIcore.mxData(WIh, DATA) === DATA.lane_matrix && WIcore.ntData(WIh, DATA) === DATA.nt_stats, "mxData/ntData at H64");
  const WImiss = {isH64: false, version: 2, cache: {}};
  for (const f of [() => WIcore.lay(WImiss, DATA, "streams", 16), () => WIcore.mxData(WImiss, DATA), () => WIcore.ntData(WImiss, DATA)]) {
    threw = false;
    try { f(); } catch (e) { threw = /missing from WI.cache/.test(e.message); }
    check(threw, "accessor must throw on a cache miss when !isH64");
  }
  log(`H64 parity checked: ${CONFIG.groupings.length} groupings x ${CONFIG.bins.length} B, laneFF max diff ${mxErr}, ntSums`);

  if (movesFile) {
    const spec = JSON.parse(fs.readFileSync(movesFile, "utf8"));
    let R = R0, ORD = nodeOf;
    const res = [];
    for (const mv of spec.moves) {
      const o = {op: mv.op};
      if (mv.op === "reset") { R = R0; ORD = nodeOf; }
      else if (mv.op === "shift") {
        const sel = maskOf(N, mv.sel);
        o.bounds = WIcore.shiftBounds(R, ORD, sel);
        t0 = performance.now();
        const r = WIcore.shift(R, ORD, sel, mv.delta);
        o.ms = ms(t0);
        Object.assign(o, {delta: r.delta, clamped: r.clamped, rigid: r.rigid, saturated: r.saturated});
        R = r.R; ORD = r.ORD;
      } else if (mv.op === "swap") {
        t0 = performance.now();
        const r = WIcore.swapGroups(R, ORD, maskOf(N, mv.A), maskOf(N, mv.B));
        o.ms = ms(t0);
        Object.assign(o, {earlier: r.earlier, interleaved: r.interleaved, warnings: r.warnings});
        R = r.R; ORD = r.ORD;
      } else throw new Error(`unknown op ${mv.op}`);
      check(same(Array.from(ORD), Array.from(ordOf(R))), `ORD inconsistent after ${mv.op}`);
      o.score = WIcore.score(R, E);
      o.rank_sha256 = sha(R);
      if (mv.layouts) {
        o.layouts = {};
        for (const [g, B] of mv.layouts) o.layouts[g + "|B" + B] = build(R, g, B);
      }
      res.push(o);
    }
    fs.writeFileSync(spec.npy_out, WIcore.ranksToNpy(R));
    console.log(JSON.stringify({h64: S0, failures, moves: res, npy: spec.npy_out}));
  }
  log(failures ? `FAILED: ${failures} checks` : "all checks passed");
  process.exitCode = failures ? 1 : 0;
}

main().catch((e) => { console.error(e); process.exitCode = 2; });

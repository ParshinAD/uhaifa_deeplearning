// What-if core of the fly streams page: edge decoding, exact FF scoring, order
// moves (rigid shift, group swap), layout recompute and the lay()/mxData()/ntData()
// accessor rule. Pure functions, no DOM. The page inlines this file
// (build_fly_html.py) and layering/tests/fly_whatif_check.mjs requires it, so
// both run the same code. Design: layering/VIZ_V2_DESIGN.md sections 2.4, 4, 5.4, 6.
const WIcore = (() => {
  "use strict";
  let pairFF = null, pairFB = null;     // reused NC*NC buffers of buildLayout

  function b64ToBytes(s) {
    s = s.trim();
    if (typeof Uint8Array.fromBase64 === "function") return Uint8Array.fromBase64(s);
    const bin = atob(s), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return u8;
  }

  async function inflate(u8) {
    const ds = new Blob([u8]).stream().pipeThrough(new DecompressionStream("deflate"));
    return new Uint8Array(await new Response(ds).arrayBuffer());
  }

  // raw: inflated payload (deg u32 | dtgt u32 | w u16, byte planes, LE).
  // Row r of the CSR is the node with H64 rank r; T holds node indices.
  function decodeEdges(raw, N, M, nodeOf, wTotal) {
    if (raw.length !== 4 * N + 6 * M)
      throw new Error(`edge payload has ${raw.length} bytes, expected ${4 * N + 6 * M}`);
    const off = new Int32Array(N + 1);
    for (let r = 0; r < N; r++) {
      const d = (raw[r] | raw[N + r] << 8 | raw[2 * N + r] << 16 | raw[3 * N + r] << 24) >>> 0;
      off[r + 1] = off[r] + d;
    }
    if (off[N] !== M) throw new Error(`edge payload degrees sum to ${off[N]}, expected ${M}`);
    const T = new Int32Array(M), W = new Uint16Array(M);
    const d0 = 4 * N, w0 = 4 * N + 4 * M;
    let wSum = 0;
    for (let r = 0; r < N; r++) {
      let t = 0;
      for (let e = off[r], e1 = off[r + 1]; e < e1; e++) {
        const d = (raw[d0 + e] | raw[d0 + M + e] << 8 | raw[d0 + 2 * M + e] << 16 |
                   raw[d0 + 3 * M + e] << 24) >>> 0;
        if (e > off[r] && d === 0) throw new Error(`edge payload: duplicate target in row ${r}`);
        t = e === off[r] ? d : t + d;
        if (t >= N) throw new Error(`edge payload: target rank ${t} out of range in row ${r}`);
        T[e] = nodeOf[t];
        const w = raw[w0 + e] | raw[w0 + M + e] << 8;
        W[e] = w;
        wSum += w;
      }
    }
    if (wTotal !== undefined && wSum !== wTotal)
      throw new Error(`edge payload weights sum to ${wSum}, expected ${wTotal}`);
    return {off, rowNode: nodeOf, T, W, N, M};
  }

  // Exact FF synapse weight; strict <, as the frozen oracle (arr[tgt] > arr[src]).
  function score(R, E) {
    const {off, rowNode, T, W, N} = E;
    let s = 0;
    for (let r = 0; r < N; r++) {
      const ru = R[rowNode[r]];
      for (let e = off[r], e1 = off[r + 1]; e < e1; e++) if (ru < R[T[e]]) s += W[e];
    }
    return s;
  }

  // p(s) = unselected nodes before s. Rigid for delta in [rigidLo, rigidHi],
  // fully saturated (S all first / all last) at satLo / satHi.
  function shiftBounds(R, ORD, sel) {
    let nS = 0, nU = 0, minP = -1, maxP = -1;
    for (let k = 0; k < ORD.length; k++) {
      if (sel[ORD[k]]) { if (minP < 0) minP = nU; maxP = nU; nS++; } else nU++;
    }
    return {nS, nU, minP, maxP, rigidLo: -minP, rigidHi: nU - maxP, satLo: -maxP, satHi: nU - minP};
  }

  // Rigid shift: delta counts unselected neurons passed (design section 4).
  function shift(R, ORD, sel, delta) {
    const N = ORD.length;
    const S = new Int32Array(N), P = new Int32Array(N), U = new Int32Array(N);
    let nS = 0, nU = 0;
    for (let k = 0; k < N; k++) {
      const v = ORD[k];
      if (sel[v]) { S[nS] = v; P[nS] = nU; nS++; } else U[nU++] = v;
    }
    if (!nS || !nU) throw new Error("shift needs a non-empty selection and a non-empty rest");
    const minP = P[0], maxP = P[nS - 1];
    const d = Math.min(Math.max(delta, -maxP), nU - minP);
    const R2 = new Int32Array(N), ORD2 = new Int32Array(N);
    let k = 0, i = 0;
    for (let j = 0; j <= nU; j++) {
      while (i < nS && Math.min(Math.max(P[i] + d, 0), nU) === j) {
        R2[S[i]] = k; ORD2[k++] = S[i]; i++;
      }
      if (j < nU) { R2[U[j]] = k; ORD2[k++] = U[j]; }
    }
    return {R: R2, ORD: ORD2, delta: d, clamped: d !== delta,
            rigid: d >= -minP && d <= nU - maxP, saturated: d === -maxP || d === nU - minP};
  }

  function medianSorted(a, n) {
    return n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2;
  }

  // Swap two disjoint groups inside the set Z of ranks they occupy: the later
  // group (larger median rank; tie -> B) is written first. Others keep their rank.
  function swapGroups(R, ORD, A, B) {
    const N = ORD.length;
    const Z = new Int32Array(N), nodesA = new Int32Array(N), nodesB = new Int32Array(N);
    const rA = new Int32Array(N), rB = new Int32Array(N);
    let nZ = 0, nA = 0, nB = 0, shared = 0;
    for (let k = 0; k < N; k++) {
      const v = ORD[k];
      if (A[v] && B[v]) { shared++; continue; }
      if (A[v]) { rA[nA] = k; nodesA[nA++] = v; Z[nZ++] = k; }
      else if (B[v]) { rB[nB] = k; nodesB[nB++] = v; Z[nZ++] = k; }
    }
    if (shared) throw new Error(`A and B share ${shared} neurons`);
    if (!nA || !nB) throw new Error("swap needs non-empty A and B");
    const medA = medianSorted(rA, nA), medB = medianSorted(rB, nB);
    const aEarlier = medA <= medB;
    const [first, nF, second, nSec] = aEarlier ? [nodesB, nB, nodesA, nA] : [nodesA, nA, nodesB, nB];
    const interleaved = aEarlier ? rA[nA - 1] > rB[0] : rB[nB - 1] > rA[0];
    const R2 = R.slice(), ORD2 = ORD.slice();
    let z = 0;
    for (let i = 0; i < nF; i++, z++) { R2[first[i]] = Z[z]; ORD2[Z[z]] = first[i]; }
    for (let i = 0; i < nSec; i++, z++) { R2[second[i]] = Z[z]; ORD2[Z[z]] = second[i]; }
    const warnings = [];
    const k = Math.abs(nA - nB);
    if (k / Math.max(nA, nB) > 0.05)
      warnings.push(`sizes differ; ${k} members of the larger group stay in its own region`);
    if (interleaved) warnings.push("A and B are interleaved; the swap also separates them.");
    return {R: R2, ORD: ORD2, earlier: aEarlier ? "A" : "B", medA, medB, interleaved, warnings};
  }

  // NumPy .npy v1.0 of the ranks as '<i4' (design section 5.4).
  function ranksToNpy(R) {
    const N = R.length;
    let hdr = `{'descr': '<i4', 'fortran_order': False, 'shape': (${N},), }`;
    hdr += " ".repeat((64 - (10 + hdr.length + 1) % 64) % 64) + "\n";
    const out = new Uint8Array(10 + hdr.length + 4 * N);
    out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, hdr.length & 255, hdr.length >> 8]);
    for (let i = 0; i < hdr.length; i++) out[10 + i] = hdr.charCodeAt(i);
    const dv = new DataView(out.buffer, 10 + hdr.length);
    for (let i = 0; i < N; i++) dv.setInt32(4 * i, R[i], true);
    return out;
  }

  // JS twin of fly_streams.build_layout + layout_stats + the per-cell part of
  // fly_artifact_export.main. cfg: {lanes: [{name, color, n}], typeNames,
  // ink_threshold_frac, export_floor_frac, W_total, topTypes}.
  function buildLayout(R, laneU8, K, B, E, sinkU8, typeU16, cfg) {
    const {off, rowNode, T, W, N} = E;
    const NC = K * B, Wtot = cfg.W_total;
    const cell = new Int32Array(N), counts = new Int32Array(NC), sink0 = new Int32Array(NC);
    for (let v = 0; v < N; v++) {
      const c = laneU8[v] * B + Math.min(Math.floor(R[v] * B / N), B - 1);
      cell[v] = c; counts[c]++;
      if (sinkU8[v]) sink0[c]++;
    }
    if (!pairFF || pairFF.length < NC * NC) {
      pairFF = new Float64Array(NC * NC); pairFB = new Float64Array(NC * NC);
    } else { pairFF.fill(0, 0, NC * NC); pairFB.fill(0, 0, NC * NC); }
    const intraFF = new Float64Array(NC), intraFB = new Float64Array(NC);
    for (let r = 0; r < N; r++) {
      const u = rowNode[r], ru = R[u], cu = cell[u];
      for (let e = off[r], e1 = off[r + 1]; e < e1; e++) {
        const v = T[e], cv = cell[v], ff = ru < R[v];
        if (cu === cv) { if (ff) intraFF[cu] += W[e]; else intraFB[cu] += W[e]; }
        else if (ff) pairFF[cu * NC + cv] += W[e];
        else pairFB[cu * NC + cv] += W[e];
      }
    }
    const thr = cfg.ink_threshold_frac * Wtot, floor = cfg.export_floor_frac * Wtot;
    const outFF = new Float64Array(NC), outFB = new Float64Array(NC);
    const inFF = new Float64Array(NC), inFB = new Float64Array(NC);
    const bundles = [];
    let ffs = 0, fbs = 0, inkedFF = 0, inkedFB = 0, sameBin = 0, sameLaneFF = 0;
    let nTotal = 0, nInkFF = 0, nInkFB = 0;
    for (let p = 0; p < NC * NC; p++) {
      const wf = pairFF[p], wb = pairFB[p];
      if (wf + wb === 0) continue;
      const s = Math.floor(p / NC), t = p - s * NC;
      nTotal++; ffs += wf; fbs += wb;
      if (wf >= thr) { inkedFF += wf; nInkFF++; }
      if (wb >= thr) { inkedFB += wb; nInkFB++; }
      if (s % B === t % B) sameBin += wf + wb;
      if (Math.floor(s / B) === Math.floor(t / B)) sameLaneFF += wf;
      outFF[s] += wf; outFB[s] += wb; inFF[t] += wf; inFB[t] += wb;
      if (wf + wb >= floor) bundles.push({s, t, ff: wf, fb: wb});
    }
    let iff = 0, ifb = 0, nonEmpty = 0;
    for (let c = 0; c < NC; c++) { iff += intraFF[c]; ifb += intraFB[c]; if (counts[c]) nonEmpty++; }
    const stats = {
      cells_nonempty: nonEmpty,
      intra_cell_pct: 100 * (iff + ifb) / Wtot,
      intra_cell_ff_pct: 100 * iff / Wtot,
      intra_cell_fb_pct: 100 * ifb / Wtot,
      inter_ff_pct_of_total: 100 * ffs / Wtot,
      inter_fb_pct_of_total: 100 * fbs / Wtot,
      same_bin_vertical_pct_of_inter: 100 * sameBin / (ffs + fbs),
      same_lane_pct_of_inter_ff: 100 * sameLaneFF / ffs,
      bundles_total: nTotal,
      bundles_inked_ff: nInkFF,
      bundles_inked_fb: nInkFB,
      inked_ff_pct_of_inter_ff: 100 * inkedFF / ffs,
      inked_fb_pct_of_inter_fb: 100 * inkedFB / fbs,
    };
    // top types: Map insertion in node-index order + stable sort = Counter.most_common
    const maps = new Array(NC);
    for (let v = 0; v < N; v++) {
      const c = cell[v], t = typeU16[v];
      const m = maps[c] || (maps[c] = new Map());
      m.set(t, (m.get(t) || 0) + 1);
    }
    const cells = [];
    for (let c = 0; c < NC; c++) {
      if (!counts[c]) continue;
      const top = [...maps[c]].sort((a, b) => b[1] - a[1]).slice(0, cfg.topTypes)
        .map(([t, n]) => [cfg.typeNames[t], n]);
      cells.push({lane: Math.floor(c / B), bin: c % B, count: counts[c],
                  intra_ff: intraFF[c], intra_fb: intraFB[c], sink0: sink0[c],
                  out_ff: outFF[c], out_fb: outFB[c], in_ff: inFF[c], in_fb: inFB[c],
                  top_types: top});
    }
    // lane medians by walking the order (even size = mean of the two middle ranks)
    const ORD = new Int32Array(N);
    for (let v = 0; v < N; v++) ORD[R[v]] = v;
    const nLane = new Int32Array(K), seen = new Int32Array(K), med = new Float64Array(K);
    for (let v = 0; v < N; v++) nLane[laneU8[v]]++;
    for (let k = 0; k < N; k++) {
      const l = laneU8[ORD[k]], i = seen[l]++, n = nLane[l];
      if (n % 2 ? i === (n - 1) / 2 : i === n / 2 - 1 || i === n / 2) med[l] += n % 2 ? k : k / 2;
    }
    const lanes = cfg.lanes.map((L, i) => ({name: L.name, color: L.color, n: L.n, med_rank: med[i]}));
    return {n_bins: B, ink_threshold_frac: cfg.ink_threshold_frac,
            export_floor_frac: cfg.export_floor_frac, lanes, cells, bundles, stats};
  }

  // Lane x lane FF share in % of total weight (source lane = row).
  function laneFF(R, laneU8, K, E, Wtot) {
    const {off, rowNode, T, W, N} = E;
    const acc = new Float64Array(K * K);
    for (let r = 0; r < N; r++) {
      const u = rowNode[r], ru = R[u], lu = laneU8[u] * K;
      for (let e = off[r], e1 = off[r + 1]; e < e1; e++) if (ru < R[T[e]]) acc[lu + laneU8[T[e]]] += W[e];
    }
    const out = [];
    for (let i = 0; i < K; i++) {
      const row = [];
      for (let j = 0; j < K; j++) row.push(100 * acc[i * K + j] / Wtot);
      out.push(row);
    }
    return out;
  }

  // FF/FB weight by the source neuron's transmitter, overall and per source lane.
  function ntSums(R, ntU8, laneU8, K, NNT, E) {
    const {off, rowNode, T, W, N} = E;
    const ff = new Float64Array(K * NNT), fb = new Float64Array(K * NNT);
    for (let r = 0; r < N; r++) {
      const u = rowNode[r], ru = R[u], c = laneU8[u] * NNT + ntU8[u];
      let sf = 0, sb = 0;
      for (let e = off[r], e1 = off[r + 1]; e < e1; e++) { if (ru < R[T[e]]) sf += W[e]; else sb += W[e]; }
      ff[c] += sf; fb[c] += sb;
    }
    const wFF = new Array(NNT).fill(0), wFB = new Array(NNT).fill(0), laneOutFF = [], laneOutFB = [];
    for (let l = 0; l < K; l++) {
      laneOutFF.push(Array.from(ff.subarray(l * NNT, (l + 1) * NNT)));
      laneOutFB.push(Array.from(fb.subarray(l * NNT, (l + 1) * NNT)));
      for (let t = 0; t < NNT; t++) { wFF[t] += ff[l * NNT + t]; wFB[t] += fb[l * NNT + t]; }
    }
    return {w_ff: wFF, w_fb: wFB, lane_out_ff: laneOutFF, lane_out_fb: laneOutFB};
  }

  // Accessor rule (design section 6): H64 reads the exported numbers; a what-if
  // order reads WI.cache, which setOrder() and the grouping/B handlers fill
  // eagerly before anything reads it. A miss is a wiring bug, so it throws.
  function cached(WI, key, what) {
    const v = WI.cache[key];
    if (v === undefined)
      throw new Error(`what-if ${what} missing from WI.cache["${key}"] (order version ${WI.version}); ` +
                      "fill the cache before reading it");
    return v;
  }
  function lay(WI, DATA, g, B) {
    return WI.isH64 ? DATA.layouts[g]["B" + B] : cached(WI, g + "|B" + B, "layout");
  }
  function mxData(WI, DATA) {
    return WI.isH64 ? DATA.lane_matrix : cached(WI, "mx", "lane matrix");
  }
  function ntData(WI, DATA) {
    return WI.isH64 ? DATA.nt_stats : cached(WI, "nt", "transmitter sums");
  }

  return {b64ToBytes, inflate, decodeEdges, score, shiftBounds, shift, swapGroups,
          buildLayout, laneFF, ntSums, ranksToNpy, lay, mxData, ntData};
})();
if (typeof module === "object") module.exports = WIcore;

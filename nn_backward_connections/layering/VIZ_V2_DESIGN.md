# Fly Streams page v2: dark-theme toggle + what-if moves of neuron groups

Design doc for branch `viz/fly-dark-drag`. Target page: `layering/outputs/fly_artifact/fly_streams.html`,
built from `layering/tools/fly_streams_page.html` by `build_fly_html.py`.
Numbers marked **[measured]** were taken on 2026-11-17 on the MacBook (Apple M4 Pro, Node v26, allen env,
scratch prototypes, not committed). **[est.]** marks estimates.

## 0. Changes after review (review: `VIZ_V2_DESIGN_REVIEW.md`) and owner decisions

| finding | resolution |
|---|---|
| B1 shift was not rigid (key R+Δ counts the group's own slots) | agreed; §4 `shift` redefined by unselected neighbours passed: `newRank(s) = oldRank(s)+Δ` exactly inside the rigid range, compression at the ends beyond it; independent Python reference (`np.insert`) + property tests in §8(b); "mean move" readout and old risk 4 removed |
| M1 stale grid next to a new score | agreed; §4.1 stale state: grid not interactive, card numbers and `thr-cov` replaced by a stale note |
| M2 how recomputed layouts reach `lay()` | agreed; §6 accessor rule, filled on commit, never lazily |
| M3 missed order-dependent text | agreed; rows added to §6 (sidebar bins hint, chem accounting sentence, matrix/chem labels, Regions banner position) |
| m1 swap with unequal sizes; wrong argument (1) | agreed; warning added; argument (1) was wrong (third parties between A and B can flip relative to A/B), replaced |
| m2 decode guards | agreed; §2.4 |
| m3 test the inlined base64 path + atob fallback | agreed; §8(a) |
| m4 moved-order parity for cell_class B=32 | agreed; §8(b) |
| m5 undo through the commit path; reset no-op at H64 | agreed; §3 |
| m6 drop the deferred/coalesced rebuild | agreed; synchronous rebuild, extra machinery only if check (e) fails |
| m7 more hard-coded colours; fate of `lum < 0.22` rule | agreed; §7 #11-14; old rule kept only as fallback for unknown hexes |
| m8 early return for click/pointerout; cancel drag on view/grouping/B/resize | agreed; §5.3 |
| n1 | kept as is (zero plane compresses away) |
| n2, n3, n5 | agreed; §2.3, §2.4, §5.1 |
| n4 | agreed; interleaving check simplified |
| owner Q1 | `edges.zlib` gitignored; final HTML committed once |
| owner Q2 | no mirror-swap op |
| owner Q3 | combine rule approved |
| owner Q4 | download of the moved order added (§5.2, format in §5.4, checked in §8(b)) |
| owner Q5 | republish only after asking the owner |
| owner Q6 | dark overrides as proposed in §7 #1 |

## 1. Scope and non-goals

In scope:
- F1: a 3-state colour-theme toggle (auto / light / dark), persisted, plus fixes for every hard-coded colour
  that breaks in dark mode.
- F2: select neurons (brush, filter, cells), then either **shift a group rigidly** along π or **swap two
  groups A and B**, and see the exact feedforward (FF) score of the new order. All order-dependent displays
  are recomputed or labelled as H64 baseline. Undo and reset.

Non-goals (v2):
- No optimizer, local search or "suggest a better move". The page only evaluates hand-made orders.
- No change to `src/mfas/metrics.py`, `eval/harness.py` or any champion artifact. The browser scorer is a
  re-implementation tested against the oracle; it never replaces it.
- No persistence of moved orders across reloads (a moved order can be downloaded, §5.4).
- No mirror-twin swap op (owner Q2).
- No recompute of statistics the page lacks data for (flow-hierarchy test) or that are documented H64 facts
  (Methods text, Regions medians, NT direction table). These are labelled as H64.
- Lane row order stays fixed at the H64 median order after moves (rows must not jump).
- MICrONS page untouched.

## 2. Data format change

### 2.1 Edge payload (new)
Node relabelling: an edge endpoint is written as its **H64 rank** (`rank[v]` from `io_utils`), so the
payload needs no node-id table. The page maps back with `nodeOf[r]` = inverse of the existing `cloud.x_u32`.

Edges are sorted by `(rank[src], rank[tgt])` (`np.lexsort((rt, rs))`). Asserted in Python [measured]: no
self-loops, no duplicate `(src, tgt)` pairs, weights in `[2, 2405]` (fit uint16).

Uncompressed payload = three sections back to back; every multi-byte value little-endian and **byte-plane
split** (plane b holds byte b of every value, planes in order b = 0, 1, ...):

| section | values | type | planes x length |
|---|---|---|---|
| A `deg` | `deg[r]` = out-degree of the node with H64 rank r, r = 0..N-1 | uint32 | 4 x N bytes |
| D `dtgt` | per edge e in sorted order: `rank[tgt_e]` if e is the first edge of its source run, else `rank[tgt_e] - rank[tgt_{e-1}]` (> 0) | uint32 | 4 x M bytes |
| W `w` | edge weight | uint16 | 2 x M bytes |

Uncompressed 4N + 6M = 34,492,906 bytes; `zlib.compress(buf, 9)` (zlib wrapper = what
`DecompressionStream("deflate")` reads). [measured] 10,552,222 bytes compressed, 14,069,632 bytes base64.
Rejected: varint coding (13.77 MB base64, saves 0.3 MB, more decode code); zlib without byte planes
(17.4 MB base64, breaks the budget).

### 2.2 Where it is produced
Extend `layering/tools/fly_artifact_export.py` (no new script):
- `CONFIG["edges_zlib_level"] = 9`.
- New `edge_payload(g, rank) -> (bytes_compressed, meta)` with a docstring. Builds the three sections,
  asserts the invariants above, returns `meta = {"n", "m", "layout": "deg_u32|dtgt_u32|w_u16, byte planes,
  LE, zlib", "raw_bytes", "zlib_bytes", "raw_sha256", "ff_h64_syn"}`.
- `ff_h64_syn = int(score_from_order(rank, g.src, g.tgt, g.weight))` via the frozen
  `src.mfas.metrics` (read-only). Expected **35,314,407** [measured; = round(0.8425817950936937 x 41,912,141)].
- `main()` writes `outputs/fly_artifact/edges.zlib` and adds `data["edges"] = meta` to `artifact_data.json`.
- `edges.zlib` is added to `.gitignore` (deterministic, regenerable). The final `fly_streams.html` is
  committed once, at the end (owner Q1).

No other new per-node arrays: lanes per grouping (`cloud.groups[g].lane_u8`), side (`side_u8`), type
(`type_u16`), transmitter (`nt_u8`), zero-out-weight flag (`sink_u8`), H64 rank (`x_u32`) already exist.

### 2.3 Inlining (single self-contained file)
`build_fly_html.py` gets two more placeholders, by analogy with `__DATA__`:
- `<script id="edges" type="application/octet-stream">__EDGES__</script>` replaced by `base64(edges.zlib)`
  (base64 has no `</`, safe inline).
- `<script>__WHATIF_CORE__</script>` replaced by the text of a new file `layering/tools/fly_whatif_core.js`
  (pure functions, no DOM). A separate file is justified because the Node test must run **the same code**
  as the page; it ends with `if (typeof module === "object") module.exports = {...};`.
- `CONFIG = {"max_html_bytes": 19 * 2**20}`; assert the output is below it, print the size.
- Assert each placeholder occurs exactly once and `"</script" not in core_js`; replace `__WHATIF_CORE__` and
  `__EDGES__` before `__DATA__` (so data text can never contain a placeholder that gets replaced).

Size budget [measured components]: current HTML 4,820,235 B + edges base64 14,069,632 B + new JS/CSS
~50 KB [est.] = **~18.94 MB = 18.06 MiB**. Artifact limit 20 MiB, build cap 19 MiB.

### 2.4 Decoding in JS (`fly_whatif_core.js`)
1. `b64ToBytes(s)`: `Uint8Array.fromBase64(s)` if present, else `atob` + byte loop.
2. `inflate(u8)`: `new Response(new Blob([u8]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer()`.
3. `decodeEdges(raw, N, M, nodeOf)` reads planes straight into the final arrays: `off` Int32Array(N+1) =
   prefix sums of `deg` (row r = node with H64 rank r), `rowNode = nodeOf`, `T` Int32Array(M) =
   `nodeOf[target H64 rank]` (node-index space), `W` Uint16Array(M). Asserts `raw.length === 4N + 6M`,
   `off[N] === M`, every decoded target rank `< N` (checked in the loop; an out-of-range index would be
   stored silently as 0), `M === DATA.m` and `ΣW === DATA.W_total`. The page additionally checks
   `score(R0) === DATA.edges.ff_h64_syn`. Any failure disables the what-if panel.
4. Then `document.getElementById("edges").textContent = ""` frees the 14 MB string.
5. The core file defines one object `const WIcore = {b64ToBytes, inflate, decodeEdges, score, shiftBounds,
   shift, swapGroups, buildLayout, laneFF, ntSums, ranksToNpy}`; no other globals.

[measured, Node] inflate 105 ms, plane decode 60 ms. Steady memory: T 22.6 MB + W 11.3 MB + off/nodeOf
1.1 MB; transient +34.5 MB inflated buffer +14 MB base64 string.

## 3. State model (JS)

```
WI = {
  ready: false, err: null,            // edges decoded and H64 parity checked
  E: {off, rowNode, T, W, N, M},
  R0: Int32Array,                      // H64 ranks (copy of cloud.x_u32)
  S0: number,                          // H64 FF synapses, must equal DATA.edges.ff_h64_syn
  R: Int32Array, ORD: Int32Array,      // current rank per node, node per rank
  score: number,                       // current FF synapses (integer held in a double)
  version: 0,                          // +1 per commit; keys all caches
  isH64: true,                         // R equals R0 elementwise (checked after each commit, O(N))
  hist: [],                            // pre-commit states {R, score, label}; cap CONFIG.historyMax = 50
  sel: Uint8Array(N), nSel: 0,         // active selection, by node index
  grpA: Uint8Array(N) | null,          // stored groups for swap, by node index
  grpB: Uint8Array(N) | null,
  tool: "explore" | "select" | "move",
  drag: null | {x0, delta, R, ORD, score},
  cache: {},                           // layouts / lane-FF matrix / NT sums for (version, grouping, B)
  stale: false                         // grid shows an older order (during a drag, until the rebuild)
}
```
Selections and groups are node sets: they survive moves, undo, reset, grouping and B changes.
Commit = push `{R, score, label}` of the current state onto `hist` (drop the oldest beyond the cap), replace
`R/ORD/score`, then `setOrder()`: `version++`, recompute `isH64`, clear `cache`, rebuild the current
layout, re-render. Undo = pop, restore `R/score`, rebuild `ORD` in O(N), then the same `setOrder()` (so
undoing back to H64 sets `isH64 = true` and drops stale caches). Reset = a commit of `R0` (undoable); a
no-op when already `isH64`. History memory: 50 x 0.55 MB = 27 MB.
All sums are integers below 2^53 (total weight 41,912,141), so accumulating in doubles is exact.

## 4. Algorithms (all in `fly_whatif_core.js`)

**score(R, E)**: `for r in 0..N-1: ru = R[rowNode[r]]; for e in off[r]..off[r+1]-1: if ru < R[T[e]]: s += W[e]`.
Strict `<`, identical to the oracle's `arr[tgt] > arr[src]`. O(M), [measured] 14.4 ms (COO form 19 ms).
Fast enough to rescore fully on every animation frame, so **no incremental delta and no Web Worker in v2**
(a prototype incident-edge delta was slower as a full scan, 65 ms; a CSR-incident version needs a further
~23 MB in-adjacency). Fallback in risk R2.

**shift(R, ORD, sel, Δ) -> {R', ORD'}** (rigid shift, the core op). Δ counts **unselected neurons passed**:
- U = unselected nodes in current rank order (`U[0..nU-1]`), S = selected nodes in current rank order.
  For s in S, `p(s)` = number of unselected nodes with rank < R[s]; target `t(s) = clamp(p(s)+Δ, 0, nU)`.
- New order: for k = 0..nU, first all s with `t(s) = k` (in S order), then `U[k]` (if k < nU).
- Consequence: `R'[s] = t(s) + (#selected before s)`, so inside the rigid range
  `Δ ∈ [-min p, nU - max p]` **every selected neuron moves by exactly Δ ranks**, spacing inside the group
  is kept, and the group passes exactly |Δ| unselected neurons. Unselected neurons keep their relative order.
- Beyond the rigid range the leading members stop at the end of the axis and the group compresses
  against it (internal order still kept); readout: "compressed at the edge". Fully saturated at
  `Δ = -max p` (all of S first) and `Δ = nU - min p` (all of S last); Δ is clamped to that range.
  For an interleaved hemisphere the rigid range is tiny (min p ≈ 0, max p ≈ nU), so most hemisphere moves
  are compressed moves; the readout says so.
- One walk over `ORD` computes p and splits S/U; one merge builds the result. O(N); [measured] 2.4 ms for
  the merge prototype of the earlier definition, same structure [est. same cost].
- Toy (N = 8, S = ranks 0-3, Δ = +1): new order `[4,0,1,2,3,5,6,7]`. Real data [measured, Python
  prototype]: largest H64 streams B=16 cell (n = 7,698), Δ = +5000: every member moves by exactly +5000.
- Disabled if S or U is empty. Δ = 0 is the identity; moves compose (each applies to the current R).

**swapGroups(R, ORD, A, B) -> {R', ORD'}** (the owner's "swap two groups"; A, B disjoint, both non-empty):
- Z = the sorted set of ranks occupied by A ∪ B. Every other neuron keeps its exact rank.
- "Earlier" group = the one with the smaller median current rank (np.median rule); tie -> A is earlier.
- Z is re-filled with the later group's members first, then the earlier group's, each in its current internal order.
- Python reference: `Z = np.sort(rank[np.r_[A, B]])`; `first, second = (B, A) if med(A) <= med(B) else (A, B)`;
  `seq = np.r_[first[np.argsort(rank[first])], second[np.argsort(rank[second])]]`; `r = rank.copy(); r[seq] = Z`.
- JS: one walk over `ORD` collects Z and both member lists in rank order; then write. O(N).
- Why this definition: (1) every neuron outside A ∪ B keeps its exact rank, so third parties do not move on
  screen (their FF relation to A/B members can still change, which is part of the effect being measured);
  (2) it is defined for unequal sizes; (3) for fully separated groups (all A before all B) applying it
  twice returns the original order.
- Unequal sizes split the larger group: with A = ranks {0,1,2}, B = {10}, the result is B@0, A@{1,2,10}.
  Warning when `||A|-|B|| / max(|A|,|B|) > 0.05`: "sizes differ; k members of the larger group stay in its own
  region" (k = the size difference; only min(|A|,|B|) members cross; review r1). MB: 2,791 vs 2,806, 0.5 %, no warning.
- Interleaved edge case: if `max rank(earlier) > min rank(later)` the groups overlap and the op also
  **de-interleaves** them (all of the later group precedes all of the earlier one inside Z). It is then not
  self-inverse (undo is the way back). Warning: "A and B are interleaved; the swap also separates them."
- Overlapping A and B: the Swap button is disabled with "A and B share k neurons".
- MB left↔right, reference [measured with the frozen oracle on this definition, reproduced by the
  reviewer]: A = mushroom body ∩ left (2,791), B = mushroom body ∩ right (2,806), P(rank_A < rank_B) = 0.839
  (so they partly overlap and the interleaved warning shows), FF 35,314,407 -> **35,208,055**
  (-106,352 syn, -0.2537 pp).

**buildLayout(R, laneU8, K, B, E, sinkU8, typeU16, cfg) -> layout**: the JS twin of
`fly_streams.build_layout` + `layout_stats` + the per-cell part of `fly_artifact_export.main`:
- `bin(v) = min(floor(R[v]*B/N), B-1)` (operands < 2^53, exact), `cell = lane*B + bin`, NC = K*B.
- counts, sink0 (members with `sinkU8 = 1`, an order-independent node flag) per cell.
- One pass over edges: FF iff `R[u] < R[v]`; same cell -> `intra_ff/intra_fb[cell]`; else into dense
  `Float64Array(NC*NC)` `pairFF/pairFB` (largest: cell_class B=32, NC = 1,600, 2 x 20.5 MB, buffers reused).
- all pairs with ff+fb > 0 feed stats and per-cell out/in totals (unthresholded, as in Python); the drawn
  bundle list keeps pairs with `ff+fb >= export_floor_frac * W_total`, matching the export.
- stats: same formulas as `layout_stats` with `thr = ink_threshold_frac * W_total`.
- top_types: per cell a `Map(type -> count)` filled in **node-index order**, stable sort by count desc,
  take 5. Matches `Counter.most_common` tie order (first insertion).
- lanes: name, colour, n unchanged; `med_rank` recomputed by walking `ORD` (even size = mean of the two middle ranks, as `np.median`).

**laneFF(R, streamsLane, K, E)** -> `W_ff_pct[K][K]` (W_pct is order-independent). O(M).
**ntSums(R, ntU8, streamsLane, E)** -> `w_ff, w_fb` per transmitter, `lane_out_ff/fb[K][NNT]` by source. O(M).
Both lazy (on opening Matrix / Chemistry), cached per `version`.

### 4.1 Time budget on drop (commit), never on pointermove
| step | time |
|---|---|
| shift or swap | 2-3 ms [measured] |
| score | 14 ms [measured] |
| buildLayout incl. top types, K=17 B=16 / K=17 B=32 / K=50 B=32 | 53 / 46 / 86 ms [measured, Node, synthetic lanes, so pair counts are an upper bound] |
| SVG rebuild of cells + bundles (`drawStreams`, unchanged code) | 20-60 ms [est.] |
| cloud redraw (136,648 dots) | 20-50 ms [est.] |
| **total** | **~110-210 ms** on the MacBook; ~0.3-0.6 s on a 3x slower laptop [est.] |

That is acceptable for a commit, so cells and bundles are rebuilt **synchronously** after every drop,
Apply, swap, undo or reset (one frame of "updating…" is acceptable). A deferred / coalesced rebuild is added
only if timing check §8(e) exceeds 400 ms.

**Stale state** (`WI.stale = true`): during a drag, and between commit and the end of the rebuild. Then:
`#g-cells/#g-ff/#g-fb` get `opacity:.25; pointer-events:none`; hover isolation and pinning are inactive;
the card shows only "Grid shows the order before this move" (no numbers); `thr-cov` is blank. The score
readout, the header and the dots are the only things that update live. On cancel (Esc) or after the
rebuild, `stale = false` and the normal card returns.

## 5. UI

### 5.1 Theme toggle (F1)
- Header, right end (`margin-left:auto`): `.seg#themeseg` (`aria-label="Colour theme"`), buttons
  `auto | light | dark`, reusing the existing `.seg` style.
- `localStorage` key `fly-theme` (`"auto" | "light" | "dark"`), read/written in `try/catch` exactly like the
  legend toggle; any failure = `auto`.
- `light`/`dark` set `documentElement.dataset.theme`; `auto` restores the value seen at page load
  (`hostTheme`, possibly none; the artifact host may set it) instead of deleting it. Apply the stored choice
  **before** the first `render()`; the existing MutationObserver handles later changes. If the observer sees a
  `data-theme` change not made by the toggle (a flag is set around the toggle's own write), the toggle
  shows `auto`.

### 5.2 What-if panel (F2): new aside block `#ctl-whatif`, Streams view only, at the top of the aside
- Tool: `.seg` `explore | select | move`. `explore` = today's behaviour.
- Selection line: "Selected: 12,345 neurons (L 6,001 / R 6,100 / other 244)" + `Clear`.
- Filter row: lane `<select>` (lanes of the current grouping + "any"), side `<select>` (any / left / right /
  other, from `side_u8`), type input reusing `#typelist`. Fields ANDed. Buttons `Select | Add | Remove`.
- Shift row: numeric `Δ ranks` + `Apply shift` (precise and reproducible; same code path as drag).
- Swap row: `Set A` / `Set B` (copy the current selection; shows count and median rank of each),
  `Swap A ↔ B`, the warnings of §4 (sizes, interleaved, overlap).
- `Undo` (also Ctrl/Cmd+Z when focus is not in a text input), `Reset to H64`, `Download order (.npy)`
  (enabled when `!isH64`, §5.4).
- Readout (monospace, `.card .row` style): `what-if FF 84.1234 %` / `vs H64 -0.1348 pp (-56,512 syn)` /
  `vs previous step ...` / `steps 3` / during drag `dragging Δ = +5,000`, plus `compressed at the edge`
  outside the rigid range and `saturated` at the clamp. Signs are text (+/-), no red/green; crimson stays
  reserved for feedback edges.
- Note under the readout once modified: "What-if order made by hand, not an optimizer result. The score is
  recomputed exactly over all 5,657,719 edges in this page; on load the page checks itself against the
  frozen oracle (H64 = 35,314,407 syn)."
- While `!WI.ready`: "loading edges…". On parity failure or missing `DecompressionStream`: the reason is
  shown, the panel stays disabled, the rest of the page works as today.

Header: when `!isH64`, a new fact `what-if FF <b>x.xxxx</b>% (H64 + n moves)` follows the champion fact,
and `order = H64 champion (pinned, oracle-verified)` becomes `order = what-if (H64 + n moves; not optimized)`.

### 5.3 Interaction
- `explore`: unchanged (hover isolation, click pin, lane labels).
- `select` / `move`: hover isolation, pinning and lane-label clicks off. The existing svg `pointerover`,
  `pointerout` **and `click`** handlers return early unless `tool === "explore"` (a `click` still fires
  after a brush pointerup under pointer capture). Pin cleared, `touch-action:none` on the SVG. If the cloud
  is `off`, switch it to `groups`.
- A pending drag is cancelled (as Esc) on view, grouping or B switch, tool switch and `resize`.
- Combine rule for brush and cell click: no modifier = replace, Shift = add, Alt/Option = remove. Filter
  uses its three buttons. No intersect; the filter's AND covers the common case.
- `select`: pointerdown on the SVG with `setPointerCapture`. Movement < `CONFIG.dragThresholdPx = 4` and the
  target is a `.cell` -> select that cell's neurons (current order, grouping, B). Otherwise a rectangle brush
  (SVG rect, dashed `--accent`); on pointerup select every node whose dot centre (same `pxy` as `drawCloud`,
  using `R`) is inside. Coordinates via `svg.getBoundingClientRect()` because `#scroller` scrolls.
- `move`: pointerdown anywhere in the plot starts a drag of the current selection (no-op if empty).
  `Δ = round((x - x0) * N / G.plotW)`, clamped to the saturation range of §4. With the rigid shift the
  selected dots follow the cursor exactly inside the rigid range. pointermove only stores Δ; one
  `requestAnimationFrame` callback per frame runs `shift` + `score` into `WI.drag`, updates the readout and
  redraws the cloud exactly from `WI.drag.R` (unselected dots in between also move, by up to |S| ranks).
  Grid stale during the drag (§4.1). pointerup with Δ ≠ 0 commits.
- Esc: during a drag cancel it (no commit); else close modals (existing); else in select/move clear the selection.
- Selected dots: drawn like today's type highlight (unselected grey at alpha 0.10, selected in their mode
  colour at 2.6 px); takes precedence over the type highlight while non-empty. While the Swap row is
  hovered, A and B are drawn instead, A in `--accent`, B in `--ink`.

### 5.4 Download of the moved order (owner Q4)
- Button `Download order (.npy)` writes the current **ranks** `R` (`R[i]` = slot of node i; node index i is
  the graph's node order from `src.mfas.io.load_dataset("connectome")`, the same index as every per-node
  array of the page and as `rank` in `io_utils`). This is exactly what `score_from_order` consumes.
- Format: NumPy `.npy` v1.0, built by `WIcore.ranksToNpy(R)`: magic `\x93NUMPY`, bytes `01 00`, uint16 LE
  header length, header `{'descr': '<i4', 'fortran_order': False, 'shape': (136648,), }` padded with spaces
  and a final `\n` so that 10 + header length is a multiple of 64, then the Int32Array bytes (little-endian).
  Delivered with `Blob` + `URL.createObjectURL` + `<a download>` (works from `file://`).
- File name `fly_whatif_ranks_ff<FF synapses>.npy`, e.g. `fly_whatif_ranks_ff35208055.npy`, so the expected
  score travels with the file.
- Re-score with the frozen oracle (run from `nn_backward_connections/`):
```
$PY -c "import sys,numpy as np; sys.path[:0]=['.','src']; from src.mfas.io import load_dataset; from src.mfas.metrics import score_from_order; g=load_dataset('connectome'); r=np.load(sys.argv[1]); assert np.array_equal(np.sort(r),np.arange(g.n_nodes)); print(int(score_from_order(r,g.src,g.tgt,g.weight)))" fly_whatif_ranks_ff35208055.npy
```
  It must print the number in the file name. The file is a what-if order, not a champion: it is never
  written to `results/champions/` and never pinned.

## 6. Order-dependent displays

"Recompute" = from the current `R` whenever `!isH64` (when `isH64`, the exported Python numbers are used,
and the Node test proves both agree). "Baseline" = kept as H64 and labelled.

How recomputed layouts reach the existing code (M2): every streams consumer already goes through `lay()`
(`drawStreams`, `drawBundles` + coverage, `showCellCard`, `updateCardDefault`, `KL()`, `laneCol()` used by
`drawCloud`). `lay()` becomes `WI.isH64 ? DATA.layouts[g]["B"+B] : WI.cache[g+"|B"+B]`. The cache entry is
filled by `setOrder()` on commit and by the grouping/B switch handlers **before** `render()`, never lazily
inside `lay()`. Lane name, colour and n are copied from `DATA` (order-independent); `med_rank` is
recomputed. Direct readers of `DATA.layouts.streams.B16.lanes` (`streamCol`, `drawChem`, `showNtCard`,
`buildRegions`) use only names and colours, except Regions' `med_rank`, which stays H64 under the banner.
Likewise `DATA.lane_matrix` / `DATA.nt_stats` reads go through `mxData()` / `ntData()` with the same rule.

| display | rule |
|---|---|
| dot x positions | recompute from `R` |
| cells: count, area, intra border, hatched-sink status, labels | recompute (`buildLayout`) |
| bundles, widths, FF/FB split | recompute |
| ink coverage readout under the slider | recompute (recomputed stats) |
| cell card: neurons, no-output, top types, intra/out/in, below-threshold | recompute |
| cell card title "ranks r0-r1k" | same formula (equal-count bins of any permutation) |
| default card: hidden intra, inter FF/FB, same-column | recompute |
| default card: hierarchy test ρ=0.263 | baseline, label "hierarchy test (H64 order)" |
| lane sub-label "n · med Xk" | n unchanged, med recompute |
| lane row order | fixed at H64 median order; axis caption says so |
| axis caption "depth = champion order π (H64)" | "depth = what-if order (H64 + n moves); rows keep H64 order" |
| header "champion feedforward 84.26%" | stays (it is the champion); what-if fact added |
| sidebar bins hint "Equal-count contiguous slices of the champion order." (l. 136) | "... of the current what-if order" when `!isH64` |
| lane matrix "% of weight" | order-independent |
| lane matrix "FF share" + tooltip | recompute (`laneFF`) |
| matrix caption (l. 656) and `mx-hint` (l. 658) | append "FF by what-if order (H64 + n moves)" when `!isH64` |
| Chemistry panels A-C, default card, NT card | recompute (`ntSums`); a subtitle line under the first panel title "FF/FB by what-if order (H64 + n moves)" when `!isH64` |
| Chemistry accounting line (l. 781-782) | recompute; compares to the current what-if FF instead of `champion_ff_pct`, and says so; the sentence "the Methods table and the Streams caption derive from these same sums" is replaced by "the Methods table describes H64, not this order" when `!isH64` |
| Methods modal (84.2582 %, 85 %/53 % ink, 11.9 %+2.5 %, medians, ρ, NT table) | baseline; banner at the top of `.mpanel` when `!isH64`: "Everything in this panel describes the H64 champion order, not your what-if order." |
| Regions modal: medians, "last neuron of the whole order" (l. 1006/1040), "the optimizer placed them early" (l. 982), group intros (l. 1071-1075, 1097) | baseline; the same banner as a separate element **above `#regions-body` and outside `#rg-intro`** (which `buildRegions` rewrites), toggled in `openRegions` |
| sink audit (2,447 / 6,651), type counts, spectral y | order-independent |

## 7. Dark-theme colour audit (F1)

WCAG contrast measured against the dark ground `#151a22`. The current `colLift` only lifts colours with
(non-linear) luminance < 0.22, i.e. descending and motor, and lifts both to near-identical grey-blue.

| # | where (template line) | problem | fix |
|---|---|---|---|
| 1 | `colLift` (355) | lobula (plate) 2.25:1, central complex 2.13:1, visual projection 2.89:1 stay below 3:1; descending and motor become `#64748f` / `#5e6a7b` (indistinct). Plain HSL lightening to 3:1 makes central complex `#3764ba` ≈ mushroom body `#3f6ab5` | explicit dark override map keyed by hex: lobula (plate) `#9a7238`, visual projection `#a07ad0`, central complex `#7fa2e6`, descending `#a9b8d9`, motor `#d6d9de`; others unchanged. [measured] min contrast 3.28:1, min pairwise CIELAB ΔE76 13.4 (light theme today: 12.4). Cell/super-class family colours inherit via the same map |
| 2 | `NT_COLORS` (338): cloud, chem, NT card | serotonin `#7a4fa3` 2.89:1 | pass NT colours through `colLift` (serotonin shares the visual-projection hex, gets the same override) |
| 3 | hatch pattern (391) `#c9c4ba`/`#6e6a62`; sink cell stroke `#6e6a62` (423) | light tiles glare on dark ground | CSS vars `--hatch-bg`, `--hatch-ln` (light: current; dark: `#3a404c` / `#8a909c`), read with `css()` |
| 4 | cell number labels (425): `#faf9f5` when f > 0.55 regardless of fill | white on yellow/orange (lamina `#efd077`) unreadable in both themes | pick by luminance of the fill composited over `--bg` at its fill-opacity: `lum > 0.45` -> `#141413`, else `#faf9f5` (the chem bars already use this rule) |
| 5 | cell card title (609) `lane.color` | raw colour | `laneCol(c.lane)` |
| 6 | NT card title (801) `NT_COLORS[k]` | raw colour | `colLift(NT_COLORS[k])` |
| 7 | matrix ramp (640) `mixHex(base, "#2f5f8f", t)` | dark ramp ends in dark blue; high cells barely differ from ground | ramp end from var `--mx-hi` (light `#2f5f8f`, dark `#7fa6d6`); number text by the luminance rule of #4 |
| 8 | matrix FF share (646-648) `#c23b3b`, `#d8d5cd`/`#39404e`, text `#f5f5f2` | hard-coded crimson ignores dark `--fb` | `css('--fb')`, var `--mx-mid`, text by luminance |
| 9 | matrix default text (637) `#e6e4df`/`#141413` | duplicates `--ink` | `css('--ink')` |
| 10 | new what-if marks (brush rect, A/B highlight) | n/a | only `--accent`, `--ink`, `--muted`; never `--fb` |
| 11 | matrix intra-lane box (651) `stroke="#e08a2e"` | fine contrast, but hard-coded | keep the hue; route through `colLift` for consistency |
| 12 | matrix `base` (634) `#1c2230`/`#ffffff` | duplicates `--panel` | `css('--panel')` |
| 13 | FF legend swatch (141) gradient of raw lane hexes | does not match lifted dark lanes | set the gradient from JS with `colLift` of the same three hexes on each render |
| 14 | cell label dark text (425) `#e6e4df` | duplicates `--ink` | covered by #4 (luminance rule uses `css('--ink')` for the light-on-dark case) |

`colLift` after the change: `isDark() ? (DARK_OVERRIDE[h] ?? oldRule(h)) : h`. The old `lum < 0.22` mix stays
only as the fallback for hexes not in the map (today none reach it; it protects future lane colours).

Fine as is: `--*` vars, side colours, lane band shading, modal backdrop, tooltip
shadow, dark `--fb` `#e0625f` (5.06:1).

## 8. Verification plan

New tests live in `layering/tests/`, next to `test_core.py`. `PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`.

(a) **Node: decode, H64 parity, layout parity**: `layering/tests/fly_whatif_check.mjs`,
`node layering/tests/fly_whatif_check.mjs layering/outputs/fly_artifact [moves.json]`. Loads
`artifact_data.json` + `edges.zlib`, `require`s `layering/tools/fly_whatif_core.js` (the page's code),
decodes, asserts `score(R0) === DATA.edges.ff_h64_syn === 35314407`. At H64 asserts `buildLayout` equals
`DATA.layouts[g]["B"+B]` for 3 groupings x 3 B (cells count/sink0/intra/out/in exact, top_types identical,
bundles above the floor identical, stats within 1e-9, med_rank exact), `laneFF` equals
`lane_matrix.W_ff_pct` (1e-9), `ntSums` equals `nt_stats` (exact). The same checks also run through the
`lay()`/`mxData()`/`ntData()` accessor rule (pure function of `isH64` and the cache, kept in the core file).
Real input path: one case extracts the `#edges` text from the **built** `fly_streams.html` and runs
`b64ToBytes -> inflate -> decodeEdges -> score`; it runs twice, once with `Uint8Array.fromBase64` and once
with it deleted, so the `atob` fallback is exercised. Given a moves file, prints JSON
`{h64, moves: [{score, rank_sha256}]}` for (b) and writes `ranksToNpy` output of the last move to a temp file.

(b) **Python reference vs JS path**: `layering/tests/test_fly_whatif.py` (pytest; skips if `node` is not on
PATH or artifact files are missing). Loads the graph via `io_utils` (~11 s), decodes selections from
`artifact_data.json` arrays (identical inputs on both sides), applies moves with **independent** numpy code,
scores with `src.mfas.metrics.score_from_order(new_rank, g.src, g.tgt, g.weight)`, runs (a) via
`subprocess` on the same moves, asserts exact equality of scores and rank vectors (sha256 of int32 bytes).

Python shift reference (insertion form; the JS uses a merge, so the two share only the definition):
```
order = np.argsort(rank, kind="stable"); s_ord = order[sel[order]]; u_ord = order[~sel[order]]
p = np.cumsum(~sel[order])[sel[order]]          # unselected before each selected node
t = np.clip(p + delta, 0, len(u_ord))
new_order = np.insert(u_ord, t, s_ord)          # equal t keep s_ord order, inserted before u_ord[t]
new_rank = np.empty_like(rank); new_rank[new_order] = np.arange(len(rank))
```
plus a pure-Python brute force for toys (build the list by walking U and emitting S members by target).
[measured] merge prototype == `np.insert` reference on 3,000 random toys (N ≤ 11, random S and Δ incl.
out-of-range); toy `[4,0,1,2,3,5,6,7]` reproduced.

Property tests (Python reference and JS, on toys and on case 3/4 orders):
(i) inside the rigid range `new_rank[S] - rank[S] == Δ` for every selected node;
(ii) a contiguous block with Δ = +1 swaps with exactly one neighbour (and Δ = -1 likewise);
(iii) relative order inside U and inside S unchanged; (iv) Δ = 0 identity; (v) at the saturation bounds
S is fully first / fully last, and Δ beyond them gives the same order as the bound.

Cases (scores from the oracle [measured in the Python prototype], recomputed by the test, never hard-coded):
1. shift Δ = 0 of the left hemisphere: 35,314,407 (identity).
2. shift of the node with H64 rank 0 by Δ = +1 (lands at rank 1).
3. shift of the largest H64 streams B=16 cell (n = 7,698) by Δ = +5000: 34,496,085; every member moves by +5000.
4. shift of the whole left hemisphere (`side_u8 == 1`, n = 68,462) by Δ = +5000: 34,357,732 (compressed move).
5. shift of the left hemisphere by Δ = -10^6 (clamped; equals the result at `-max p`). Also Δ = -5000: 34,413,313.
6. **MB left↔right swap** (named case): A = streams lane "mushroom body" ∩ left, B = ∩ right: 35,208,055 (-106,352 syn).
7. swap of two separated groups twice = identity (A = H64 ranks 0-999, B = ranks 5000-5999).
8. composition: case 3, then the right hemisphere by Δ = -2000, then case 6.
9. download: the `.npy` written by `ranksToNpy` for case 8 loads with `np.load`, is a permutation, and the
   frozen oracle scores it to the JS score; also run the §5.4 one-liner on it once by hand.

Layout parity on moved orders: Python `fly_streams.build_layout(new_rank, …)` + `layout_stats` vs JS
`buildLayout` for case 3 at streams B=16 **and** cell_class B=32 (K = 50, NC = 1,600, the largest pair
buffer): counts, intra, sink0, bundle sums, stats.
Run: `$PY -m pytest layering/tests/test_fly_whatif.py -q`.

(c) **Headless screenshots** (flag checked [measured]: `--force-dark-mode` sets `prefers-color-scheme: dark`):
```
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
F=file://$PWD/layering/outputs/fly_artifact/fly_streams.html
"$CH" --headless=new --disable-gpu --window-size=1600,1000 --virtual-time-budget=8000 --screenshot=$SCRATCH/light.png "$F"
"$CH" --headless=new --disable-gpu --force-dark-mode --window-size=1600,1000 --virtual-time-budget=8000 --screenshot=$SCRATCH/dark.png "$F"
```
Inspect both. Toggle states, persistence, drag, brush, swap, undo and Esc are checked by hand in Chrome from
`file://` and reported as manual checks. Contrast: a few lines print every lifted lane/NT colour's contrast
vs `#151a22` and the min pairwise ΔE76; pass = min contrast >= 3.0 and min ΔE >= 12.4.

(d) **Size**: `build_fly_html.py` prints the size and asserts < 19 MiB. Expected ~18.9 MB.

(e) **Drop timing**: in Chrome DevTools, time one commit of case 4 (`performance.now()` around commit,
logged to the console once) and report it next to §4.1.

Also `$PY -m pytest layering/tests -q` to show nothing else broke.

## 9. Risks (ranked)

1. **Size headroom**: ~0.9 MiB below the build cap, ~2 MiB below the artifact limit. Growth of
   `artifact_data.json` eats it. Mitigation: the assert; fallback = deflate the JSON too and decode it the same way.
2. **Slower client machines**: rescore 14 ms on an M4 Pro [measured]; 45-70 ms per frame on a 3-5x
   slower laptop [est.], ~15 fps, usable. Fallback: `shift`+`score` in a Worker created from a Blob URL of
   the core script text (works from `file://`); not planned for v2.
3. **Cloud redraw during drag is unmeasured** (136,648 `fillRect` with style changes). If > ~40 ms, draw only
   the selected dots offset by Δ in pixels during the drag; exact redraw on drop. Measure in step 8 of §11.
4. **Compressed moves of spread-out selections**: for a selection spanning most of the axis (a hemisphere)
   the rigid range is tiny and most drags compress the group at an edge. Correct by definition, but
   possibly surprising; the readout says "compressed at the edge" and the Methods modal gets one sentence.
5. **Hemisphere swap semantics**: per the owner's measurement, L/R are interleaved along π in almost every
   lane (whole brain P(rank_L < rank_R) = 0.513); the visible split is on y, which does not affect the score.
   Only the mushroom body lane is mostly separated (0.839). Swapping overlapping groups also de-interleaves
   them (§4), which the panel states. The mirror-twin swap (`exp_mirror_swap.py`, -2.21 pp) is out of scope (Q2).
6. **Memory**: ~35 MB edges + up to 41 MB pair buffers + 27 MB history + ~50 MB transient at load [est.].
   Fine on laptops; phones are not a target.
7. **Artifact host vs toggle**: handled by showing `auto` on a foreign `data-theme` change (§5.1); untested
   against the real host until the owner approves a republish.
8. `DecompressionStream` needs Chrome 80+, Firefox 113+, Safari 16.4+; older browsers get a disabled panel.

## 10. Owner decisions (recorded 2026-11-17) and what remains open

- Q1: `edges.zlib` gitignored; the final `fly_streams.html` (~19 MB) is committed once, at the end.
- Q2: no mirror-swap op.
- Q3: combine rule replace / Shift add / Alt remove approved.
- Q4: download of the moved order added (§5.4).
- Q5: republish the artifact only after asking the owner.
- Q6: dark overrides as in §7 #1 (architect's call).
- Open: none blocking. If check (e) shows commit > 400 ms on the MacBook, ask before adding the deferred rebuild.

## 11. Implementation steps (each checkable alone)

1. `fly_artifact_export.py`: `edge_payload()` + `data["edges"]`; `.gitignore` entry for `edges.zlib`.
   Check: zlib size 10,552,222 B, `ff_h64_syn = 35314407`.
2. `fly_whatif_core.js` (`WIcore`): `b64ToBytes`, `inflate`, `decodeEdges` with guards, `score`;
   `fly_whatif_check.mjs` H64 parity. Check: prints 35314407.
3. Core: `shift` (with bounds), `swapGroups`, `ranksToNpy`; `test_fly_whatif.py` property tests and cases 1-9.
   Check: pytest green.
4. Core: `buildLayout`, `laneFF`, `ntSums`, accessor rule; parity at H64 (a) and on moved orders, streams
   B=16 and cell_class B=32 (b). Check: pytest green.
5. `build_fly_html.py`: inline core + edges, placeholder asserts, size assert; then the built-HTML decode
   case of (a). Check: size < 19 MiB; page opens from `file://` unchanged.
6. Theme toggle + audit fixes §7 #1-9 and #11-14. Check: screenshots (c), contrast script.
7. Page wiring without selection UI: async load + guards, readout, `Apply shift` on a temporary hard-coded
   selection (removed after the check), `setOrder()` for commit/undo/reset, `lay()` accessor, §6 labels,
   header facts, stale state, synchronous rebuild, download. Check: case 3 by hand matches the pytest value;
   downloaded file passes the §5.4 one-liner; timing (e).
8. Selection: filter, cell click, brush, rendering; handler early returns (click/pointerover/pointerout);
   then the `move` tool with live drag, Esc and cancel-on-switch. Check by hand; time `drawCloud` (risk 3).
9. Swap row (Set A / Set B / Swap, warnings). Check: MB case by hand equals case 6.
10. Docs: one paragraph in the Methods modal (what-if, exactness, rigid shift and compression, swap
    semantics, download), a short section in `layering/README.md` and the "Driving the page" part of
    `FLY_FINDINGS.md`. Final: all tests, screenshots, size; commit the HTML once (Q1); ask the owner before
    any republish (Q5).

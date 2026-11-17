# Review of VIZ_V2_DESIGN.md (dark toggle + what-if moves)

Reviewer: independent architecture review, read-only on the repo. Target code: worktree
`/Users/abed359/IdeaProjects/university/deeplearning_thesis_viz/nn_backward_connections` (branch `viz/fly-dark-drag`).
Template line numbers refer to `layering/tools/fly_streams_page.html`.

## What I re-checked myself

| claim | result | how |
|---|---|---|
| H64 FF = 35,314,407 syn | confirmed | frozen `score_from_order(rank, ...)`, `scratchpad/rev/swap_check.py` |
| oracle takes `rank` (not `order`) | confirmed | `io_utils.load_graph_and_champion_order` docstring + call |
| weights int64 in [2, 2405], 0 self-loops, 0 duplicate (src,tgt) | confirmed | same script |
| MB left<->right swap = 35,208,055 (-106,352), sizes 2,791 / 2,806 | confirmed with the §4 reference definition | same script |
| 4N+6M = 34,492,906; base64(10,552,222 B) = 14,069,632 B; 4,820,235 + 14,069,632 + 50k = 18.06 MiB | arithmetic consistent | by hand; current HTML size from `ls` |
| dark overrides: min contrast 3.28:1, min ΔE76 13.4 (light 12.4), NT min 3.94:1 | confirmed | `scratchpad/rev/col.py` |
| Node 26 has `Uint8Array.fromBase64`, `DecompressionStream`, `Blob`, `Response` | confirmed | `node -e` |
| zlib size 10,552,222 B; all timings (14 ms score, 46-86 ms layout, ...) | not re-measured | |

## BLOCKER

### B1. §4 `shift`: the definition is not a rigid shift. Dense selections get stretched and interleaved
Problem. Key = `R[v] + Δ` for selected nodes counts Δ in old ranks, and those old ranks include the
selection's own slots. A selected node therefore moves by the number of unselected nodes in `(R, R+Δ]`, not by Δ.
Members at the leading edge move by Δ and members at the trailing edge move much less. The group spreads out
and mixes with its neighbours. The owner asked for "rigid shift of the selected set by Δ ranks, internal order
preserved". The doc lists the symptom as risk 4 and adds a "mean move" readout, but does not change the definition.

Evidence.
- Toy: N = 8, S = ranks {0,1,2,3}, Δ = +1. Doc definition gives order `[0,1,2,4,3,5,6,7]`. Only the last
  member moved and node 4 now sits inside the block. A rigid shift gives `[4,0,1,2,3,5,...]`.
- Real data (`scratchpad/rev/shift_check.py`): largest streams B=16 cell (n = 7,698), Δ = +5000. Under the doc
  definition member displacement is min 330 / median 1,151 / max 5,000 ranks, and the cell's rank span grows
  from 8,523 to 12,888. The left MB, Δ = +5000: min 245, median 4,374.
- The tests cannot catch this. The Python reference in §8(b) uses the same `lexsort` key, and the listed
  properties (single neuron lands at R+Δ, Δ = 0 identity) hold under both definitions.

Fix (same O(N) merge, same code size). Count Δ in unselected neighbours passed. Let `p(s)` = number of
unselected nodes before s and `nU = |U|`. Put s in front of unselected item number `clamp(p(s)+Δ, 0, nU)`, and
for equal targets keep S's internal order. Then `newRank(s) = oldRank(s) + Δ` exactly for every selected node
until the edge is reached. Spacing inside the group is kept, and every member passes exactly Δ unselected neurons.
On the same data every member of the 7,698-cell moves by exactly +5000.
Bounds: the shift is rigid for `Δ ∈ [-min p, nU - max p]`. Beyond that range the group compresses against
the end, and it is fully saturated at `-max p` / `nU - min p`. Either clamp to the rigid range or show
"compressed at the edge". Python reference: `key = where(sel, clip(p+Δ,0,nU), rankAmongU)`,
`tb = where(sel, 0, 1)`, `lexsort((tb, key))`.
Side effects of the fix. Dots of the selection follow the cursor exactly (`Δ = round(dx*N/plotW)`). Risk 4 and
the "mean move" readout can go. A Δ ≠ 0 inside the bounds always changes the order, so no empty history steps.
Add property tests: (i) `newRank[S] - oldRank[S] == Δ` inside the rigid bounds; (ii) a contiguous block with
Δ = 1 swaps with exactly one neighbour; (iii) relative order inside U is unchanged.

## MAJOR

### M1. §4.1 / §5.3: stale grid is shown next to a new score without a label
Problem. During a drag ("grid dimmed (stale)") and while a deferred or coalesced rebuild is pending, the cells,
bundles, coverage readout (`thr-cov`, l. 495) and hover/pinned cards (l. 594-616, still live through
`pointerover`, l. 910) show numbers of the previous order. The readout and header already show the new order.
Dimming is not a label. This breaks "nothing may present numbers of one order as describing another".
Fix. While stale: `pointer-events:none` on `#g-cells/#g-ff/#g-fb`, hide the card numbers or replace them with
"grid shows the order before this drag", and blank `thr-cov`. Also consider dropping the deferral and the
`slowGridMs` coalescing: the measured total is about 110-210 ms (see m6).

### M2. §6 / §3: how recomputed layouts reach the existing code is not specified
Problem. Every streams consumer reads `lay()` = `DATA.layouts[g]["B"+B]` (l. 316). That includes
`drawStreams`, `drawBundles` + coverage (l. 487-490), `showCellCard`, `updateCardDefault`, `KL()`, and
`laneCol()` → `lay().lanes[i].color` (l. 357). `drawCloud` calls `laneCol` on every frame (l. 520). The doc
never says that `lay()` returns the what-if layout. If `lay()` builds lazily, the deferred rebuild becomes
synchronous on the first `laneCol` call. If it does not, half the views read H64. `buildRegions` (l. 1081) and
`streamCol`/`drawChem`/`showNtCard` read `DATA.layouts.streams.B16.lanes` directly. That is fine for
names and colours, but `med_rank` in Regions stays H64 and must sit under the banner, which it does.
Fix. One sentence in §6 plus code: `lay() = WI.isH64 ? DATA.layouts[g][B] : WI.cache[g+B]`, where the cache
is filled on commit (or by the rebuild task) and never lazily inside `lay()`. Lane name, colour and n always
come from `DATA` (they do not depend on the order). Test this accessor at H64 with the §8(a) parity.

### M3. §6 table misses order-dependent text
Grep of the template finds these. None is in the table or covered by a banner:
- l. 136 sidebar hint "Equal-count contiguous slices of the champion order." Change it to "of the current
  order" when `!isH64`.
- l. 782 Chemistry accounting: "the Methods table and the Streams caption derive from these same sums". After a
  move this is false, because the Methods NT table is H64 under the banner. Reword when `!isH64`.
- Matrix and Chemistry views get no what-if label of their own. Only the header changes. The matrix caption
  (l. 656), `mx-hint` (l. 658) and the Chemistry title area should say "FF by what-if order (H64 + n moves)".
- Regions modal texts that are facts about H64 ("the very last neuron of the whole order", l. 1006/1040;
  "the optimizer placed them early", l. 982; "median position ... in the optimized order", l. 1071-1075 and
  l. 1097). The doc's Regions banner covers these. Confirm the banner sits above `regions-body`, not only in
  `rg-intro`, because `buildRegions` rewrites `rg-intro` on every call (l. 1084).

## MINOR

m1. §4 `swapGroups`, unequal sizes split the larger group. A = ranks {0,1,2}, B = {10}: the result is B@0,
A@{1,2,10}, so one A member jumps alone into B's old place. The definition is sound and self-inverse for
separated groups, and the owner's MB case (2,791 vs 2,806) is fine. Show a warning when
`||A|-|B|| / max > 0.05`: "sizes differ; k members of the larger group land in the other group's region".
Also, argument (1) under "Why this definition" is misleading. The score depends only on relative order, and a
block move also keeps the relative order of third parties. Drop that argument or reword it.

m2. §2.4 decode guards. `T[e] = nodeOf[rank]` with a corrupt rank ≥ N gives `undefined`, which an Int32Array
silently stores as 0. Add `maxRank < N` in the loop (cheap), `Σ W === DATA.W_total` and `M === DATA.m`
next to the S0 parity check. S0 alone does not catch weights permuted among edges of the same FF class.

m3. §8(a) does not run the page's real input path. The Node test reads `edges.zlib` bytes, so `b64ToBytes`
and the build's inlining are never exercised. Add one case that extracts the `#edges` text from the built
`fly_streams.html` and runs `b64ToBytes → inflate → decodeEdges → score`. Also test the `atob` fallback branch.

m4. §8(b) layout parity on a moved order only covers streams B=16. Add cell_class B=32 (K=50, NC=1,600, the
largest pair buffer and the only one where `pair` indices near 2.56M are exercised).

m5. §3 undo. "Pop and restore" must take the same path as commit (`version++`, recompute `isH64`, clear
the cache, re-render). Otherwise an undo back to H64 keeps `isH64 = false` and stale caches. "Reset" when
already H64 should be a no-op, not a history step.

m6. Scope. The deferred rebuild with the `slowGridMs = 400` threshold, the coalescing and two banner texts is
built for a problem that the measured 110-210 ms does not show. Start with a synchronous rebuild after commit
(one frame of "updating…"). Add the rest only if timing check (e) fails. The "mean move" readout goes away with B1.

m7. §7 audit misses hard-coded colours in the template: l. 651 intra-lane matrix box `stroke="#e08a2e"`;
l. 634 matrix `base` `#1c2230`/`#ffffff` (duplicates `--panel`; use `css('--panel')`); l. 141 FF legend
swatch gradient with raw lane hexes (it will not match lifted dark lane colours); l. 425 `#e6e4df` (= `--ink`,
covered by #4 only implicitly). Also say whether the old `lum < 0.22` rule in `colLift` stays as a fallback
or is replaced by the map. It currently catches only the two overridden colours.

m8. §5.3 + existing handlers. The `svg` click handler (l. 943) pins and unpins cells and opens Regions on a
lane-label click. With `setPointerCapture` on the svg, a `click` still fires after the brush pointerup. The
early return must cover `click`, `pointerover` and `pointerout`, not only hover. Grouping or B switches call
`render()`, which clears `pinned` (l. 826). That is fine, but a pending drag must be cancelled on view,
grouping or B switch and on `resize` (l. 1158).

## NIT

n1. §2.1: `dtgt` values are < N < 2^18, so the 4th byte plane is all zeros. It compresses to nearly nothing,
so leave it, but the "uint32, 4 planes" wording costs nothing to keep.
n2. §2.3: assert that each placeholder appears exactly once, replace `__WHATIF_CORE__` and `__EDGES__` before
`__DATA__`, and assert `"</script" not in core_js`.
n3. §2.4: core functions become globals in the page (`score`, `shift`, ...). Wrap them in one object
(`const WIcore = {...}`) to avoid collisions with page names.
n4. §4 swap: the `P(rank_A < rank_B)` readout is fine, but a simpler "A and B are interleaved" check (max
A rank > min B rank) gives the same warning.
n5. §5.1: risk 7 (host changes `data-theme` after the user's choice). When the MutationObserver sees a change
not made by the toggle, set the toggle to `auto`.

## Answers to the review criteria (short)

1. Algorithms. Shift is wrong for dense sets (B1). Swap is correct and self-inverse for separated groups,
   and splits the larger group for unequal sizes (m1). Re-binning `floor(R*B/N)` is exact in doubles and matches
   `(rank*n_bins)//n`. The bundle, intra, stats and `top_types` tie order match `fly_streams.build_layout` /
   `layout_stats` / `Counter.most_common`. `most_common(n)` uses `heapq.nlargest`, which is stable on ties,
   so insertion order holds.
2. Exactness. Encoding is sound: deltas are > 0 within a run because there are no duplicates, weights fit
   uint16, all sums are < 2^53. Guards need m2.
3. Measurements are plausible. Those I could check reproduce exactly (table above). The Node test runs the same
   core file as the page, but not the inlined-base64 path (m3) or the page wiring (M2).
4. Labelling: gaps in M1 and M3.
5. Interactions: m8, M1.
6. Performance and memory: the numbers look reasonable. Cloud redraw during a drag is unmeasured, as risk 3 says.
   After B1 the unselected dots between the old and new spot also shift (by up to |S| ranks), so a "translate
   only the selected layer" preview is approximate. Keep the exact redraw on drop.
7. Scope: m6 (deferral machinery, mean-move readout). Nothing the owner asked for is missing.
8. Dark theme: m7.

## Verdict

**REVISE**, limited to §4 `shift` (B1: definition, bounds, Python reference, property tests). With B1 fixed and
M1-M3 applied, the rest of the design is APPROVE-WITH-CHANGES: data format, exactness, size budget, swap
and the colour audit are sound.

---

## Re-review (revised design with §0 resolution table)

Scope: new `shift`, M1-M3 resolutions, §5.4 download, regressions. Owner decisions in §0/§10 are not reopened.
Check script: `scratchpad/rev/rereview.py` (plus `shift_check.py` from the first round).

### What I re-ran

| check | result |
|---|---|
| §8(b) `np.insert` reference vs my own merge (written independently for B1) on 3,000 random toys (N ≤ 11, Δ in [-15, 15], incl. beyond saturation) | identical on all |
| toy N=8, S = ranks 0-3, Δ=+1 | `[4,0,1,2,3,5,6,7]` reproduced |
| case 3 (largest streams B16 cell, Δ=+5000) | 34,496,085 by the frozen oracle; every member moves by exactly +5000 |
| case 4 (left hemisphere +5000) / case 5 (-5000) | 34,357,732 / 34,413,313, as stated in the doc |
| case 5 clamp Δ=-10^6 == result at `-max p`, S fully first | true; left hemisphere `min p = 4`, `max p = 68,184`, `nU = 68,186`, so the rigid range is [-4, +2] (the doc says "tiny", correct) |
| §5.4 `.npy` format: I wrote a file by hand from the §5.4 byte spec (not `np.save`), case-3 order | `np.load` reads it; the §5.4 one-liner run verbatim from `nn_backward_connections/` prints 34,496,085 = the expected value |
| one-liner vs signatures | `load_dataset('connectome')` → `g.n_nodes/src/tgt/weight`; `score_from_order(order, src, tgt, weights)` treats its first argument as positions (`arr[tgt] > arr[src]`), so passing ranks is correct, same as `io_utils` |

### Findings on the focus points

1. **Shift.** Correct. `cumsum(~sel)` is inclusive, but it is evaluated at selected slots, so it counts the
   unselected nodes strictly before. `np.insert` breaks ties stably (and `t` is non-decreasing anyway).
   The saturation bounds `[-max p, nU - min p]` are tight, and Δ beyond them returns the bound's order.
   Beyond the rigid range the members whose target hits 0 or nU pile up at the edge in their internal order.
   That is the stated "compressed" behaviour. **Independence:** the Python reference shares only the
   definition with the JS (insert vs merge), and the property tests (i)-(v) check the definition itself, so a
   wrong definition would now be caught. Sufficient.
2. **M1** resolved. With a synchronous rebuild, stale only happens during a drag. **M2** resolved (accessor
   rule, filled eagerly). **M3** resolved (all four missed items are in §6, and the Regions banner sits outside
   `#rg-intro`).
3. **Download** is correct, see the table above.
4. Newly introduced or remaining issues (all MINOR):
   - r1 (§4 swap warning text is inverted). With unequal sizes, `||A|-|B||` members of the **larger group stay
     in their own region**. The doc says they "land in the other group's region". Only `min(|A|,|B|)` of them
     cross. Toy: A={0,1,2}, B={10} gives A@{1,2,10}, so 2 = |A|-|B| A members stay and 1 crosses. Fix the
     sentence: "k members of the larger group stay in its own region".
   - r2 (§6 accessor). The `binseg`/`groupseg` handlers read `lay().ink_threshold_frac` and `syncThr()`
     reads `lay().export_floor_frac` before `render()` (template l. 850-853, 904-907). The cache entry must
     carry `n_bins`, `ink_threshold_frac` and `export_floor_frac`, and must be filled before those reads, not
     only before `render()`. Make `lay()` throw a clear error on a cache miss when `!isH64`, rather than
     returning `undefined`.
   - r3 (§5.3 drag lifecycle). Also cancel on `pointercancel` / `lostpointercapture`. pointerup with Δ = 0 must
     clear `stale`, the same as Esc. Otherwise the grid stays dimmed and non-interactive.
   - r4 (NIT). §4.1 "stale ... between commit and the end of the rebuild" is unobservable with a synchronous
     rebuild unless a frame is yielded for "updating…". Fine either way, but say which.

### Verdict

**APPROVE-WITH-CHANGES.** No remaining BLOCKER or MAJOR. B1 and M1-M3 are resolved, and the new definition,
the case numbers and the download path reproduce exactly against the frozen oracle. Apply r1-r3 during
implementation (wording plus two small guards).

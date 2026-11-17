# Progress / resume point: fly viz v2 (dark theme + drag/swap what-if)

Worktree: /Users/abed359/IdeaProjects/university/deeplearning_thesis_viz/nn_backward_connections
Branch: viz/fly-dark-drag (from layering/v1 @ 0139df7). Nothing committed yet.
Why a new session: the project hook .claude/hooks/protect-frozen-files.sh only allows Edit/Write
inside the session's project dir; the worktree must be the session root. Do NOT bypass the hook.

## Files to copy into the worktree first (from this scratchpad dir)
- VIZ_V2_DESIGN.md        -> layering/VIZ_V2_DESIGN.md   (architect's design)
- VIZ_V2_DESIGN_REVIEW.md -> keep as layering/VIZ_V2_DESIGN_REVIEW.md (architecture review)
- this HANDOFF.md         -> layering/VIZ_V2_PROGRESS.md (resume point; keep updated)

## Process the owner asked for (each stage by subagents)
architect -> architecture reviewer -> (back to architect if needed) -> implementation -> final review.
Status: architect DONE; architecture review: see "Review status" below.

## Owner decisions (2026-11-17)
- Exact score via full edge list in the page: YES.
- Selection: brush on cloud + filter (lane/group + hemisphere + cell type) + click/shift-click cells.
  Combine rule: plain = replace, Shift = add, Alt = remove (no intersect): YES.
- Move = rigid shift by delta ranks (internal order kept). Plus explicit swap of two groups
  (motivating case: mushroom body left vs right, AUC 0.839, medians 100,930 vs 104,101;
  everywhere else L/R are interleaved along pi, AUC ~0.5).
- After a move: score + deltas, dots move, full cells/bundles recompute on drop (if fast), undo/reset.
- Q1 git: do not commit the edge binary (gitignore edges.zlib); commit the final ~19 MB HTML once.
- Q2 mirror-swap op: NOT now.
- Q4 download of the moved order (for re-scoring with the frozen oracle in Python): ADD it.
- Q5 republish the artifact (slug 1d49a8d0-8a16-440e-89fc-b7cb067e7027): ASK the owner after the build.
- Q6 dark-theme colours: implementer/reviewer decide; show light+dark screenshots to the owner.
- Commits: per global rules, stage + propose message; commit only when the owner says so.

## Key measured numbers (architect)
- Oracle FF for H64 = 35,314,407 syn (84.25817950936937 %).
- Full JS rescore of 5,657,719 edges: 14.4 ms (Node, M4 Pro).
- Edge payload zlib9 byte-planes: 10,552,222 B; base64 14,069,632 B; HTML est. ~18.9 MB.
- MB left<->right swap reference (oracle): 35,208,055 syn (-106,352, -0.2537 pp).

## Review status
- Round 1: REVISE (B1: key rank+delta was not a rigid shift; M1-M3 stale/labelling/lay() wiring).
- Architect revised the doc in place (§0 maps every finding -> resolution; all accepted).
- Round 2 (re-review): APPROVE-WITH-CHANGES, no BLOCKER/MAJOR left. Oracle-verified references:
  largest B16 cell +5000 = 34,496,085; left hemisphere +5000 = 34,357,732; -5000 = 34,413,313;
  MB L<->R swap = 35,208,055. Left-hemisphere rigid range is only [-4, +2] -> "compressed at edge" is real.
- Remaining minors to apply DURING implementation (see "Re-review" section of the review file):
  r1 swap unequal-size warning text is inverted (k = ||A|-|B|| members of the larger group STAY in their region);
  r2 lay() cache entry must carry n_bins / ink_threshold_frac / export_floor_frac, be filled before the
     B/grouping handlers and syncThr() read it; a cache miss when !isH64 must throw, not return undefined;
  r3 cancel drag on pointercancel / lostpointercapture; pointerup with delta 0 must clear `stale`;
  r4 say whether a frame is yielded before the synchronous grid rebuild (stale state visibility).
- Reviewer check scripts: scratchpad/rev/ (shift_check.py, swap_check.py, col.py, rereview.py).

## Next step
Implementation by subagent(s) following the design's implementation steps, applying r1-r4;
then an independent final review (code + tests + light/dark screenshots + HTML size), fix, re-review.
Then show screenshots to the owner and ASK before republishing the artifact. Stage, propose commit message.

## Log
- 2026-11-17 (session 2, root = worktree): design, review, progress copied into layering/.
  Implementation plan: phase 1 in parallel: agent A = design §11 steps 1-4 (export, core JS, tests; no template/build),
  agent B = step 6 (theme toggle + §7 colour audit, template only). Phase 2: agent C = steps 5, 7-10 (build, page wiring,
  selection, swap, docs). Then independent final review.
- 2026-11-17: agent B (step 6) DONE: only tools/fly_streams_page.html changed. Toggle in header (localStorage "fly-theme",
  foreign data-theme change -> shows auto), §7 #1-9, #11-14 applied. Contrast min 3.28, ΔE76 min 13.4 dark / 12.4 light (pass).
  Screenshots /private/tmp/claude-501/fly_v2_theme/*.png. Not checked: real Chrome clicks, live OS theme switch, real artifact host.
  Waiting for agent A (steps 1-4).
- 2026-11-17: agent A (steps 1-4) DONE. export -> edges.zlib 10,552,222 B (byte-identical to prototype), ff_h64_syn 35314407;
  new tools/fly_whatif_core.js, tests/fly_whatif_check.mjs, tests/test_fly_whatif.py. pytest layering/tests: 21 passed, 1 skipped
  (built-HTML decode, waits for step 5). Cases 3/4/5/6 reproduce oracle refs; case 8 = 33,785,637; .npy one-liner OK.
  r1 applied in code and in design §4 wording; r2 accessor in core (cache fill before binseg/groupseg/syncThr = page, step 7).
  Moves-file format: {"npy_out","moves":[{op:shift,sel,delta},{op:swap,A,B},{op:reset}]}.
  Next: agent C = steps 5, 7-10.
- 2026-11-17: owner (going to sleep): after the final review, rebuild the page, commit and push the branch. Work autonomously.
  Artifact republish (Q5) still waits for the owner's explicit OK (not covered by "rebuild").
- 2026-11-17: agent C (steps 5, 7-10) DONE, nothing committed or staged. build_fly_html.py inlines core + edges
  (18,929,487 B = 18.05 MiB < 19 MiB cap); built-HTML decode test now runs (22 passed). Page wiring, selection
  (filter / cell click / brush, replace-Shift-Alt), move tool with live drag, swap row, download, §6 labels, Methods
  paragraph, README section, "Driving the page" in FLY_FINDINGS.md and .ru.md. r2-r4 applied (r4: no frame is yielded
  before the synchronous rebuild, said in a code comment and in the Methods text).
  Headless Chrome via CDP (scratch /private/tmp/claude-501/fly_v2_page/drive.mjs): case 3 = 34,496,085, MB swap =
  35,208,055, case 4 = 34,357,732, undo/reset/Ctrl+Z = 35,314,407; downloaded .npy re-scored by the oracle = 34,496,085.
  Commit timing 84-116 ms (case 4: 99-103 ms); drawCloud 37-39 ms during drag (< 40 ms, no fallback).
  Not checked: real mouse/trackpad by hand, a real artifact host, slower machines.
  Next: independent final review, then show screenshots to the owner, ask before republish, propose commit.
- 2026-11-17: agent C (steps 5, 7-10) DONE. HTML 18,929,487 B (18.05 MiB). pytest 22 passed; page via CDP: case3 34,496,085,
  MB swap 35,208,055, case4 34,357,732, undo/reset 35,314,407; commit 84-116 ms; drawCloud 37-39 ms headless (near 40 ms trigger).
  Screenshots /private/tmp/claude-501/fly_v2_page/. Final review launched: code reviewer + UI/docs reviewer (fresh context).
- 2026-11-17: UI/docs review: PASS-WITH-FIXES, no wrong numbers. Accepted to fix: 1 matrix caption clipped (MAJOR),
  2 header wrap, 3 readout wraps, 4 Methods jargon, 5 EN sentence, 6 RU wording, 7 commit VIZ_V2_DESIGN.md with HTML,
  9 Methods scrollTop, 10 aside wraps (cheap ones), 11 readout at H64, 12 Methods phrase. Not fixing 8 (italics follow the
  file's existing style for UI names). Waiting for code review to batch fixes.
- 2026-11-17: code review: PASS-WITH-FIXES, no BLOCKER/MAJOR. Accepted: 1 three weak test asserts (satHi, 0.05 warn threshold,
  .npy 64-byte padding), 3 built-HTML test must fail not skip once build exists, 4 size check before write + assert no
  placeholders in core, 5 tool-* class off in Matrix/Chemistry, 7 declare WI.nA/nB, console.info once. Not taken: 6 (design
  only asks to show auto), 7 shiftBounds unused R / lambdas / {size:,} (cosmetic). 2 drawCloud 60-65 ms headless --disable-gpu
  (agent C measured 37-39): conflicting software-render numbers; fallback not added, flagged for owner to check in real Chrome.
  Next: one fixer agent for both reviews' accepted items, then re-review.
- 2026-11-17: fixer agent applied both reviews' accepted items, nothing committed or staged. Page: matrix caption on two
  lines (also fixes the FF-share base caption overflow), header facts shortened (what-if fact without the move tag, no
  "(dragging)" suffix) + #themeseg flex:none, header 47.6 px before and after moves; readout "vs prev." + nowrap values,
  shift shape note on its own row; "H64 hierarchy test"; side filter 84 px so the lane select shows "mushroom body";
  "H64 FF" readout label at H64; Methods text (A6) and scrollTop=0 on open; tool-* class off in Matrix/Chemistry;
  WI.nA/nB declared, commit console.info once. Build: size cap before write, no placeholders in core. Tests: satHi flag,
  0.05 warning threshold (6 % warns, 4 % not), .npy 64-byte header padding, built-HTML test fails if the payload is missing;
  each new assert fails on a scratch mutant (/private/tmp/claude-501/fly_v2_fix/mut). Docs: EN Undo/Reset sentence, RU wording.
  HTML 18,929,855 B; pytest 24 passed; node check all passed; CDP light+dark: case 3 = 34,496,085, undo = 35,314,407.
  Left as is: "Selected: … (L / R / other)" still wraps to 2 lines; card rows "hidden intra-cell" / "inter-cell FF / FB"
  wrap (not in the fix list). Screenshots /private/tmp/claude-501/fly_v2_fix/*.png. Next: re-review.
- 2026-11-17: fix pass for both reviews DONE (HTML 18,929,855 B; pytest 24 passed; mutants for new asserts fail as expected).
  Page "left hemisphere +5000" = 33,676,470 was a composed move after case 3; oracle check of case3 -> left+5000 gives
  33,676,470 (script /private/tmp/claude-501/fly_v2_final/compose_check.py). Re-review launched.
- 2026-11-17: re-review PASS (no regressions; page values case3 34,496,085, MB swap 35,208,055, undo 35,314,407 in light+dark).
  Applied its nit (Methods unequal-size sentence), rebuilt (18,929,856 B), pytest 24 passed, node check passed.
  Committed and pushed branch viz/fly-dark-drag (owner OK 2026-11-17).

## Open for the owner
- Republish artifact 1d49a8d0-8a16-440e-89fc-b7cb067e7027 with the new fly_streams.html: waits for explicit OK (Q5).
- drawCloud during drag: 37-39 ms (agent C) vs 60-65 ms (code reviewer), both headless --disable-gpu; design threshold
  ~40 ms for the "draw only selected dots" fallback. Not added. Check by hand in real Chrome; add fallback if laggy.
- Not verified: real mouse/trackpad by hand, Safari/Firefox, real artifact host, live OS theme switch.
- Known cosmetic: "Selected: ... (L/R/other)" line and two card rows wrap to 2 lines in the aside.

# Certificate format v1

A certificate proves LB <= minimum feedback weight of the connectome (dataset `connectome`,
loaded by `src.mfas.io.load_dataset`). Edge id e = index into that loader's `src`, `tgt`,
`weight` arrays. One `.npz` file per certificate, all integer arrays int64, no pickles.

Header (all required):
- `meta`: JSON string with `dataset`, `n_nodes`, `n_edges`, `total_weight`, `edge_checksum`
  (see `cert_io.edge_checksum`), `method`, `git_commit`, `created`, free-form `notes`.
  Stored as a 0-d unicode array (no pickle). `edge_checksum` = sha256 hex of the bytes of
  src, then tgt, then weight, each cast to little-endian int64. The checksum binds the edge
  ids: a certificate for another edge order or other weights fails.
- `D`: 0-d int64 array, common denominator (>= 1). All loads, capacities and claims are in units of 1/D
  synapses.

K1 cycle packing (may be empty):
- `cyc_ptr` (C+1,), `cyc_edges` (cyc_ptr[-1],): cycle i is edges cyc_edges[cyc_ptr[i]:cyc_ptr[i+1]]
  in traversal order: tgt of each edge = src of the next, tgt of the last = src of the first,
  no node visited twice, length >= 2.
- `y` (C,): multiplicity of each cycle, y >= 1.

K2 piece packing (may be empty):
- `piece_ptr` (P+1,), `piece_edges`, `piece_cap`: piece j owns edges
  piece_edges[piece_ptr[j]:piece_ptr[j+1]] with capacities piece_cap (same slice), cap >= 1,
  no edge twice inside one piece.
- `piece_claim` (P,): claimed minimum, over all node orders of the piece's node set, of the
  capacity-weighted backward edges of the piece. Verified by the verifier's own exact solver;
  pieces too large for it are rejected (default limit 20 nodes per piece). claim >= 0.
  Self-loops cost 0 in the verifier's solver (the metric counts them as always backward, so
  this can only lower the recomputed minimum).

Strictness: every array must be exactly int64 with the stated shape; any other dtype (float,
int32, uint64, bool, object) or an unknown array name fails the certificate. The graph must
have integer weights; float-weight graphs (mouse) are rejected, not converted.

Feasibility (checked for every edge e):
    sum of y over cycles containing e + sum of piece_cap over pieces containing e <= D * w(e)

Bound:
    LB = floor((sum(y) + sum(piece_claim)) / D)      (synapses, integer)
    ceiling_pct = 100 * (W - LB) / W

Why it is valid: any order's backward edge set B hits every cycle, so the feedback weight is
at least sum_C y_C * [C hit] / D; the capacities split each edge's weight among the cycles and
pieces, and on each piece the order restricted to the piece's nodes pays at least its claim.

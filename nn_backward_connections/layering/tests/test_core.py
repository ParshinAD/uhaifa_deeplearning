"""Unit tests for layering.core - small synthetic fixtures plus one integration
test against the pinned mouse champion (skipped if the artifact is not pinned).

The toy graph is the T2 example from the task discussion: order pi = [p, q, a, c]
with FF chain p->q->a and feedback edge c->a. Method A (longest-path) puts c in
layer 0 and a in layer 2, so c->a becomes forward-by-layer (RECLAIMED); method B
(pi-slices) keeps c in the last slice, where c->a stays non-forward.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layering import core


# --- toy fixture: p=0, q=1, a=2, c=3 ---------------------------------------
TOY_ORDER = np.array([0, 1, 2, 3])
TOY_SRC = np.array([0, 1, 3])
TOY_TGT = np.array([1, 2, 2])
TOY_W = np.array([1.0, 1.0, 1.0])


def toy():
    rank = core.rank_from_order(TOY_ORDER)
    ff = core.ff_mask_by_order(TOY_SRC, TOY_TGT, rank)
    return rank, ff


def test_ff_split_toy():
    rank, ff = toy()
    assert ff.tolist() == [True, True, False]


def test_longest_path_layers_and_reclamation():
    rank, ff = toy()
    layer = core.layers_longest_path(TOY_SRC, TOY_TGT, ff, TOY_ORDER)
    assert layer.tolist() == [0, 1, 2, 0]
    cls = core.classify_by_layers(TOY_SRC, TOY_TGT, layer)
    # the pi-feedback edge c->a goes layer 0 -> layer 2: reclaimed
    assert cls.tolist() == [core.FORWARD, core.FORWARD, core.FORWARD]
    stats = core.layout_stats(TOY_SRC, TOY_TGT, TOY_W, rank, layer, "A")
    assert stats["reclaimed_edges"] == 1
    assert stats["edges_intra"] == 0


def test_pi_slices_layers_no_reclamation():
    rank, ff = toy()
    layer = core.layers_pi_slices(TOY_SRC, TOY_TGT, ff, TOY_ORDER)
    assert layer.tolist() == [0, 1, 2, 2]
    cls = core.classify_by_layers(TOY_SRC, TOY_TGT, layer)
    assert cls.tolist() == [core.FORWARD, core.FORWARD, core.INTRA]
    stats = core.layout_stats(TOY_SRC, TOY_TGT, TOY_W, rank, layer, "B")
    assert stats["reclaimed_edges"] == 0
    # layer index must be non-decreasing along pi
    assert np.all(np.diff(layer[TOY_ORDER]) >= 0)


def test_invariants_random_graph():
    rng = np.random.default_rng(0)
    n, m = 30, 150
    src = rng.integers(0, n, m)
    tgt = rng.integers(0, n, m)
    keep = src != tgt                      # drop self-loops
    src, tgt = src[keep], tgt[keep]
    order = rng.permutation(n)
    rank = core.rank_from_order(order)
    ff = core.ff_mask_by_order(src, tgt, rank)
    layer_a = core.layers_longest_path(src, tgt, ff, order)
    layer_b = core.layers_pi_slices(src, tgt, ff, order)
    for layer in (layer_a, layer_b):
        # HARD invariant: every FF edge strictly increases the layer
        assert np.all(layer[src[ff]] < layer[tgt[ff]])
    # slices are monotone along pi; longest-path is pointwise minimal
    assert np.all(np.diff(layer_b[order]) >= 0)
    assert np.all(layer_a <= layer_b)


def test_mutual_edge_mask():
    src = np.array([0, 1, 0, 2])
    tgt = np.array([1, 0, 2, 3])
    assert core.mutual_edge_mask(src, tgt).tolist() == [True, True, False, False]


def test_ff_path_exists_mask():
    # FF chain 0 -> 1 -> 2, plus an FB edge 2 -> 0 excluded by the mask
    src = np.array([0, 1, 2])
    tgt = np.array([1, 2, 0])
    ff = np.array([True, True, False])
    res = core.ff_path_exists_mask(src, tgt, ff,
                                   np.array([0, 2, 1, 1]),
                                   np.array([2, 0, 1, 0]))
    # 0->2 via the chain; 2->0 impossible (FB edge masked out); 1->1 trivial;
    # 1->0 impossible (edges point away)
    assert res.tolist() == [True, False, True, False]


# --- crossing counter and barycenter ---------------------------------------

def two_layer_fixture(crossed: bool):
    # nodes 0,1 in layer 0 (slots 0,1); nodes 2,3 in layer 1 (slots 0,1)
    layer = np.array([0, 0, 1, 1])
    slot = np.array([0, 1, 0, 1])
    if crossed:
        src, tgt = np.array([0, 1]), np.array([3, 2])
    else:
        src, tgt = np.array([0, 1]), np.array([2, 3])
    return src, tgt, layer, slot


def test_count_crossings_basic():
    for crossed, expected in ((True, 1), (False, 0)):
        src, tgt, layer, slot = two_layer_fixture(crossed)
        x = layer.astype(float)
        y = core.y_coords(layer, slot)
        mask = np.ones(src.shape[0], dtype=bool)
        assert core.count_crossings(src, tgt, mask, x, y) == expected


def test_count_crossings_shared_endpoint_is_zero():
    layer = np.array([0, 1, 1])
    slot = np.array([0, 0, 1])
    src, tgt = np.array([0, 0]), np.array([1, 2])
    mask = np.ones(2, dtype=bool)
    x, y = layer.astype(float), core.y_coords(layer, slot)
    assert core.count_crossings(src, tgt, mask, x, y) == 0


def test_barycenter_uncrosses_and_never_worsens():
    src, tgt, layer, slot = two_layer_fixture(crossed=True)
    mask = np.ones(src.shape[0], dtype=bool)
    best_slot, best_c = core.barycenter_order(src, tgt, mask, layer, slot)
    assert best_c == 0
    # idempotent on an already-perfect layout
    _, again_c = core.barycenter_order(src, tgt, mask, layer, best_slot)
    assert again_c == 0


def test_initial_slots_follow_pi():
    layer = np.array([0, 1, 0, 1])
    order = np.array([2, 0, 3, 1])          # pi puts node 2 before node 0
    rank = core.rank_from_order(order)
    slot = core.initial_slots_from_order(layer, rank)
    # layer 0 holds nodes {0, 2}: node 2 is earlier in pi -> slot 0
    assert slot[2] == 0 and slot[0] == 1
    # layer 1 holds nodes {1, 3}: node 3 earlier in pi -> slot 0
    assert slot[3] == 0 and slot[1] == 1


# --- integration: the pinned mouse champion --------------------------------

def test_mouse_integration():
    from layering import io_utils
    npz = REPO_ROOT / io_utils.CHAMPION_ORDER_NPZ["mouse"]
    if not npz.exists():
        pytest.skip("mouse champion ordering not pinned (tools/pin_champion.py)")
    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order("mouse")
    assert g.n_nodes == 148 and g.n_edges == 583
    ff = core.ff_mask_by_order(g.src, g.tgt, rank)
    layer_a = core.layers_longest_path(g.src, g.tgt, ff, order)
    layer_b = core.layers_pi_slices(g.src, g.tgt, ff, order)
    assert int(layer_a.max()) + 1 == 26     # measured 2026-08-31, H63 order
    assert int(layer_b.max()) + 1 == 49
    for layer in (layer_a, layer_b):
        assert np.all(layer[g.src[ff]] < layer[g.tgt[ff]])
    assert np.all(layer_a <= layer_b)

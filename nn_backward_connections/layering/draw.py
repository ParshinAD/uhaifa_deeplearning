"""Matplotlib rendering of a layered layout.

Layers are columns (like an ANN diagram): x = layer index, y = centered slot
within the layer. Feedforward-by-layer edges always run left -> right, so they
are drawn as straight segments without arrowheads; feedback-by-layer edges are
red arcs with arrowheads; intra-layer edges (possible only for pairs with no FF
path, see core.classify_by_layers) are orange arcs.

Line width encodes edge weight (sqrt-scaled); the CROSSING OBJECTIVE stays
unweighted per decision 4 - width is presentation only.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from .core import classify_by_layers, y_coords, FORWARD, INTRA, BACKWARD

__all__ = ["auto_figsize", "edge_linewidths", "draw_layered"]

COL_FWD = "#5b7fa6"     # muted steel blue
COL_BACK = "#c23b3b"    # red: the feedback we study
COL_INTRA = "#e08a2e"   # orange
COL_NODE = "#1f2a44"

# Presentation defaults (draw-only; no effect on any reported number).
FIG_W_RANGE = (8.0, 34.0)       # inches, min/max figure width
FIG_H_RANGE = (5.0, 18.0)       # inches, min/max figure height
FIG_BASE = 1.5                  # inches added before per-layer/per-slot scaling
FIG_PER_LAYER = 0.55            # inches of width per layer
FIG_PER_SLOT = 0.30             # inches of height per node in the widest layer
NODE_SIZE = 30                  # scatter marker area
ARROW_SCALE = 7                 # FancyArrowPatch mutation_scale for arcs


def auto_figsize(layer: np.ndarray) -> tuple:
    """Figure size scaled to layer count (width) and max layer width (height)."""
    n_layers = int(layer.max()) + 1
    max_width = int(np.bincount(layer).max())
    return (min(FIG_W_RANGE[1], max(FIG_W_RANGE[0], FIG_BASE + FIG_PER_LAYER * n_layers)),
            min(FIG_H_RANGE[1], max(FIG_H_RANGE[0], FIG_BASE + FIG_PER_SLOT * max_width)))


def edge_linewidths(weight, lw_range=(0.4, 2.8)) -> np.ndarray:
    """Sqrt-scaled min-max mapping of edge weight to line width."""
    w = np.asarray(weight, dtype=np.float64)
    lo, hi = float(w.min()), float(w.max())
    if hi <= lo:
        return np.full(w.shape, float(np.mean(lw_range)))
    unit = np.sqrt((w - lo) / (hi - lo))
    return lw_range[0] + unit * (lw_range[1] - lw_range[0])


def draw_layered(src, tgt, weight, layer, slot, title=None, ax=None,
                 annotate: bool = False, arc_rad: float = 0.35, dpi: int = 120):
    """Draw the layered graph; returns ``(fig, ax)``.

    Parameters mirror core's conventions: ``layer``/``slot`` define the layout,
    edge classes are recomputed here via ``classify_by_layers`` so the picture
    can never disagree with the reported statistics.
    """
    src = np.asarray(src)
    tgt = np.asarray(tgt)
    layer = np.asarray(layer)
    x = layer.astype(np.float64)
    y = y_coords(layer, np.asarray(slot))
    cls = classify_by_layers(src, tgt, layer)
    if ax is None:
        fig, ax = plt.subplots(figsize=auto_figsize(layer), dpi=dpi)
    else:
        fig = ax.figure
    lw = edge_linewidths(weight)

    fwd = cls == FORWARD
    if fwd.any():
        segs = np.stack([np.stack([x[src[fwd]], y[src[fwd]]], axis=1),
                         np.stack([x[tgt[fwd]], y[tgt[fwd]]], axis=1)], axis=1)
        ax.add_collection(LineCollection(segs, colors=COL_FWD,
                                         linewidths=lw[fwd], alpha=0.55, zorder=1))
    for mask, col, rad in ((cls == BACKWARD, COL_BACK, arc_rad),
                           (cls == INTRA, COL_INTRA, 0.6)):
        for e in np.where(mask)[0].tolist():
            u, v = int(src[e]), int(tgt[e])
            ax.add_patch(FancyArrowPatch(
                (x[u], y[u]), (x[v], y[v]),
                connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                mutation_scale=ARROW_SCALE, color=col, lw=float(lw[e]), alpha=0.75,
                zorder=2, shrinkA=3, shrinkB=3))

    ax.scatter(x, y, s=NODE_SIZE, c=COL_NODE, edgecolors="white",
               linewidths=0.6, zorder=3)
    if annotate:
        for v in range(x.shape[0]):
            ax.annotate(str(v), (x[v], y[v]), fontsize=4.5, ha="center",
                        va="center", color="white", zorder=4)

    handles = [Line2D([], [], color=COL_FWD, lw=2, label="feedforward (by layer)"),
               Line2D([], [], color=COL_BACK, lw=2, label="feedback (by layer)")]
    if bool((cls == INTRA).any()):
        handles.append(Line2D([], [], color=COL_INTRA, lw=2, label="intra-layer"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    n_layers = int(layer.max()) + 1
    ax.set_xlabel("layer")
    ax.set_ylabel("position within layer (centered)")
    if title:
        ax.set_title(title)
    ax.set_xticks(np.arange(n_layers))
    ax.tick_params(axis="x", labelsize=6 if n_layers > 30 else 8)
    ax.margins(x=0.02, y=0.05)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax

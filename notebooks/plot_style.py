"""Shared plotting style for the Yelp sentiment MLOps notebooks.

Every notebook that produces a chart should start with:

    from plot_style import (
        apply_style,
        PURPLE_PRIMARY, PURPLE_DARK, PURPLE_LIGHT,
        PURPLE_SENTIMENT, PURPLE_SPLITS, PURPLE_CMAP,
        purple_sequential,
    )
    apply_style()

That guarantees every figure in the project shares the same purple-gradient
palette and seaborn theme, so the final report has visual consistency from
EDA all the way through model evaluation plots.

To restyle the entire project, edit the constants below and re-run the
downstream notebooks.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

PURPLE_PRIMARY = "#6a51a3"
PURPLE_DARK = "#54278f"
PURPLE_LIGHT = "#bcbddc"
PURPLE_ACCENT = "#9e9ac8"

PURPLE_CMAP = "Purples"
PURPLE_CMAP_R = "Purples_r"

PURPLE_SENTIMENT = {
    "positive": PURPLE_DARK,
    "negative": PURPLE_LIGHT,
}

PURPLE_SPLITS = {
    "train": PURPLE_DARK,
    "validation": PURPLE_PRIMARY,
    "test": PURPLE_ACCENT,
    "production": PURPLE_LIGHT,
}

PURPLE_CONFUSION = {
    "true_negative": PURPLE_LIGHT,
    "false_positive": PURPLE_ACCENT,
    "false_negative": PURPLE_ACCENT,
    "true_positive": PURPLE_DARK,
}


def purple_sequential(n: int, reverse: bool = False) -> list:
    """Return ``n`` purple shades from light to dark.

    Use for ordered categorical axes (e.g. 5 star ratings, top-N word bars).
    Pass ``reverse=True`` to flip the order (darkest first).
    """
    if n <= 0:
        return []
    palette = sns.color_palette(PURPLE_CMAP, n_colors=n + 1)[1:]
    if reverse:
        palette = list(reversed(palette))
    return palette


def apply_style() -> None:
    """Apply the shared seaborn/matplotlib theme.

    Call this once at the top of each notebook (after importing seaborn).
    Idempotent — safe to call multiple times.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.labelweight": "regular",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#e6e6f0",
            "grid.linewidth": 0.6,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )

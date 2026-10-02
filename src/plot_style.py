"""Shared chart style so every figure in the project looks the same."""

import matplotlib.pyplot as plt

# Colors for the two classes. This blue and orange pair stays distinguishable for
# the common types of color blindness.
NO_CKD = "#2a78d6"
CKD = "#eb6834"

TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e6e3"
AXIS = "#8a8984"


def apply():
    """Call once at the top of a notebook."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": TEXT,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "legend.frameon": False,
    })

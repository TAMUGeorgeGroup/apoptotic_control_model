import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def finish(fig, output):
    output = Path(output)
    if root := os.environ.get("APOPTOTIC_OUTPUT_ROOT"):
        output = Path(root) / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.15)
    return output


def panel_label(ax, label):
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def heatmap(ax, values, states, parameters, *, cmap="viridis", vmin=None, vmax=None):
    return ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        extent=[states[0], states[-1], parameters[0], parameters[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


def symmetric_limits(values):
    bound = np.nanmax(np.abs(values))
    return -bound, bound


def use_paper_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )
import argparse

import matplotlib.pyplot as plt
import numpy as np

from apoptotic_control.parameters import FIGURE_S2
from apoptotic_control.plotting import finish, use_paper_style
from apoptotic_control.simulation import gillespie
from apoptotic_control.solvers import solve_quadratic


def compute(quick=False):
    count = 12 if quick else 40

    # Number of stochastic trajectories per parameter pair
    n_reps = 10 if quick else 100

    c1_values = np.linspace(0.001, 0.03, count)
    c3_values = np.linspace(0.2, 5.0, count)

    median_maxima = np.zeros((count, count))

    base_seed = 123

    for row, c1 in enumerate(c1_values):
        for column, c3 in enumerate(c3_values):

            model, _, policy = solve_quadratic(
                dict(FIGURE_S2, c1=c1, c3=c3)
            )

            seed_sequence = np.random.SeedSequence(
                [base_seed, row, column]
            )
            replicate_seeds = seed_sequence.spawn(n_reps)

            replicate_maxima = np.empty(n_reps)

            for rep, seed in enumerate(replicate_seeds):
                rng = np.random.default_rng(seed)

                _, states = gillespie(
                    model,
                    policy,
                    i0=10,
                    final_time=300,
                    rng=rng,
                )

                replicate_maxima[rep] = states.max()

            median_maxima[row, column] = np.median(replicate_maxima)

    return c1_values, c3_values, median_maxima


def plot(data):
    c1, c3, median_maxima = data

    use_paper_style()

    fig, ax = plt.subplots(figsize=(7, 5))

    image = ax.imshow(
        median_maxima,
        origin="lower",
        aspect="auto",
        extent=[c3[0], c3[-1], c1[0], c1[-1]],
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Median maximum population reached",
    )

    ax.set(
        title=r"Growth/suppression phase map: $c_1$ vs $c_3$",
        xlabel=r"$c_3$ growth reward",
        ylabel=r"$c_1$ quadratic penalty",
    )

    ax.title.set_fontweight("bold")

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    finish(
        plot(compute(args.quick)),
        "figures/supplementary/Figure_S3.png",
    )
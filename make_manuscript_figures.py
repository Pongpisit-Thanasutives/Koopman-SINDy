#!/usr/bin/env python3
"""Regenerate manuscript figures with an optional SciencePlots style.

The script uses SciencePlots when installed (``pip install SciencePlots``) and
falls back to a conservative serif matplotlib style otherwise. It regenerates
only the manuscript figures from already defined synthetic systems/results; it
does not rerun the benchmark experiments.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import koopman_sindy_ode_benchmark as ode
import koopman_sindy_pde_benchmark as pde
from koopman_sindy_advection_diffusion_benchmark import advection_diffusion_system, write_dataset_figure
from koopman_sindy_model_selection_experiment import plot_pareto


def apply_publication_plot_style() -> None:
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "nature", "no-latex"])
    except Exception:
        plt.rcParams.update({
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        })
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.4,
        "savefig.dpi": 300,
    })


def dataset_examples(outpath: str = "figures/dataset_examples.png") -> None:
    apply_publication_plot_style()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    lor = ode.lorenz_system()
    vdp = ode.vanderpol_system()
    t_l, X_l = ode.integrate_system(lor, dt=0.01)
    t_v, X_v = ode.integrate_system(vdp, dt=0.01)

    burg = pde.burgers_system(n=64)
    fish = pde.fisher_kpp_system(n=64, ic_variant="default")
    t_b, U_b = pde.integrate_pde(burg, dt=0.01)
    t_f, U_f = pde.integrate_pde(fish, dt=0.01)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(X_l[:, 0], X_l[:, 2], lw=0.9)
    idx = np.arange(0, len(X_l), max(1, len(X_l)//35))
    rng = np.random.default_rng(7)
    X_l_obs = ode.add_noise(X_l[idx], 0.03, rng)
    ax.scatter(X_l_obs[:, 0], X_l_obs[:, 2], s=14, alpha=0.60)
    ax.set_title("(a) Lorenz--63")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$z$")

    ax = axes[0, 1]
    ax.plot(X_v[:, 0], X_v[:, 1], lw=0.9)
    idx = np.arange(0, len(X_v), max(1, len(X_v)//35))
    rng = np.random.default_rng(11)
    X_v_obs = ode.add_noise(X_v[idx], 0.03, rng)
    ax.scatter(X_v_obs[:, 0], X_v_obs[:, 1], s=14, alpha=0.60)
    ax.set_title("(b) Van der Pol")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    for ax, title, t, U, x in [
        (axes[1, 0], "(c) Burgers field", t_b, U_b, burg.x),
        (axes[1, 1], "(d) Fisher--KPP field", t_f, U_f, fish.x),
    ]:
        im = ax.imshow(
            U.T,
            origin="lower",
            aspect="auto",
            extent=[float(t[0]), float(t[-1]), float(x[0]), float(x[-1])],
        )
        obs = t[::16]
        for tt in obs:
            ax.axvline(float(tt), color="white", lw=0.6, alpha=0.65)
        ax.set_title(title)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=10)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def advection_diffusion_example(outpath: str = "figures/advection_diffusion_dataset.png") -> None:
    sys = advection_diffusion_system()
    t, U = pde.integrate_pde(sys, dt=0.01)
    write_dataset_figure(t, U, sys.x, outpath=outpath)


def model_selection_pareto() -> None:
    front_path = "results/model_selection_publication/model_selection_pareto_front.csv"
    outdir = "results/model_selection_publication"
    if os.path.exists(front_path):
        front = pd.read_csv(front_path)
        plot_pareto(front, outdir)
        src = os.path.join(outdir, "model_selection_pareto_equationwise.png")
        dst = "figures/model_selection_pareto_equationwise.png"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        import shutil
        shutil.copyfile(src, dst)


def main() -> None:
    dataset_examples()
    advection_diffusion_example()
    model_selection_pareto()


if __name__ == "__main__":
    main()

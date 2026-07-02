#!/usr/bin/env python3
"""Sensitivity study for the upsampling factor q and POD rank r.

This script is intended to generate Appendix-A-style robustness evidence for the
fixed hyperparameters used in the manuscript.  It does not replace the main
benchmark tables.  Instead, it asks whether the reported choices, especially
q=5 and the system-specific PDE POD ranks, are reasonable under a representative
sparse/noisy setting.

Default appendix setting
------------------------
ODE: Lorenz--63 and Van der Pol, EDMD-polynomial, noise=3%, sparse factor 16,
     q in {1, 3, 5, 7}, five seeds.
PDE: Burgers, Fisher--KPP, and advection--diffusion, POD-EDMD-RBF, noise=3%,
     sparse factor 8, q in {1, 3, 5, 7}, r in {1,2,4,6,8,10,12}, five seeds.

Outputs are written to results/qr_sensitivity by default:
  qr_sensitivity_raw.csv
  qr_sensitivity_summary.csv
  ode_q_sensitivity_summary.csv
  pde_qr_sensitivity_summary.csv
  pde_default_vs_best.csv
  appendix_a_qr_sensitivity.tex
  appendix_q_sensitivity_ode.png
  appendix_qr_sensitivity_pde.png

The generated LaTeX fragment can be included in the manuscript appendix after
checking the locally regenerated values.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import koopman_sindy_ode_benchmark as ode
import koopman_sindy_pde_benchmark as pde

EPS = 1e-12
ODE_THRESHOLDS = np.array([0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
PDE_THRESHOLDS = np.array([
    0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3,
    1e-2, 1.3e-2, 1.5e-2, 2e-2, 3e-2, 5e-2, 7e-2, 1e-1, 2e-1, 3e-1
])
PDE_DEFAULT_RANKS = {"burgers": 8, "fisher_kpp": 2, "advection_diffusion": 4}
SYSTEM_LABELS = {
    "lorenz63": "Lorenz-63",
    "vanderpol_mu2": "Van der Pol",
    "burgers": "Burgers",
    "fisher_kpp": "Fisher--KPP",
    "advection_diffusion": "Advection--diffusion",
}
METHOD_LABELS = {
    "baseline": "Baseline",
    "edmd_poly3": "EDMD-polynomial",
    "pod_edmd_rbf": "POD-EDMD-RBF",
}


def parse_int_list(s: str | None, default: Sequence[int]) -> List[int]:
    if s is None or str(s).strip() == "":
        return list(default)
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def parse_float_list(s: str | None, default: Sequence[float]) -> List[float]:
    if s is None or str(s).strip() == "":
        return list(default)
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def apply_plot_style() -> None:
    """Use SciencePlots when available; otherwise use a safe serif style."""
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
        "lines.linewidth": 1.5,
        "savefig.dpi": 300,
    })


def row_seed(setting: str, system: str, sparse_factor: int, noise: float, seed: int, q: int, rank: int = 0) -> int:
    # Deterministic seed that changes when q/r changes but is stable across runs.
    base = seed + 1000 * sparse_factor + 100000 * int(round(noise * 1000))
    if setting == "PDE":
        base += 10000000 + 100 * q + 10000 * rank + (abs(hash(system)) % 997)
    else:
        base += 200 * q + (abs(hash(system)) % 997)
    return int(base % (2**32 - 1))


def status_row(**kwargs) -> Dict[str, object]:
    defaults = {
        "support_f1": np.nan,
        "coef_error": np.nan,
        "practical_score": np.nan,
        "threshold": np.nan,
        "status": "not_run",
    }
    defaults.update(kwargs)
    return defaults


def run_ode_sensitivity(args: argparse.Namespace) -> List[Dict[str, object]]:
    systems = [ode.lorenz_system(), ode.vanderpol_system()]
    rows: List[Dict[str, object]] = []
    for sys in systems:
        print(f"[ODE] integrating {sys.name}", flush=True)
        t_true, X_true = ode.integrate_system(sys, args.ode_dt)
        _, feature_names = ode.polynomial_library(X_true[:2], sys.var_names, degree=3)
        Xi_true = sys.true_coefficients(feature_names)
        ind = np.arange(0, len(t_true), args.ode_sparse_factor)
        if ind[-1] != len(t_true) - 1:
            ind = np.r_[ind, len(t_true) - 1]
        t_obs = t_true[ind]
        X_clean_obs = X_true[ind]
        # Baseline is independent of q, so run once per seed.
        for seed in args.seeds:
            rng = np.random.default_rng(row_seed("ODE", sys.name, args.ode_sparse_factor, args.noise, seed, q=1))
            X_obs = ode.add_noise(X_clean_obs, args.noise, rng)
            try:
                res = ode.fit_sindy_oracle(X_obs, t_obs, sys.var_names, 3, Xi_true, ODE_THRESHOLDS)
                rows.append(status_row(
                    setting="ODE", system=sys.name, system_label=SYSTEM_LABELS.get(sys.name, sys.name),
                    method="baseline", method_label=METHOD_LABELS["baseline"], q=1, pod_rank=0,
                    sparse_factor=args.ode_sparse_factor, noise=args.noise, seed=seed,
                    support_f1=float(res["f1"]), coef_error=float(res["coef_error"]),
                    practical_score=float(res["score"]), threshold=float(res["threshold"]), status="ok",
                ))
            except Exception as exc:
                rows.append(status_row(
                    setting="ODE", system=sys.name, system_label=SYSTEM_LABELS.get(sys.name, sys.name),
                    method="baseline", method_label=METHOD_LABELS["baseline"], q=1, pod_rank=0,
                    sparse_factor=args.ode_sparse_factor, noise=args.noise, seed=seed,
                    status=f"fail: {type(exc).__name__}: {exc}",
                ))
        for q in args.q_values:
            for seed in args.seeds:
                rng = np.random.default_rng(row_seed("ODE", sys.name, args.ode_sparse_factor, args.noise, seed, q=q))
                X_obs = ode.add_noise(X_clean_obs, args.noise, rng)
                t_new = ode.make_tnew(t_obs, q)
                try:
                    X_use = ode.edmd_reconstruct(
                        X_obs, t_obs, t_new,
                        kind="poly", degree=3, var_names=sys.var_names, rng=rng,
                    )
                    if len(t_new) < 5 or not np.all(np.isfinite(X_use)):
                        raise FloatingPointError("non-finite or too-short reconstruction")
                    res = ode.fit_sindy_oracle(X_use, t_new, sys.var_names, 3, Xi_true, ODE_THRESHOLDS)
                    rows.append(status_row(
                        setting="ODE", system=sys.name, system_label=SYSTEM_LABELS.get(sys.name, sys.name),
                        method="edmd_poly3", method_label=METHOD_LABELS["edmd_poly3"], q=int(q), pod_rank=0,
                        sparse_factor=args.ode_sparse_factor, noise=args.noise, seed=seed,
                        support_f1=float(res["f1"]), coef_error=float(res["coef_error"]),
                        practical_score=float(res["score"]), threshold=float(res["threshold"]), status="ok",
                    ))
                except Exception as exc:
                    rows.append(status_row(
                        setting="ODE", system=sys.name, system_label=SYSTEM_LABELS.get(sys.name, sys.name),
                        method="edmd_poly3", method_label=METHOD_LABELS["edmd_poly3"], q=int(q), pod_rank=0,
                        sparse_factor=args.ode_sparse_factor, noise=args.noise, seed=seed,
                        status=f"fail: {type(exc).__name__}: {exc}",
                    ))
    return rows


def get_pde_systems(args: argparse.Namespace) -> List[pde.PDESystem]:
    n = int(args.pde_n)
    systems = []
    for name in args.pde_systems:
        if name == "burgers":
            systems.append(pde.burgers_system(n=n))
        elif name == "fisher_kpp":
            systems.append(pde.fisher_kpp_system(n=n, ic_variant=args.fisher_ic))
        elif name == "advection_diffusion":
            systems.append(pde.advection_diffusion_system(n=n))
        else:
            raise ValueError(f"unknown PDE system: {name}")
    return systems


def run_pde_sensitivity(args: argparse.Namespace) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for sys in get_pde_systems(args):
        print(f"[PDE] integrating {sys.name}", flush=True)
        k = pde.periodic_wavenumbers(len(sys.x), sys.params["L"])
        t_true, U_true = pde.integrate_pde(sys, args.pde_dt)
        _, feature_names = pde.pde_library(U_true[:2], k)
        xi_true = sys.true_coefficients(feature_names)
        ind = np.arange(0, len(t_true), args.pde_sparse_factor)
        if ind[-1] != len(t_true) - 1:
            ind = np.r_[ind, len(t_true) - 1]
        t_obs = t_true[ind]
        U_clean_obs = U_true[ind]
        # Baseline is independent of q and r, so run once per seed.
        for seed in args.seeds:
            rng = np.random.default_rng(row_seed("PDE", sys.name, args.pde_sparse_factor, args.noise, seed, q=1, rank=0))
            U_obs = pde.add_noise(U_clean_obs, args.noise, rng)
            try:
                res = pde.fit_pdefind_oracle(U_obs, t_obs, k, xi_true, PDE_THRESHOLDS, rng, max_rows=args.max_rows)
                rows.append(status_row(
                    setting="PDE", system=sys.name, system_label=SYSTEM_LABELS.get(sys.name, sys.name),
                    method="baseline", method_label=METHOD_LABELS["baseline"], q=1, pod_rank=0,
                    sparse_factor=args.pde_sparse_factor, noise=args.noise, seed=seed,
                    support_f1=float(res["f1"]), coef_error=float(res["coef_error"]),
                    practical_score=float(res["score"]), threshold=float(res["threshold"]), status="ok",
                ))
            except Exception as exc:
                rows.append(status_row(
                    setting="PDE", system=sys.name, system_label=SYSTEM_LABELS.get(sys.name, sys.name),
                    method="baseline", method_label=METHOD_LABELS["baseline"], q=1, pod_rank=0,
                    sparse_factor=args.pde_sparse_factor, noise=args.noise, seed=seed,
                    status=f"fail: {type(exc).__name__}: {exc}",
                ))
        for q in args.q_values:
            t_new = pde.make_tnew(t_obs, q)
            for rank in args.r_values:
                for seed in args.seeds:
                    rng = np.random.default_rng(row_seed("PDE", sys.name, args.pde_sparse_factor, args.noise, seed, q=q, rank=rank))
                    U_obs = pde.add_noise(U_clean_obs, args.noise, rng)
                    try:
                        U_use = pde.pod_edmd_reconstruct(
                            U_obs, t_obs, t_new, int(rank), "rbf", rng, rbf_centers=args.rbf_centers
                        )
                        if len(t_new) < 5 or not np.all(np.isfinite(U_use)):
                            raise FloatingPointError("non-finite or too-short reconstruction")
                        res = pde.fit_pdefind_oracle(U_use, t_new, k, xi_true, PDE_THRESHOLDS, rng, max_rows=args.max_rows)
                        rows.append(status_row(
                            setting="PDE", system=sys.name, system_label=SYSTEM_LABELS.get(sys.name, sys.name),
                            method="pod_edmd_rbf", method_label=METHOD_LABELS["pod_edmd_rbf"],
                            q=int(q), pod_rank=int(rank), sparse_factor=args.pde_sparse_factor,
                            noise=args.noise, seed=seed, support_f1=float(res["f1"]),
                            coef_error=float(res["coef_error"]), practical_score=float(res["score"]),
                            threshold=float(res["threshold"]), status="ok",
                        ))
                    except Exception as exc:
                        rows.append(status_row(
                            setting="PDE", system=sys.name, system_label=SYSTEM_LABELS.get(sys.name, sys.name),
                            method="pod_edmd_rbf", method_label=METHOD_LABELS["pod_edmd_rbf"],
                            q=int(q), pod_rank=int(rank), sparse_factor=args.pde_sparse_factor,
                            noise=args.noise, seed=seed, status=f"fail: {type(exc).__name__}: {exc}",
                        ))
    return rows


def summarize(raw: pd.DataFrame, outdir: str) -> pd.DataFrame:
    ok = raw[raw["status"] == "ok"].copy()
    group_cols = ["setting", "system", "system_label", "method", "method_label", "q", "pod_rank", "sparse_factor", "noise"]
    summary = ok.groupby(group_cols, as_index=False).agg(
        support_f1_mean=("support_f1", "mean"),
        support_f1_median=("support_f1", "median"),
        coef_error_median=("coef_error", "median"),
        coef_error_mean=("coef_error", "mean"),
        practical_score_mean=("practical_score", "mean"),
        practical_score_median=("practical_score", "median"),
        n_ok=("status", "size"),
    )
    summary.to_csv(os.path.join(outdir, "qr_sensitivity_summary.csv"), index=False)
    summary[summary.setting == "ODE"].to_csv(os.path.join(outdir, "ode_q_sensitivity_summary.csv"), index=False)
    summary[summary.setting == "PDE"].to_csv(os.path.join(outdir, "pde_qr_sensitivity_summary.csv"), index=False)

    # Default-vs-best summaries for manuscript appendix.
    rows = []
    for system, g in summary[(summary.setting == "PDE") & (summary.method == "pod_edmd_rbf")].groupby("system"):
        g_ok = g.dropna(subset=["coef_error_median", "practical_score_mean"]).copy()
        if g_ok.empty:
            continue
        best_score = g_ok.sort_values(["practical_score_mean", "coef_error_median"], ascending=[False, True]).iloc[0]
        default_rank = PDE_DEFAULT_RANKS.get(system, np.nan)
        default = g_ok[(g_ok.q == 5) & (g_ok.pod_rank == default_rank)]
        default_row = default.iloc[0] if len(default) else None
        rows.append({
            "system": system,
            "system_label": SYSTEM_LABELS.get(system, system),
            "default_q": 5,
            "default_rank": default_rank,
            "default_f1_mean": np.nan if default_row is None else float(default_row.support_f1_mean),
            "default_coef_error_median": np.nan if default_row is None else float(default_row.coef_error_median),
            "default_score_mean": np.nan if default_row is None else float(default_row.practical_score_mean),
            "best_q": int(best_score.q),
            "best_rank": int(best_score.pod_rank),
            "best_f1_mean": float(best_score.support_f1_mean),
            "best_coef_error_median": float(best_score.coef_error_median),
            "best_score_mean": float(best_score.practical_score_mean),
        })
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "pde_default_vs_best.csv"), index=False)
    return summary


def plot_ode_q(summary: pd.DataFrame, outdir: str) -> None:
    apply_plot_style()
    d = summary[(summary.setting == "ODE") & (summary.method.isin(["baseline", "edmd_poly3"]))].copy()
    if d.empty:
        return
    systems = list(d["system"].drop_duplicates())
    fig, axes = plt.subplots(1, len(systems), figsize=(4.3 * len(systems), 3.8), squeeze=False, constrained_layout=False)
    for ax, system in zip(axes.ravel(), systems):
        g = d[d.system == system]
        base = g[g.method == "baseline"]
        edmd = g[g.method == "edmd_poly3"].sort_values("q")
        if len(base):
            bval = float(base.coef_error_median.iloc[0])
            ax.axhline(bval, color="black", linestyle="-", linewidth=1.3, label="Baseline")
        if len(edmd):
            ax.plot(edmd.q, edmd.coef_error_median, color="tab:blue", linestyle="--", marker="o", markerfacecolor="none", markeredgecolor="tab:blue", markeredgewidth=1.0, label="EDMD-polynomial")
            if 5 in set(edmd.q):
                y5 = float(edmd.loc[edmd.q == 5, "coef_error_median"].iloc[0])
                ax.scatter([5], [y5], s=70, facecolors="none", edgecolors="tab:blue", linewidths=1.2, zorder=5)
        ax.set_title(SYSTEM_LABELS.get(system, system))
        ax.set_xlabel("upsampling factor $q$")
        ax.set_ylabel("median coefficient error")
        ax.set_xticks(sorted(set(edmd.q.astype(int))) if len(edmd) else [])
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    for ax in axes.ravel():
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    path = os.path.join(outdir, "appendix_q_sensitivity_ode.png")
    fig.subplots_adjust(bottom=0.22, wspace=0.30)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    shutil.copyfile(path, os.path.join("figures", "appendix_q_sensitivity_ode.png"))
    plt.close(fig)


def plot_pde_qr(summary: pd.DataFrame, outdir: str) -> None:
    apply_plot_style()
    d = summary[(summary.setting == "PDE") & (summary.method == "pod_edmd_rbf")].copy()
    if d.empty:
        return
    systems = list(d["system"].drop_duplicates())
    n = len(systems)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.6), squeeze=False, constrained_layout=True)
    vmax = np.nanpercentile(d["coef_error_median"].to_numpy(), 95) if np.isfinite(d["coef_error_median"]).any() else 1.0
    for ax, system in zip(axes.ravel(), systems):
        g = d[d.system == system]
        pivot = g.pivot(index="pod_rank", columns="q", values="coef_error_median").sort_index().sort_index(axis=1)
        arr = pivot.to_numpy(dtype=float)
        im = ax.imshow(arr, origin="lower", aspect="auto", vmin=0.0, vmax=max(vmax, EPS))
        ax.set_title(SYSTEM_LABELS.get(system, system))
        ax.set_xlabel("upsampling factor $q$")
        ax.set_ylabel("POD rank $r$")
        ax.set_xticks(np.arange(pivot.shape[1]), labels=[str(int(c)) for c in pivot.columns])
        ax.set_yticks(np.arange(pivot.shape[0]), labels=[str(int(r)) for r in pivot.index])
        default_rank = PDE_DEFAULT_RANKS.get(system)
        if default_rank in set(pivot.index) and 5 in set(pivot.columns):
            x = list(pivot.columns).index(5)
            y = list(pivot.index).index(default_rank)
            ax.scatter([x], [y], marker="*", s=130, facecolor="white", edgecolor="black", linewidth=1.0, zorder=4)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("median coefficient error")
    path = os.path.join(outdir, "appendix_qr_sensitivity_pde.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    shutil.copyfile(path, os.path.join("figures", "appendix_qr_sensitivity_pde.png"))
    plt.close(fig)


def latex_float(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def write_appendix_fragment(summary: pd.DataFrame, outdir: str, args: argparse.Namespace) -> None:
    """Write a reproducible Appendix A fragment from sensitivity outputs."""
    ode_sum = summary[(summary.setting == "ODE") & (summary.method.isin(["baseline", "edmd_poly3"]))].copy()
    pde_best_path = os.path.join(outdir, "pde_default_vs_best.csv")
    pde_best = pd.read_csv(pde_best_path) if os.path.exists(pde_best_path) else pd.DataFrame()

    def bold_fmt(value: float, target: float, mode: str, digits: int = 3) -> str:
        txt = latex_float(value, digits)
        if pd.isna(value) or pd.isna(target):
            return txt
        if round(float(value), digits) == round(float(target), digits):
            return r"\textbf{" + txt + "}"
        return txt

    lines: List[str] = []
    lines.append(r"\clearpage")
    lines.append(r"\section{Upsampling-factor and POD-rank sensitivity}")
    lines.append(r"\label{app:qr_sensitivity}")
    lines.append("")
    lines.append(
        "This appendix examines the fixed interpolation factor $q$ and POD rank $r$ used by the "
        "assisted preprocessors. For PDEs, the $q=1$ column is a POD-only denoising control: "
        "the noisy sparse snapshots are projected to rank $r$ and reconstructed at the observed times, "
        "but no temporal points are inserted. Values with $q>1$ combine the same POD projection with "
        "DMD/EDMD temporal upsampling. The study is not used for oracle selection in the main tables; "
        "it is a representative robustness check at the same synthetic-equation level. Unless otherwise stated, "
        f"the sensitivity run uses relative noise {args.noise:g}, ODE sparse factor {args.ode_sparse_factor}, "
        f"PDE sparse factor {args.pde_sparse_factor}, and {len(args.seeds)} random seeds."
    )
    lines.append("")
    lines.extend([
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\textwidth]{figures/appendix_q_sensitivity_ode.png}",
        r"\caption{ODE sensitivity to the upsampling factor $q$ for EDMD-polynomial preprocessing. Horizontal black lines show the no-upsampling baseline; dashed blue curves show EDMD-polynomial. Markers at $q=5$ indicate the value used in the main benchmark.}",
        r"\label{fig:appendix_q_sensitivity_ode}",
        r"\end{figure}",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\TBL{\caption{ODE $q$ sensitivity for Lorenz--63 and Van der Pol under EDMD-polynomial preprocessing. The sensitivity run uses relative noise 0.03, sparse factor 16, and five random seeds. F1 and practical score are means; coefficient error is the median over valid evaluation records. Boldface marks the best value at reported precision within each system.\label{tab:appendix_ode_q_sensitivity}}}",
        r"{\begin{tabular}{@{}llcccc@{}}\toprule",
        r"\TCH{System} & \TCH{Method} & \TCH{$q$} & \TCH{F1} & \TCH{Coeff. err.} & \TCH{Score} \\ \midrule",
    ])
    first_system = True
    for system, g in ode_sum.sort_values(["system_label", "method", "q"]).groupby("system_label", sort=False):
        if not first_system:
            lines.append(r"\midrule")
        first_system = False
        targets = {
            "f1": g.support_f1_mean.max(),
            "coef": g.coef_error_median.min(),
            "score": g.practical_score_mean.max(),
        }
        for j, (_, row) in enumerate(g.iterrows()):
            q = "--" if row.method == "baseline" else str(int(row.q))
            sys_label = system if j == 0 else ""
            lines.append(
                f"{sys_label} & {row.method_label} & {q} & "
                f"{bold_fmt(row.support_f1_mean, targets['f1'], 'max')} & "
                f"{bold_fmt(row.coef_error_median, targets['coef'], 'min')} & "
                f"{bold_fmt(row.practical_score_mean, targets['score'], 'max')} " + r"\\"
            )
    lines.extend([r"\botrule", r"\end{tabular}}", r"\end{table}", ""])

    lines.extend([
        r"Figure~\ref{fig:appendix_q_sensitivity_ode} and Table~\ref{tab:appendix_ode_q_sensitivity} show that the usefulness of increasing $q$ is system-dependent. For Van der Pol, EDMD-polynomial is already substantially better than the baseline for $q\geq3$, and the default $q=5$ is near the best observed value. For Lorenz--63, the sensitivity is less monotone, which is consistent with the chaotic trajectory and the difficulty of estimating accurate derivatives from sparse noisy measurements. The default $q=5$ gives the lowest median coefficient error among the EDMD-polynomial values tested for Lorenz--63, although the practical score varies only modestly across the grid.",
        "",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\textwidth]{figures/appendix_qr_sensitivity_pde.png}",
        r"\caption{PDE sensitivity of POD-EDMD-RBF preprocessing to the interpolation factor $q$ and POD rank $r$. Each heatmap reports median coefficient error at relative noise 0.03 and PDE sparse factor 8; white stars mark the default values used in the main benchmark. The $q=1$ column corresponds to POD-only denoising without temporal insertion.}",
        r"\label{fig:appendix_qr_sensitivity_pde}",
        r"\end{figure}",
        "",
    ])

    if not pde_best.empty:
        lines.extend([
            r"\begin{table}[H]",
            r"\centering",
            r"\TBL{\caption{Default-versus-best comparison over the PDE $q$--$r$ sensitivity grid for advection--diffusion, Burgers, and Fisher--KPP. The best grid point is selected by mean practical score, with coefficient error used only as a tie-breaker. F1 and practical score are means; coefficient error is the median over five seeds. Boldface marks the better value between the default and best grid point at reported precision.\label{tab:appendix_pde_qr_sensitivity}}}",
            r"{\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lcccccccc@{}}\toprule",
            r"\TCH{System} & \multicolumn{4}{c}{\TCH{Default}} & \multicolumn{4}{c}{\TCH{Best}} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
            r" & \TCH{$(q,r)$} & \TCH{F1} & \TCH{Coeff. err.} & \TCH{Score} & \TCH{$(q,r)$} & \TCH{F1} & \TCH{Coeff. err.} & \TCH{Score} \\ \midrule",
        ])
        for _, row in pde_best.sort_values("system_label").iterrows():
            f1_best = max(row.default_f1_mean, row.best_f1_mean)
            coef_best = min(row.default_coef_error_median, row.best_coef_error_median)
            score_best = max(row.default_score_mean, row.best_score_mean)
            lines.append(
                f"{row.system_label} & ({int(row.default_q)},{int(row.default_rank)}) & "
                f"{bold_fmt(row.default_f1_mean, f1_best, 'max')} & "
                f"{bold_fmt(row.default_coef_error_median, coef_best, 'min')} & "
                f"{bold_fmt(row.default_score_mean, score_best, 'max')} & "
                f"({int(row.best_q)},{int(row.best_rank)}) & "
                f"{bold_fmt(row.best_f1_mean, f1_best, 'max')} & "
                f"{bold_fmt(row.best_coef_error_median, coef_best, 'min')} & "
                f"{bold_fmt(row.best_score_mean, score_best, 'max')} " + r"\\"
            )
        lines.extend([
            r"\botrule",
            r"\end{tabular}%",
            r"}}",
            r"\end{table}",
            "",
        ])
    lines.append(
        r"The PDE grid in Figure~\ref{fig:appendix_qr_sensitivity_pde} and Table~\ref{tab:appendix_pde_qr_sensitivity} supports the use of fixed system-specific ranks in the main study. The $q=1$ column provides the requested POD-only control, while $q>1$ combines the same low-rank spatial projection with temporal upsampling. The default advection--diffusion choice $(q,r)=(5,4)$ is close to the best grid point $(3,4)$; the Burgers default $(5,8)$ is not the lowest-error point in this sensitivity slice but remains in a stable region of the heatmap; and Fisher--KPP has a broad low-error region in which the default rank $r=2$ remains competitive. These results do not imply that $q$ and $r$ are universally optimal. They show that the main conclusions are not based on an isolated unstable choice of interpolation factor or POD rank."
    )
    with open(os.path.join(outdir, "appendix_a_qr_sensitivity.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def configure_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "smoke":
        args.seeds = [0] if args.seeds is None else args.seeds
        args.q_values = [1, 5] if args.q_values is None else args.q_values
        args.r_values = [2, 4] if args.r_values is None else args.r_values
        args.noise = 0.03 if args.noise is None else args.noise
        args.ode_sparse_factor = 16 if args.ode_sparse_factor is None else args.ode_sparse_factor
        args.pde_sparse_factor = 8 if args.pde_sparse_factor is None else args.pde_sparse_factor
        args.pde_systems = ["burgers"] if args.pde_systems is None else args.pde_systems
        args.max_rows = 3000 if args.max_rows is None else args.max_rows
    elif args.preset == "appendix":
        args.seeds = [0, 1, 2, 3, 4] if args.seeds is None else args.seeds
        args.q_values = [1, 3, 5, 7] if args.q_values is None else args.q_values
        args.r_values = [1, 2, 4, 6, 8, 10, 12] if args.r_values is None else args.r_values
        args.noise = 0.03 if args.noise is None else args.noise
        args.ode_sparse_factor = 16 if args.ode_sparse_factor is None else args.ode_sparse_factor
        args.pde_sparse_factor = 8 if args.pde_sparse_factor is None else args.pde_sparse_factor
        args.pde_systems = ["burgers", "fisher_kpp", "advection_diffusion"] if args.pde_systems is None else args.pde_systems
        args.max_rows = 10000 if args.max_rows is None else args.max_rows
    else:
        raise ValueError(args.preset)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Run q/r sensitivity experiments for the Koopman-SINDy manuscript appendix.")
    parser.add_argument("--preset", choices=["smoke", "appendix"], default="appendix")
    parser.add_argument("--outdir", default="results/qr_sensitivity")
    parser.add_argument("--seeds", type=lambda s: parse_int_list(s, []), default=None, help="Comma-separated seeds.")
    parser.add_argument("--q-values", type=lambda s: parse_int_list(s, []), default=None, help="Comma-separated upsampling factors.")
    parser.add_argument("--r-values", type=lambda s: parse_int_list(s, []), default=None, help="Comma-separated POD ranks for PDEs.")
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--ode-sparse-factor", type=int, default=None)
    parser.add_argument("--pde-sparse-factor", type=int, default=None)
    parser.add_argument("--ode-dt", type=float, default=0.01)
    parser.add_argument("--pde-dt", type=float, default=0.01)
    parser.add_argument("--pde-n", type=int, default=64)
    parser.add_argument("--pde-systems", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=None)
    parser.add_argument("--fisher-ic", choices=["default", "front", "rich"], default="default")
    parser.add_argument("--rbf-centers", type=int, default=12)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-ode", action="store_true")
    parser.add_argument("--skip-pde", action="store_true")
    args = configure_preset(parser.parse_args())

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    start = time.time()
    rows: List[Dict[str, object]] = []
    if not args.skip_ode:
        rows.extend(run_ode_sensitivity(args))
    if not args.skip_pde:
        rows.extend(run_pde_sensitivity(args))
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(args.outdir, "qr_sensitivity_raw.csv"), index=False)
    summary = summarize(raw, args.outdir)
    plot_ode_q(summary, args.outdir)
    plot_pde_qr(summary, args.outdir)
    write_appendix_fragment(summary, args.outdir, args)
    elapsed = time.time() - start
    print(f"Wrote {args.outdir}")
    print(f"Finished {len(raw)} cases in {elapsed/60:.2f} min")
    fail = raw[raw.status.astype(str).str.startswith("fail")]
    if len(fail):
        print(f"WARNING: {len(fail)} cases failed; inspect qr_sensitivity_raw.csv", flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()

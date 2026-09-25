#!/usr/bin/env python3
"""Appendix C comparison with classical interpolation techniques.

The script runs a compact set of non-dynamical temporal upsampling baselines
and formats the resulting CSV, Markdown, and LaTeX summaries.  Tuning uses only
the sparse noisy observations.

Publication-level comparison:
  ODE: baseline, linear interpolation, tuned smoothing spline, EDMD-polynomial (optional: --include-gp)
  PDE: baseline, linear interpolation, tuned smoothing spline, POD-EDMD-RBF (optional: --include-gp)

Typical use:
  python classical_interpolation_baselines.py --preset publication \
      --outdir results/classical_interpolation_publication

To regenerate only tables from existing raw CSV files:
  python classical_interpolation_baselines.py --skip-run \
      --outdir results/classical_interpolation_publication
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ODE_METHODS = ["baseline", "linear_interp", "smoothing_spline_cv", "edmd_poly3"]
PDE_METHODS = ["baseline", "linear_interp", "smoothing_spline_cv", "pod_edmd_rbf"]

METHOD_LABELS = {
    "baseline": "Raw FD",
    "linear_interp": "Linear interpolation",
    "smoothing_spline": "Smoothing spline",
    "smoothing_spline_cv": "Tuned smoothing spline",
    "spline": "Smoothing spline",
    "gp_smoothing_cv": "Tuned GP smoothing",
    "gaussian_process_cv": "Tuned GP smoothing",
    "edmd_poly3": "EDMD-polynomial",
    "pod_edmd_rbf": "POD-EDMD-RBF",
}

SYSTEM_LABELS = {
    "lorenz63": "Lorenz--63",
    "vanderpol_mu2": "Van der Pol",
    "burgers": "Burgers",
    "fisher_kpp": "Fisher--KPP",
    "advection_diffusion": "Advection--diffusion",
}


def run_command(cmd: Sequence[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def run_benchmarks(args: argparse.Namespace, root: Path) -> None:
    outdir = Path(args.outdir)
    ode_dir = outdir / "ode"
    pde_dir = outdir / "pde"
    ode_dir.mkdir(parents=True, exist_ok=True)
    pde_dir.mkdir(parents=True, exist_ok=True)

    ode_methods = ODE_METHODS + (["gp_smoothing_cv"] if args.include_gp else [])
    pde_methods = PDE_METHODS + (["gp_smoothing_cv"] if args.include_gp else [])
    if args.run in {"both", "ode"}:
        cmd = [
            sys.executable,
            "koopman_sindy_ode_benchmark.py",
            "--preset", args.preset,
            "--outdir", str(ode_dir),
            "--methods", ",".join(ode_methods),
            "--upsample", str(args.upsample),
            "--checkpoint", str(args.checkpoint),
            "--spline-cv-folds", str(args.spline_cv_folds),
            "--gp-cv-folds", str(args.gp_cv_folds),
        ]
        if args.spline_alpha_grid:
            cmd.extend(["--spline-alpha-grid", str(args.spline_alpha_grid)])
        if args.gp_kernel_grid:
            cmd.extend(["--gp-kernel-grid", str(args.gp_kernel_grid)])
        if args.resume:
            cmd.append("--resume")
        if args.no_progress:
            cmd.append("--no-progress")
        run_command(cmd, cwd=root)

    if args.run in {"both", "pde"}:
        cmd = [
            sys.executable,
            "koopman_sindy_pde_benchmark.py",
            "--preset", args.preset,
            "--outdir", str(pde_dir),
            "--methods", ",".join(pde_methods),
            "--upsample", str(args.upsample),
            "--checkpoint", str(args.checkpoint),
            "--spline-cv-folds", str(args.spline_cv_folds),
            "--systems", "burgers,fisher_kpp,advection_diffusion",
            "--rank-mode", args.pde_rank_mode,
            "--rank-cv-folds", str(args.rank_cv_folds),
            "--gp-cv-folds", str(args.gp_cv_folds),
            "--gp-cv-spatial-points", str(args.gp_cv_spatial_points),
        ]
        if args.rank_grid:
            cmd.extend(["--rank-grid", str(args.rank_grid)])
        if args.spline_alpha_grid:
            cmd.extend(["--spline-alpha-grid", str(args.spline_alpha_grid)])
        if args.gp_kernel_grid:
            cmd.extend(["--gp-kernel-grid", str(args.gp_kernel_grid)])
        if args.resume:
            cmd.append("--resume")
        if args.no_progress:
            cmd.append("--no-progress")
        run_command(cmd, cwd=root)


def read_raw(path: Path, setting: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {setting} raw-results file: {path}")
    df = pd.read_csv(path)
    df.insert(0, "setting", setting)
    return df


def valid_records(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["spline_alpha", "spline_cv_error", "spline_cv_folds", "gp_kernel", "gp_cv_error", "gp_cv_folds", "gp_cv_spatial_points", "pod_rank", "rank_cv_error", "rank_cv_folds"]:
        if col not in df.columns:
            df[col] = np.nan
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No valid records were found in the benchmark outputs.")
    ok["method_label"] = ok["method"].map(METHOD_LABELS).fillna(ok["method"])
    ok["system_label"] = ok["system"].map(SYSTEM_LABELS).fillna(ok["system"])
    return ok


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ok = valid_records(df)
    summary = (
        ok.groupby(["setting", "method", "method_label"], as_index=False)
        .agg(
            support_f1_mean=("support_f1", "mean"),
            coef_error_median=("coef_error", "median"),
            practical_score_mean=("practical_score", "mean"),
            upsample_median=("upsample", "median"),
            pod_rank_median=("pod_rank", "median"),
            spline_alpha_median=("spline_alpha", "median"),
            spline_cv_error_median=("spline_cv_error", "median"),
            rank_cv_error_median=("rank_cv_error", "median"),
            gp_cv_error_median=("gp_cv_error", "median"),
            n_ok=("status", "size"),
        )
    )
    by_system = (
        ok.groupby(["setting", "system", "system_label", "method", "method_label"], as_index=False)
        .agg(
            support_f1_mean=("support_f1", "mean"),
            coef_error_median=("coef_error", "median"),
            practical_score_mean=("practical_score", "mean"),
            upsample_median=("upsample", "median"),
            pod_rank_median=("pod_rank", "median"),
            spline_alpha_median=("spline_alpha", "median"),
            spline_cv_error_median=("spline_cv_error", "median"),
            rank_cv_error_median=("rank_cv_error", "median"),
            gp_cv_error_median=("gp_cv_error", "median"),
            n_ok=("status", "size"),
        )
    )
    base = ok[ok["method"] == "baseline"][[
        "setting", "system", "sparse_factor", "noise", "seed",
        "support_f1", "coef_error", "practical_score",
    ]].rename(columns={
        "support_f1": "support_f1_base",
        "coef_error": "coef_error_base",
        "practical_score": "practical_score_base",
    })
    win_rows = []
    for (setting, method), d_method in ok[ok["method"] != "baseline"].groupby(["setting", "method"]):
        d = d_method.merge(base, on=["setting", "system", "sparse_factor", "noise", "seed"], how="inner")
        win_rows.append({
            "setting": setting,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "paired_cases": len(d),
            "support_f1_win_rate": float((d["support_f1"] > d["support_f1_base"]).mean()) if len(d) else np.nan,
            "coef_error_win_rate": float((d["coef_error"] < d["coef_error_base"]).mean()) if len(d) else np.nan,
            "practical_score_win_rate": float((d["practical_score"] > d["practical_score_base"]).mean()) if len(d) else np.nan,
        })
    winrate = pd.DataFrame(win_rows)
    return summary, by_system, winrate


def ordered_methods(setting: str) -> list[str]:
    return (ODE_METHODS if setting == "ODE" else PDE_METHODS) + ["gp_smoothing_cv"]


def fmt_float(x: float) -> str:
    return "--" if pd.isna(x) else f"{float(x):.3f}"


def latex_fmt(x: float, best: bool) -> str:
    val = fmt_float(x)
    return rf"\textbf{{{val}}}" if best and val != "--" else val


def method_qr_cell(row: pd.Series, method: str, setting: str) -> tuple[str, str]:
    q_cell = "--" if method == "baseline" else str(int(round(float(row.get("upsample_median", 5)))))
    r_val = row.get("pod_rank_median", np.nan)
    if setting != "PDE" or method in {"baseline", "linear_interp", "smoothing_spline", "smoothing_spline_cv", "spline", "gp_smoothing_cv", "gaussian_process_cv"} or pd.isna(r_val) or float(r_val) <= 0:
        r_cell = "--"
    else:
        r_cell = str(int(round(float(r_val))))
    return q_cell, r_cell


def latex_table(by_system: pd.DataFrame, setting: str, label: str, caption: str) -> str:
    d = by_system[by_system["setting"] == setting].copy()
    methods = ordered_methods(setting)
    systems = [s for s in SYSTEM_LABELS if s in set(d["system"])]
    lines: list[str] = []
    lines.append(r"\begin{table}[!htbp]" if setting == "PDE" else r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(rf"\TBL{{\caption{{{caption}\label{{{label}}}}}}}")
    lines.append(r"{%")
    if setting == "ODE":
        lines.append(r"\begin{tabular}{@{}llcccc@{}}")
        lines.append(r"\toprule")
        lines.append(r"\TCH{System} & \TCH{Method} & \TCH{$q$} & \TCH{F1} & \TCH{Coeff. err.} & \TCH{Score} \\")
    else:
        lines.append(r"\begin{tabular}{@{}llccccc@{}}")
        lines.append(r"\toprule")
        lines.append(r"\TCH{System} & \TCH{Method} & \TCH{$q$} & \TCH{$r$} & \TCH{F1} & \TCH{Coeff. err.} & \TCH{Score} \\")
    lines.append(r"\midrule")
    for si, system in enumerate(systems):
        ds = d[d["system"] == system].set_index("method")
        if ds.empty:
            continue
        f1_best = ds["support_f1_mean"].max()
        err_best = ds["coef_error_median"].min()
        score_best = ds["practical_score_mean"].max()
        first = True
        for method in methods:
            if method not in ds.index:
                continue
            row = ds.loc[method]
            system_cell = row["system_label"] if first else ""
            q_cell, r_cell = method_qr_cell(row, method, setting)
            f1 = latex_fmt(row["support_f1_mean"], np.isclose(row["support_f1_mean"], f1_best, rtol=0, atol=5e-4))
            err = latex_fmt(row["coef_error_median"], np.isclose(row["coef_error_median"], err_best, rtol=0, atol=5e-4))
            score = latex_fmt(row["practical_score_mean"], np.isclose(row["practical_score_mean"], score_best, rtol=0, atol=5e-4))
            if setting == "ODE":
                lines.append(rf"{system_cell} & {row['method_label']} & {q_cell} & {f1} & {err} & {score} \\")
            else:
                lines.append(rf"{system_cell} & {row['method_label']} & {q_cell} & {r_cell} & {f1} & {err} & {score} \\")
            first = False
        if si != len(systems) - 1:
            lines.append(r"\midrule")
    lines.append(r"\botrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def markdown_table(by_system: pd.DataFrame, setting: str) -> str:
    d = by_system[by_system["setting"] == setting].copy()
    methods = ordered_methods(setting)
    systems = [s for s in SYSTEM_LABELS if s in set(d["system"])]
    if setting == "ODE":
        lines = [
            f"## {setting} classical interpolation comparison",
            "",
            "| System | Method | q | Mean F1 | Median coeff. err. | Mean score | n |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            f"## {setting} classical interpolation comparison",
            "",
            "| System | Method | q | r | Mean F1 | Median coeff. err. | Mean score | n |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    for system in systems:
        ds = d[d["system"] == system].set_index("method")
        for method in methods:
            if method not in ds.index:
                continue
            row = ds.loc[method]
            q_cell, r_cell = method_qr_cell(row, method, setting)
            if setting == "ODE":
                lines.append(
                    f"| {row['system_label']} | {row['method_label']} | {q_cell} | "
                    f"{fmt_float(row['support_f1_mean'])} | {fmt_float(row['coef_error_median'])} | "
                    f"{fmt_float(row['practical_score_mean'])} | {int(row['n_ok'])} |"
                )
            else:
                lines.append(
                    f"| {row['system_label']} | {row['method_label']} | {q_cell} | {r_cell} | "
                    f"{fmt_float(row['support_f1_mean'])} | {fmt_float(row['coef_error_median'])} | "
                    f"{fmt_float(row['practical_score_mean'])} | {int(row['n_ok'])} |"
                )
    return "\n".join(lines)


def markdown_winrate_table(winrate: pd.DataFrame) -> str:
    """Format the matched win rates without pandas' optional tabulate dependency."""
    lines = [
        "| Setting | Method | Paired cases | F1 win rate | Coefficient-error win rate | Score win rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in winrate.itertuples(index=False):
        lines.append(
            f"| {row.setting} | {row.method_label} | {int(row.paired_cases)} | "
            f"{fmt_float(row.support_f1_win_rate)} | {fmt_float(row.coef_error_win_rate)} | "
            f"{fmt_float(row.practical_score_win_rate)} |"
        )
    return "\n".join(lines)


def build_appendix_fragment(by_system: pd.DataFrame) -> str:
    def descriptive_winners(setting: str) -> str:
        """Describe current per-system winners without frozen dominance claims."""
        blocks = []
        for system in SYSTEM_LABELS:
            frame = by_system[(by_system.setting == setting) & (by_system.system == system)]
            if frame.empty:
                continue
            score_best = frame.practical_score_mean.max()
            error_best = frame.coef_error_median.min()
            score_names = ", ".join(frame.loc[np.isclose(frame.practical_score_mean,score_best,rtol=0,atol=5e-4), "method_label"])
            error_names = ", ".join(frame.loc[np.isclose(frame.coef_error_median,error_best,rtol=0,atol=5e-4), "method_label"])
            blocks.append(f"For {SYSTEM_LABELS[system]}, the highest mean Score is attained by {score_names}; the lowest median coefficient error is attained by {error_names}.")
        return " ".join(blocks) + " Near-ties use the same numerical tolerance as the table boldface. These comparisons describe the tested grid and metrics, rather than establishing a general ordering of preprocessing methods."

    has_gp = "gp_smoothing_cv" in set(by_system["method"])
    if has_gp:
        ode_alt = "Linear interpolation, the validation-tuned smoothing spline, and validation-tuned GP smoothing insert"
        ode_tuning = "The spline smoothing parameter and GP kernel are selected only from sparse noisy observations."
        pde_alt = "The linear, validation-tuned smoothing-spline, and validation-tuned GP techniques apply"
        interp_list = "componentwise smoothing-spline interpolation, componentwise Gaussian-process smoothing, and the main assisted preprocessors"
        tuning_list = "the smoothing-spline parameter, the GP kernel, and the PDE POD rank $r$"
    else:
        ode_alt = "Linear interpolation and the validation-tuned smoothing spline insert"
        ode_tuning = "The spline smoothing parameter is selected only from sparse noisy observations."
        pde_alt = "The linear and validation-tuned smoothing-spline techniques apply"
        interp_list = "componentwise smoothing-spline interpolation, and the main assisted preprocessors"
        tuning_list = "the smoothing-spline parameter and the PDE POD rank $r$"

    ode_caption = (
        "Classical non-dynamical interpolation techniques for the ODE benchmark. "
        f"{ode_alt} the same number of temporal points as the assisted method "
        "but do not estimate a flow map or Koopman operator. "
        f"{ode_tuning} F1 and Score are means, Coeff. err. is the median, "
        "and boldface marks the best value within each system and metric."
    )
    pde_caption = (
        "Classical non-dynamical interpolation techniques and observation-validated Koopman-based preprocessing for the PDE benchmark. "
        f"{pde_alt} separate temporal interpolants at each spatial grid point, with one validation-selected "
        + ("smoothing level or kernel" if has_gp else "smoothing level") + " shared across the field; they do not use POD, DMD, or EDMD. "
        "For POD-EDMD-RBF, the POD rank $r$ is selected by deterministic interior holdout validation on the sparse noisy snapshots only; the displayed $r$ is its median across records. "
        "All non-baseline methods use the same fixed interpolation factor $q$. F1 and Score are means, Coeff. err. is the median, and boldface marks the best value within each system and metric."
    )
    parts = [
        r"\clearpage",
        r"\section{Classical non-dynamical interpolation techniques}",
        r"\label{app:classical_interpolation}",
        "",
        f"This appendix evaluates whether the improvements from Koopman-based preprocessing can be explained by inserting additional time points before sparse regression. The comparison focuses on direct non-dynamical alternatives: the no-upsampling baseline, componentwise piecewise-linear interpolation, {interp_list}. The ODE assisted method is EDMD-polynomial. The PDE assisted method is POD-EDMD-RBF.",
        "",
        r"The ODE comparison uses the main ODE grid in Table~\ref{tab:settings}. All three PDEs in this appendix, including advection--diffusion, use $n_x=64$, base $\Delta t=0.005$, $T=2$, sparse factors $\{4,8,16,32\}$, noise levels $\{0,0.01,0.03,0.05,0.10\}$, eight seeds, and $q=5$. POD-rank validation uses three interior folds, candidate ranks $\{1,2,3,4,6,8\}$, and at most 30 RBF centres. Spline validation uses four interior folds and the smoothing-parameter grid $\{0,10\}\cup\{10^j,3\cdot10^j:j=-8,\ldots,0\}$.",
        "",
        f"The interpolation factor $q$ is fixed across all non-baseline methods rather than tuned by validation, because it controls the output grid resolution used for subsequent derivative estimation. Observation-only validation at the original sparse measurement times cannot identify a preferred dense-grid spacing without using downstream SINDy/PDE-FIND information. In contrast, {tuning_list} directly affect reconstruction of held-out sparse noisy measurements, so these parameters are selected by deterministic interior holdout validation using only the observed sparse noisy trajectories or snapshots.",
        "",
        r"The POD basis and RBF centers are fitted using the training observations only. The EDMD map is fitted using training pairs separated by the original observation interval; pairs touching held-out snapshots or spanning a longer gap are excluded. Each held-out snapshot is predicted from the nearest preceding training anchor using the actual elapsed time. No dense clean trajectory, true coefficient vector, threshold oracle, or downstream SINDy/PDE-FIND score is used to tune any Appendix~C preprocessing parameter. The appendix therefore provides a fair check against the most direct alternative explanation: ordinary temporal interpolation plus classical smoothing. The Appendix-C advection--diffusion comparison follows this grid, which differs from the main advection--diffusion experiment reported in Table~\ref{tab:pde_system}.",
        "",
        latex_table(by_system, "ODE", "tab:appendix_c_ode_classical", ode_caption),
        "",
        descriptive_winners("ODE"),
        "",
        latex_table(by_system, "PDE", "tab:appendix_c_pde_classical", pde_caption),
        "",
        descriptive_winners("PDE"),
        "",
    ]
    return "\n".join(parts)


def write_outputs(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    frames = []
    ode_raw = outdir / "ode" / "ode_raw_results.csv"
    pde_raw = outdir / "pde" / "pde_raw_results.csv"
    if args.run in {"both", "ode"} or ode_raw.exists():
        frames.append(read_raw(ode_raw, "ODE"))
    if args.run in {"both", "pde"} or pde_raw.exists():
        frames.append(read_raw(pde_raw, "PDE"))
    if not frames:
        raise FileNotFoundError(f"No raw benchmark files found under {outdir}")
    raw_all = pd.concat(frames, ignore_index=True)
    summary, by_system, winrate = summarize(raw_all)
    summary.to_csv(outdir / "classical_interpolation_summary.csv", index=False)
    by_system.to_csv(outdir / "classical_interpolation_by_system.csv", index=False)
    winrate.to_csv(outdir / "classical_interpolation_winrate_vs_baseline.csv", index=False)

    md_parts = [
        "# Appendix C: classical non-dynamical interpolation techniques",
        "",
        markdown_table(by_system, "ODE") if (by_system["setting"] == "ODE").any() else "",
        "",
        markdown_table(by_system, "PDE") if (by_system["setting"] == "PDE").any() else "",
        "",
        "## Win rate versus no-upsampling baseline",
        "",
        markdown_winrate_table(winrate),
        "",
    ]
    (outdir / "appendix_c_classical_interpolation.md").write_text("\n".join(md_parts))
    (outdir / "appendix_c_classical_interpolation.tex").write_text(build_appendix_fragment(by_system))
    print(f"\nWrote Appendix C outputs to: {outdir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["quick", "publication"], default="publication")
    parser.add_argument("--outdir", default="results/classical_interpolation_publication")
    parser.add_argument("--run", choices=["both", "ode", "pde"], default="both")
    parser.add_argument("--include-gp", action="store_true", help="Run optional GP extension; not part of the archived four-method comparison.")
    parser.add_argument("--upsample", type=int, default=5)
    parser.add_argument("--checkpoint", type=int, default=25)
    parser.add_argument(
        "--spline-alpha-grid",
        type=str,
        default=None,
        help="Comma-separated dimensionless smoothing levels for smoothing_spline_cv.",
    )
    parser.add_argument(
        "--spline-cv-folds",
        type=int,
        default=4,
        help="Number of deterministic interior folds for smoothing_spline_cv tuning.",
    )
    parser.add_argument(
        "--gp-kernel-grid",
        type=str,
        default=None,
        help="Comma-separated GP kernel specs family:length_scale:noise for gp_smoothing_cv.",
    )
    parser.add_argument(
        "--gp-cv-folds",
        type=int,
        default=4,
        help="Number of deterministic interior folds for gp_smoothing_cv tuning.",
    )
    parser.add_argument(
        "--gp-cv-spatial-points",
        type=int,
        default=8,
        help="Number of deterministic spatial grid points used for PDE GP-kernel validation.",
    )
    parser.add_argument(
        "--pde-rank-mode",
        choices=["fixed", "system", "energy", "cv"],
        default="cv",
        help="PDE POD-rank strategy for POD-EDMD-RBF in Appendix C.",
    )
    parser.add_argument(
        "--rank-grid",
        type=str,
        default="1,2,3,4,6,8",
        help="Comma-separated candidate POD ranks for observation-only PDE rank CV.",
    )
    parser.add_argument("--rank-cv-folds", type=int, default=3, help="Number of deterministic interior folds for PDE POD-rank tuning.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Only format existing raw CSV outputs.")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    if not args.skip_run:
        run_benchmarks(args, root=root)
    write_outputs(args)


if __name__ == "__main__":
    main()

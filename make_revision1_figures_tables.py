#!/usr/bin/env python3
"""Reproduce manuscript reporting from the current, corrected benchmark records.

No discovery benchmark is rerun. Only two representative interpolations are
reconstructed for the data illustration. Run from any directory:
  OPENBLAS_NUM_THREADS=1 python make_revision1_figures_tables.py

Mean F1 and Score retain the original available-record estimand. Their standard
errors cluster all noise/sparsity records by seed (CR1 sandwich for an intercept).
Coefficient-error curves display arithmetic mean and sample SD of the valid
records at each x value; SD is dispersion across the other grid axis and seeds,
not a standard error or confidence interval. Outliers are never removed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LABELS = {
    "baseline": "Raw FD", "edmd_poly3": "EDMD-polynomial",
    "edmd_rbf": "EDMD-RBF", "pydmd_optdmd": "optDMD",
    "optdmd": "optDMD", "pod_edmd_rbf": "POD-EDMD-RBF",
}
SYSTEMS = {
    "lorenz63": "Lorenz--63", "vanderpol_mu2": "Van der Pol",
    "burgers": "Burgers", "fisher_kpp": "Fisher--KPP",
    "advection_diffusion": "Advection--diffusion",
}
ORDER = {
    "lorenz63": ["baseline", "edmd_poly3", "edmd_rbf"],
    "vanderpol_mu2": ["baseline", "edmd_poly3", "edmd_rbf"],
    "burgers": ["baseline", "pydmd_optdmd", "pod_edmd_rbf"],
    "fisher_kpp": ["baseline", "pydmd_optdmd", "pod_edmd_rbf"],
    "advection_diffusion": ["baseline", "optdmd", "pod_edmd_rbf"],
}
COLORS = ["#222222", "#0072B2", "#D55E00"]
MARKERS = ["o", "s", "^"]
KEYS = ["system", "sparse_factor", "noise", "seed"]
CONTROL_RANKS = {"burgers": 8, "fisher_kpp": 5, "advection_diffusion": 4}


def read_raw() -> pd.DataFrame:
    paths = [
        "results/ode_publication/ode_raw_results.csv",
        "results/pde_publication/pde_raw_results.csv",
        "results/advection_diffusion_benchmark/advection_diffusion_raw.csv",
    ]
    parts = []
    for path in paths:
        df = pd.read_csv(ROOT / path)
        df["source_file"] = path
        parts.append(df)
    raw = pd.concat(parts, ignore_index=True)
    assert not raw.duplicated(KEYS + ["method"]).any(), "Duplicate evaluation record"
    raw["valid"] = (raw.status == "ok") & np.isfinite(raw[["support_f1", "coef_error", "practical_score"]]).all(axis=1)
    return raw


def clustered_se(frame: pd.DataFrame, column: str) -> float:
    """CR1 SE of record-weighted mean; balanced case = SD(seed means)/sqrt(G)."""
    n, g = len(frame), frame.seed.nunique()
    if g < 2:
        return float("nan")
    residuals = frame[column] - frame[column].mean()
    sums = residuals.groupby(frame.seed).sum().to_numpy()
    return float(np.sqrt(g / (g - 1) * np.sum(sums**2)) / n)


def statistics(raw: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, wins = [], []
    for system, methods in ORDER.items():
        for method in methods:
            records = raw[(raw.system == system) & (raw.method == method)]
            ok = records[records.valid]
            rows.append(dict(system=system, method=method, n_valid=len(ok), n_total=len(records),
                             n_seeds=ok.seed.nunique(),
                             f1_mean=ok.support_f1.mean(), f1_se=clustered_se(ok, "support_f1"),
                             coef_error_median=ok.coef_error.median(),
                             score_mean=ok.practical_score.mean(), score_se=clustered_se(ok, "practical_score")))
            if method != "baseline":
                base = raw[(raw.system == system) & (raw.method == "baseline") & raw.valid]
                paired = ok.merge(base, on=KEYS, suffixes=("", "_base"), validate="one_to_one")
                n = len(paired)
                wc = int((paired.coef_error < paired.coef_error_base).sum())
                ws = int((paired.practical_score > paired.practical_score_base).sum())
                wins.append(dict(system=system, method=method, paired_n=n,
                                 coefficient_wins=wc, coefficient_win_percent=100 * wc / n,
                                 score_wins=ws, score_win_percent=100 * ws / n))
    stats, win = pd.DataFrame(rows), pd.DataFrame(wins)
    stats.to_csv(out / "main_table_seed_cluster_se.csv", index=False)
    win.to_csv(out / "paired_win_rates_by_system.csv", index=False)
    raw[~raw.valid].to_csv(out / "invalid_main_records.csv", index=False)
    raw[raw.valid].sort_values("coef_error",ascending=False).groupby(["system","method"]).head(1).to_csv(out / "main_largest_errors.csv",index=False)
    previous = ROOT / "results/pre_revision_archive/reporting_before_complex_correction/main_table_seed_cluster_se.csv"
    if previous.exists():
        comparison = pd.read_csv(previous).merge(stats,on=["system","method"],suffixes=("_before","_corrected"),validate="one_to_one")
        for metric in ["f1_mean","coef_error_median","score_mean"]:
            comparison[metric+"_change"] = comparison[metric+"_corrected"]-comparison[metric+"_before"]
        comparison.to_csv(out / "main_table_complex_correction_changes.csv",index=False)
    return stats, win


def pod_control_reporting(raw_path: Path, tables: Path, out: Path) -> pd.DataFrame:
    """Report paired q=1/q=5 controls at fixed main ranks; no experiments run.

    Raw FD is retained as a third matched control. Only complete, finite triples
    are reported, so method summaries and paired differences use identical seed
    keys. The sensitivity grid supplies the numerical values; no best q or rank
    is selected for this attribution comparison.
    """
    raw = pd.read_csv(raw_path)
    keys = ["system", "sparse_factor", "noise", "seed"]
    required = keys + ["method", "q", "pod_rank", "status", "support_f1", "coef_error", "practical_score"]
    assert set(required).issubset(raw.columns), "Incomplete q/r sensitivity schema"
    assert not raw.duplicated(keys + ["method", "q", "pod_rank"]).any()
    raw = raw[(raw.setting == "PDE")].copy()
    raw["valid"] = (raw.status == "ok") & np.isfinite(raw[["support_f1", "coef_error", "practical_score"]]).all(axis=1)
    rows, paired_rows, retained = [], [], []
    control_labels = {"raw": "Raw FD", "pod_only": "POD only", "pod_edmd": "POD+EDMD-RBF"}
    for system, rank in CONTROL_RANKS.items():
        block = raw[raw.system == system]
        selections = {
            "raw": block[block.method == "baseline"],
            "pod_only": block[(block.method == "pod_edmd_rbf") & (block.q == 1) & (block.pod_rank == rank)],
            "pod_edmd": block[(block.method == "pod_edmd_rbf") & (block.q == 5) & (block.pod_rank == rank)],
        }
        assert all(len(frame) for frame in selections.values()), f"Missing fixed-rank controls for {system}"
        common = None
        for frame in selections.values():
            index = pd.MultiIndex.from_frame(frame.loc[frame.valid, keys])
            common = index if common is None else common.intersection(index)
        assert common is not None and len(common) >= 2, f"Too few matched controls for {system}"
        data = {}
        for control, frame in selections.items():
            selected = frame.set_index(keys).loc[common].reset_index().sort_values(keys)
            # The promoted control is one fixed noise/sampling slice, not a
            # mixture that would obscure within-seed differences.
            assert selected.noise.nunique() == selected.sparse_factor.nunique() == 1
            assert selected.seed.nunique() == len(selected)
            selected["control"] = control
            retained.append(selected)
            data[control] = selected
            rows.append(dict(system=system, fixed_pod_rank=rank, control=control,
                q=1 if control != "pod_edmd" else 5, n_paired=len(selected),
                n_available=int(frame.valid.sum()), n_total=len(frame),
                noise=float(selected.noise.iloc[0]), sparse_factor=int(selected.sparse_factor.iloc[0]),
                f1_mean=selected.support_f1.mean(), f1_se=clustered_se(selected, "support_f1"),
                coef_error_mean=selected.coef_error.mean(), coef_error_se=clustered_se(selected, "coef_error"),
                coef_error_median=selected.coef_error.median(),
                score_mean=selected.practical_score.mean(), score_se=clustered_se(selected, "practical_score")))
        q1, q5 = data["pod_only"], data["pod_edmd"]
        assert q1[keys].equals(q5[keys])
        diff = q5[keys].copy()
        for metric in ["support_f1", "coef_error", "practical_score"]:
            diff[metric] = q5[metric].to_numpy() - q1[metric].to_numpy()
        paired_rows.append(dict(system=system, fixed_pod_rank=rank, n_paired=len(diff),
            delta_f1_mean=diff.support_f1.mean(), delta_f1_se=clustered_se(diff, "support_f1"),
            delta_coef_error_mean=diff.coef_error.mean(), delta_coef_error_se=clustered_se(diff, "coef_error"),
            delta_score_mean=diff.practical_score.mean(), delta_score_se=clustered_se(diff, "practical_score"),
            coefficient_wins=int((diff.coef_error < 0).sum()), coefficient_ties=int((diff.coef_error == 0).sum()),
            score_wins=int((diff.practical_score > 0).sum()), score_ties=int((diff.practical_score == 0).sum())))
    summary, paired = pd.DataFrame(rows), pd.DataFrame(paired_rows)
    pd.concat(retained, ignore_index=True).to_csv(out / "pod_controls_matched_raw.csv", index=False)
    summary.to_csv(out / "pod_controls_summary.csv", index=False)
    paired.to_csv(out / "pod_controls_paired_differences.csv", index=False)
    caption = (r"Matched fixed-rank control for spatial POD denoising and temporal insertion. "
        r"Raw FD differentiates the noisy observations; POD only reconstructs them at the same times ($q=1$); "
        r"POD+EDMD-RBF inserts points at $q=5$ after the identical rank-$r$ projection. "
        r"The representative sensitivity setting uses 3\% noise, sparse factor eight, $n_x=64$, "
        r"base $\Delta t=0.01$, $T=2$, and 12 RBF centres; $r=8,5,4$ for Burgers, Fisher--KPP, and advection--diffusion. "
        r"$N$ counts complete matched seed triples. F1 and Score are means $\pm$ standard error over those seeds; "
        r"coefficient error is the median. Ranks are fixed rather than selected from the sensitivity grid. "
        r"Boldface marks the best displayed value within each system")
    lines = [r"\begin{table}[!htbp]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
        r"\TBL{\caption{" + caption + r"\label{tab:pod_attribution}}}",
        r"{\begin{tabular}{@{}llrccc@{}}\toprule",
        r"\TCH{System} & \TCH{Control} & \TCH{$N$} & \TCH{F1} & \TCH{Coeff. err.} & \TCH{Score} \\\midrule"]
    for si, system in enumerate(CONTROL_RANKS):
        block = summary[summary.system == system]
        for ri, row in enumerate(block.itertuples()):
            label = SYSTEMS[system]
            if system == "advection_diffusion":
                label = r"\makecell[l]{Advection--\\diffusion}"
            prefix = rf"\multirow{{3}}{{*}}{{{label}}}" if ri == 0 else ""
            cells = []
            for value_col, se_col, maximize in [("f1_mean", "f1_se", True),
                    ("coef_error_median", None, False), ("score_mean", "score_se", True)]:
                value = getattr(row, value_col)
                best = block[value_col].max() if maximize else block[value_col].min()
                cell = f"{value:.3f}" + (rf"\pm {getattr(row, se_col):.3f}" if se_col else "")
                if f"{value:.3f}" == f"{best:.3f}":
                    cell = r"\mathbf{" + cell + "}"
                cells.append("$" + cell + "$")
            lines.append(prefix + f" & {control_labels[row.control]} & {row.n_paired} & "
                + " & ".join(cells) + r" \\")
        if si < len(CONTROL_RANKS)-1:
            lines.append(r"\midrule")
    lines.extend([r"\botrule\end{tabular}}", r"\end{table}", ""])
    (tables / "revision_pod_attribution.tex").write_text("\n".join(lines))
    return summary


def reporting_manifest(raw: pd.DataFrame, qr_path: Path, out: Path, interpolation: bool) -> None:
    """Fingerprint every raw input and the reconstruction code used for figures."""
    paths = [ROOT / path for path in raw.source_file.unique()]
    paths += [qr_path, Path(__file__), ROOT / "koopman_sindy_ode_benchmark.py", ROOT / "koopman_sindy_pde_benchmark.py",
        ROOT / "koopman_propagation.py", ROOT / "sync_revision_outputs.py", ROOT / "koopman_sindy_qr_sensitivity.py",
        ROOT / "classical_interpolation_baselines.py",
        ROOT / "results/upsampling_strategy_publication/upsampling_strategy_raw.csv",
        ROOT / "results/model_selection_publication/model_selection_selected_by_equation.csv",
        ROOT / "results/classical_interpolation_publication/ode/ode_raw_results.csv",
        ROOT / "results/classical_interpolation_publication/pde/pde_raw_results.csv",
        ROOT / "results/revision_tv_high_noise/summary_high_noise.csv"]
    files = {os.path.relpath(path, ROOT): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (out / "reporting_manifest.json").write_text(json.dumps(dict(
        sha256=files, path_base="repository root; ../ components permit external inputs",
        main_records=len(raw), main_valid_records=int(raw.valid.sum()),
        representative_interpolation_regenerated=interpolation,
        uncertainty="Seed-cluster CR1 standard error; regime figures show record SD",
        control_estimand="Complete matched seed triples, fixed representative q/r setting"), indent=2)+"\n")


def latex_tables(stats: pd.DataFrame, wins: pd.DataFrame, tables: Path) -> None:
    for group, systems in [("ode", list(SYSTEMS)[:2]), ("pde", list(SYSTEMS)[2:])]:
        seed_text = "ten seeds" if group == "ode" else "eight seeds for Burgers and Fisher--KPP and five for advection--diffusion"
        invalid_text = "" if group == "ode" else " Unsuccessful optDMD runs are excluded rather than assigned a score."
        caption = (f"{group.upper()} benchmark results. F1 and Score are record-weighted means $\\pm$ seed-cluster standard error "
                   f"({seed_text}); the noise/sparsity grid is held fixed. Coeff. err. is the median. "
                   "$N$ is the number of valid records." + invalid_text + " " +
                   "Boldface denotes the best displayed mean or median.")
        text = [r"\begin{table}[!htbp]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
                r"\TBL{\caption{" + caption + r"\label{tab:" + group + r"_system}}}",
                r"{\begin{tabular}{@{}llrccc@{}}\toprule",
                r"\TCH{System} & \TCH{Method} & \TCH{$N$} & \TCH{F1} & \TCH{Coeff. err.} & \TCH{Score} \\\midrule"]
        for si, system in enumerate(systems):
            block = stats[stats.system == system]
            for ri, row in enumerate(block.itertuples()):
                system_label = SYSTEMS[system]
                if system == "advection_diffusion":
                    system_label = r"\makecell[l]{Advection--\\diffusion}"
                prefix = rf"\multirow{{3}}{{*}}{{{system_label}}}" if ri == 0 else ""
                cells = []
                for val_col, se_col, maximize in [("f1_mean", "f1_se", True), ("coef_error_median", None, False), ("score_mean", "score_se", True)]:
                    val = getattr(row, val_col)
                    best = block[val_col].max() if maximize else block[val_col].min()
                    # Compare displayed values, preserving the original tied 0.039 medians.
                    bold = round(val, 3) == round(best, 3)
                    cell = f"{val:.3f}" + (rf"\pm {getattr(row, se_col):.3f}" if se_col else "")
                    cells.append("$" + (r"\mathbf{" + cell + "}" if bold else cell) + "$")
                text.append(prefix + f" & {LABELS[row.method]} & {row.n_valid} & " + " & ".join(cells) + r" \\")
            if si < len(systems) - 1:
                text.append(r"\midrule")
        text.extend([r"\botrule\end{tabular}}", r"\end{table}", ""])
        (tables / f"revision_{group}_system.tex").write_text("\n".join(text))
    text = [r"\begin{table}[!htbp]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}",
            r"\TBL{\caption{Strict improvement over the raw baseline on matched valid records. $N$ is the number of matched system/sparsity/noise/seed records, and the last two columns are percentages with lower coefficient error or higher Score. Ties are not wins. Percentages describe this benchmark grid, not independent repeated trials. Boldface marks the best displayed value within each system.\label{tab:paired_wins}}}",
            r"{\begin{tabular}{@{}llrrr@{}}\toprule",
            r"\TCH{System} & \TCH{Method} & \TCH{$N$} & \TCH{Error wins (\%)} & \TCH{Score wins (\%)} \\\midrule"]
    for si, system in enumerate(SYSTEMS):
        block = wins[wins.system == system]
        for ri, row in enumerate(block.itertuples()):
            label = SYSTEMS[system]
            if system == "advection_diffusion":
                label = r"\makecell[l]{Advection--\\diffusion}"
            prefix = rf"\multirow{{2}}{{*}}{{{label}}}" if ri == 0 else ""
            cells = []
            for column in ["coefficient_win_percent", "score_win_percent"]:
                cell = f"{getattr(row, column):.1f}"
                if cell == f"{block[column].max():.1f}":
                    cell = r"$\mathbf{" + cell + "}$"
                cells.append(cell)
            text.append(prefix + f" & {LABELS[row.method]} & {row.paired_n} & "
                + " & ".join(cells) + r" \\")
        if si < len(SYSTEMS) - 1:
            text.append(r"\midrule")
    text.extend([r"\botrule\end{tabular}}", r"\end{table}", ""])
    (tables / "revision_paired_wins.tex").write_text("\n".join(text))


def style() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 11,
                         "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.dpi": 250, "pdf.fonttype": 42})


def save(fig: plt.Figure, directory: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(directory / (name + "." + ext), bbox_inches="tight")
    plt.close(fig)


def performance_figures(raw: pd.DataFrame, figures: Path, out: Path) -> None:
    ok = raw[raw.valid]
    summaries = []
    for axis in ["noise", "sparse_factor"]:
        frame = ok.groupby(["system", "method", axis]).agg(
            mean=("coef_error", "mean"), sd=("coef_error", "std"), n=("coef_error", "size"),
            median=("coef_error", "median"), maximum=("coef_error", "max")).reset_index()
        frame["axis"] = axis
        frame["axis_value"] = frame[axis]
        summaries.append(frame)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / "coefficient_error_regime_curves.csv", index=False)
    for group, systems in [("ode", list(SYSTEMS)[:2]), ("pde", list(SYSTEMS)[2:])]:
        fig, axes = plt.subplots(len(systems), 2, figsize=(8.0, 2.6 * len(systems)), squeeze=False)
        for si, system in enumerate(systems):
            ymax = 0.0
            for ci, axis in enumerate(["noise", "sparse_factor"]):
                ax = axes[si, ci]
                for mi, method in enumerate(ORDER[system]):
                    sub = summary[(summary.system == system) & (summary.method == method) & (summary.axis == axis)].sort_values("axis_value")
                    x, y, sd = sub.axis_value.to_numpy(), sub["mean"].to_numpy(), sub.sd.to_numpy()
                    if axis == "noise":
                        x = x * 100
                    lo, hi = np.maximum(0, y - sd), y + sd
                    # Slightly horizontal offset errorbar caps are unnecessary with different markers.
                    ax.plot(x, y, color=COLORS[mi], marker=MARKERS[mi], ms=4.5, lw=1.2, label=LABELS[method])
                    ax.fill_between(x, lo, hi, color=COLORS[mi], alpha=0.11, linewidth=0)
                    ax.plot(x, hi, color=COLORS[mi], lw=0.7, ls=":", alpha=.8)
                    ax.plot(x, lo, color=COLORS[mi], lw=0.7, ls=":", alpha=.8)
                    ymax = max(ymax, float(hi.max()))
                ax.set_yscale("symlog", linthresh=0.01, linscale=0.6)
                ax.grid(axis="y", color="#d9d9d9", lw=.5)
                ax.set_xlabel("Relative noise (%)" if axis == "noise" else "Temporal sparse factor")
                ax.set_xticks(sorted(x))
                ax.set_title(f"({chr(97 + 2*si + ci)}) {SYSTEMS[system].replace('--', '–')}", loc="left")
                if ci == 0:
                    ax.set_ylabel("Coefficient error")
            for ax in axes[si]:
                ax.set_ylim(0, ymax * 1.18)
                ticks = [0, .01, .1, 1, 10, 100, 1000]
                ax.set_yticks([v for v in ticks if v < ymax * 1.18])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.52, 1.0), fontsize=10)
        fig.subplots_adjust(left=.105, right=.99, top=.915 if group == "ode" else .94, bottom=.09 if group == "ode" else .065, hspace=.55, wspace=.24)
        save(fig, figures, f"revision_{group}_noise_sparsity")


def interpolation_figure(figures: Path, out: Path) -> None:
    import koopman_sindy_ode_benchmark as ode
    import koopman_sindy_pde_benchmark as pde
    sf, noise, seed, q = 16, .03, 0, 5
    records = []
    examples = []
    sys = ode.vanderpol_system()
    t, y = ode.integrate_system(sys, .005)
    obs_i = np.arange(0, len(t), sf)
    assert obs_i[-1] == len(t) - 1
    rng = np.random.default_rng(seed + 1000*sf + 100000*int(noise*1000))
    obs = ode.add_noise(y[obs_i], noise, rng)
    tn = ode.make_tnew(t[obs_i], q)
    rec = ode.edmd_reconstruct(obs, t[obs_i], tn, "poly", 3, sys.var_names, rng)
    examples.append(("Van der Pol: EDMD-polynomial", t, y[:, 0], t[obs_i], obs[:, 0], tn, rec[:, 0], (2.40, 3.20), r"$x(t)$"))
    sys = pde.burgers_system(n=64)
    t, y = pde.integrate_pde(sys, .005)
    obs_i = np.arange(0, len(t), sf)
    assert obs_i[-1] == len(t) - 1
    rng = np.random.default_rng(seed + 1000*sf + 100000*int(noise*1000))
    obs = pde.add_noise(y[obs_i], noise, rng)
    tn = pde.make_tnew(t[obs_i], q)
    rec = pde.pod_edmd_reconstruct(obs, t[obs_i], tn, 8, "rbf", rng, 30)
    ix = 16  # x = pi/2, fixed a priori.
    examples.append((r"Burgers: POD-EDMD-RBF", t, y[:, ix], t[obs_i], obs[:, ix], tn, rec[:, ix], (.40, 1.20), r"$u(t,\pi/2)$"))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.35))
    for panel, (ax, ex) in enumerate(zip(axes, examples)):
        title, t, clean, to, obs, tn, rec, lim, ylabel = ex
        ax.plot(t, clean, color="#222222", lw=1.6, zorder=1)
        ax.scatter(to, obs, marker="o", s=25, facecolor="white", edgecolor="#D55E00", linewidth=1.1, zorder=4)
        inserted = np.arange(len(tn)) % q != 0
        ax.scatter(tn[inserted], rec[inserted], marker=".", s=19, color="#0072B2", zorder=3)
        ax.scatter(tn[~inserted], rec[~inserted], marker="x", s=21, linewidth=1.0, color="#0072B2", zorder=5)
        # Keep each local-reset segment separate: do not draw a smooth bridge to
        # the next anchor, which would falsely imply continuous interpolation.
        for j in range(len(to) - 1):
            sl = slice(q*j, q*j+q)
            ax.plot(tn[sl], rec[sl], color="#0072B2", lw=.8, alpha=.8, zorder=2)
        ax.set_xlim(*lim)
        values = np.r_[clean[(t >= lim[0]) & (t <= lim[1])], obs[(to >= lim[0]) & (to <= lim[1])], rec[(tn >= lim[0]) & (tn <= lim[1])]]
        margin = np.ptp(values)*.12
        ax.set_ylim(values.min()-margin, values.max()+margin)
        ax.set_title(f"({chr(97+panel)}) {title}", loc="left")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", lw=.5)
        for kind, times, vals in [("clean_reference", t, clean), ("noisy_observation", to, obs), ("reconstruction", tn, rec)]:
            records.extend({"panel": panel+1, "kind": kind, "time": tt, "value": value} for tt, value in zip(times, vals))
    handles = [Line2D([], [], color="#222222", lw=1.6, label="Clean reference"),
               Line2D([], [], color="#D55E00", marker="o", mfc="white", ls="none", label="Noisy observations"),
               Line2D([], [], color="#0072B2", marker=".", ls="none", label="Inserted points"),
               Line2D([], [], color="#0072B2", marker="x", ls="none", label="Reset values")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5, bbox_to_anchor=(.52, -.015))
    fig.subplots_adjust(left=.09, right=.99, top=.89, bottom=.31, wspace=.29)
    save(fig, figures, "revision_interpolated_data")
    pd.DataFrame(records).to_csv(out / "representative_interpolation_data.csv", index=False)
    (out / "representative_interpolation_settings.json").write_text(json.dumps(dict(
        seed=seed, sparse_factor=sf, relative_noise=noise, upsample=q, base_dt=.005,
        vanderpol=dict(degree=3, fitting_window=[0, 20], displayed_window=[2.4, 3.2]),
        burgers=dict(nx=64, rank=8, rbf_centers=30, spatial_index=16, fitting_window=[0, 2], displayed_window=[.4, 1.2])), indent=2)+"\n")


def figure_fragments(tables: Path) -> None:
    interpolation = r"""\begin{figure}[!htbp]
\centering
\includegraphics[width=\textwidth]{figures/revision_interpolated_data.pdf}
\caption{Representative within-window reconstructions: (a) Van der Pol with polynomial EDMD and (b) a Burgers time trace at $x=\pi/2$ with POD-EDMD-RBF (rank eight). Both use 3\% noise, sparse factor 16, base $\Delta t=0.005$, $q=5$, and seed zero. Curves are fitted on the complete observed windows ($[0,20]$ and $[0,2]$); the panels zoom into fixed subintervals. Four points are inserted between each observation pair. Blue crosses show resets, which coincide with the noisy observations in (a) and with their POD projections in (b). Blue segments are drawn separately between resets: the procedure does not enforce temporal continuity across anchors. The clean reference is shown for evaluation only}
\label{fig:interpolated_data}
\end{figure}
"""
    (tables / "revision_interpolated_data_figure.tex").write_text(interpolation)
    for group in ["ode", "pde"]:
        counts = ("Each noise point pools 40 records per method (four sparse factors and ten seeds); each sparse-factor point pools 50 records (five noise levels and ten seeds)." if group == "ode" else
                  r"Burgers and Fisher--KPP have 32 records per noise point and 40 per sparse-factor point, except for one missing optDMD record in Burgers at 10\% noise/sparse factor 32 and one in Fisher--KPP at 3\%/16. Advection--diffusion has 15 records per noise point and 20 per sparse-factor point.")
        caption = (f"{group.upper()} coefficient error versus relative noise (left) and temporal sparse factor (right). "
                   "Markers/solid lines show arithmetic means; shading and dotted boundaries show one sample standard deviation across valid records, "
                   "pooling seeds and the other grid axis. The lower boundary is truncated at zero. "
                   "The vertical axis is linear from zero to 0.01 and logarithmic above 0.01; all finite outliers are retained. "
                   "These bands describe record dispersion, not confidence intervals. " + counts)
        size = r"width=\textwidth,height=0.67\textheight,keepaspectratio" if group == "pde" else r"width=\textwidth"
        placement = "!htbp"
        text = (r"\begin{figure}[" + placement + "]\n" + r"\centering" + "\n" +
                r"\includegraphics["+size+r"]{figures/revision_"+group+r"_noise_sparsity.pdf}" + "\n" +
                r"\caption{"+caption.rstrip(".")+"}\n" + r"\label{fig:"+group+r"_noise_sparsity}"+"\n"+r"\end{figure}"+"\n")
        (tables / f"revision_{group}_regime_figure.tex").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latex-dir", type=Path, default=ROOT / "generated_assets",
                        help="Asset destination; relative paths use the invoking working directory.")
    parser.add_argument("--skip-interpolation", action="store_true",
                        help="Reuse the included illustration instead of reconstructing it.")
    parser.add_argument("--pod-control-raw", type=Path, default=ROOT / "results/qr_sensitivity/qr_sensitivity_raw.csv")
    args = parser.parse_args()
    args.latex_dir = args.latex_dir.expanduser().resolve()
    args.pod_control_raw = args.pod_control_raw.expanduser().resolve()
    saved_interpolation = ROOT / "figures/revision_interpolated_data.pdf"
    if args.skip_interpolation and not saved_interpolation.is_file():
        parser.error(f"--skip-interpolation requires the saved illustration: {saved_interpolation}")
    out, figures, tables = ROOT / "results/revision1_reporting", args.latex_dir / "figures", args.latex_dir / "tables"
    for directory in (out, figures, tables):
        directory.mkdir(parents=True, exist_ok=True)
    raw = read_raw()
    stats, wins = statistics(raw, out)
    latex_tables(stats, wins, tables)
    style()
    performance_figures(raw, figures, out)
    if not args.skip_interpolation:
        interpolation_figure(figures, out)
    elif saved_interpolation != figures / saved_interpolation.name:
        shutil.copy2(saved_interpolation, figures / saved_interpolation.name)
    figure_fragments(tables)
    control = pod_control_reporting(args.pod_control_raw, tables, out)
    reporting_manifest(raw, args.pod_control_raw, out, not args.skip_interpolation)
    print(stats.to_string(index=False))
    print(wins.to_string(index=False))
    print(control.to_string(index=False))
    print(f"Saved reporting artifacts to {out}")


if __name__ == "__main__":
    main()

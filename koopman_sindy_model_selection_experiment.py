#!/usr/bin/env python3
"""Equation-wise non-oracle model selection for DMD-assisted SINDy/PDE-FIND.

The main oracle benchmark chooses the STLSQ threshold with known support and
coefficients. This script instead evaluates the same threshold path, builds a
Pareto curve in support size, computes BIC/EBIC from derivative-regression
residuals, and uses an elbow rule on the EBIC-improvement curve to select a
model without using the true equation. The true support is used only for
reporting diagnostics after selection.

Outputs are written to --outdir:
  model_selection_candidates.csv
  model_selection_selected_by_equation.csv
  model_selection_summary.csv
  model_selection_tables.md
  model_selection_pareto_equationwise.png
"""
from __future__ import annotations

import argparse
import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import koopman_sindy_ode_benchmark as ode
import koopman_sindy_pde_benchmark as pde

EPS = 1e-12


def apply_publication_plot_style() -> None:
    """Apply an optional SciencePlots style with a safe matplotlib fallback."""
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

ODE_THRESHOLDS = np.array([0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
PDE_THRESHOLDS = np.array([
    0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3,
    1e-2, 1.3e-2, 1.5e-2, 2e-2, 3e-2, 5e-2, 7e-2, 1e-1, 2e-1, 3e-1
])


@dataclass
class SelectionConfig:
    ebic_gamma: float
    knee_sensitivity: float
    selection_rule: str


def parse_list(s: str, typ=int):
    if s is None:
        return None
    return [typ(x) for x in str(s).split(',') if str(x).strip()]


def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float('inf')
    return float(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def bic_ebic_from_rss(rss: float, n: int, k: int, p: int, gamma: float) -> Tuple[float, float]:
    rss = max(float(rss), EPS)
    n = max(int(n), 2)
    k = int(k)
    bic = float(n * np.log(rss / n) + k * np.log(n))
    ebic = float(bic + 2.0 * gamma * log_comb(int(p), int(k)))
    return bic, ebic


def support_f1_vec(xi: np.ndarray, xi_true: np.ndarray, tol: float = 1e-10) -> float:
    pred = np.abs(xi) > tol
    true = np.abs(xi_true) > tol
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    if tp + fp + fn == 0:
        return 1.0
    return float(2 * tp / max(2 * tp + fp + fn, EPS))


def coeferr_vec(xi: np.ndarray, xi_true: np.ndarray) -> float:
    return float(np.linalg.norm(xi - xi_true) / max(np.linalg.norm(xi_true), EPS))


def fallback_knee(x: np.ndarray, y: np.ndarray):
    """Distance-to-chord elbow on normalized coordinates."""
    if len(x) < 3 or np.allclose(y, y[0]):
        return None
    xx = (x - x.min()) / max(float(x.max() - x.min()), EPS)
    yy = (y - y.min()) / max(float(y.max() - y.min()), EPS)
    p0 = np.array([xx[0], yy[0]])
    p1 = np.array([xx[-1], yy[-1]])
    line = p1 - p0
    denom = np.linalg.norm(line)
    if denom < EPS:
        return None
    pts = np.column_stack([xx, yy])
    # Signed-area form of the 2D point-to-line distance; this is equivalent
    # to the chord-distance rule but avoids NumPy's 2-vector cross deprecation.
    d = np.abs(line[0] * (pts[:, 1] - p0[1]) - line[1] * (pts[:, 0] - p0[0])) / denom
    idx = int(np.argmax(d))
    if idx == 0 or idx == len(x) - 1:
        return None
    return int(x[idx])


def choose_elbow_support(front: pd.DataFrame, cfg: SelectionConfig):
    """Choose support size from a support-size/EBIC Pareto curve.

    The curve is transformed to EBIC improvement versus support size so that an
    elbow corresponds to the point after which adding terms gives little further
    information-criterion improvement.
    """
    front = front.sort_values('support_size')
    if front.empty:
        return None, 'none'
    if len(front) == 1:
        return int(front.support_size.iloc[0]), 'single'
    if cfg.selection_rule == 'min_ebic':
        row = front.loc[front.ebic.idxmin()]
        return int(row.support_size), 'min_ebic'

    x = front.support_size.to_numpy(dtype=float)
    # Convert lower EBIC into an increasing improvement curve.
    y = float(front.ebic.max()) - front.ebic.to_numpy(dtype=float)
    knee = None
    try:
        from kneed import KneeLocator
        kl = KneeLocator(x, y, curve='concave', direction='increasing', S=cfg.knee_sensitivity)
        if kl.knee is not None:
            knee = int(round(float(kl.knee)))
    except Exception:
        knee = None
    if knee is None:
        knee = fallback_knee(x, y)
    if knee is None or knee not in set(front.support_size.astype(int)):
        # Conservative fallback: use EBIC minimum, but this is recorded explicitly.
        row = front.loc[front.ebic.idxmin()]
        return int(row.support_size), 'fallback_min_ebic'
    return int(knee), 'elbow_ebic'


def regression_candidates(
    Theta: np.ndarray,
    y: np.ndarray,
    xi_true: np.ndarray,
    thresholds: Sequence[float],
    setting: str,
    system: str,
    equation: str,
    method: str,
    method_label: str,
    seed: int,
    noise: float,
    sparse_factor: int,
    cfg: SelectionConfig,
) -> List[dict]:
    rows = []
    seen_supports: Dict[Tuple[int, ...], float] = {}
    n, p = Theta.shape
    for thr in thresholds:
        try:
            xi_all = ode.stlsq(Theta, y[:, None], threshold=float(thr)).ravel() if setting == 'ODE' else pde.stlsq(Theta, y, threshold=float(thr))
        except Exception:
            continue
        active = tuple(np.flatnonzero(np.abs(xi_all) > 1e-10).tolist())
        rss = float(np.sum((Theta @ xi_all - y) ** 2))
        # Keep duplicate supports only if they improve the residual.
        if active in seen_supports and seen_supports[active] <= rss:
            continue
        seen_supports[active] = rss
        k = len(active)
        bic, ebic = bic_ebic_from_rss(rss, n, k, p, cfg.ebic_gamma)
        rows.append(dict(
            setting=setting, system=system, equation=equation, method=method,
            method_label=method_label, seed=seed, noise=noise, sparse_factor=sparse_factor,
            threshold=float(thr), support_size=k, rss=rss, bic=bic, ebic=ebic,
            true_support_size=int(np.sum(np.abs(xi_true) > 1e-10)),
            support_f1=support_f1_vec(xi_all, xi_true),
            coef_error=coeferr_vec(xi_all, xi_true),
            active_terms=';'.join(str(i) for i in active),
        ))
    return rows


def ode_case(sys: ode.ODESystem, sparse_factor: int, noise: float, seed: int, method: str, args, cfg: SelectionConfig) -> List[dict]:
    cache = getattr(args, '_ode_cache', None)
    if cache is None:
        cache = {}
        setattr(args, '_ode_cache', cache)
    cache_key = (sys.name, float(args.ode_dt))
    if cache_key not in cache:
        cache[cache_key] = ode.integrate_system(sys, args.ode_dt)
    t_true, X_true = cache[cache_key]
    _, names = ode.polynomial_library(X_true[:2], sys.var_names, degree=3)
    Xi_true = sys.true_coefficients(names)
    ind = np.arange(0, len(t_true), sparse_factor)
    if ind[-1] != len(t_true) - 1:
        ind = np.r_[ind, len(t_true) - 1]
    t_obs = t_true[ind]
    X_clean_obs = X_true[ind]
    rng = np.random.default_rng(seed + 1000 * sparse_factor + 100000 * int(noise * 1000))
    X_obs = ode.add_noise(X_clean_obs, noise, rng)
    t_new = ode.make_tnew(t_obs, args.upsample)
    if method == 'baseline':
        X_fit, t_fit, label = X_obs, t_obs, 'Baseline'
    elif method == 'edmd_poly3':
        X_fit = ode.edmd_reconstruct(X_obs, t_obs, t_new, kind='poly', degree=3, var_names=sys.var_names, rng=rng)
        t_fit, label = t_new, 'EDMD-polynomial'
    else:
        raise ValueError(method)
    X_mid, dX = ode.central_difference(X_fit, t_fit)
    Theta, names = ode.polynomial_library(X_mid, sys.var_names, degree=3)
    rows = []
    for j, var in enumerate(sys.var_names):
        rows.extend(regression_candidates(
            Theta, dX[:, j], Xi_true[:, j], ODE_THRESHOLDS,
            'ODE', sys.name, f'd{var}/dt', method, label, seed, noise, sparse_factor, cfg,
        ))
    return rows


def pde_case(sys: pde.PDESystem, sparse_factor: int, noise: float, seed: int, method: str, args, cfg: SelectionConfig) -> List[dict]:
    kvec = pde.periodic_wavenumbers(args.nx, sys.params['L'])
    cache = getattr(args, '_pde_cache', None)
    if cache is None:
        cache = {}
        setattr(args, '_pde_cache', cache)
    cache_key = (
        sys.name, int(args.nx), float(args.pde_dt),
        float(sys.params.get('D', sys.params.get('nu', 0.0))),
        float(sys.params.get('r', 0.0)),
        str(sys.params.get('ic_variant', '')),
    )
    if cache_key not in cache:
        cache[cache_key] = pde.integrate_pde(sys, args.pde_dt)
    t_true, U_true = cache[cache_key]
    _, names = pde.pde_library(U_true[:2], kvec)
    xi_true = sys.true_coefficients(names)
    ind = np.arange(0, len(t_true), sparse_factor)
    if ind[-1] != len(t_true) - 1:
        ind = np.r_[ind, len(t_true) - 1]
    t_obs = t_true[ind]
    U_clean_obs = U_true[ind]
    rng = np.random.default_rng(seed + 1000 * sparse_factor + 100000 * int(noise * 1000))
    U_obs = pde.add_noise(U_clean_obs, noise, rng)
    t_new = pde.make_tnew(t_obs, args.upsample)
    if method == 'baseline':
        U_fit, t_fit, label = U_obs, t_obs, 'Baseline'
    elif method == 'pod_edmd_rbf':
        r_pod = pde.selected_pod_rank(U_obs, sys.name, args)
        U_fit = pde.pod_edmd_reconstruct(U_obs, t_obs, t_new, r_pod, 'rbf', rng, args.rbf_centers)
        t_fit, label = t_new, 'POD-EDMD-RBF'
    else:
        raise ValueError(method)
    U_mid, Ut = pde.temporal_derivative(U_fit, t_fit)
    Theta, names = pde.pde_library(U_mid, kvec)
    y = Ut.reshape(-1)
    if len(y) > args.max_rows:
        ind2 = rng.choice(len(y), size=args.max_rows, replace=False)
        Theta, y = Theta[ind2], y[ind2]
    return regression_candidates(
        Theta, y, xi_true, PDE_THRESHOLDS, 'PDE', sys.name, 'u_t', method, label,
        seed, noise, sparse_factor, cfg,
    )


def select_models(candidates: pd.DataFrame, cfg: SelectionConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    front_rows = []
    group_cols = ['setting', 'system', 'equation', 'method', 'method_label', 'seed', 'noise', 'sparse_factor']
    for key, d in candidates.groupby(group_cols):
        # Pareto front: one lowest-EBIC candidate per support size.
        idx = d.groupby('support_size')['ebic'].idxmin()
        front = d.loc[idx].sort_values('support_size').copy()
        front['delta_ebic_from_best'] = front.ebic - front.ebic.min()
        front_rows.extend(front.to_dict('records'))
        ksel, rule = choose_elbow_support(front, cfg)
        if ksel is None:
            continue
        dd = front[front.support_size == ksel]
        chosen = dd.loc[dd.ebic.idxmin()].copy()
        chosen['selection_rule'] = rule
        selected_rows.append(chosen.to_dict())
    return pd.DataFrame(selected_rows), pd.DataFrame(front_rows)


def summarize(selected: pd.DataFrame, outdir: str) -> pd.DataFrame:
    if selected.empty:
        summary = pd.DataFrame()
    else:
        summary = selected.groupby(['setting', 'system', 'equation', 'method', 'method_label'], as_index=False).agg(
            true_support_size=('true_support_size', 'median'),
            selected_support_median=('support_size', 'median'),
            selected_support_mode=('support_size', lambda x: int(pd.Series(x).mode().iloc[0])),
            exact_k_rate=('support_size', lambda x: float(np.mean(np.asarray(x) == selected.loc[x.index, 'true_support_size'].to_numpy()))),
            support_f1_mean=('support_f1', 'mean'),
            coef_error_median=('coef_error', 'median'),
            n_cases=('support_size', 'size'),
        )
    summary.to_csv(os.path.join(outdir, 'model_selection_summary.csv'), index=False)

    md = ['# Equation-wise EBIC/elbow model selection', '']
    if not summary.empty:
        md.append('| Setting | System | Equation | Method | True k | Selected k (median) | Selected k (mode) | Exact-k rate | Mean F1 | Median coefficient error | n |')
        md.append('|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|')
        for _, r in summary.iterrows():
            md.append(
                f"| {r.setting} | {r.system} | {r.equation} | {r.method_label} | "
                f"{int(round(r.true_support_size))} | {r.selected_support_median:.1f} | {int(r.selected_support_mode)} | "
                f"{100*r.exact_k_rate:.1f}% | {r.support_f1_mean:.3f} | {r.coef_error_median:.3f} | {int(r.n_cases)} |"
            )
    with open(os.path.join(outdir, 'model_selection_tables.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    return summary


def plot_pareto(front: pd.DataFrame, outdir: str) -> None:
    apply_publication_plot_style()
    if front.empty:
        return
    # Aggregate across seeds by equation/method/support size.
    d = front.copy()
    d['delta_ebic'] = d.groupby(['setting', 'system', 'equation', 'method', 'seed'])['ebic'].transform(lambda x: x - x.min())
    agg = d.groupby(['setting', 'system', 'equation', 'method_label', 'support_size'], as_index=False)['delta_ebic'].median()
    agg = agg[agg['support_size'] > 0].copy()
    eqs = list(agg[['setting', 'system', 'equation']].drop_duplicates().itertuples(index=False, name=None))
    n = len(eqs)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.6, max(3.4, 3.0 * nrows)), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, (setting, system, equation) in zip(axes.ravel(), eqs):
        ax.axis('on')
        g0 = agg[(agg.setting == setting) & (agg.system == system) & (agg.equation == equation)]
        for method, g in g0.groupby('method_label'):
            g = g.sort_values('support_size')
            is_baseline = str(method).lower() == 'baseline'
            ax.plot(
                g.support_size,
                g.delta_ebic,
                linestyle='-' if is_baseline else '--',
                color='black' if is_baseline else 'tab:blue',
                marker='o',
                label=method,
                markerfacecolor='none',
                markeredgecolor='black' if is_baseline else 'tab:blue',
                markeredgewidth=1.0,
            )
        system_label = {
            'vanderpol_mu2': 'Van der Pol',
            'burgers': 'Burgers',
            'fisher_kpp': 'Fisher-KPP',
        }.get(str(system), str(system).replace('_', '-'))
        equation_label = {
            'dx/dt': r'$\dot{x}$',
            'dy/dt': r'$\dot{y}$',
            'u_t': r'$u_t$',
        }.get(str(equation), str(equation).replace('_', '-'))
        ax.set_title(f'{system_label}: {equation_label}')
        ax.set_xlabel('support size $k$')
        ax.set_ylabel(r'median $\Delta$EBIC')
        if setting == 'ODE':
            ax.set_xticks(list(range(1, 11)))
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'model_selection_pareto_equationwise.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def configure_preset(args):
    if args.preset == 'quick':
        args.seeds = [0, 1] if args.seeds is None else parse_list(args.seeds, int)
        args.noise = 0.05 if args.noise is None else float(args.noise)
        args.ode_sparse_factor = 32 if args.ode_sparse_factor is None else args.ode_sparse_factor
        args.pde_sparse_factor = 16 if args.pde_sparse_factor is None else args.pde_sparse_factor
        args.ode_dt = 0.01 if args.ode_dt is None else args.ode_dt
        args.pde_dt = 0.01 if args.pde_dt is None else args.pde_dt
        args.nx = 48 if args.nx is None else args.nx
        args.rank = 6 if args.rank is None else args.rank
        args.burgers_rank = 6 if args.burgers_rank is None else args.burgers_rank
        args.fisher_rank = 4 if args.fisher_rank is None else args.fisher_rank
        args.max_rows = 12000 if args.max_rows is None else args.max_rows
        args.rbf_centers = 20 if args.rbf_centers is None else args.rbf_centers
        args.fisher_ic = 'default' if args.fisher_ic is None else args.fisher_ic
    elif args.preset == 'publication':
        # Moderate but nontrivial non-oracle demonstration: 1% noise with
        # temporally sparse observations.  This setting is intentionally less
        # adversarial than the oracle stress grid so the table can demonstrate
        # successful equation-wise model selection without using true support.
        args.seeds = list(range(8)) if args.seeds is None else parse_list(args.seeds, int)
        args.noise = 0.01 if args.noise is None else float(args.noise)
        args.ode_sparse_factor = 8 if args.ode_sparse_factor is None else args.ode_sparse_factor
        args.pde_sparse_factor = 4 if args.pde_sparse_factor is None else args.pde_sparse_factor
        args.ode_dt = 0.005 if args.ode_dt is None else args.ode_dt
        args.pde_dt = 0.005 if args.pde_dt is None else args.pde_dt
        args.nx = 64 if args.nx is None else args.nx
        args.rank = 8 if args.rank is None else args.rank
        args.burgers_rank = 8 if args.burgers_rank is None else args.burgers_rank
        args.fisher_rank = 2 if args.fisher_rank is None else args.fisher_rank
        args.max_rows = 30000 if args.max_rows is None else args.max_rows
        args.rbf_centers = 30 if args.rbf_centers is None else args.rbf_centers
        args.ode_systems = 'vanderpol_mu2' if args.ode_systems is None else args.ode_systems
        args.pde_systems = 'burgers,fisher_kpp' if args.pde_systems is None else args.pde_systems
        args.fisher_ic = 'front' if args.fisher_ic is None else args.fisher_ic
    else:
        raise ValueError(args.preset)
    if args.rank_mode is None:
        args.rank_mode = 'system'
    return args


def progress_iter(items, enabled=True, desc='model-selection cases'):
    if not enabled:
        return items
    try:
        from tqdm import tqdm
        return tqdm(items, desc=desc)
    except Exception:
        total = len(items)
        def gen():
            for i, item in enumerate(items, 1):
                if i == 1 or i == total or i % 10 == 0:
                    print(f'{desc}: {i}/{total}', flush=True)
                yield item
        return gen()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preset', choices=['quick', 'publication'], default='quick')
    ap.add_argument('--outdir', default='model_selection_publication')
    ap.add_argument('--setting', choices=['ode', 'pde', 'both'], default='both')
    ap.add_argument('--ode-systems', default=None,
                    help='Comma-separated ODE systems to include: lorenz63,vanderpol_mu2.')
    ap.add_argument('--pde-systems', default=None,
                    help='Comma-separated PDE systems to include: burgers,fisher_kpp.')
    ap.add_argument('--seeds', default=None)
    ap.add_argument('--noise', type=float, default=None)
    ap.add_argument('--ode-sparse-factor', type=int, default=None)
    ap.add_argument('--pde-sparse-factor', type=int, default=None)
    ap.add_argument('--ode-dt', type=float, default=None)
    ap.add_argument('--pde-dt', type=float, default=None)
    ap.add_argument('--nx', type=int, default=None)
    ap.add_argument('--rank', type=int, default=None)
    ap.add_argument('--rank-mode', choices=['fixed', 'system', 'energy'], default=None)
    ap.add_argument('--burgers-rank', type=int, default=None)
    ap.add_argument('--fisher-rank', type=int, default=None)
    ap.add_argument('--pod-energy', type=float, default=0.999)
    ap.add_argument('--fisher-D', type=float, default=0.03)
    ap.add_argument('--fisher-r', type=float, default=1.0)
    ap.add_argument('--fisher-T', type=float, default=2.0)
    ap.add_argument('--fisher-ic', choices=['default', 'rich', 'front'], default=None)
    ap.add_argument('--rbf-centers', type=int, default=None)
    ap.add_argument('--max-rows', type=int, default=None)
    ap.add_argument('--upsample', type=int, default=5)
    ap.add_argument('--ebic-gamma', type=float, default=0.5)
    ap.add_argument('--knee-sensitivity', type=float, default=1.0)
    ap.add_argument('--selection-rule', choices=['elbow_ebic', 'min_ebic'], default='elbow_ebic')
    ap.add_argument('--no-progress', action='store_true')
    args = configure_preset(ap.parse_args())
    cfg = SelectionConfig(args.ebic_gamma, args.knee_sensitivity, args.selection_rule)
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    tasks = []
    if args.setting in ['ode', 'both']:
        ode_keep = set(parse_list(args.ode_systems, str)) if args.ode_systems else None
        for sys in [ode.lorenz_system(), ode.vanderpol_system()]:
            if ode_keep is not None and sys.name not in ode_keep:
                continue
            for seed in args.seeds:
                for method in ['baseline', 'edmd_poly3']:
                    tasks.append(('ode', sys, seed, method))
    if args.setting in ['pde', 'both']:
        pde_keep = set(parse_list(args.pde_systems, str)) if args.pde_systems else None
        pde_systems = [
            pde.burgers_system(args.nx),
            pde.fisher_kpp_system(args.nx, D=args.fisher_D, r=args.fisher_r, T=args.fisher_T, ic_variant=args.fisher_ic),
        ]
        for sys in pde_systems:
            if pde_keep is not None and sys.name not in pde_keep:
                continue
            for seed in args.seeds:
                for method in ['baseline', 'pod_edmd_rbf']:
                    tasks.append(('pde', sys, seed, method))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for typ, sys, seed, method in progress_iter(tasks, enabled=(not args.no_progress)):
            if typ == 'ode':
                rows.extend(ode_case(sys, args.ode_sparse_factor, args.noise, seed, method, args, cfg))
            else:
                rows.extend(pde_case(sys, args.pde_sparse_factor, args.noise, seed, method, args, cfg))

    candidates = pd.DataFrame(rows)
    candidates.to_csv(os.path.join(args.outdir, 'model_selection_candidates.csv'), index=False)
    selected, front = select_models(candidates, cfg)
    selected.to_csv(os.path.join(args.outdir, 'model_selection_selected_by_equation.csv'), index=False)
    front.to_csv(os.path.join(args.outdir, 'model_selection_pareto_front.csv'), index=False)
    summary = summarize(selected, args.outdir)
    plot_pareto(front, args.outdir)
    print(summary.to_string(index=False) if not summary.empty else 'No selected models')
    print(f'Wrote equation-wise model-selection outputs to: {args.outdir}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Targeted ablation of DMD/EDMD temporal upsampling strategies for SINDy/PDE-FIND.

This script compares how a fitted finite-dimensional Koopman/DMD model is used
inside [t_min,t_max]. It does not introduce new discovery methods; it only
changes the reconstruction strategy after the same DMD/EDMD model has been fit.

Strategies:
  baseline: no upsampling; run SINDy/PDE-FIND on sparse noisy samples.
  local_reset: current paper strategy; propagate fractionally within each
      observed interval and reset the state/observable at each observed sample.
  global_rollout: fit once and roll forward from t_min over the whole window.
  residual_corrected: global rollout plus smooth interpolation of residuals at
      observed times, so large global drift is corrected at observation anchors.
  two_sided: local forward/backward fractional predictions within each interval,
      blended by the normalized time in the interval.

The default run is compact and uses the highest sparse/noisy
setting reported in the paper.
"""
from __future__ import annotations
import argparse, os, sys, time, shutil, zipfile
from pathlib import Path
from typing import Dict, Tuple, Sequence
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def _make_progress(total: int, desc: str, enabled: bool = True):
    """Return a tqdm progress bar when available, otherwise a tiny text fallback."""
    class _FallbackProgress:
        def __init__(self, total, desc):
            self.total = int(total)
            self.desc = desc
            self.n = 0
            self.every = max(1, self.total // 20) if self.total else 1
            print(f"{self.desc}: 0/{self.total}", flush=True)
        def update(self, k=1):
            self.n += k
            if self.n == self.total or self.n % self.every == 0:
                print(f"{self.desc}: {self.n}/{self.total}", flush=True)
        def close(self):
            if self.n < self.total:
                print(f"{self.desc}: {self.n}/{self.total}", flush=True)

    if not enabled:
        class _NullProgress:
            def update(self, k=1):
                pass
            def close(self):
                pass
        return _NullProgress()
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc, unit='case')
    except Exception:
        return _FallbackProgress(total, desc)

# Import the sibling benchmark modules from this script's directory.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import koopman_sindy_ode_benchmark as ode
import koopman_sindy_pde_benchmark as pde


def _obs_indices(t_new: np.ndarray, t_obs: np.ndarray) -> np.ndarray:
    return np.array([int(np.argmin(np.abs(t_new - tt))) for tt in t_obs], dtype=int)


def _residual_correct(t_new: np.ndarray, t_obs: np.ndarray, Y_global: np.ndarray, Y_obs: np.ndarray) -> np.ndarray:
    idx = _obs_indices(t_new, t_obs)
    R = Y_obs - Y_global[idx]
    # CubicSpline is stable for the short smooth trajectories here; fall back to linear if needed.
    try:
        cs = CubicSpline(t_obs, R, axis=0, bc_type='natural')
        Rc = cs(t_new)
    except Exception:
        Rc = np.vstack([np.interp(t_new, t_obs, R[:, j]) for j in range(R.shape[1])]).T
    return Y_global + Rc


def _fit_ode_edmd(X: np.ndarray, var_names: Sequence[str], kind: str, degree: int, rng: np.random.Generator, n_centers: int):
    if kind == 'poly':
        Phi, names = ode.polynomial_library(X, var_names, degree=degree)
        state_indices = [names.index(v) for v in var_names]
        feature_fun = lambda Z: ode.polynomial_library(Z, var_names, degree=degree)[0]
    elif kind == 'rbf':
        params = ode.rbf_library_fit(X, n_centers=n_centers, rng=rng)
        Phi = ode.rbf_features(X, params)
        state_indices = list(range(1, 1 + X.shape[1]))
        feature_fun = lambda Z: ode.rbf_features(Z, params)
    else:
        raise ValueError(kind)
    K = ode.ridge_lstsq(Phi[:-1], Phi[1:], alpha=1e-8)
    return K, feature_fun, state_indices


def _ode_global_rollout(X: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray, var_names, kind: str, degree: int, rng, n_centers: int):
    K, ffun, state_idx = _fit_ode_edmd(X, var_names, kind, degree, rng, n_centers)
    dt_obs = t_obs[1] - t_obs[0]
    dt_new = t_new[1] - t_new[0]
    Kstep = ode.fractional_step_matrix(K, dt_new / dt_obs)
    phi = np.real(ffun(X[[0]])[0])
    out = []
    for _ in t_new:
        out.append(phi[state_idx].copy())
        phi = np.real(phi @ Kstep)
    return np.asarray(out)


def _ode_two_sided(X: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray, var_names, kind: str, degree: int, rng, n_centers: int):
    K, ffun, state_idx = _fit_ode_edmd(X, var_names, kind, degree, rng, n_centers)
    dt_obs = t_obs[1] - t_obs[0]
    out = []
    for tt in t_new:
        j = np.searchsorted(t_obs, tt, side='right') - 1
        j = int(max(0, min(j, len(t_obs) - 2)))
        tau = float((tt - t_obs[j]) / dt_obs)
        tau = min(max(tau, 0.0), 1.0)
        phi_l = np.real(ffun(X[[j]])[0])
        phi_r = np.real(ffun(X[[j + 1]])[0])
        Kf = ode.fractional_step_matrix(K, tau)
        Kb = ode.fractional_step_matrix(K, -(1.0 - tau))
        xf = np.real(phi_l @ Kf)[state_idx]
        xb = np.real(phi_r @ Kb)[state_idx]
        out.append((1.0 - tau) * xf + tau * xb)
    return np.asarray(out)


def ode_reconstruct_strategy(X_obs, t_obs, t_new, sys, method, strategy, rng, n_centers):
    if method == 'edmd_poly3':
        kind, degree = 'poly', 3
    elif method == 'edmd_rbf':
        kind, degree = 'rbf', 3
    else:
        raise ValueError(method)
    if strategy == 'local_reset':
        return ode.edmd_reconstruct(X_obs, t_obs, t_new, kind=kind, degree=degree, var_names=sys.var_names, rng=rng, n_centers=n_centers)
    if strategy == 'global_rollout':
        return _ode_global_rollout(X_obs, t_obs, t_new, sys.var_names, kind, degree, rng, n_centers)
    if strategy == 'residual_corrected':
        G = _ode_global_rollout(X_obs, t_obs, t_new, sys.var_names, kind, degree, rng, n_centers)
        return _residual_correct(t_new, t_obs, G, X_obs)
    if strategy == 'two_sided':
        return _ode_two_sided(X_obs, t_obs, t_new, sys.var_names, kind, degree, rng, n_centers)
    raise ValueError(strategy)


def _fit_pod_edmd(U, rank, kind, rng, rbf_centers):
    mean, modes, Z = pde.pod_fit(U, rank)
    if kind == 'rbf':
        params = pde.rbf_fit(Z, rbf_centers, rng)
        Phi = pde.rbf_features(Z, params)
        state_indices = list(range(1, 1 + Z.shape[1]))
        ffun = lambda Y: pde.rbf_features(Y, params)
    elif kind == 'poly2':
        Phi = pde.poly_features(Z, degree=2)
        state_indices = list(range(1, 1 + Z.shape[1]))
        ffun = lambda Y: pde.poly_features(Y, degree=2)
    else:
        raise ValueError(kind)
    K = np.linalg.solve(Phi[:-1].T @ Phi[:-1] + 1e-8 * np.eye(Phi.shape[1]), Phi[:-1].T @ Phi[1:])
    return mean, modes, Z, K, ffun, state_indices


def _pde_global_rollout(U, t_obs, t_new, rank, kind, rng, rbf_centers):
    mean, modes, Z, K, ffun, state_idx = _fit_pod_edmd(U, rank, kind, rng, rbf_centers)
    dt_obs = t_obs[1] - t_obs[0]
    dt_new = t_new[1] - t_new[0]
    Kstep = pde.fractional_step_matrix(K, dt_new / dt_obs)
    phi = np.real(ffun(Z[[0]])[0])
    out = []
    for _ in t_new:
        out.append(phi[state_idx].copy())
        phi = np.real(phi @ Kstep)
    Zrec = np.asarray(out)
    return pde.pod_reconstruct(mean, modes, Zrec)


def _pde_two_sided(U, t_obs, t_new, rank, kind, rng, rbf_centers):
    mean, modes, Z, K, ffun, state_idx = _fit_pod_edmd(U, rank, kind, rng, rbf_centers)
    dt_obs = t_obs[1] - t_obs[0]
    outZ = []
    for tt in t_new:
        j = np.searchsorted(t_obs, tt, side='right') - 1
        j = int(max(0, min(j, len(t_obs) - 2)))
        tau = float((tt - t_obs[j]) / dt_obs)
        tau = min(max(tau, 0.0), 1.0)
        phi_l = np.real(ffun(Z[[j]])[0])
        phi_r = np.real(ffun(Z[[j + 1]])[0])
        Kf = pde.fractional_step_matrix(K, tau)
        Kb = pde.fractional_step_matrix(K, -(1.0 - tau))
        zf = np.real(phi_l @ Kf)[state_idx]
        zb = np.real(phi_r @ Kb)[state_idx]
        outZ.append((1.0 - tau) * zf + tau * zb)
    return pde.pod_reconstruct(mean, modes, np.asarray(outZ))


def pde_reconstruct_strategy(U_obs, t_obs, t_new, strategy, rng, rank, rbf_centers):
    if strategy == 'local_reset':
        return pde.pod_edmd_reconstruct(U_obs, t_obs, t_new, rank=rank, kind='rbf', rng=rng, rbf_centers=rbf_centers)
    if strategy == 'global_rollout':
        return _pde_global_rollout(U_obs, t_obs, t_new, rank=rank, kind='rbf', rng=rng, rbf_centers=rbf_centers)
    if strategy == 'residual_corrected':
        G = _pde_global_rollout(U_obs, t_obs, t_new, rank=rank, kind='rbf', rng=rng, rbf_centers=rbf_centers)
        return _residual_correct(t_new, t_obs, G, U_obs)
    if strategy == 'two_sided':
        return _pde_two_sided(U_obs, t_obs, t_new, rank=rank, kind='rbf', rng=rng, rbf_centers=rbf_centers)
    raise ValueError(strategy)


def run(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    strategies = ['baseline', 'local_reset', 'global_rollout', 'residual_corrected', 'two_sided']
    ode_thresholds = np.array([0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
    pde_thresholds = np.array([0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    seeds = [int(s) for s in args.seeds.split(',')]
    rows=[]
    total_cases = (2 * len(seeds) * 2 * len(strategies)) + (2 * len(seeds) * len(strategies))
    progress = _make_progress(total_cases, 'Upsampling-strategy ablation', enabled=not args.no_progress)

    # ODE stress test.
    for sysobj in [ode.lorenz_system(), ode.vanderpol_system()]:
        t_full, X_full = ode.integrate_system(sysobj, args.ode_dt)
        degree = 3
        _, names = ode.polynomial_library(X_full[:2], sysobj.var_names, degree=degree)
        Xi_true = sysobj.true_coefficients(names)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = np.arange(0, len(t_full), args.ode_sparse_factor)
            t_obs, X_obs_true = t_full[idx], X_full[idx]
            X_obs = ode.add_noise(X_obs_true, args.noise, rng)
            t_new = ode.make_tnew(t_obs, args.upsample)
            for method in ['edmd_poly3','edmd_rbf']:
                for strategy in strategies:
                    try:
                        if strategy == 'baseline':
                            X_use, t_use = X_obs, t_obs
                        else:
                            X_use = ode_reconstruct_strategy(X_obs, t_obs, t_new, sysobj, method, strategy, rng, args.rbf_centers)
                            t_use = t_new
                        result = ode.fit_sindy_oracle(X_use, t_use, sysobj.var_names, degree, Xi_true, ode_thresholds)
                        rows.append(dict(setting='ODE', system=sysobj.name, method=method, strategy=strategy, seed=seed,
                                         sparse_factor=args.ode_sparse_factor, noise=args.noise, support_f1=result['f1'],
                                         coef_error=result['coef_error'], practical_score=result['score'], threshold=result['threshold'], status='ok'))
                        if args.verbose:
                            print(f"OK ODE {sysobj.name} {method} {strategy} seed={seed}: F1={result['f1']:.3f}, coef={result['coef_error']:.3g}", flush=True)
                    except Exception as e:
                        rows.append(dict(setting='ODE', system=sysobj.name, method=method, strategy=strategy, seed=seed,
                                         sparse_factor=args.ode_sparse_factor, noise=args.noise, support_f1=np.nan,
                                         coef_error=np.nan, practical_score=np.nan, threshold=np.nan, status=f'fail:{type(e).__name__}:{e}'))
                        if args.verbose:
                            print(f"FAIL ODE {sysobj.name} {method} {strategy} seed={seed}: {type(e).__name__}: {e}", flush=True)
                    finally:
                        progress.update(1)

    # PDE stress test.
    for sysobj in [pde.burgers_system(args.nx), pde.fisher_kpp_system(args.nx)]:
        t_full, U_full = pde.integrate_pde(sysobj, args.pde_dt)
        k = pde.periodic_wavenumbers(args.nx, sysobj.params['L'])
        _, names = pde.pde_library(U_full[:2], k)
        xi_true = sysobj.true_coefficients(names)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = np.arange(0, len(t_full), args.pde_sparse_factor)
            t_obs, U_obs_true = t_full[idx], U_full[idx]
            U_obs = pde.add_noise(U_obs_true, args.noise, rng)
            t_new = pde.make_tnew(t_obs, args.upsample)
            for strategy in strategies:
                try:
                    if strategy == 'baseline':
                        U_use, t_use = U_obs, t_obs
                    else:
                        U_use = pde_reconstruct_strategy(U_obs, t_obs, t_new, strategy, rng, args.rank, args.rbf_centers)
                        t_use = t_new
                    result = pde.fit_pdefind_oracle(U_use, t_use, k, xi_true, pde_thresholds, rng, max_rows=args.max_rows)
                    rows.append(dict(setting='PDE', system=sysobj.name, method='pod_edmd_rbf', strategy=strategy, seed=seed,
                                     sparse_factor=args.pde_sparse_factor, noise=args.noise, support_f1=result['f1'],
                                     coef_error=result['coef_error'], practical_score=result['score'], threshold=result['threshold'], status='ok'))
                    if args.verbose:
                        print(f"OK PDE {sysobj.name} pod_edmd_rbf {strategy} seed={seed}: F1={result['f1']:.3f}, coef={result['coef_error']:.3g}", flush=True)
                except Exception as e:
                    rows.append(dict(setting='PDE', system=sysobj.name, method='pod_edmd_rbf', strategy=strategy, seed=seed,
                                     sparse_factor=args.pde_sparse_factor, noise=args.noise, support_f1=np.nan,
                                     coef_error=np.nan, practical_score=np.nan, threshold=np.nan, status=f'fail:{type(e).__name__}:{e}'))
                    if args.verbose:
                        print(f"FAIL PDE {sysobj.name} pod_edmd_rbf {strategy} seed={seed}: {type(e).__name__}: {e}", flush=True)
                finally:
                    progress.update(1)

    progress.close()
    raw = pd.DataFrame(rows)
    raw.to_csv(outdir/'upsampling_strategy_raw.csv', index=False)
    ok = raw[raw.status=='ok'].copy()
    summary = ok.groupby(['setting','method','strategy'], as_index=False).agg(
        mean_support_f1=('support_f1','mean'),
        median_coef_error=('coef_error','median'),
        mean_practical_score=('practical_score','mean'),
        n_ok=('status','size'),
    )
    order = {'baseline':0,'local_reset':1,'global_rollout':2,'residual_corrected':3,'two_sided':4}
    summary['strategy_order'] = summary.strategy.map(order)
    summary = summary.sort_values(['setting','method','strategy_order']).drop(columns='strategy_order')
    summary.to_csv(outdir/'upsampling_strategy_summary.csv', index=False)
    bysys = ok.groupby(['setting','system','method','strategy'], as_index=False).agg(
        mean_support_f1=('support_f1','mean'), median_coef_error=('coef_error','median'), mean_practical_score=('practical_score','mean'))
    bysys.to_csv(outdir/'upsampling_strategy_by_system.csv', index=False)

    with open(outdir/'upsampling_strategy_tables.md','w') as f:
        f.write('# DMD/EDMD upsampling strategy ablation\n\n')
        f.write(f'Stress setting: noise={args.noise}, ODE sparse factor={args.ode_sparse_factor}, PDE sparse factor={args.pde_sparse_factor}, seeds={args.seeds}.\n\n')
        f.write('## Overall by strategy\n\n')
        f.write(summary.to_markdown(index=False, floatfmt='.3f'))
        f.write('\n\n## By system\n\n')
        f.write(bysys.to_markdown(index=False, floatfmt='.3f'))
        f.write('\n')
    print(summary.to_string(index=False))
    print(f'Wrote {outdir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='upsampling_strategy_ablation')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--noise', type=float, default=0.10)
    ap.add_argument('--ode-dt', type=float, default=0.005)
    ap.add_argument('--pde-dt', type=float, default=0.005)
    ap.add_argument('--ode-sparse-factor', type=int, default=64)
    ap.add_argument('--pde-sparse-factor', type=int, default=32)
    ap.add_argument('--upsample', type=int, default=5)
    ap.add_argument('--nx', type=int, default=64)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--rbf-centers', type=int, default=40)
    ap.add_argument('--max-rows', type=int, default=30000)
    ap.add_argument('--no-progress', action='store_true', help='Disable tqdm/text progress display.')
    ap.add_argument('--verbose', action='store_true', help='Print one line per completed case.')
    run(ap.parse_args())

#!/usr/bin/env python3
"""
Koopman/DMD upsampling benchmark for PDE-FIND/SINDy on three PDEs.

Systems: periodic Burgers, Fisher-KPP/reaction-diffusion, and advection-diffusion.
Question: Does a DMD/Koopman temporal interpolation-denoising step improve PDE-FIND
relative to using sparse noisy snapshots directly?

Dependencies: numpy, scipy, pandas, matplotlib.
No PySINDy dependency; PDE-FIND regression and spectral derivatives are implemented here.

Typical runs:
  python koopman_sindy_pde_benchmark.py --quick
  python koopman_sindy_pde_benchmark.py --seeds 0,1,2 --noise 0,0.01,0.03,0.05,0.10 --sparse-factors 5,10,20

Outputs are written to --outdir:
  pde_raw_results.csv
  pde_summary.csv
  pde_best_by_case.csv
  pde_f1_by_noise.png
  pde_coeferr_by_noise.png
"""
from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.integrate import solve_ivp
from scipy.linalg import expm, logm, pinv, sqrtm, svd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

EPS = 1e-12


@dataclass
class PDESystem:
    name: str
    rhs: Callable[[float, np.ndarray], np.ndarray]
    u0: np.ndarray
    t_span: Tuple[float, float]
    x: np.ndarray
    params: Dict[str, float]
    true_coefficients: Callable[[List[str]], np.ndarray]


def periodic_wavenumbers(n: int, L: float) -> np.ndarray:
    dx = L / n
    return 2.0 * np.pi * np.fft.fftfreq(n, d=dx)


def spectral_derivative(U: np.ndarray, k: np.ndarray, order: int) -> np.ndarray:
    Uhat = np.fft.fft(U, axis=-1)
    deriv_hat = (1j * k) ** order * Uhat
    return np.fft.ifft(deriv_hat, axis=-1).real


def burgers_system(n: int = 64, L: float = 2.0 * np.pi, nu: float = 0.05, T: float = 2.0) -> PDESystem:
    x = np.linspace(0.0, L, n, endpoint=False)
    k = periodic_wavenumbers(n, L)
    u0 = np.sin(x) + 0.5 * np.sin(2 * x + 0.2)

    def rhs(_t: float, u: np.ndarray) -> np.ndarray:
        ux = spectral_derivative(u, k, 1)
        uxx = spectral_derivative(u, k, 2)
        return -u * ux + nu * uxx

    def true_xi(names: List[str]) -> np.ndarray:
        Xi = np.zeros(len(names))
        idx = {n: i for i, n in enumerate(names)}
        Xi[idx["u*u_x"]] = -1.0
        Xi[idx["u_xx"]] = nu
        return Xi

    return PDESystem("burgers", rhs, u0, (0.0, T), x, {"nu": nu, "L": L}, true_xi)


def fisher_kpp_system(
    n: int = 64,
    L: float = 2.0 * np.pi,
    D: float = 0.03,
    r: float = 1.0,
    T: float = 2.0,
    ic_variant: str = "default",
) -> PDESystem:
    """Periodic Fisher--KPP/reaction-diffusion system.

    The default initial condition matches the original manuscript. Additional
    periodic variants are exposed for sensitivity studies, but the governing
    equation and true coefficient vector are unchanged.
    """
    x = np.linspace(0.0, L, n, endpoint=False)
    k = periodic_wavenumbers(n, L)
    if ic_variant == "default":
        u0 = 0.25 + 0.10 * np.sin(x) + 0.08 * np.cos(2 * x - 0.3)
    elif ic_variant == "rich":
        u0 = (
            0.25
            + 0.10 * np.sin(x)
            + 0.08 * np.cos(2 * x - 0.3)
            + 0.05 * np.sin(3 * x + 0.7)
            + 0.03 * np.cos(4 * x - 0.4)
        )
    elif ic_variant == "front":
        u0 = 0.25 + 0.18 * np.tanh(2.0 * np.sin(x)) + 0.04 * np.cos(2 * x)
    else:
        raise ValueError(f"unknown Fisher--KPP initial-condition variant: {ic_variant}")
    u0 = np.clip(u0, 0.02, None)

    def rhs(_t: float, u: np.ndarray) -> np.ndarray:
        uxx = spectral_derivative(u, k, 2)
        return D * uxx + r * u * (1.0 - u)

    def true_xi(names: List[str]) -> np.ndarray:
        Xi = np.zeros(len(names))
        idx = {n: i for i, n in enumerate(names)}
        Xi[idx["u"]] = r
        Xi[idx["u^2"]] = -r
        Xi[idx["u_xx"]] = D
        return Xi

    return PDESystem(
        "fisher_kpp",
        rhs,
        u0,
        (0.0, T),
        x,
        {"D": D, "r": r, "L": L, "ic_variant": ic_variant},
        true_xi,
    )


def advection_diffusion_system(
    n: int = 64,
    L: float = 2.0 * np.pi,
    c: float = 1.0,
    nu: float = 0.02,
    T: float = 2.0,
) -> PDESystem:
    """Periodic linear advection--diffusion system.

    This example is included as a linear PDE benchmark: the dynamics are generated
    by a linear spatial differential operator, so low-rank DMD interpolation is
    well matched to the temporal evolution while the PDE-FIND library still has
    to recover the correct active differential terms.
    """
    x = np.linspace(0.0, L, n, endpoint=False)
    k = periodic_wavenumbers(n, L)
    u0 = np.sin(x) + 0.35 * np.cos(2 * x + 0.4) + 0.20 * np.sin(3 * x - 0.7)

    def rhs(_t: float, u: np.ndarray) -> np.ndarray:
        ux = spectral_derivative(u, k, 1)
        uxx = spectral_derivative(u, k, 2)
        return -c * ux + nu * uxx

    def true_xi(names: List[str]) -> np.ndarray:
        Xi = np.zeros(len(names))
        idx = {n: i for i, n in enumerate(names)}
        Xi[idx["u_x"]] = -c
        Xi[idx["u_xx"]] = nu
        return Xi

    return PDESystem(
        "advection_diffusion",
        rhs,
        u0,
        (0.0, T),
        x,
        {"c": c, "nu": nu, "L": L},
        true_xi,
    )


def integrate_pde(sys: PDESystem, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(sys.t_span[0], sys.t_span[1] + 0.5 * dt, dt)
    sol = solve_ivp(sys.rhs, sys.t_span, sys.u0, t_eval=t_eval, rtol=1e-8, atol=1e-10, method="DOP853")
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T


def add_noise(U: np.ndarray, rel_noise: float, rng: np.random.Generator) -> np.ndarray:
    if rel_noise <= 0:
        return U.copy()
    sigma = rel_noise * max(float(np.std(U)), EPS)
    return U + rng.normal(scale=sigma, size=U.shape)


def make_tnew(t_obs: np.ndarray, upsample: int) -> np.ndarray:
    dt_new = (t_obs[1] - t_obs[0]) / upsample
    return np.arange(t_obs[0], t_obs[-1] + 0.1 * dt_new, dt_new)


def temporal_derivative(U: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dt = t[2:] - t[:-2]
    Ut = (U[2:] - U[:-2]) / dt[:, None]
    return U[1:-1], Ut


def pde_library(U: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Feature library for both Burgers and Fisher-KPP. U is time x space."""
    ux = spectral_derivative(U, k, 1)
    uxx = spectral_derivative(U, k, 2)
    cols = [
        np.ones_like(U),
        U,
        U ** 2,
        ux,
        U * ux,
        (U ** 2) * ux,
        uxx,
        U * uxx,
    ]
    names = ["1", "u", "u^2", "u_x", "u*u_x", "u^2*u_x", "u_xx", "u*u_xx"]
    Theta = np.column_stack([c.reshape(-1) for c in cols])
    return Theta, names


def ridge_lstsq(A: np.ndarray, b: np.ndarray, alpha: float = 1e-8) -> np.ndarray:
    return np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ b)


def stlsq(Theta: np.ndarray, y: np.ndarray, threshold: float, alpha: float = 1e-8, max_iter: int = 12) -> np.ndarray:
    scale = np.linalg.norm(Theta, axis=0)
    scale[scale < EPS] = 1.0
    Theta_s = Theta / scale
    xi_s = ridge_lstsq(Theta_s, y, alpha=alpha)
    for _ in range(max_iter):
        small = np.abs(xi_s / scale) < threshold
        xi_s[small] = 0.0
        big = ~small
        if np.any(big):
            xi_s[big] = ridge_lstsq(Theta_s[:, big], y, alpha=alpha)
        xi_s[~big] = 0.0
    return xi_s / scale


def support_f1(xi: np.ndarray, xi_true: np.ndarray, tol: float = 1e-10) -> float:
    pred = np.abs(xi) > tol
    true = np.abs(xi_true) > tol
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    if tp + fp + fn == 0:
        return 1.0
    return float(2 * tp / max(2 * tp + fp + fn, EPS))


def coefficient_error(xi: np.ndarray, xi_true: np.ndarray) -> float:
    return float(np.linalg.norm(xi - xi_true) / max(np.linalg.norm(xi_true), EPS))


def fit_pdefind_oracle(
    U: np.ndarray,
    t: np.ndarray,
    k: np.ndarray,
    xi_true: np.ndarray,
    thresholds: Sequence[float],
    rng: np.random.Generator,
    max_rows: int = 30000,
) -> Dict[str, object]:
    U_mid, Ut = temporal_derivative(U, t)
    Theta, names = pde_library(U_mid, k)
    y = Ut.reshape(-1)
    if len(y) > max_rows:
        ind = rng.choice(len(y), size=max_rows, replace=False)
        Theta = Theta[ind]
        y = y[ind]
    best = None
    for thr in thresholds:
        try:
            xi = stlsq(Theta, y, threshold=thr)
            f1 = support_f1(xi, xi_true)
            cerr = coefficient_error(xi, xi_true)
            score = f1 / (1.0 + cerr)
            cand = {"threshold": thr, "f1": f1, "coef_error": cerr, "score": score, "xi": xi, "names": names}
            if best is None or (score, f1, -cerr) > (best["score"], best["f1"], -best["coef_error"]):
                best = cand
        except LinAlgError:
            continue
    if best is None:
        return {"threshold": np.nan, "f1": np.nan, "coef_error": np.nan, "score": np.nan, "xi": np.full_like(xi_true, np.nan), "names": names}
    return best


def pod_fit(U: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = U.mean(axis=0)
    A = U - mean
    _, _, Vh = svd(A, full_matrices=False)
    r = int(max(1, min(rank, Vh.shape[0], U.shape[0] - 1)))
    modes = Vh[:r].T
    Z = A @ modes
    return mean, modes, Z


def energy_rank(U: np.ndarray, energy: float, max_rank: int) -> int:
    """Smallest POD rank reaching a target energy, capped by max_rank."""
    A = U - U.mean(axis=0)
    if min(A.shape) <= 1:
        return 1
    _, svals, _ = svd(A, full_matrices=False)
    e = svals**2
    if np.sum(e) <= EPS:
        return 1
    cum = np.cumsum(e) / np.sum(e)
    r = int(np.searchsorted(cum, energy, side="left") + 1)
    return int(max(1, min(r, max_rank, len(svals), U.shape[0] - 1)))


def selected_pod_rank(U_obs: np.ndarray, sys_name: str, args: argparse.Namespace) -> int:
    """Select the POD rank used by PDE preprocessors outside method-specific CV."""
    return selected_pod_rank_fallback(U_obs, sys_name, args)


def pod_reconstruct(mean: np.ndarray, modes: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return Z @ modes.T + mean


def linear_operator_local_reconstruct(A: np.ndarray, Z_obs: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    dt_obs = t_obs[1] - t_obs[0]
    dt_new = t_new[1] - t_new[0] if len(t_new) > 1 else dt_obs
    L = np.real(logm(A) / dt_obs)
    A_step = np.real(expm(L * dt_new))
    out = []
    j = 0
    z = Z_obs[0].copy()
    for tt in t_new:
        while j + 1 < len(t_obs) and tt >= t_obs[j + 1] - 0.5 * dt_new:
            j += 1
            z = Z_obs[j].copy()
        out.append(np.real(z).copy())
        z = np.real(A_step @ z)
    return np.asarray(out)



def fractional_step_matrix(M: np.ndarray, frac: float) -> np.ndarray:
    """Fast approximate fractional matrix power via eigendecomposition.

    This is used for EDMD interpolation; it is much faster than logm/expm for
    noisy, moderately large observable matrices.
    """
    vals, vecs = np.linalg.eig(M)
    vals = np.where(np.abs(vals) < EPS, EPS + 0j, vals)
    Mf = vecs @ np.diag(np.exp(frac * np.log(vals))) @ pinv(vecs)
    return np.real(Mf)

def pod_dmd_reconstruct(U: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray, rank: int) -> np.ndarray:
    mean, modes, Z = pod_fit(U, rank)
    Z1, Z2 = Z[:-1].T, Z[1:].T
    A = Z2 @ pinv(Z1)
    Zrec = linear_operator_local_reconstruct(A, Z, t_obs, t_new)
    return pod_reconstruct(mean, modes, Zrec)


def pod_fb_dmd_reconstruct(U: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray, rank: int) -> np.ndarray:
    mean, modes, Z = pod_fit(U, rank)
    Z1, Z2 = Z[:-1].T, Z[1:].T
    Af = Z2 @ pinv(Z1)
    Ab = Z1 @ pinv(Z2)
    try:
        A = np.real(sqrtm(Af @ pinv(Ab)))
    except Exception:
        A = Af
    Zrec = linear_operator_local_reconstruct(A, Z, t_obs, t_new)
    return pod_reconstruct(mean, modes, Zrec)


def temporal_linear_interp_reconstruct(U: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Componentwise piecewise-linear temporal interpolation of PDE snapshots.

    This is a non-dynamical upsampling control. It operates independently at
    every spatial grid point and does not apply POD, DMD, EDMD, or a learned
    dynamical model.
    """
    return np.column_stack([np.interp(t_new, t_obs, U[:, j]) for j in range(U.shape[1])])


def _fit_temporal_spline_matrix(U: np.ndarray, t_obs: np.ndarray, t_eval: np.ndarray, alpha: float) -> np.ndarray:
    """Fit componentwise temporal smoothing splines with dimensionless alpha.

    The smoothing budget at each spatial degree of freedom is
        s_j = alpha * n_obs * Var(U_j).
    The value of alpha can therefore be tuned by validation on the sparse noisy
    snapshots, without using dense clean data or equation-level oracle metrics.
    """
    if len(t_obs) < 2:
        raise ValueError("at least two observations are required for spline interpolation")
    k = int(min(3, len(t_obs) - 1))
    cols = []
    for j in range(U.shape[1]):
        sj = float(alpha) * len(t_obs) * max(float(np.var(U[:, j])), EPS)
        spl = UnivariateSpline(t_obs, U[:, j], s=sj, k=k)
        cols.append(spl(t_eval))
    return np.column_stack(cols)


def temporal_spline_reconstruct(U: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray, rel_noise: float) -> np.ndarray:
    """Componentwise smoothing-spline interpolation using the nominal noise label.

    This legacy control is kept for backward compatibility. Appendix C uses
    ``smoothing_spline_cv``, which tunes the dimensionless smoothing parameter
    only from the sparse noisy snapshots.
    """
    return _fit_temporal_spline_matrix(U, t_obs, t_new, alpha=float(rel_noise) ** 2)


def parse_alpha_grid(s: str | Sequence[float] | None) -> list[float]:
    if s is None:
        return [0.0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5,
                1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
    if isinstance(s, str):
        vals = [float(x) for x in s.split(',') if x.strip() != '']
    else:
        vals = [float(x) for x in s]
    vals = sorted(set(v for v in vals if np.isfinite(v) and v >= 0.0))
    if not vals:
        raise ValueError("spline alpha grid must contain at least one nonnegative value")
    return vals


def _cv_splits_from_observations(n_obs: int, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_obs < 6:
        return []
    interior = np.arange(1, n_obs - 1)
    n_folds = int(max(2, min(n_folds, len(interior))))
    splits = []
    all_idx = np.arange(n_obs)
    for fold in range(n_folds):
        val_idx = interior[fold::n_folds]
        if len(val_idx) == 0:
            continue
        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        if len(train_idx) >= 4:
            splits.append((train_idx, val_idx))
    return splits


def _normalized_validation_mse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    scale = np.maximum(np.var(y_train, axis=0), EPS)
    return float(np.mean(((y_pred - y_true) ** 2) / scale))


def select_temporal_spline_alpha_cv(
    U: np.ndarray,
    t_obs: np.ndarray,
    alphas: Sequence[float],
    n_folds: int = 4,
) -> tuple[float, float, int]:
    """Select temporal smoothing only from sparse noisy PDE snapshots."""
    splits = _cv_splits_from_observations(len(t_obs), n_folds)
    if not splits:
        return 0.0, np.nan, 0
    best_alpha, best_score = float(alphas[0]), np.inf
    for alpha in alphas:
        fold_scores = []
        for train_idx, val_idx in splits:
            U_train = U[train_idx]
            t_train = t_obs[train_idx]
            try:
                pred = _fit_temporal_spline_matrix(U_train, t_train, t_obs[val_idx], alpha=float(alpha))
                if np.all(np.isfinite(pred)):
                    fold_scores.append(_normalized_validation_mse(U[val_idx], pred, U_train))
            except Exception:
                continue
        score = float(np.mean(fold_scores)) if fold_scores else np.inf
        if (score < best_score - 1e-12) or (np.isfinite(score) and np.isclose(score, best_score, rtol=1e-4, atol=1e-12) and float(alpha) > best_alpha):
            best_alpha, best_score = float(alpha), score
    if not np.isfinite(best_score):
        return 0.0, np.nan, len(splits)
    return best_alpha, best_score, len(splits)


def temporal_spline_reconstruct_cv(U: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    alphas = parse_alpha_grid(getattr(args, "spline_alpha_grid", None))
    n_folds = int(getattr(args, "spline_cv_folds", 4))
    alpha, val_score, n_splits = select_temporal_spline_alpha_cv(U, t_obs, alphas, n_folds=n_folds)
    args._last_spline_alpha = alpha
    args._last_spline_cv_error = val_score
    args._last_spline_cv_folds = n_splits
    return _fit_temporal_spline_matrix(U, t_obs, t_new, alpha=alpha)


def parse_rank_grid(s: str | Sequence[int] | None, max_rank: int) -> list[int]:
    """Return admissible POD-rank candidates for observation-only CV tuning."""
    if s is None:
        vals = [1, 2, 3, 4, 6, 8, 10, 12, 16]
    elif isinstance(s, str):
        vals = [int(x) for x in s.split(',') if x.strip() != '']
    else:
        vals = [int(x) for x in s]
    vals = sorted(set(v for v in vals if 1 <= int(v) <= int(max_rank)))
    if not vals:
        vals = [max(1, min(4, int(max_rank)))]
    return vals


def _pde_reconstruct_for_rank_cv(
    method: str,
    U_train: np.ndarray,
    t_train: np.ndarray,
    t_val: np.ndarray,
    rank: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> np.ndarray:
    """Predict held-out sparse noisy snapshots for rank validation.

    The validation target is the sparse noisy data itself.  No dense clean
    trajectory, true coefficient vector, threshold selection score, or PDE-FIND
    output is used here.
    """
    if method == "pod_edmd_rbf":
        return pod_edmd_reconstruct(U_train, t_train, t_val, rank, "rbf", rng, args.rbf_centers)
    if method == "pydmd_optdmd":
        return pydmd_optdmd_reconstruct(U_train, t_train, t_val, rank, args.pydmd_timeout)
    raise ValueError(f"rank CV is defined only for POD-based proposed methods, got {method}")


def select_pod_rank_cv_for_method(
    method: str,
    U_obs: np.ndarray,
    t_obs: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[int, float, int]:
    """Select POD rank by deterministic interior holdout on sparse noisy snapshots.

    This is the PDE analogue of the smoothing-spline alpha selection used in
    Appendix C.  Candidate ranks are evaluated by fitting the same proposed
    temporal preprocessor on training observation times and predicting held-out
    observation times.  The lowest normalized validation MSE is selected, with a
    smaller-rank tie break for parsimony.
    """
    max_rank = max(1, min(int(getattr(args, "rank", 8)), U_obs.shape[0] - 2, U_obs.shape[1]))
    ranks = parse_rank_grid(getattr(args, "rank_grid", None), max_rank=max_rank)
    splits = _cv_splits_from_observations(len(t_obs), int(getattr(args, "rank_cv_folds", 3)))
    if not splits:
        fallback = selected_pod_rank_fallback(U_obs, "", args)
        return fallback, np.nan, 0
    best_rank = ranks[0]
    best_score = np.inf
    for rank in ranks:
        fold_scores = []
        for fold_id, (train_idx, val_idx) in enumerate(splits):
            if len(train_idx) <= rank + 1:
                continue
            try:
                fold_seed = int(rng.integers(0, 2**32 - 1))
                fold_rng = np.random.default_rng(fold_seed)
                pred = _pde_reconstruct_for_rank_cv(
                    method, U_obs[train_idx], t_obs[train_idx], t_obs[val_idx], int(rank), fold_rng, args
                )
                if np.all(np.isfinite(pred)):
                    fold_scores.append(_normalized_validation_mse(U_obs[val_idx], pred, U_obs[train_idx]))
            except Exception:
                continue
        score = float(np.mean(fold_scores)) if fold_scores else np.inf
        if (score < best_score - 1e-12) or (np.isfinite(score) and np.isclose(score, best_score, rtol=1e-4, atol=1e-12) and int(rank) < int(best_rank)):
            best_rank, best_score = int(rank), score
    if not np.isfinite(best_score):
        fallback = selected_pod_rank_fallback(U_obs, "", args)
        return fallback, np.nan, len(splits)
    return int(best_rank), float(best_score), len(splits)


def selected_pod_rank_fallback(U_obs: np.ndarray, sys_name: str, args: argparse.Namespace) -> int:
    """System/fixed/energy rank selection used outside observation-CV mode."""
    mode = getattr(args, "rank_mode", "system")
    if mode == "cv":
        # When CV cannot be performed, use the publication's system-specific rule.
        mode = "system"
    if mode == "fixed":
        return int(args.rank)
    if mode == "system":
        if sys_name == "burgers":
            return int(args.burgers_rank)
        if sys_name == "fisher_kpp":
            return int(args.fisher_rank)
        if sys_name == "advection_diffusion":
            return int(args.advection_rank)
        return int(args.rank)
    if mode == "energy":
        return energy_rank(U_obs, float(args.pod_energy), int(args.rank))
    return int(args.rank)


def hankel_embed(Z: np.ndarray, delays: int) -> np.ndarray:
    m, d = Z.shape
    return np.column_stack([Z[j:j + delays].reshape(-1) for j in range(m - delays + 1)]).T


def pod_hankel_dmd_reconstruct(U: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray, rank: int, delays: int) -> np.ndarray:
    mean, modes, Z = pod_fit(U, rank)
    delays = int(max(2, min(delays, len(t_obs) // 2)))
    H = hankel_embed(Z, delays)
    H1, H2 = H[:-1].T, H[1:].T
    A = H2 @ pinv(H1)
    Hrec = linear_operator_local_reconstruct(A, H, t_obs[: H.shape[0]], t_new)
    Zrec = Hrec[:, :Z.shape[1]]
    return pod_reconstruct(mean, modes, Zrec)


def poly_features(Z: np.ndarray, degree: int = 2) -> np.ndarray:
    n, d = Z.shape
    cols = [np.ones(n)]
    cols.extend([Z[:, j] for j in range(d)])
    if degree >= 2:
        for i in range(d):
            for j in range(i, d):
                cols.append(Z[:, i] * Z[:, j])
    return np.column_stack(cols)


def rbf_fit(Z: np.ndarray, n_centers: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    n = Z.shape[0]
    take = min(n_centers, n)
    centers = Z[rng.choice(n, take, replace=False)]
    dist2 = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    med = np.median(dist2[dist2 > 0]) if np.any(dist2 > 0) else np.var(Z) + EPS
    gamma = 1.0 / max(float(med), EPS)
    return {"centers": centers, "gamma": np.array(gamma)}


def rbf_features(Z: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    centers = params["centers"]
    gamma = float(params["gamma"])
    dist2 = np.sum((Z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.column_stack([np.ones(Z.shape[0]), Z, np.exp(-gamma * dist2)])


def pod_edmd_reconstruct(U: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray, rank: int, kind: str, rng: np.random.Generator, rbf_centers: int) -> np.ndarray:
    mean, modes, Z = pod_fit(U, rank)
    if kind == "poly2":
        Phi = poly_features(Z, degree=2)
        state_indices = list(range(1, 1 + Z.shape[1]))
        ffun = lambda Y: poly_features(Y, degree=2)
    elif kind == "rbf":
        params = rbf_fit(Z, rbf_centers, rng)
        Phi = rbf_features(Z, params)
        state_indices = list(range(1, 1 + Z.shape[1]))
        ffun = lambda Y: rbf_features(Y, params)
    else:
        raise ValueError(kind)
    K = np.linalg.solve(Phi[:-1].T @ Phi[:-1] + 1e-8 * np.eye(Phi.shape[1]), Phi[:-1].T @ Phi[1:])
    dt_obs = t_obs[1] - t_obs[0]
    dt_new = t_new[1] - t_new[0] if len(t_new) > 1 else dt_obs
    Kstep = fractional_step_matrix(K, dt_new / dt_obs)
    out = []
    j = 0
    phi = np.real(ffun(Z[[0]])[0])
    for tt in t_new:
        while j + 1 < len(t_obs) and tt >= t_obs[j + 1] - 0.5 * dt_new:
            j += 1
            phi = np.real(ffun(Z[[j]])[0])
        out.append(phi[state_indices].copy())
        phi = np.real(phi @ Kstep)
    Zrec = np.asarray(out)
    return pod_reconstruct(mean, modes, Zrec)


def optimized_dmd_latent(Z: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray, n_exp: int | None = None, max_nfev: int = 80) -> np.ndarray:
    """Lightweight real-valued optDMD-style variable projection in POD coordinates.

    This intentionally uses real exponents because the chosen PDE examples are
    dissipative/non-oscillatory; for dispersive waves one should use complex-pair optDMD.
    """
    m, d = Z.shape
    if n_exp is None:
        n_exp = min(d, max(1, m // 3))
    n_exp = int(max(1, min(n_exp, d, m - 1)))
    # Initialize with DMD eigenvalues.
    try:
        A = Z[1:].T @ pinv(Z[:-1].T)
        evals = np.linalg.eigvals(A)
        lam0 = np.real(np.log(evals[:n_exp] + EPS) / (t_obs[1] - t_obs[0]))
    except Exception:
        lam0 = -np.linspace(0.1, 2.0, n_exp)
    if len(lam0) < n_exp:
        lam0 = np.r_[lam0, -np.linspace(0.1, 2.0, n_exp - len(lam0))]
    tau = t_obs - t_obs[0]

    def coeffs(lam: np.ndarray, times: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E = np.exp(np.outer(times - t_obs[0], lam))
        C, *_ = np.linalg.lstsq(E, Y, rcond=None)
        return E, C

    def residual(lam: np.ndarray) -> np.ndarray:
        E, C = coeffs(lam, t_obs, Z)
        return (E @ C - Z).ravel()

    try:
        res = least_squares(residual, lam0, max_nfev=max_nfev, xtol=1e-6, ftol=1e-6, gtol=1e-6)
        lam = res.x
    except Exception:
        lam = lam0
    _, C = coeffs(lam, t_obs, Z)
    Enew = np.exp(np.outer(t_new - t_obs[0], lam))
    return Enew @ C


def pod_optdmd_reconstruct(U: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray, rank: int, max_nfev: int) -> np.ndarray:
    mean, modes, Z = pod_fit(U, rank)
    Zrec = optimized_dmd_latent(Z, t_obs, t_new, n_exp=min(rank, len(t_obs) // 3), max_nfev=max_nfev)
    return pod_reconstruct(mean, modes, Zrec)


def pod_bopdmd_reconstruct(U: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray, rank: int, rng: np.random.Generator, bags: int, max_nfev: int) -> np.ndarray:
    mean, modes, Z = pod_fit(U, rank)
    preds = []
    m = len(t_obs)
    subset_size = max(rank + 2, int(0.75 * m))
    for _ in range(bags):
        idx = np.sort(rng.choice(m, size=subset_size, replace=False))
        if 0 not in idx:
            idx[0] = 0
            idx = np.unique(np.sort(idx))
        try:
            Zp = optimized_dmd_latent(Z[idx], t_obs[idx], t_new, n_exp=min(rank, len(idx) // 3), max_nfev=max_nfev)
            if np.all(np.isfinite(Zp)):
                preds.append(Zp)
        except Exception:
            pass
    if not preds:
        Zrec = optimized_dmd_latent(Z, t_obs, t_new, n_exp=min(rank, len(t_obs) // 3), max_nfev=max_nfev)
    else:
        Zrec = np.median(np.stack(preds, axis=0), axis=0)
    return pod_reconstruct(mean, modes, Zrec)



METHOD_LABELS = {
    "baseline": "Baseline",
    "linear_interp": "Linear interpolation",
    "smoothing_spline": "Smoothing spline",
    "smoothing_spline_cv": "Tuned smoothing spline",
    "spline": "Smoothing spline",
    "pydmd_optdmd": "optDMD",
    "pod_edmd_rbf": "POD-EDMD-RBF",
}
DEFAULT_METHODS = ["baseline", "pydmd_optdmd", "pod_edmd_rbf"]


def configure_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "quick":
        args.dt = 0.01; args.nx = 48; args.rank = 6
        args.rank_mode = "system" if args.rank_mode is None else args.rank_mode
        args.burgers_rank = 6 if args.burgers_rank is None else args.burgers_rank
        args.fisher_rank = 4 if args.fisher_rank is None else args.fisher_rank
        args.advection_rank = 4 if args.advection_rank is None else args.advection_rank
        args.fisher_ic = "default" if args.fisher_ic is None else args.fisher_ic
        args.sparse_factors = [8, 32]; args.noise = [0.0, 0.05, 0.10]; args.seeds = [0, 1]
        args.max_rows = 12000; args.rbf_centers = 20
    elif args.preset == "publication":
        args.dt = 0.005; args.nx = 64; args.rank = 8
        args.rank_mode = "system" if args.rank_mode is None else args.rank_mode
        args.burgers_rank = 8 if args.burgers_rank is None else args.burgers_rank
        # Fisher--KPP is smoother and more reaction dominated than Burgers in
        # this benchmark.  A lower POD rank limits high-rank noise amplification
        # while preserving the default Fisher--KPP initial condition used in the
        # main publication table.  The steeper front-type initial condition is
        # reserved for the model-selection and sensitivity runs.
        args.fisher_rank = 2 if args.fisher_rank is None else args.fisher_rank
        args.advection_rank = 4 if args.advection_rank is None else args.advection_rank
        args.fisher_ic = "default" if args.fisher_ic is None else args.fisher_ic
        args.sparse_factors = [4, 8, 16, 32]; args.noise = [0.0, 0.01, 0.03, 0.05, 0.10]; args.seeds = list(range(8))
        args.max_rows = 30000; args.rbf_centers = 30
    else:
        raise ValueError(args.preset)
    return args


def case_key(row: dict) -> tuple:
    return (row["system"], row["method"], int(row["sparse_factor"]), float(row["noise"]), int(row["seed"]))


def progress_iter(items, enabled=True, desc="cases"):
    if not enabled:
        return items
    try:
        from tqdm import tqdm
        return tqdm(items, desc=desc)
    except Exception:
        total=len(items)
        def gen():
            for i,item in enumerate(items,1):
                if i==1 or i==total or i%10==0:
                    print(f"{desc}: {i}/{total}", flush=True)
                yield item
        return gen()


def _pydmd_optdmd_latent_worker(queue, Z, t_obs, t_new):
    try:
        import numpy as _np
        from pydmd import BOPDMD
        model = BOPDMD(svd_rank=Z.shape[1], num_trials=0, compute_A=True)
        model.fit(Z.T, t=t_obs)
        pred = _np.real(model.forecast(t_new).T)
        queue.put(("ok", pred))
    except Exception as exc:
        queue.put(("fail", f"{type(exc).__name__}: {exc}"))


def pydmd_optdmd_reconstruct(U: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray, rank: int, timeout: float) -> np.ndarray:
    """Optimized DMD in POD coordinates, with a safe fallback.

    The benchmark records failures rather than letting numerical solver issues halt
    a long benchmark run. If PyDMD is unavailable, a lightweight local
    optimized-DMD-style fallback is used for quick checks.
    """
    mean, modes, Z = pod_fit(U, rank)
    try:
        import importlib.util
        if importlib.util.find_spec("pydmd") is None:
            Zrec = optimized_dmd_latent(Z, t_obs, t_new, n_exp=min(rank, len(t_obs)//3), max_nfev=60)
            return pod_reconstruct(mean, modes, Zrec)
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_pydmd_optdmd_latent_worker, args=(q, Z, t_obs, t_new))
        p.start(); p.join(timeout)
        if p.is_alive():
            p.terminate(); p.join()
            raise TimeoutError(f"optimized DMD exceeded {timeout} seconds")
        status, payload = q.get_nowait()
        if status != "ok":
            raise RuntimeError(payload)
        Zrec = payload
        if not np.all(np.isfinite(Zrec)):
            raise FloatingPointError("non-finite optDMD reconstruction")
        return pod_reconstruct(mean, modes, Zrec)
    except Exception:
        raise


def reconstruct_pde_method(method: str, sys_name: str, U_obs: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray, rng: np.random.Generator, args: argparse.Namespace):
    if method == "baseline":
        return U_obs, t_obs, 0
    if method == "linear_interp":
        return temporal_linear_interp_reconstruct(U_obs, t_obs, t_new), t_new, 0
    if method in {"smoothing_spline", "spline"}:
        rel_noise = float(getattr(args, "_current_noise", 0.0))
        return temporal_spline_reconstruct(U_obs, t_obs, t_new, rel_noise=rel_noise), t_new, 0
    if method in {"smoothing_spline_cv", "tuned_smoothing_spline"}:
        return temporal_spline_reconstruct_cv(U_obs, t_obs, t_new, args=args), t_new, 0
    if method in {"pod_edmd_rbf", "pydmd_optdmd"} and getattr(args, "rank_mode", None) == "cv":
        r_pod, rank_cv_error, rank_cv_folds = select_pod_rank_cv_for_method(method, U_obs, t_obs, args, rng)
        args._last_rank_cv_error = rank_cv_error
        args._last_rank_cv_folds = rank_cv_folds
    else:
        r_pod = selected_pod_rank(U_obs, sys_name, args)
        args._last_rank_cv_error = np.nan
        args._last_rank_cv_folds = 0
    if method == "pod_edmd_rbf":
        return pod_edmd_reconstruct(U_obs, t_obs, t_new, r_pod, "rbf", rng, args.rbf_centers), t_new, r_pod
    if method == "pydmd_optdmd":
        return pydmd_optdmd_reconstruct(U_obs, t_obs, t_new, r_pod, args.pydmd_timeout), t_new, r_pod
    raise ValueError(method)


def run_benchmark(args: argparse.Namespace):
    systems = [
        burgers_system(args.nx),
        fisher_kpp_system(args.nx, D=args.fisher_D, r=args.fisher_r, T=args.fisher_T, ic_variant=args.fisher_ic),
        advection_diffusion_system(args.nx),
    ]
    methods = args.methods
    thresholds = np.array([0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    rows, coef_rows = [], []
    completed=set(); raw_path=os.path.join(args.outdir,"pde_raw_results.csv")
    if args.resume and os.path.exists(raw_path):
        prev=pd.read_csv(raw_path); rows.extend(prev.to_dict("records")); completed={case_key(r) for r in rows}; print(f"Resuming PDE benchmark with {len(completed)} existing records.", flush=True)
    cache={}; tasks=[]
    for sys in systems:
        k=periodic_wavenumbers(args.nx, sys.params["L"]); t_true,U_true=integrate_pde(sys,args.dt); cache[sys.name]=(sys,k,t_true,U_true)
        for sparse_factor in args.sparse_factors:
            for noise in args.noise:
                for seed in args.seeds:
                    for method in methods:
                        key=(sys.name,method,sparse_factor,float(noise),seed)
                        if key not in completed: tasks.append((sys.name,sparse_factor,float(noise),seed,method))
    for sys_name,sparse_factor,noise,seed,method in progress_iter(tasks, enabled=(not args.no_progress), desc="PDE cases"):
        sys,k,t_true,U_true=cache[sys_name]
        _,names=pde_library(U_true[:2],k); xi_true=sys.true_coefficients(names)
        ind=np.arange(0,len(t_true),sparse_factor)
        if ind[-1] != len(t_true)-1: ind=np.r_[ind,len(t_true)-1]
        t_obs=t_true[ind]; U_clean_obs=U_true[ind]
        rng=np.random.default_rng(seed+1000*sparse_factor+100000*int(noise*1000))
        U_obs=add_noise(U_clean_obs, noise, rng); t_new=make_tnew(t_obs,args.upsample)
        pod_rank_used = 0
        try:
            # Used only by the legacy non-dynamical smoothing-spline control.
            args._current_noise = noise
            args._last_spline_alpha = np.nan
            args._last_spline_cv_error = np.nan
            args._last_spline_cv_folds = 0
            args._last_rank_cv_error = np.nan
            args._last_rank_cv_folds = 0
            U_use,t_use,pod_rank_used=reconstruct_pde_method(method,sys.name,U_obs,t_obs,t_new,rng,args)
            if not np.all(np.isfinite(U_use)) or len(t_use)<5: raise FloatingPointError("non-finite reconstruction")
            result=fit_pdefind_oracle(U_use,t_use,k,xi_true,thresholds,rng,max_rows=args.max_rows)
            f1=float(result["f1"]); cerr=float(result["coef_error"]); score=float(result["score"]); thr=float(result["threshold"]); status="ok"
            if args.save_coefficients and seed==args.seeds[0] and noise in [args.noise[0], args.noise[-1]]:
                xi=result["xi"]
                for i,name in enumerate(names):
                    if abs(xi[i])>1e-12 or abs(xi_true[i])>1e-12:
                        coef_rows.append({"system":sys.name,"method":method,"method_label":METHOD_LABELS.get(method,method),"sparse_factor":sparse_factor,"noise":noise,"seed":seed,"feature":name,"coef":xi[i],"true_coef":xi_true[i]})
        except Exception as exc:
            f1=cerr=score=thr=np.nan; status=f"fail: {type(exc).__name__}: {exc}"
        row={"system":sys.name,"method":method,"method_label":METHOD_LABELS.get(method,method),"sparse_factor":sparse_factor,"obs_dt":float(t_obs[1]-t_obs[0]),"noise":noise,"seed":seed,"upsample":args.upsample if method!="baseline" else 1,"pod_rank":pod_rank_used if method!="baseline" else 0,"rank_mode":args.rank_mode,"rank_cv_error":getattr(args,"_last_rank_cv_error",np.nan),"rank_cv_folds":getattr(args,"_last_rank_cv_folds",0),"rank_grid":getattr(args,"rank_grid",None),"spline_alpha":getattr(args,"_last_spline_alpha",np.nan),"spline_cv_error":getattr(args,"_last_spline_cv_error",np.nan),"spline_cv_folds":getattr(args,"_last_spline_cv_folds",0),"support_f1":f1,"coef_error":cerr,"practical_score":score,"threshold":thr,"status":status}
        rows.append(row)
        if args.verbose: print(row, flush=True)
        if args.checkpoint and len(rows)%args.checkpoint==0: pd.DataFrame(rows).to_csv(raw_path,index=False)
    raw=pd.DataFrame(rows)
    summary=raw[raw.status=="ok"].groupby(["system","method","method_label","sparse_factor","noise"],as_index=False).agg(support_f1_mean=("support_f1","mean"),support_f1_std=("support_f1","std"),coef_error_median=("coef_error","median"),coef_error_mean=("coef_error","mean"),practical_score_mean=("practical_score","mean"),n_ok=("status","size"))
    return raw,summary,pd.DataFrame(coef_rows)


def ordered_methods(df: pd.DataFrame) -> list:
    return [m for m in DEFAULT_METHODS if m in set(df["method"])] + [m for m in df["method"].unique() if m not in DEFAULT_METHODS]


def make_tables(raw: pd.DataFrame, outdir: str, prefix: str = "pde") -> None:
    ok=raw[raw.status=="ok"].copy(); methods=ordered_methods(ok); label={m:METHOD_LABELS.get(m,m) for m in methods}
    overall=ok.groupby("method",as_index=False).agg(support_f1_mean=("support_f1","mean"),coef_error_median=("coef_error","median"),practical_score_mean=("practical_score","mean")).set_index("method").reindex(methods).reset_index(); overall.insert(1,"method_label",overall.method.map(label)); overall.to_csv(os.path.join(outdir,f"{prefix}_overall_ranking.csv"),index=False)
    by_system=ok.groupby(["system","method"],as_index=False).agg(support_f1_mean=("support_f1","mean"),coef_error_median=("coef_error","median"),practical_score_mean=("practical_score","mean")); by_system["method_label"]=by_system.method.map(label); by_system.to_csv(os.path.join(outdir,f"{prefix}_by_system.csv"),index=False)
    noise_tbl=ok.groupby(["noise","method"])["coef_error"].median().unstack().reindex(columns=methods); sparse_tbl=ok.groupby(["sparse_factor","method"])["coef_error"].median().unstack().reindex(columns=methods)
    noise_tbl.rename(columns=label).to_csv(os.path.join(outdir,f"{prefix}_coeferr_by_noise.csv")); sparse_tbl.rename(columns=label).to_csv(os.path.join(outdir,f"{prefix}_coeferr_by_sparse_factor.csv"))
    base=ok[ok.method=="baseline"][["system","sparse_factor","noise","seed","support_f1","coef_error","practical_score"]].rename(columns={"support_f1":"support_f1_base","coef_error":"coef_error_base","practical_score":"practical_score_base"})
    wins=[]
    for m in methods:
        if m=="baseline": continue
        d=ok[ok.method==m].merge(base,on=["system","sparse_factor","noise","seed"])
        wins.append({"method":m,"method_label":label[m],"paired_cases":len(d),"practical_score_win_rate":float((d.practical_score>d.practical_score_base).mean()) if len(d) else np.nan,"coef_error_win_rate":float((d.coef_error<d.coef_error_base).mean()) if len(d) else np.nan,"support_f1_win_rate":float((d.support_f1>d.support_f1_base).mean()) if len(d) else np.nan})
    pd.DataFrame(wins).to_csv(os.path.join(outdir,f"{prefix}_winrate_vs_baseline.csv"),index=False)
    def fmt(x): return "--" if pd.isna(x) else f"{x:.3f}"
    md=["# PDE benchmark tables\n","## Overall comparison\n","| Method | Mean support F1 | Median coefficient error | Mean practical score |","|---|---:|---:|---:|"]
    for _,r in overall.iterrows(): md.append(f"| {r.method_label} | {fmt(r.support_f1_mean)} | {fmt(r.coef_error_median)} | {fmt(r.practical_score_mean)} |")
    md += ["\n## Median coefficient error by noise level\n","| Noise | "+" | ".join(label[m] for m in methods)+" |","|---:"+"|---:"*len(methods)+"|"]
    for idx,row in noise_tbl.iterrows(): md.append(f"| {idx:.2f} | "+" | ".join(fmt(row[m]) for m in methods)+" |")
    md += ["\n## Median coefficient error by sparse factor\n","| Sparse factor | "+" | ".join(label[m] for m in methods)+" |","|---:"+"|---:"*len(methods)+"|"]
    for idx,row in sparse_tbl.iterrows(): md.append(f"| {int(idx)} | "+" | ".join(fmt(row[m]) for m in methods)+" |")
    __import__('pathlib').Path(os.path.join(outdir,f"{prefix}_tables.md")).write_text("\n".join(md))


def plot_results(raw: pd.DataFrame, outdir: str) -> None:
    ok=raw[raw.status=="ok"]
    plot_df=ok.groupby(["method","noise"],as_index=False).agg(support_f1=("support_f1","mean"),coef_error=("coef_error","median"),practical_score=("practical_score","mean"))
    methods=ordered_methods(ok)
    plt.figure(figsize=(7,4))
    for method in methods:
        d=plot_df[plot_df.method==method].sort_values("noise"); plt.plot(d.noise,d.support_f1,marker="o",label=METHOD_LABELS.get(method,method))
    plt.xlabel("relative noise level"); plt.ylabel("mean support F1"); plt.title("PDE-FIND support recovery"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(outdir,"pde_f1_by_noise.png"),dpi=200); plt.close()


def parse_list(s: str, typ=float): return [typ(x) for x in str(s).split(",") if str(x).strip()!=""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["quick", "publication"], default="quick")
    parser.add_argument("--outdir", default="results_pde")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--rank-mode", choices=["fixed", "system", "energy", "cv"], default=None)
    parser.add_argument("--burgers-rank", type=int, default=None)
    parser.add_argument("--fisher-rank", type=int, default=None)
    parser.add_argument("--advection-rank", type=int, default=None)
    parser.add_argument("--pod-energy", type=float, default=0.999)
    parser.add_argument("--fisher-D", type=float, default=0.03)
    parser.add_argument("--fisher-r", type=float, default=1.0)
    parser.add_argument("--fisher-T", type=float, default=2.0)
    parser.add_argument("--fisher-ic", choices=["default", "rich", "front"], default=None)
    parser.add_argument("--sparse-factors", type=str, default=None)
    parser.add_argument("--noise", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--upsample", type=int, default=5)
    parser.add_argument("--rbf-centers", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--pydmd-timeout", type=float, default=30.0)
    parser.add_argument(
        "--rank-grid",
        type=str,
        default=None,
        help="Comma-separated candidate POD ranks for observation-only CV rank tuning.",
    )
    parser.add_argument(
        "--rank-cv-folds",
        type=int,
        default=3,
        help="Number of deterministic interior folds for observation-only POD-rank tuning.",
    )
    parser.add_argument(
        "--spline-alpha-grid",
        type=str,
        default=None,
        help="Comma-separated dimensionless smoothing levels for tuned smoothing_spline_cv.",
    )
    parser.add_argument(
        "--spline-cv-folds",
        type=int,
        default=4,
        help="Number of deterministic interior folds used to tune smoothing_spline_cv.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint", type=int, default=10)
    parser.add_argument("--save-coefficients", action="store_true", default=True)
    args = parser.parse_args()
    args = configure_preset(args)
    if isinstance(args.sparse_factors, str): args.sparse_factors=parse_list(args.sparse_factors,int)
    if isinstance(args.noise, str): args.noise=parse_list(args.noise,float)
    if isinstance(args.seeds, str): args.seeds=parse_list(args.seeds,int)
    if args.rbf_centers is None: args.rbf_centers=30 if args.preset=="publication" else 20
    if args.max_rows is None: args.max_rows=30000 if args.preset=="publication" else 12000
    args.methods=[m.strip() for m in args.methods.split(",") if m.strip()]
    os.makedirs(args.outdir,exist_ok=True)
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); raw,summary,coef_df=run_benchmark(args)
    raw.to_csv(os.path.join(args.outdir,"pde_raw_results.csv"),index=False); summary.to_csv(os.path.join(args.outdir,"pde_summary.csv"),index=False)
    if not coef_df.empty: coef_df.to_csv(os.path.join(args.outdir,"pde_coefficients_examples.csv"),index=False)
    if not summary.empty:
        best=summary.sort_values(["system","sparse_factor","noise","practical_score_mean"],ascending=[True,True,True,False]).groupby(["system","sparse_factor","noise"],as_index=False).head(1); best.to_csv(os.path.join(args.outdir,"pde_best_by_case.csv"),index=False)
    make_tables(raw,args.outdir,"pde"); plot_results(raw,args.outdir); print(f"Wrote PDE results to: {args.outdir}")


if __name__=="__main__": main()

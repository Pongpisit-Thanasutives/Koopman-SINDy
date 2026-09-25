#!/usr/bin/env python3
"""
Koopman-based DMD/EDMD upsampling benchmark for SINDy on two ODEs.

Systems: Lorenz-63 and Van der Pol oscillator.
Question: Does a Koopman-based DMD/EDMD interpolation-denoising step improve sparse-regression
recovery relative to using the sparse noisy samples directly?

Dependencies: numpy, scipy, pandas, matplotlib.
No PySINDy dependency; STLSQ and feature libraries are implemented here.

Typical runs:
  python koopman_sindy_ode_benchmark.py --preset quick
  python koopman_sindy_ode_benchmark.py --seeds 0,1,2,3,4 --noise 0,0.01,0.03,0.05,0.10 --sparse-factors 10,20,40

Outputs are written to --outdir:
  ode_raw_results.csv
  ode_summary.csv
  ode_best_by_case.csv
  ode_f1_by_noise.png
  ode_coeferr_by_noise.png
"""
from __future__ import annotations

import argparse
import itertools
import math
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
import matplotlib.pyplot as plt

from koopman_propagation import (FractionalEvolution, fractional_step_matrix,
                                 nominal_observation_pairs, local_observable_reconstruction)

EPS = 1e-12


@dataclass
class ODESystem:
    name: str
    dim: int
    var_names: List[str]
    rhs: Callable[[float, np.ndarray], np.ndarray]
    x0: np.ndarray
    t_span: Tuple[float, float]
    true_coefficients: Callable[[List[str]], np.ndarray]


def lorenz_system() -> ODESystem:
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def rhs(_t: float, x: np.ndarray) -> np.ndarray:
        return np.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ])

    def true_xi(names: List[str]) -> np.ndarray:
        idx = {n: i for i, n in enumerate(names)}
        Xi = np.zeros((len(names), 3))
        Xi[idx["x"], 0] = -sigma
        Xi[idx["y"], 0] = sigma
        Xi[idx["x"], 1] = rho
        Xi[idx["y"], 1] = -1.0
        Xi[idx["x*z"], 1] = -1.0
        Xi[idx["z"], 2] = -beta
        Xi[idx["x*y"], 2] = 1.0
        return Xi

    return ODESystem(
        name="lorenz63",
        dim=3,
        var_names=["x", "y", "z"],
        rhs=rhs,
        x0=np.array([-8.0, 7.0, 27.0]),
        t_span=(0.0, 10.0),
        true_coefficients=true_xi,
    )


def vanderpol_system(mu: float = 2.0) -> ODESystem:
    def rhs(_t: float, x: np.ndarray) -> np.ndarray:
        return np.array([x[1], mu * (1.0 - x[0] ** 2) * x[1] - x[0]])

    def true_xi(names: List[str]) -> np.ndarray:
        idx = {n: i for i, n in enumerate(names)}
        Xi = np.zeros((len(names), 2))
        Xi[idx["y"], 0] = 1.0
        Xi[idx["x"], 1] = -1.0
        Xi[idx["y"], 1] = mu
        Xi[idx["x*x*y"], 1] = -mu
        return Xi

    return ODESystem(
        name="vanderpol_mu2",
        dim=2,
        var_names=["x", "y"],
        rhs=rhs,
        x0=np.array([2.0, 0.0]),
        t_span=(0.0, 20.0),
        true_coefficients=true_xi,
    )


def polynomial_library(X: np.ndarray, var_names: Sequence[str], degree: int = 3) -> Tuple[np.ndarray, List[str]]:
    """Polynomial feature library up to total degree, using combinations with replacement."""
    X = np.asarray(X)
    n, d = X.shape
    cols = [np.ones(n)]
    names = ["1"]
    for deg in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(d), deg):
            col = np.prod(X[:, combo], axis=1)
            name = "*".join(var_names[i] for i in combo)
            cols.append(col)
            names.append(name)
    return np.column_stack(cols), names


def rbf_library_fit(X: np.ndarray, n_centers: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    n = X.shape[0]
    take = min(n_centers, n)
    ind = rng.choice(n, size=take, replace=False)
    centers = X[ind].copy()
    # Median heuristic for RBF width.
    if take > 1:
        diffs = centers[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diffs * diffs, axis=2)
        med = np.median(dist2[dist2 > 0]) if np.any(dist2 > 0) else 1.0
    else:
        med = np.var(X) + EPS
    gamma = 1.0 / max(med, EPS)
    return {"centers": centers, "gamma": np.array(gamma)}


def rbf_features(X: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    centers = params["centers"]
    gamma = float(params["gamma"])
    dist2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.column_stack([np.ones(X.shape[0]), X, np.exp(-gamma * dist2)])


def central_difference(X: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return states and central finite-difference derivative on interior times."""
    X = np.asarray(X)
    t = np.asarray(t)
    dt = t[2:] - t[:-2]
    dX = (X[2:] - X[:-2]) / dt[:, None]
    return X[1:-1], dX


def ridge_lstsq(A: np.ndarray, B: np.ndarray, alpha: float = 1e-8) -> np.ndarray:
    ATA = A.T @ A
    ATB = A.T @ B
    return np.linalg.solve(ATA + alpha * np.eye(ATA.shape[0]), ATB)


def stlsq(Theta: np.ndarray, Y: np.ndarray, threshold: float, alpha: float = 1e-8, max_iter: int = 12) -> np.ndarray:
    """Sequential thresholded least squares with column normalization."""
    scale = np.linalg.norm(Theta, axis=0)
    scale[scale < EPS] = 1.0
    # Do not let the intercept scale become a numerical outlier.
    Theta_s = Theta / scale
    Xi_s = ridge_lstsq(Theta_s, Y, alpha=alpha)
    for _ in range(max_iter):
        small = np.abs(Xi_s / scale[:, None]) < threshold
        Xi_s[small] = 0.0
        for k in range(Y.shape[1]):
            big = ~small[:, k]
            if np.any(big):
                Xi_s[big, k] = ridge_lstsq(Theta_s[:, big], Y[:, [k]], alpha=alpha).ravel()
            Xi_s[~big, k] = 0.0
    return Xi_s / scale[:, None]


def support_f1(Xi: np.ndarray, Xi_true: np.ndarray, tol: float = 1e-10) -> float:
    pred = np.abs(Xi) > tol
    true = np.abs(Xi_true) > tol
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    if tp + fp + fn == 0:
        return 1.0
    return float(2 * tp / max(2 * tp + fp + fn, EPS))


def coefficient_error(Xi: np.ndarray, Xi_true: np.ndarray) -> float:
    return float(np.linalg.norm(Xi - Xi_true) / max(np.linalg.norm(Xi_true), EPS))


def fit_sindy_oracle(
    X: np.ndarray,
    t: np.ndarray,
    var_names: Sequence[str],
    degree: int,
    Xi_true: np.ndarray,
    thresholds: Sequence[float],
) -> Dict[str, object]:
    X_mid, dX = central_difference(X, t)
    Theta, names = polynomial_library(X_mid, var_names, degree=degree)
    best = None
    for thr in thresholds:
        try:
            Xi = stlsq(Theta, dX, threshold=thr)
            f1 = support_f1(Xi, Xi_true)
            cerr = coefficient_error(Xi, Xi_true)
            score = f1 / (1.0 + cerr)
            cand = {"threshold": thr, "f1": f1, "coef_error": cerr, "score": score, "Xi": Xi, "names": names}
            if best is None or (score, f1, -cerr) > (best["score"], best["f1"], -best["coef_error"]):
                best = cand
        except LinAlgError:
            continue
    if best is None:
        return {"threshold": np.nan, "f1": np.nan, "coef_error": np.nan, "score": np.nan, "Xi": np.full_like(Xi_true, np.nan), "names": []}
    return best


def add_noise(X: np.ndarray, rel_noise: float, rng: np.random.Generator) -> np.ndarray:
    if rel_noise <= 0:
        return X.copy()
    sigma = rel_noise * np.std(X, axis=0, ddof=1)
    sigma[sigma < EPS] = rel_noise * max(float(np.std(X)), EPS)
    return X + rng.normal(scale=sigma, size=X.shape)


def robust_rank(X: np.ndarray, max_rank: int) -> int:
    return int(max(1, min(max_rank, min(X.shape) - 1)))


def linear_operator_local_reconstruct(A: np.ndarray, X_obs: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Continuous interpolation from a discrete operator, reset at each observed sample.

    This avoids long-horizon DMD forecast drift and makes the method an
    interpolation/denoising preprocessor rather than a stand-alone forecaster.
    """
    dt_obs = t_obs[1] - t_obs[0]
    dt_new = t_new[1] - t_new[0] if len(t_new) > 1 else dt_obs
    L = np.real(logm(A) / dt_obs)
    A_step = np.real(expm(L * dt_new))
    out = []
    j = 0
    x = X_obs[0].copy()
    for tt in t_new:
        while j + 1 < len(t_obs) and tt >= t_obs[j + 1] - 0.5 * dt_new:
            j += 1
            x = X_obs[j].copy()
        out.append(np.real(x).copy())
        x = np.real(A_step @ x)
    return np.asarray(out)


def state_dmd_reconstruct(X: np.ndarray, t: np.ndarray, t_new: np.ndarray, rank: int | None = None) -> np.ndarray:
    X1 = X[:-1].T
    X2 = X[1:].T
    U, s, Vh = svd(X1, full_matrices=False)
    if rank is None:
        rank = robust_rank(X1, min(10, X1.shape[0], X1.shape[1]))
    rank = min(rank, len(s))
    Ur, sr, Vr = U[:, :rank], s[:rank], Vh.conj().T[:, :rank]
    A = X2 @ Vr @ np.diag(1.0 / (sr + EPS)) @ Ur.T
    return linear_operator_local_reconstruct(np.real(A), X, t, t_new)


def fb_dmd_reconstruct(X: np.ndarray, t: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    X1 = X[:-1].T
    X2 = X[1:].T
    Af = X2 @ pinv(X1)
    Ab = X1 @ pinv(X2)
    try:
        A = sqrtm(Af @ pinv(Ab))
        A = np.real(A)
    except Exception:
        A = Af
    return linear_operator_local_reconstruct(A, X, t, t_new)


def tls_dmd_reconstruct(X: np.ndarray, t: np.ndarray, t_new: np.ndarray, rank: int | None = None) -> np.ndarray:
    X1 = X[:-1].T
    X2 = X[1:].T
    Z = np.vstack([X1, X2])
    _, _, Vh = svd(Z, full_matrices=False)
    if rank is None:
        rank = robust_rank(X1, min(10, X1.shape[0], X1.shape[1]))
    Vr = Vh.conj().T[:, :rank]
    X1p = X1 @ Vr @ Vr.T
    X2p = X2 @ Vr @ Vr.T
    A = X2p @ pinv(X1p)
    return linear_operator_local_reconstruct(A, X, t, t_new)


def hankel_embed(X: np.ndarray, delays: int) -> np.ndarray:
    m, d = X.shape
    cols = []
    for j in range(m - delays + 1):
        cols.append(X[j:j + delays].reshape(-1))
    return np.column_stack(cols)  # d*delays by columns


def hankel_dmd_reconstruct(X: np.ndarray, t: np.ndarray, t_new: np.ndarray, delays: int = 8, rank: int | None = None) -> np.ndarray:
    delays = int(max(2, min(delays, len(t) // 2)))
    H = hankel_embed(X, delays).T
    t_h = t[: H.shape[0]]
    Hrec = state_dmd_reconstruct(H, t_h, t_new, rank=rank)
    d = X.shape[1]
    return Hrec[:, :d]


def edmd_reconstruct(
    X: np.ndarray,
    t: np.ndarray,
    t_new: np.ndarray,
    kind: str,
    degree: int,
    var_names: Sequence[str],
    rng: np.random.Generator,
    n_centers: int = 40,
    alpha: float = 1e-8,
) -> np.ndarray:
    if kind == "poly":
        Phi, names = polynomial_library(X, var_names, degree=degree)
        state_indices = [names.index(v) for v in var_names]
        feature_fun = lambda Z: polynomial_library(Z, var_names, degree=degree)[0]
    elif kind == "rbf":
        params = rbf_library_fit(X, n_centers=n_centers, rng=rng)
        Phi = rbf_features(X, params)
        state_indices = list(range(1, 1 + X.shape[1]))
        feature_fun = lambda Z: rbf_features(Z, params)
    else:
        raise ValueError(kind)
    nominal_dt, usable_pairs = nominal_observation_pairs(t)
    Phi0, Phi1 = Phi[:-1][usable_pairs], Phi[1:][usable_pairs]
    K = ridge_lstsq(Phi0, Phi1, alpha=alpha)
    return local_observable_reconstruction(K, Phi, t, t_new, state_indices, nominal_dt)



def linear_interp_reconstruct(X: np.ndarray, t: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Componentwise piecewise-linear temporal interpolation.

    This is a non-dynamical upsampling control: it inserts intermediate
    snapshots using only the observed samples and does not estimate a flow map,
    Koopman operator, POD latent dynamics, or governing equation.
    """
    return np.column_stack([np.interp(t_new, t, X[:, j]) for j in range(X.shape[1])])


def _fit_spline_matrix(X: np.ndarray, t: np.ndarray, t_eval: np.ndarray, alpha: float) -> np.ndarray:
    """Fit componentwise smoothing splines with a dimensionless smoothing level.

    The smoothing budget for coordinate j is
        s_j = alpha * n_obs * Var(X_j).
    Thus alpha=0 gives an interpolating spline, while larger alpha values allow
    stronger smoothing. The parameter is dimensionless and can be selected by
    validation without using the clean dense trajectory or true equation.
    """
    if len(t) < 2:
        raise ValueError("at least two observations are required for spline interpolation")
    k = int(min(3, len(t) - 1))
    out = []
    for j in range(X.shape[1]):
        sj = float(alpha) * len(t) * max(float(np.var(X[:, j])), EPS)
        spl = UnivariateSpline(t, X[:, j], s=sj, k=k)
        out.append(spl(t_eval))
    return np.column_stack(out)


def spline_reconstruct(X: np.ndarray, t: np.ndarray, t_new: np.ndarray, rel_noise: float) -> np.ndarray:
    """Componentwise smoothing-spline interpolation using the nominal noise label.

    This legacy control is kept for backward compatibility. The Appendix C
    publication script uses ``smoothing_spline_cv`` instead, which tunes the
    dimensionless smoothing parameter only from sparse noisy observations.
    """
    return _fit_spline_matrix(X, t, t_new, alpha=float(rel_noise) ** 2)


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
    """Deterministic interior holdout splits for sparse observed time series.

    Endpoints are always retained in the training subset to avoid turning the
    validation problem into extrapolation. Held-out points are interior observed
    snapshots only.
    """
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


def parse_gp_kernel_grid(s: str | Sequence[str] | None) -> list[str]:
    """Return candidate GP kernels for observation-only validation.

    Each specification has the form ``family:length_scale:noise`` in normalized
    time units.  Supported families are ``rbf``, ``matern32``, ``matern52``, and
    ``rq``.  The targets are standardized coordinatewise before fitting, so the
    noise value is dimensionless.
    """
    if s is None:
        vals = [
            "rbf:0.15:1e-4",
            "rbf:0.30:1e-3",
            "matern32:0.30:1e-3",
            "matern52:0.30:1e-3",
            "rq:0.30:1e-3",
        ]
    elif isinstance(s, str):
        vals = [x.strip() for x in s.split(',') if x.strip()]
    else:
        vals = [str(x).strip() for x in s if str(x).strip()]
    if not vals:
        raise ValueError("GP kernel grid must contain at least one kernel specification")
    return vals


def _make_gp_kernel(spec: str):
    """Create a fixed scikit-learn GP kernel from a compact specification."""
    try:
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, RationalQuadratic, WhiteKernel
    except Exception as exc:  # pragma: no cover - only triggered if sklearn is absent
        raise ImportError("gp_smoothing_cv requires scikit-learn. Install requirements.txt.") from exc

    parts = spec.split(':')
    family = parts[0].strip().lower()
    length = float(parts[1]) if len(parts) > 1 and parts[1] else 0.30
    noise = float(parts[2]) if len(parts) > 2 and parts[2] else 1e-3
    length = max(length, 1e-6)
    noise = max(noise, 1e-10)

    if family == "rbf":
        base = RBF(length_scale=length, length_scale_bounds="fixed")
    elif family in {"matern32", "matern3/2"}:
        base = Matern(length_scale=length, length_scale_bounds="fixed", nu=1.5)
    elif family in {"matern52", "matern5/2"}:
        base = Matern(length_scale=length, length_scale_bounds="fixed", nu=2.5)
    elif family in {"rq", "rational_quadratic"}:
        base = RationalQuadratic(length_scale=length, alpha=1.0, length_scale_bounds="fixed", alpha_bounds="fixed")
    else:
        raise ValueError(f"Unknown GP kernel family in specification '{spec}'")
    return ConstantKernel(1.0, constant_value_bounds="fixed") * base + WhiteKernel(noise_level=noise, noise_level_bounds="fixed")


def _normalize_time_for_gp(t: np.ndarray, reference_t: np.ndarray) -> np.ndarray:
    reference_t = np.asarray(reference_t, dtype=float)
    scale = max(float(reference_t[-1] - reference_t[0]), EPS)
    return ((np.asarray(t, dtype=float) - reference_t[0]) / scale).reshape(-1, 1)


def _fit_gp_matrix(X: np.ndarray, t: np.ndarray, t_eval: np.ndarray, kernel_spec: str) -> np.ndarray:
    """Fit independent scalar GP smoothers for each coordinate and evaluate them."""
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
    except Exception as exc:  # pragma: no cover - only triggered if sklearn is absent
        raise ImportError("gp_smoothing_cv requires scikit-learn. Install requirements.txt.") from exc

    tt = _normalize_time_for_gp(t, t)
    te = _normalize_time_for_gp(t_eval, t)
    kernel = _make_gp_kernel(kernel_spec)
    out = []
    for j in range(X.shape[1]):
        y = np.asarray(X[:, j], dtype=float)
        mu = float(np.mean(y))
        sig = max(float(np.std(y)), EPS)
        y_std = (y - mu) / sig
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, optimizer=None, normalize_y=False, copy_X_train=False)
        gp.fit(tt, y_std)
        out.append(mu + sig * gp.predict(te))
    return np.column_stack(out)


def select_gp_kernel_cv(
    X: np.ndarray,
    t: np.ndarray,
    kernel_specs: Sequence[str],
    n_folds: int = 4,
) -> tuple[str, float, int]:
    """Select a GP kernel by validation on sparse noisy observations only."""
    splits = _cv_splits_from_observations(len(t), n_folds)
    if not splits:
        return str(kernel_specs[0]), np.nan, 0
    best_spec, best_score = str(kernel_specs[0]), np.inf
    for spec in kernel_specs:
        fold_scores = []
        for train_idx, val_idx in splits:
            X_train = X[train_idx]
            t_train = t[train_idx]
            try:
                pred = _fit_gp_matrix(X_train, t_train, t[val_idx], str(spec))
                if np.all(np.isfinite(pred)):
                    fold_scores.append(_normalized_validation_mse(X[val_idx], pred, X_train))
            except Exception:
                continue
        score = float(np.mean(fold_scores)) if fold_scores else np.inf
        if score < best_score - 1e-12:
            best_spec, best_score = str(spec), score
    if not np.isfinite(best_score):
        return str(kernel_specs[0]), np.nan, len(splits)
    return best_spec, best_score, len(splits)


def gp_smoothing_reconstruct_cv(X: np.ndarray, t: np.ndarray, t_new: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    kernel_specs = parse_gp_kernel_grid(getattr(args, "gp_kernel_grid", None))
    n_folds = int(getattr(args, "gp_cv_folds", 4))
    spec, val_score, n_splits = select_gp_kernel_cv(X, t, kernel_specs, n_folds=n_folds)
    args._last_gp_kernel = spec
    args._last_gp_cv_error = val_score
    args._last_gp_cv_folds = n_splits
    return _fit_gp_matrix(X, t, t_new, kernel_spec=spec)


def select_smoothing_spline_alpha_cv(
    X: np.ndarray,
    t: np.ndarray,
    alphas: Sequence[float],
    n_folds: int = 4,
) -> tuple[float, float, int]:
    """Select spline smoothing by validation on sparse noisy observations only."""
    splits = _cv_splits_from_observations(len(t), n_folds)
    if not splits:
        return 0.0, np.nan, 0
    best_alpha, best_score = float(alphas[0]), np.inf
    for alpha in alphas:
        fold_scores = []
        for train_idx, val_idx in splits:
            X_train = X[train_idx]
            t_train = t[train_idx]
            try:
                pred = _fit_spline_matrix(X_train, t_train, t[val_idx], alpha=float(alpha))
                if np.all(np.isfinite(pred)):
                    fold_scores.append(_normalized_validation_mse(X[val_idx], pred, X_train))
            except Exception:
                continue
        score = float(np.mean(fold_scores)) if fold_scores else np.inf
        # Tie-break toward the smoother model when validation scores are effectively equal.
        if (score < best_score - 1e-12) or (np.isfinite(score) and np.isclose(score, best_score, rtol=1e-4, atol=1e-12) and float(alpha) > best_alpha):
            best_alpha, best_score = float(alpha), score
    if not np.isfinite(best_score):
        return 0.0, np.nan, len(splits)
    return best_alpha, best_score, len(splits)


def spline_reconstruct_cv(X: np.ndarray, t: np.ndarray, t_new: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    alphas = parse_alpha_grid(getattr(args, "spline_alpha_grid", None))
    n_folds = int(getattr(args, "spline_cv_folds", 4))
    alpha, val_score, n_splits = select_smoothing_spline_alpha_cv(X, t, alphas, n_folds=n_folds)
    args._last_spline_alpha = alpha
    args._last_spline_cv_error = val_score
    args._last_spline_cv_folds = n_splits
    return _fit_spline_matrix(X, t, t_new, alpha=alpha)


def integrate_system(sys: ODESystem, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(sys.t_span[0], sys.t_span[1] + 0.5 * dt, dt)
    sol = solve_ivp(sys.rhs, sys.t_span, sys.x0, t_eval=t_eval, rtol=1e-10, atol=1e-12, method="DOP853")
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T


def make_tnew(t_obs: np.ndarray, upsample: int) -> np.ndarray:
    dt_new = (t_obs[1] - t_obs[0]) / upsample
    return np.arange(t_obs[0], t_obs[-1] + 0.1 * dt_new, dt_new)



METHOD_LABELS = {
    "baseline": "Baseline",
    "linear_interp": "Linear interpolation",
    "smoothing_spline": "Smoothing spline",
    "smoothing_spline_cv": "Tuned smoothing spline",
    "spline": "Smoothing spline",
    "gp_smoothing_cv": "Tuned GP smoothing",
    "gaussian_process_cv": "Tuned GP smoothing",
    "edmd_poly3": "EDMD-polynomial",
    "edmd_rbf": "EDMD-RBF",
}
DEFAULT_METHODS = ["baseline", "edmd_poly3", "edmd_rbf"]


def configure_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply defaults only; explicit command-line values take precedence."""
    presets = {
        "quick": dict(dt=0.01, sparse_factors=[16, 64], noise=[0.0, 0.05, 0.10], seeds=[0, 1], rbf_centers=25),
        "publication": dict(dt=0.005, sparse_factors=[8, 16, 32, 64], noise=[0.0, 0.01, 0.03, 0.05, 0.10], seeds=list(range(10)), rbf_centers=40),
    }
    for name, value in presets[args.preset].items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
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
        total = len(items)
        def gen():
            for i, item in enumerate(items, 1):
                if i == 1 or i == total or i % 25 == 0:
                    print(f"{desc}: {i}/{total}", flush=True)
                yield item
        return gen()


def reconstruct_ode_method(method: str, X_obs: np.ndarray, t_obs: np.ndarray, t_new: np.ndarray, sys: ODESystem, rng: np.random.Generator, args: argparse.Namespace, noise: float):
    if method == "baseline":
        return X_obs, t_obs
    if method == "linear_interp":
        return linear_interp_reconstruct(X_obs, t_obs, t_new), t_new
    if method in {"smoothing_spline", "spline"}:
        return spline_reconstruct(X_obs, t_obs, t_new, rel_noise=noise), t_new
    if method in {"smoothing_spline_cv", "tuned_smoothing_spline"}:
        return spline_reconstruct_cv(X_obs, t_obs, t_new, args=args), t_new
    if method in {"gp_smoothing_cv", "gaussian_process_cv"}:
        return gp_smoothing_reconstruct_cv(X_obs, t_obs, t_new, args=args), t_new
    if method == "edmd_poly3":
        return edmd_reconstruct(X_obs, t_obs, t_new, kind="poly", degree=3, var_names=sys.var_names, rng=rng), t_new
    if method == "edmd_rbf":
        return edmd_reconstruct(X_obs, t_obs, t_new, kind="rbf", degree=3, var_names=sys.var_names, rng=rng, n_centers=args.rbf_centers), t_new
    raise ValueError(method)


def run_benchmark(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    systems = [lorenz_system(), vanderpol_system()]
    thresholds = np.array([0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
    methods = args.methods
    rows = []
    coef_rows = []
    completed = set()
    raw_path = os.path.join(args.outdir, "ode_raw_results.csv")
    if args.resume and os.path.exists(raw_path):
        prev = pd.read_csv(raw_path)
        rows.extend(prev.to_dict("records"))
        completed = {case_key(r) for r in rows}
        print(f"Resuming ODE benchmark with {len(completed)} existing records.", flush=True)

    tasks = []
    cache = {}
    for sys in systems:
        t_true, X_true = integrate_system(sys, args.dt)
        cache[sys.name] = (sys, t_true, X_true)
        for sparse_factor in args.sparse_factors:
            ind = np.arange(0, len(t_true), sparse_factor)
            if ind[-1] != len(t_true) - 1:
                ind = np.r_[ind, len(t_true) - 1]
            for noise in args.noise:
                for seed in args.seeds:
                    for method in methods:
                        key = (sys.name, method, sparse_factor, float(noise), seed)
                        if key not in completed:
                            tasks.append((sys.name, sparse_factor, float(noise), seed, method))

    for sys_name, sparse_factor, noise, seed, method in progress_iter(tasks, enabled=(not args.no_progress), desc="ODE cases"):
        sys, t_true, X_true = cache[sys_name]
        _, feature_names = polynomial_library(X_true[:2], sys.var_names, degree=3)
        Xi_true = sys.true_coefficients(feature_names)
        ind = np.arange(0, len(t_true), sparse_factor)
        if ind[-1] != len(t_true) - 1:
            ind = np.r_[ind, len(t_true) - 1]
        t_obs = t_true[ind]
        X_clean_obs = X_true[ind]
        rng = np.random.default_rng(seed + 1000 * sparse_factor + 100000 * int(noise * 1000))
        X_obs = add_noise(X_clean_obs, noise, rng)
        t_new = make_tnew(t_obs, args.upsample)
        try:
            args._last_spline_alpha = np.nan
            args._last_spline_cv_error = np.nan
            args._last_spline_cv_folds = 0
            args._last_gp_kernel = ""
            args._last_gp_cv_error = np.nan
            args._last_gp_cv_folds = 0
            X_use, t_use = reconstruct_ode_method(method, X_obs, t_obs, t_new, sys, rng, args, noise)
            if not np.all(np.isfinite(X_use)) or len(t_use) < 5:
                raise FloatingPointError("non-finite reconstruction")
            result = fit_sindy_oracle(X_use, t_use, sys.var_names, 3, Xi_true, thresholds)
            f1 = float(result["f1"]); cerr = float(result["coef_error"]); score = float(result["score"]); thr = float(result["threshold"])
            status = "ok"
            if args.save_coefficients and seed == args.seeds[0] and noise in [args.noise[0], args.noise[-1]]:
                Xi = result["Xi"]
                for i, name in enumerate(feature_names):
                    for j, outvar in enumerate(sys.var_names):
                        if abs(Xi[i, j]) > 1e-12 or abs(Xi_true[i, j]) > 1e-12:
                            coef_rows.append({"system": sys.name, "method": method, "method_label": METHOD_LABELS.get(method, method), "sparse_factor": sparse_factor, "noise": noise, "seed": seed, "feature": name, "equation": f"d{outvar}/dt", "coef": Xi[i, j], "true_coef": Xi_true[i, j]})
        except Exception as exc:
            f1 = cerr = score = thr = np.nan
            status = f"fail: {type(exc).__name__}: {exc}"
        row = {"system": sys.name, "method": method, "method_label": METHOD_LABELS.get(method, method), "sparse_factor": sparse_factor, "obs_dt": float(t_obs[1]-t_obs[0]), "noise": noise, "seed": seed, "upsample": args.upsample if method != "baseline" else 1, "spline_alpha": getattr(args, "_last_spline_alpha", np.nan), "spline_cv_error": getattr(args, "_last_spline_cv_error", np.nan), "spline_cv_folds": getattr(args, "_last_spline_cv_folds", 0), "gp_kernel": getattr(args, "_last_gp_kernel", ""), "gp_cv_error": getattr(args, "_last_gp_cv_error", np.nan), "gp_cv_folds": getattr(args, "_last_gp_cv_folds", 0), "support_f1": f1, "coef_error": cerr, "practical_score": score, "threshold": thr, "status": status}
        rows.append(row)
        if args.verbose:
            print(row, flush=True)
        if args.checkpoint and len(rows) % args.checkpoint == 0:
            pd.DataFrame(rows).to_csv(raw_path, index=False)

    raw = pd.DataFrame(rows)
    summary = raw[raw["status"] == "ok"].groupby(["system","method","method_label","sparse_factor","noise"], as_index=False).agg(
        support_f1_mean=("support_f1","mean"), support_f1_std=("support_f1","std"), coef_error_median=("coef_error","median"), coef_error_mean=("coef_error","mean"), practical_score_mean=("practical_score","mean"), n_ok=("status","size"))
    return raw, summary, pd.DataFrame(coef_rows)


def ordered_methods(df: pd.DataFrame) -> list:
    return [m for m in DEFAULT_METHODS if m in set(df["method"])] + [m for m in df["method"].unique() if m not in DEFAULT_METHODS]


def make_tables(raw: pd.DataFrame, outdir: str, prefix: str = "ode") -> None:
    ok = raw[raw["status"] == "ok"].copy()
    methods = ordered_methods(ok)
    label = {m: METHOD_LABELS.get(m,m) for m in methods}
    overall = ok.groupby("method", as_index=False).agg(
        support_f1_mean=("support_f1","mean"), coef_error_median=("coef_error","median"), practical_score_mean=("practical_score","mean")
    ).set_index("method").reindex(methods).reset_index()
    overall.insert(1,"method_label", overall["method"].map(label))
    overall.to_csv(os.path.join(outdir, f"{prefix}_overall_ranking.csv"), index=False)
    by_system = ok.groupby(["system","method"], as_index=False).agg(
        support_f1_mean=("support_f1","mean"), coef_error_median=("coef_error","median"), practical_score_mean=("practical_score","mean")
    )
    by_system["method_label"] = by_system["method"].map(label)
    by_system.to_csv(os.path.join(outdir, f"{prefix}_by_system.csv"), index=False)
    noise_tbl = ok.groupby(["noise","method"])["coef_error"].median().unstack().reindex(columns=methods)
    sparse_tbl = ok.groupby(["sparse_factor","method"])["coef_error"].median().unstack().reindex(columns=methods)
    noise_tbl.rename(columns=label).to_csv(os.path.join(outdir, f"{prefix}_coeferr_by_noise.csv"))
    sparse_tbl.rename(columns=label).to_csv(os.path.join(outdir, f"{prefix}_coeferr_by_sparse_factor.csv"))

    base = ok[ok.method=="baseline"][["system","sparse_factor","noise","seed","support_f1","coef_error","practical_score"]].rename(columns={"support_f1":"support_f1_base","coef_error":"coef_error_base","practical_score":"practical_score_base"})
    wins=[]
    for m in methods:
        if m=="baseline":
            continue
        d=ok[ok.method==m].merge(base,on=["system","sparse_factor","noise","seed"])
        wins.append({"method":m,"method_label":label[m],"paired_cases":len(d),"practical_score_win_rate":float((d.practical_score>d.practical_score_base).mean()) if len(d) else np.nan,"coef_error_win_rate":float((d.coef_error<d.coef_error_base).mean()) if len(d) else np.nan,"support_f1_win_rate":float((d.support_f1>d.support_f1_base).mean()) if len(d) else np.nan})
    pd.DataFrame(wins).to_csv(os.path.join(outdir, f"{prefix}_winrate_vs_baseline.csv"), index=False)

    def fmt(x):
        if pd.isna(x): return "--"
        return f"{x:.3f}"
    md=[]
    md.append("# ODE benchmark tables\n")
    md.append("## Overall comparison\n")
    md.append("| Method | Mean support F1 | Median coefficient error | Mean practical score |\n|---|---:|---:|---:|")
    for _,r in overall.iterrows():
        md.append(f"| {r.method_label} | {fmt(r.support_f1_mean)} | {fmt(r.coef_error_median)} | {fmt(r.practical_score_mean)} |")
    md.append("\n## Median coefficient error by noise level\n")
    md.append("| Noise | " + " | ".join(label[m] for m in methods) + " |")
    md.append("|---:" + "|---:"*len(methods) + "|")
    for idx,row in noise_tbl.iterrows():
        md.append(f"| {idx:.2f} | " + " | ".join(fmt(row[m]) for m in methods) + " |")
    md.append("\n## Median coefficient error by sparse factor\n")
    md.append("| Sparse factor | " + " | ".join(label[m] for m in methods) + " |")
    md.append("|---:" + "|---:"*len(methods) + "|")
    for idx,row in sparse_tbl.iterrows():
        md.append(f"| {int(idx)} | " + " | ".join(fmt(row[m]) for m in methods) + " |")
    Path = __import__('pathlib').Path
    Path(os.path.join(outdir, f"{prefix}_tables.md")).write_text("\n".join(md))


def plot_results(raw: pd.DataFrame, outdir: str) -> None:
    ok = raw[raw["status"] == "ok"]
    plot_df = ok.groupby(["method", "noise"], as_index=False).agg(support_f1=("support_f1", "mean"), coef_error=("coef_error", "median"), practical_score=("practical_score", "mean"))
    methods = ordered_methods(ok)
    plt.figure(figsize=(7,4))
    for method in methods:
        d = plot_df[plot_df["method"] == method].sort_values("noise")
        plt.plot(d["noise"], d["support_f1"], marker="o", label=METHOD_LABELS.get(method,method))
    plt.xlabel("relative noise level"); plt.ylabel("mean support F1"); plt.title("ODE SINDy support recovery"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(outdir,"ode_f1_by_noise.png"), dpi=200); plt.close()


def parse_list(s: str, typ=float):
    return [typ(x) for x in str(s).split(",") if str(x).strip() != ""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["quick","publication"], default="quick")
    parser.add_argument("--outdir", default="results_ode")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--sparse-factors", type=str, default=None)
    parser.add_argument("--noise", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--upsample", type=int, default=5)
    parser.add_argument("--rbf-centers", type=int, default=None)
    parser.add_argument("--spline-alpha-grid", type=str, default=None, help="Comma-separated dimensionless smoothing levels for tuned smoothing_spline_cv.")
    parser.add_argument("--spline-cv-folds", type=int, default=4, help="Number of deterministic interior folds used to tune smoothing_spline_cv.")
    parser.add_argument("--gp-kernel-grid", type=str, default=None, help="Comma-separated GP kernel specs family:length_scale:noise for gp_smoothing_cv.")
    parser.add_argument("--gp-cv-folds", type=int, default=4, help="Number of deterministic interior folds used to tune gp_smoothing_cv.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint", type=int, default=25)
    parser.add_argument("--save-coefficients", action="store_true", default=True)
    args = parser.parse_args()
    args = configure_preset(args)
    if isinstance(args.sparse_factors, str): args.sparse_factors = parse_list(args.sparse_factors, int)
    if isinstance(args.noise, str): args.noise = parse_list(args.noise, float)
    if isinstance(args.seeds, str): args.seeds = parse_list(args.seeds, int)
    if args.rbf_centers is None: args.rbf_centers = 40 if args.preset == "publication" else 25
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    os.makedirs(args.outdir, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw, summary, coef_df = run_benchmark(args)
    raw.to_csv(os.path.join(args.outdir,"ode_raw_results.csv"), index=False)
    summary.to_csv(os.path.join(args.outdir,"ode_summary.csv"), index=False)
    if not coef_df.empty: coef_df.to_csv(os.path.join(args.outdir,"ode_coefficients_examples.csv"), index=False)
    if not summary.empty:
        best = summary.sort_values(["system","sparse_factor","noise","practical_score_mean"], ascending=[True,True,True,False]).groupby(["system","sparse_factor","noise"], as_index=False).head(1)
        best.to_csv(os.path.join(args.outdir,"ode_best_by_case.csv"), index=False)
    make_tables(raw, args.outdir, "ode")
    plot_results(raw, args.outdir)
    print(f"Wrote ODE results to: {args.outdir}")


if __name__ == "__main__":
    main()

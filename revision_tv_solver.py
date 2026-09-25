"""Convex discrete integral-TV smoothing using the existing NumPy/SciPy stack.

The minimizer solves 0.5*sum(mask*(z-y)**2) + lam*sum(abs(B@z)),
where B differences adjacent interval slopes in tau=(t-t[0])/median(diff(t)).
Thus z is an integrated piecewise-constant derivative with a fitted intercept;
its discrete derivative's total variation is penalized. Missing observations
remain variables and never enter the data-fidelity term. No ground truth is
used. Channels share a banded factorization but otherwise solve independently.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded


def _derivative(z, t):
    slopes = np.diff(z, axis=0) / np.diff(t)[:, None]
    out = np.empty_like(z)
    out[0], out[-1] = slopes[0], slopes[-1]
    out[1:-1] = (slopes[:-1] + slopes[1:]) / 2
    return out


def smooth_tv(y, t, lam, mask=None, tol=1e-5, max_iter=10000):
    """Return ``(smoothed_states, physical_time_derivative, diagnostics)``.

    Parameters
    ----------
    y : (n,) or (n,d) array
        States, preferably scaled using training observations only. Masked-out
        values, including NaNs, are ignored completely.
    t : (n,) array
        Strictly increasing observation times; irregular grids are supported.
    lam : nonnegative float
        Penalty for slope variation on the median-step-normalized time grid.
    mask : optional (n,) boolean array
        Whole-time training mask; at least two times must be observed.
    tol : float
        Maximum relative primal-dual gap, scaled by max(1, primal objective),
        required separately for every channel.
    max_iter : int
        Maximum ADMM iterations. Failure to certify convergence raises
        RuntimeError rather than returning an unconverged result silently.

    Notes
    -----
    A banded ADMM solve is certified using a dual-feasible vector reconstructed
    from the fitted residual after removing its two affine moments. Scaling
    that vector into the l-infinity ball preserves the masked stationarity
    constraints. The returned gap is therefore a convex optimality certificate,
    up to explicitly reported floating-point stationarity error. At lam=0 the
    unobserved states are assigned by linear interpolation/extrapolation.
    """
    started = time.perf_counter()
    y = np.asarray(y, dtype=float)
    vector = y.ndim == 1
    if vector:
        y = y[:, None]
    if y.ndim != 2:
        raise ValueError('y must have shape (n,) or (n,d)')
    t = np.asarray(t, dtype=float)
    n, d = y.shape
    if t.shape != (n,) or n < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('t must contain at least two finite, strictly increasing times')
    if not np.isfinite(lam) or lam < 0 or not np.isfinite(tol) or tol <= 0 or max_iter < 1:
        raise ValueError('lam >= 0, tol > 0 and max_iter >= 1 are required')
    obs = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if obs.shape != (n,) or np.count_nonzero(obs) < 2:
        raise ValueError('mask must have shape (n,) and at least two observed times')
    if not np.all(np.isfinite(y[obs])):
        raise ValueError('observed y values must be finite')
    yy = np.where(obs[:, None], y, 0.0)
    tau = (t - t[0]) / np.median(np.diff(t))
    h = np.diff(tau)
    # Center and scale the affine nullspace for well-conditioned moment fitting.
    x = (tau - np.mean(tau[obs])) / max(np.ptp(tau), 1.0)
    A = np.column_stack((np.ones(n), x))
    gram = A[obs].T @ A[obs]
    linear = A @ np.linalg.solve(gram, A[obs].T @ yy[obs])
    a, c = 1.0 / h[:-1], 1.0 / h[1:]
    b = -a - c

    def B(z):
        return a[:, None]*z[:-2] + b[:, None]*z[1:-1] + c[:, None]*z[2:]

    def BT(p):
        out = np.zeros((n, p.shape[1]))
        out[:-2] += a[:, None]*p
        out[1:-1] += b[:, None]*p
        out[2:] += c[:, None]*p
        return out

    def certify(z):
        # Affine correction changes no Bz, and enforces both residual moments.
        z = z - A @ np.linalg.solve(gram, A[obs].T @ (z[obs] - yy[obs]))
        residual = np.where(obs[:, None], z - yy, 0.0)
        p = -np.cumsum(h[:, None] * np.cumsum(residual, axis=0)[:-1], axis=0)[:-1]
        largest = np.max(np.abs(p), axis=0) if n > 2 else np.zeros(d)
        factor = np.ones_like(largest)
        np.divide(lam, largest, out=factor, where=largest > lam)
        p *= factor[None, :] * (1.0 - 8*np.finfo(float).eps)
        bt = BT(p)
        primal = 0.5*np.sum(residual**2, axis=0) + lam*np.sum(np.abs(B(z)), axis=0)
        dual = np.sum(yy[obs]*bt[obs] - 0.5*bt[obs]**2, axis=0)
        gap = np.maximum(primal - dual, 0.0)
        stationarity = np.max(np.abs(residual + bt), axis=0)
        bz = B(z)
        complementarity = np.maximum(lam*np.sum(np.abs(bz), axis=0) - np.sum(p*bz, axis=0), 0.0)
        prox_residual = float(np.max(np.abs(p-np.clip(p+bz, -lam, lam)), initial=0.0))
        missing_stationarity = float(np.max(np.abs(bt[~obs]))) if np.any(~obs) else 0.0
        relative_gap = gap / np.maximum(1.0, primal)
        return z, dict(primal_objective=primal.tolist(), dual_objective=dual.tolist(),
                       primal_dual_gap=gap.tolist(), relative_primal_dual_gap=relative_gap.tolist(),
                       max_relative_gap=float(np.max(relative_gap)),
                       kkt_stationarity_max=float(np.max(stationarity)),
                       kkt_complementarity_gap=complementarity.tolist(),
                       kkt_subgradient_prox_max=prox_residual,
                       masked_dual_stationarity_max=missing_stationarity,
                       dual_box_violation=float(max(0.0, np.max(np.abs(p), initial=0)-lam)),
                       dual_scale_min=float(np.min(factor)))

    def finish(z, info, iterations, rho, method):
        dz = _derivative(z, t)
        info.update(converged=True, iterations=int(iterations), rho=float(rho),
                    tolerance=float(tol), method=method,
                    elapsed_seconds=time.perf_counter()-started,
                    n_observed=int(obs.sum()), n_times=n, n_channels=d,
                    objective='0.5*sum_train((z-y)^2)+lambda*sum(abs(Bz))')
        return (z[:, 0], dz[:, 0], info) if vector else (z, dz, info)

    if lam == 0 or n == 2:
        ids = np.flatnonzero(obs)
        z = np.column_stack([np.interp(t, t[obs], yy[obs, j]) for j in range(d)])
        left = t < t[ids[0]]
        right = t > t[ids[-1]]
        z[left] = yy[ids[0]] + (t[left]-t[ids[0]])[:, None]*(yy[ids[1]]-yy[ids[0]])/(t[ids[1]]-t[ids[0]])
        z[right] = yy[ids[-1]] + (t[right]-t[ids[-1]])[:, None]*(yy[ids[-1]]-yy[ids[-2]])/(t[ids[-1]]-t[ids[-2]])
        z, info = certify(z)
        return finish(z, info, 0, 0.0, 'zero-penalty interpolation' if lam == 0 else 'affine exact solution')

    zlin, info = certify(linear)
    if info['max_relative_gap'] <= tol:
        return finish(zlin, info, 0, 0.0, 'affine exact solution')

    diag = np.zeros(n)
    diag[:-2] += a*a
    diag[1:-1] += b*b
    diag[2:] += c*c
    off1 = np.zeros(n-1)
    off1[:-1] += a*b
    off1[1:] += b*c
    off2 = a*c
    rho = max(0.01, min(1e5, float(lam)*n/10))
    rho_floor = rho / 2

    def factorize(rho):
        band = np.zeros((3, n))
        band[0] = obs.astype(float) + rho*diag
        band[1, :-1] = rho*off1
        band[2, :-2] = rho*off2
        return cholesky_banded(band, lower=True, check_finite=False)

    factorization = factorize(rho)
    z = linear.copy()
    v = B(z)
    u = np.zeros_like(v)
    for iteration in range(1, int(max_iter)+1):
        z = cho_solve_banded((factorization, True), yy + rho*BT(v-u), check_finite=False)
        bz = B(z)
        previous_v = v
        relaxed_bz = 1.8*bz - 0.8*previous_v
        w = relaxed_bz + u
        v = np.sign(w)*np.maximum(np.abs(w)-lam/rho, 0.0)
        u += relaxed_bz-v
        if iteration % 25 == 0 or iteration == max_iter:
            zcert, info = certify(z)
            if info['max_relative_gap'] <= tol:
                return finish(zcert, info, iteration, rho, 'banded ADMM with dual certificate')
            # Residual balancing is global; channels still have separate objectives.
            primal_residual = np.linalg.norm(bz-v)
            dual_residual = rho*np.linalg.norm(BT(v-previous_v))
            multiplier = 2.0 if primal_residual > 10*dual_residual else (0.5 if dual_residual > 10*primal_residual else 1.0)
            new_rho = min(1e8, max(rho_floor, rho*multiplier))
            if new_rho != rho:
                u *= rho/new_rho
                rho = new_rho
                factorization = factorize(rho)
    raise RuntimeError(f'TV solver did not certify convergence after {max_iter} iterations: '
                       f'max relative primal-dual gap={info["max_relative_gap"]:.3e}, '
                       f'target={tol:.3e}, n={n}, d={d}, lambda={lam:g}, rho={rho:g}')

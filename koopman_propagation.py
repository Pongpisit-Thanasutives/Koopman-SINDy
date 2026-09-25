"""Complex-valued, principal-branch propagation shared by the revision scripts.

A complex Schur form avoids inversion of a possibly ill-conditioned eigenvector
matrix. Eigenvalues smaller than ``spectral_floor`` are regularized to that
positive floor; this is recorded on the propagator. Physical outputs, not the
latent evolution, are projected to real values by the callers.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import fractional_matrix_power, schur


class FractionalEvolution:
    def __init__(self, matrix: np.ndarray, spectral_floor: float = 1e-12):
        matrix = np.asarray(matrix, dtype=complex)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("the evolution matrix must be square")
        if not np.all(np.isfinite(matrix)):
            raise FloatingPointError("non-finite fitted evolution matrix")
        self.triangular, self.basis = schur(matrix, output="complex")
        diagonal = np.diag(self.triangular).copy()
        near_real = np.abs(diagonal.imag) <= 100 * np.finfo(float).eps * np.maximum(1.0, np.abs(diagonal))
        diagonal[near_real] = diagonal[near_real].real.astype(complex)
        self.regularized_eigenvalues = int(np.count_nonzero(np.abs(diagonal) < spectral_floor))
        replace = np.abs(diagonal) < spectral_floor
        diagonal[replace] = complex(spectral_floor)
        self.triangular[np.diag_indices_from(self.triangular)] = diagonal
        self.regularized_matrix = self.basis @ self.triangular @ self.basis.conj().T
        self.regularization_relative_change = float(
            np.linalg.norm(self.regularized_matrix - matrix) / max(np.linalg.norm(matrix), 1e-12)
        )
        self._cache = {0.0: np.eye(matrix.shape[0], dtype=complex)}

    def power(self, fraction: float) -> np.ndarray:
        fraction = float(fraction)
        if not np.isfinite(fraction):
            raise ValueError("the elapsed-time fraction must be finite")
        # Times are sampled on deterministic grids; round only floating-point
        # arithmetic noise so equal substeps reuse an identical matrix.
        key = round(fraction, 12)
        if key not in self._cache:
            if key.is_integer():
                triangular_power = np.linalg.matrix_power(self.triangular, int(key))
            else:
                triangular_power = fractional_matrix_power(self.triangular, key)
            value = self.basis @ triangular_power @ self.basis.conj().T
            if not np.all(np.isfinite(value)):
                raise FloatingPointError("non-finite principal fractional evolution")
            self._cache[key] = value
        return self._cache[key]


def fractional_step_matrix(matrix: np.ndarray, fraction: float) -> np.ndarray:
    """Return a complex principal fractional power; never discard its phase."""
    return FractionalEvolution(matrix).power(fraction)


def nominal_observation_pairs(times: np.ndarray) -> tuple[float, np.ndarray]:
    """Identify transitions with the nominal lag, excluding a short final gap."""
    times = np.asarray(times, dtype=float)
    gaps = np.diff(times)
    if len(gaps) == 0 or not np.all(np.isfinite(times)) or np.any(gaps <= 0):
        raise ValueError("observation times must be finite and strictly increasing")
    nominal = float(np.median(gaps))
    usable = np.isclose(gaps, nominal, rtol=1e-7, atol=1e-12)
    if not np.any(usable):
        raise ValueError("no observation transitions have the nominal lag")
    return nominal, usable


def local_observable_reconstruction(
    matrix: np.ndarray,
    observed_observables: np.ndarray,
    observation_times: np.ndarray,
    query_times: np.ndarray,
    state_indices: np.ndarray | list[int],
    nominal_dt: float,
) -> np.ndarray:
    """Advance each query from the preceding anchor at its actual elapsed time."""
    times = np.asarray(observation_times, dtype=float)
    queries = np.asarray(query_times, dtype=float)
    phi = np.asarray(observed_observables, dtype=complex)
    if np.any(queries < times[0] - 1e-10) or np.any(queries > times[-1] + 1e-10):
        raise ValueError("local reconstruction is restricted to the observed window")
    evolution = FractionalEvolution(matrix)
    out = []
    for tt in queries:
        j = int(np.searchsorted(times, tt, side="right") - 1)
        j = max(j, 0)
        # Floating-point grid representations of an exact anchor may differ by
        # a few ulps. This tolerance never anticipates a genuinely later anchor.
        if j + 1 < len(times) and abs(times[j + 1] - tt) <= 1e-10 * max(1.0, abs(tt)):
            j += 1
        fraction = max(0.0, float((tt - times[j]) / nominal_dt))
        latent = phi[j] @ evolution.power(fraction)
        out.append(np.real(latent[state_indices]))
    result = np.asarray(out)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("non-finite physical reconstruction")
    return result

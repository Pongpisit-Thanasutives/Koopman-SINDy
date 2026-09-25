# Complex propagation validation

All 6 correctness checks passed.

The checks cover a negative-real eigenvalue, a real rotation with conjugate eigenpairs, a defective Jordan block, an irregular final observation interval, spectral regularization, and non-finite input rejection.

Complex Schur evaluation avoids an eigenvector inverse. Schur diagonal values whose imaginary part is at most 100 times machine epsilon times max(1,abs(lambda)) are made exactly real; eigenvalues of magnitude below 1e-12 are replaced by positive 1e-12. Fractional powers use the principal complex branch. The reported round trips are relative to the explicitly regularized matrix.

The eight diagnostic fits use seed0, base dt0.005, q5, ODE sparse factor32 and PDE factor8, Burgers/Fisher ranks8/5, and clean or3% noisy observations. They establish consistency with the fitted map, not better equation recovery.

| System | Noise | Real-step round trip | Complex round trip | Real-step state defect | Complex state defect |
|---|---:|---:|---:|---:|---:|
| lorenz63 | 0.00 | 1.37e-12 | 1.66e-12 | 1.48e-12 | 9.89e-13 |
| vanderpol_mu2 | 0.00 | 4.93e-15 | 5.14e-15 | 9.28e-15 | 1.11e-14 |
| burgers | 0.00 | 1.15e-06 | 3.09e-14 | 2.9e-14 | 3.14e-14 |
| fisher_kpp | 0.00 | 1.25e-06 | 5.62e-14 | 1.49e-13 | 1.66e-13 |
| lorenz63 | 0.03 | 2.77 | 6.65e-12 | 0.0523 | 6.16e-13 |
| vanderpol_mu2 | 0.03 | 4.14e-15 | 6.04e-15 | 8.19e-15 | 9.34e-15 |
| burgers | 0.03 | 0.194 | 6.58e-11 | 0.0124 | 3e-10 |
| fisher_kpp | 0.03 | 0.735 | 3.3e-10 | 0.00216 | 4.68e-10 |

Ordinary conjugate eigenpairs often already yield real fractional powers; projection is harmless there. Negative-real eigenvalues lie on the branch cut. The branch convention does not uniquely identify a continuous-time generator. Local-reset jumps, finite-dictionary bias, aliasing and noise remain.

Observation arrays and nominal dense output grids are unchanged. Map fitting excludes unequal-lag final transitions. Queries use actual elapsed time from their preceding anchor.

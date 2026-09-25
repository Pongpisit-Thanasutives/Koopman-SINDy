# Weak-library validation

All 19 checks passed.

Checks cover exact polynomial-weight quadrature on noisy irregular nodal data, temporal integration-by-parts signs, all eight conservative PDE identities across periodic boundaries, and clean benchmark support/identity recovery at both planned observation densities. The clean sparse VdP case has a 3.08% weak identity residual and 2.04% coefficient error; the remaining clean coefficient errors are at most 1.27e-5. This discretization limit is retained in the study.

Clean diagnostics validate implementation and expose discretization error; no parameters were tuned using these outcomes. Exact quadrature applies to nodal-feature interpolants, not the unknown continuous trajectory.

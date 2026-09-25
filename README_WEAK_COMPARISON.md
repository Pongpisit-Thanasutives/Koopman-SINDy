# Focused weak-form preprocessing experiment

This small diagnostic asks whether the corrected local-reset EDMD reconstruction adds value before an integral sparse-regression fit. It compares preprocessing choices **within a fixed weak library**. It is not a benchmark or replication of the complete PySINDy or Messenger–Bortz WSINDy algorithms, and its sparsity thresholds remain oracle-selected. All planned conditions are reported, including unfavorable results.

## Reproduce

From the repository root with the versions in `requirements_revision.txt` (use a disposable copy to preserve the delivered outputs):

```bash
python -m pip install -r requirements_revision.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python validate_revision_weak.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python revision_weak_comparison.py --jobs 4
```

The 60 noisy records produced all 300 planned method results in about 21 seconds with four workers in the supplied runtime. Wall time will vary. No GPU or external weak-SINDy package is required. To regenerate only summaries from saved results:

```bash
python revision_weak_comparison.py --summarize-only
```

To generate the manuscript table from the saved results:

```bash
python make_revision_weak_table.py
```

The table defaults to `generated_assets/tables/revision_weak_comparison.tex`. Use `--output /path/to/table.tex` for another location; relative paths are supported. Table-generation provenance is refreshed in the results directory.

The experiment output directory is `results/revision_weak_comparison`. `revision_weak_protocol.json` was fixed before inspecting noisy identification outcomes. `manifest.json` records the full protocol, task grid, runtime, input-source hashes checked before and after computation, and hashes of numerical outputs. The initially absent optional Markdown dependency `tabulate==0.9.0` affected only final Markdown reporting; numerical results, coefficient/CV records, CSV summaries and execution manifest had already been written. Installing the already-pinned dependency and using `--summarize-only` completes reporting without repeating any fits.

## Fixed comparison

| Setting | Van der Pol, mu=2 | Burgers, viscosity=0.05 |
|---|---|---|
| Observation spacings | 0.08, 0.32 | 0.04, 0.16 |
| Common observation interval | [0,19.84] | [0,1.92] |
| Noise levels | 5%, 10%, 20% | 5%, 10%, 20% |
| Independent seeds | 0–4 | 0–4 |
| State dimension/grid | 2 | 64 periodic positions on [0,2 pi) |
| Time test halfwidth; center spacing | 1.28; 0.32 | 0.48; 0.16 |
| Number of temporal test centers | 55 | 7 |
| Spatial test halfwidth; centers | not applicable | pi/2; 16 uniformly spaced periodic centers |
| Weak rows | 55, each with 2 targets | 112 |
| EDMD | degree-3 polynomial, q=5 | POD rank8, up to 30 RBF centers, q=5 |
| Arms | raw, linear q5, EDMD, TV states | raw, linear q5, POD-EDMD, TV states, POD only, POD+TV states |

The existing numerical generators, noise definitions, EDMD code, STLSQ solver and TV observation-only cross-validation are reused. Observation grids are uniformly cropped to a common horizon within each system. Every arm in a record receives an identical noisy array, checked by a saved SHA256. Noise seeds follow `seed + 1000*factor + 100000*int(noise*1000)`; EDMD's center sampling uses that generator after the noise draw. The RBF-center count is capped by the number of observed snapshots (13 at the coarsest Burgers spacing). Raw, TV, POD-only and POD+TV use the observation grid; linear-state interpolation and EDMD use the same q=5 grid. All arms use identical physical test supports and centers. Extra reconstructed points change the numerical representation, not the number of physical test windows or independent records.

The coarsest observations provide nine time samples per VdP support and seven per Burgers support. Supports were chosen to keep multiple observations in the shortest windows while retaining variation across the trajectory; no width/degree search or outcome-driven adjustment was performed. The compact degree parameter is p=4 throughout.

TV regularization uses the existing three deterministic interior folds, observation-only prediction error, nine penalties from 0 through 1000, and final relative primal–dual gap tolerance 1e-5. Raw PDE CV uses eight spatial positions; POD is refit from training observations within each fold. Only the reconstructed states enter this experiment. The returned TV derivatives are discarded. This distinction avoids relabeling the previous TV differentiation experiment as weak SINDy.

All arms use the original ODE or PDE STLSQ threshold grid, normalized-column ridge parameter 1e-8 and 12 support-update iterations. Physical coefficient thresholds are selected by maximizing `(F1/(1+relative coefficient error), F1, -relative coefficient error)`; exact ties retain the first ascending threshold. Thus this is a controlled oracle diagnostic, not a claim of deployable non-oracle weak-model selection.

## Mathematical construction

Let `phi(s)=(1-s^2)^4` on `[-1,1]` and zero outside. With time center c and halfwidth H, `psi(t)=phi((t-c)/H)`. The ODE target and feature row are

\[
b_k=-\int x(t)\psi'_k(t)\,dt,\qquad
G_{kj}=\int f_j(x(t))\psi_k(t)\,dt.
\]

The ODE library contains the same ten monomials through total degree3 as the VdP benchmark. For the PDE, the test is the product of compact time and space tests; spatial supports wrap periodically. Writing the space-time pairing as brackets, the target is `-<u,psi_t>`. The eight feature columns are

\[
\big[\langle1,\psi\rangle,\langle u,\psi\rangle,
\langle u^2,\psi\rangle,-\langle u,\psi_x\rangle,
-\tfrac12\langle u^2,\psi_x\rangle,
-\tfrac13\langle u^3,\psi_x\rangle,
\langle u,\psi_{xx}\rangle,\langle u^2,\psi_{xx}\rangle\big].
\]

These represent `[1,u,u²,u_x,u*u_x,u²*u_x,u_xx,(u²)_xx]`. Burgers coefficients remain -1 for `u*u_x` and 0.05 for `u_xx`. The final nuisance candidate **explicitly replaces** the main study's `u*u_xx`. They are not equivalent: `(u²)_xx=2*u*u_xx+2*u_x²`. Keeping the original term would require a remaining data derivative. This common conservative library removes all observed-state derivatives from every weak arm; it also means that a direct strong-versus-weak comparison would not isolate preprocessing alone.

`weak_integral_library.py` computes exact polynomial moments of each cell's linear nodal-feature interpolant. In normalized coordinates, cell endpoint weights use integrals of `phi^(d)(s)` and `s*phi^(d)(s)`, with physical scale `H^(1-d)`. Tensor products give the space-time weights. “Exact” here refers to integration of these specified interpolants, not exact integration of the unknown trajectory. In particular, the interpolant of nodal nonlinear features is generally different from the nonlinear features of an interpolated state. Consequently the linear-state q5 control tests the effect of the denser representation as well as ordinary interpolation.

No temporal or spatial derivative of an observed or reconstructed state is used in the weak library. A common normalization by test mass is applied to both target and features. Overlapping windows are regression rows, never treated as independent statistical replicates.

## Validation and reporting

`validate_revision_weak.py` provides 19 checks: analytic moment weights against independent adaptive quadrature on noisy irregular nodal data; integration-by-parts signs; all conservative PDE identities across periodic boundaries; and clean-data support/weak-identity recovery at both planned densities. All pass. Clean coarse VdP retains a 3.08% weak-identity residual and 2.04% coefficient error, showing a finite-resolution limit without tuning it away. Other clean coefficient errors are at most 1.27e-5.

`raw_results.csv` retains every arm/condition/seed, preprocessing and solver details, observation hashes and failures if present. `coefficients.csv` stores all estimated and true coefficients; `tv_cv.csv` stores all fold/penalty evaluations. `summary_by_condition.csv` reports all 60 method-condition cells with five-seed mean/SE where appropriate; coefficient-error medians are explicitly distinguished from means. Pooled SEs in `pooled_seed_summary.csv` first average the six conditions within each seed. `paired_differences.csv` and `paired_summary.csv` compare EDMD with every corresponding control on matching records. Consult `findings.md` for the complete numerical interpretation.

## Primary implementation references consulted

- [PySINDy WeakPDELibrary source](https://pysindy.readthedocs.io/en/stable/_modules/pysindy/feature_library/weak_pde_library.html), stable2.1.0 documentation, accessed 24 September2026. The reference documents compact polynomial tests and analytic integration of piecewise-linear nodal features. Our independent implementation fixes physical supports deterministically; it does not reproduce the package's random support selection, support shrinking or mixed-derivative feature machinery. The requested main-branch [GitHub location](https://github.com/dynamicslab/pysindy/blob/main/pysindy/feature_library/weak_pde_library.py) was also checked; the accessible stable source supplies the documented quadrature reference.
- [SethMinor/PyWSINDy-for-PDEs](https://github.com/SethMinor/PyWSINDy-for-PDEs), particularly [wsindy.py](https://github.com/SethMinor/PyWSINDy-for-PDEs/blob/main/wsindy.py), accessed24September2026. Its separable compact-polynomial tests and conservative derivative-of-nonlinearity library clarify the fully weak construction. Its convolution implementation, support selection, scaling and modified sequential-thresholding/model-selection procedure are not reproduced here. Our fixed library and original oracle STLSQ are chosen to isolate preprocessing within this paper's diagnostic framework.

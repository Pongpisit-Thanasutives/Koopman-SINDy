# Reproduce the added TV/high-noise comparison

The revision experiment is complete and its numerical outputs are included. No local rerun is required to use the revised manuscript. The commands below reproduce the experiment independently. These outputs use the corrected common Koopman propagation implementation; the original uploaded archive remains unchanged.

From the repository root:

```bash
python -m venv .venv_revision
source .venv_revision/bin/activate
python -m pip install -r requirements_revision.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python revision_tv_comparison.py --jobs 4
```

On Windows PowerShell, replace environment activation and the final line with:

```powershell
.venv_revision\Scripts\Activate.ps1
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
python revision_tv_comparison.py --jobs 4
```

Use `--jobs 1` for serial execution. Every record contains all methods using the same noisy observations. No GPU is used. The completed run used Python 3.12.14 with the package versions in `requirements_revision.txt`; the computational run took about 20 seconds using four workers in this environment, excluding figure rendering. Runtime depends on the machine.

## Scope and fixed settings

| System | Base time step | Spatial grid | Sparse factors | POD rank | RBF centers |
|---|---:|---:|---|---:|---:|
| Lorenz–63 | 0.005 | — | 16, 64 | — | — |
| Van der Pol | 0.005 | — | 16, 64 | — | — |
| Burgers | 0.005 | 64 | 8, 32 | 8 | 30 |
| Fisher–KPP | 0.005 | 64 | 8, 32 | 5 | 30 |
| Advection–diffusion | 0.01 | 48 | 4, 16 | 4 | 12 |

All five systems retain their original governing coefficients, clean integration horizons, initial conditions and noise definitions. In this new comparison, every method uses the identical uniform sparse observation grid; an incomplete terminal interval is discarded instead of appending a shorter interval to a fixed-step EDMD fit. The observed window ends at 9.92 for Lorenz factor 64, 19.84 for Van der Pol factor 64, and 1.92 for the coarser PDE factors; all other windows reach the original endpoint. The q=5 dense grid contains both observed endpoints exactly. Archived main results retain their original endpoint convention. Fisher–KPP uses the original smooth initial condition, not the front sensitivity condition. POD ranks match the archived main results. Relative noise levels are 0.05, 0.10, 0.20 and 0.50; seeds are 0–4. Gaussian noise is scaled componentwise by the clean subsampled ODE standard deviation (ddof=1) or by the single global PDE field standard deviation (ddof=0), exactly as in the original generation functions. Every method receives the same noisy record, using the original seed formula.

This yields 200 noisy datasets and 720 method records: raw central differences, polynomial EDMD, and TV for ODEs; raw central differences, POD-EDMD-RBF, TV, and POD+TV for PDEs. EDMD uses upsampling factor five; TV estimates states and derivatives directly on the observed time grid. Endpoints are excluded in every derivative regression. The original polynomial ODE and spectral-spatial PDE libraries, ridge coefficient, STLSQ implementation, and oracle threshold grids are reused. All available regression rows are used in this targeted extension (at most 15,936 per fit), avoiding extra random subsampling within the comparison.

## TV objective, tuning and numerical validation

`revision_tv_solver.py` solves the convex discrete integral-TV objective

`0.5 * sum_train((z-y)^2) + gamma * sum(abs(diff(interval_slopes)))`,

where interval slopes are differences of the estimated states `z` divided by time increments normalized by the median observed increment. This is TV regularization of the derivative, not TV denoising of the state. Equivalently `z = c + A v`, where `v` is a piecewise-constant derivative and `A` integrates it. Physical-time nodal derivatives are the arithmetic averages of adjacent interval slopes, rescaled by the median time step; first and last nodes are discarded. The integrated denoised states enter the discovery library, so the TV comparison does not retain noisy state features unnecessarily.

The implementation follows the regularization principle of Chartrand (2011), DOI `10.5402/2011/164564`; it is a new discrete convex implementation, not the original TVRegDiff software. It uses only NumPy/SciPy, banded ADMM and a per-channel feasible-dual optimality certificate. Any nonconverged solve raises an exception and is recorded as a failure. Every completed final fit and CV solve has relative primal–dual gap at most 1e-5 (gap divided by max(1, primal objective)).

The regularization grid is `{0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000}`. It was expanded from `{1e-4,...,100}` after an observation-only boundary diagnostic to include unregularized and fully affine limits. The final reported experiment uses the expanded grid throughout. Each record selects one regularizer by minimum normalized prediction error across three deterministic interior time holdouts; the endpoints remain in training. Smaller penalties break exactly equal CV scores. No clean states, derivatives, equation support, coefficients, or nominal noise level enter this selection. ODE components are standardized separately using training data; raw PDE columns use a common field standard deviation. For raw PDE TV, CV uses eight deterministically spaced spatial points, then refits all points. POD+TV uses the same fixed rank as POD-EDMD-RBF and a common latent standard deviation. Each fold's POD basis, mean and scaling are fitted only on training observations. Ground truth is used only in the common downstream oracle threshold-selection diagnostic and final accuracy evaluation.

Meaningful solver validation (analytic constant/linear/sine derivatives, agreement with independent constrained quadratic optimization, held-out-response exclusion and explicit nonconvergence handling):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python results/revision_tv_validation/validate_solver_publication.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validate_revision_tv_outputs.py
```

The solver validation report is `results/revision_tv_validation/validation_report_publication.json`. The completed-experiment checks are in `results/revision_tv_high_noise/validation_audit.json`: all 720 records succeed, all 8,640 CV solves and 320 final TV fits meet tolerance, every sparse grid is uniform, and the q=5 grid retains its anchors and exact endpoints. Validation certifies numerical correctness; it does not imply statistical superiority or support-recovery guarantees.

## Outputs and partial reruns

The complete experiment is under `results/revision_tv_high_noise/`: per-record metrics and coefficients, every TV fold/regularizer diagnostic, summaries by noise and system, final hyperparameter ranges and boundary counts, environment provenance, and coefficient-error/support-F1 figures. No records failed. Of the 320 final TV fits, eleven select zero regularization and none select the upper grid boundary.

A small implementation smoke run, which is not a replacement for the full paper experiment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python revision_tv_comparison.py --jobs 2 --systems lorenz63,burgers --noise 0.10 --seeds 0 --outdir results/revision_tv_smoke
```

Regenerate summaries and figures without rerunning numerical experiments:

```bash
python revision_tv_comparison.py --summarize-only
```

Resume a genuinely interrupted full run with identical settings:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python revision_tv_comparison.py --jobs 4 --resume
```

The new evidence is intentionally reported without a universal superiority claim. TV and POD+TV outperform EDMD in several high-noise regimes. At 20–50% noise, exact-support recovery is zero for every method on Lorenz–63 and all three PDEs; Van der Pol recovery is 40% for raw differences, 25% for polynomial EDMD, and 60% for TV. Weak-form SINDy is not implemented or benchmarked in this focused extension.

The run manifest preserves the SHA-256 values of the source files actually executed. A separate `final_package_source_sha256` mapping records the delivered files after a documentation-only correction to the PDE comparator docstring. The change does not alter numerical code or the reported experiment.

# Koopman-assisted preprocessing for sparse equation discovery

This standalone repository contains the corrected implementation, completed benchmark outputs, and revision experiments for “Dynamics-aware identification of governing equations from sparse and noisy data” by Pongpisit Thanasutives and Yoshinobu Kawahara (DCE-2026-0088). Numerical experiments and reporting commands run without manuscript sources. The separately delivered submission package contains `latex_source/`, `submission/`, and `local_checking/`.

## Use the delivered results

All numerical work for the submitted revision is complete. Current summaries are generated from the per-record CSVs in `results/`; historical records remain in explicitly named archive directories. The release changes reporting destinations, fixes weak-table output-path handling, and clarifies standalone use. Numerical methods and saved experimental evidence are unchanged. `RELEASE_MANIFEST.json` and `RELEASE_VALIDATION.json` document the release files and preservation checks. See `GITHUB_RELEASE.md` for the local commit and tag procedure.

The final implementation retains complex fractional evolution until physical readout, fits a nominal-step map only to equal-lag observation pairs, and evaluates queries from their preceding anchor using actual elapsed times. A complex Schur matrix-function implementation handles spectral propagation without an eigenvector inverse. These are numerical consistency corrections; the dictionary, POD ranks, local-reset design, libraries, and STLSQ estimator are retained. See `CODE_AUDIT_AND_REPRODUCIBILITY.md` for details and scope.

## Environment

Run commands from the repository root. Use Python 3.12 and the pinned revision dependencies:

```bash
python3.12 -m venv .venv_revision
source .venv_revision/bin/activate
python -m pip install -r requirements_revision.txt
```

## Standalone checks and reporting

The following checks need no manuscript files:

```bash
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python validate_complex_propagation.py --outdir generated_assets/propagation_validation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python validate_revision_weak.py
python validate_revision_tv_outputs.py
```

The propagation and weak checks execute small implementation diagnostics; they do not rerun the noisy benchmark grids. The weak and TV checks rewrite their validation reports under `results/`. Run checks, reporting regeneration, and experiment reruns in a disposable copy or clean checkout if you want to preserve the delivered files exactly.

To regenerate presentation assets from saved records without fitting new models:

```bash
python regenerate_presentation_assets.py
python make_revision_weak_table.py
```

Figures, tables, and appendix fragments are written to `generated_assets/` by default. Derived summaries, plots, and reporting manifests under `results/` and `figures/` are also refreshed; protected numerical records and execution manifests are checked for preservation. No sibling `latex_source/` directory is required. The weak-table command reads saved weak-study results and writes `generated_assets/tables/revision_weak_comparison.tex`.

For an explicit manuscript destination in a working copy of the complete package:

```bash
python regenerate_presentation_assets.py --latex-dir /path/to/working_copy/latex_source
python make_revision_weak_table.py --output /path/to/working_copy/latex_source/tables/revision_weak_comparison.tex
```

`make_revision1_figures_tables.py` and `sync_revision_outputs.py` also accept `--latex-dir`. Relative destinations are interpreted from the invoking working directory. `README_REVISION1_REPORTING.md` describes the individual reporting steps and their outputs.

The archived `results/final_validation/validate_package.py` is a **full-submission-package validator**, not a standalone repository test. It requires the frozen manuscript, response, local-checking files, and historical source fingerprints from the submitted package. Its historical report does not certify this modified release. Use the original full package for that audit; see `results/final_validation/README.md`.

## Reproduce the revision experiments

In a disposable copy with the revision environment activated:

```bash
bash run_revision_experiments.sh 4
```

This command reruns affected EDMD evidence, the TV/high-noise and non-oracle TV comparisons, and the targeted weak-form study, then regenerates the presentation assets in `generated_assets/`. It writes numerical and reporting outputs into this copy of the repository. It reuses shipped unaffected main raw/optDMD and classical-method records. Use `1` instead of `4` for serial TV and weak-study execution. For individual stages:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python rerun_complex_revision.py --help
```

The named stages are `main_ode`, `main_pde`, `advection`, `appendix_a`, `appendix_c`, `model_selection`, `fisher_front`, `strategy`, and `finalize`. Use `--stage NAME` to rerun one stage. The default reruns all stages. `--reuse-completed` is only for resuming isolated outputs produced by the same corrected source and settings; it is not a general configuration-aware cache.

Do not use the obsolete `sync_revision_outputs.py --merge-classical` path; Appendix C updates are handled by the corrected rerun driver.

## Main scripts

| Script | Purpose |
|---|---|
| `koopman_propagation.py` | Shared complex fractional evolution and timestamp-aware reconstruction |
| `koopman_sindy_ode_benchmark.py` | Lorenz–63 and Van der Pol benchmark |
| `koopman_sindy_pde_benchmark.py` | PDE infrastructure, Burgers/Fisher benchmark and rank validation |
| `koopman_sindy_advection_diffusion_benchmark.py` | Separate advection–diffusion benchmark |
| `koopman_sindy_model_selection_experiment.py` | Equation-wise non-oracle EBIC/Pareto model selection |
| `koopman_sindy_qr_sensitivity.py` | Paired upsampling/rank sensitivity, including POD-only controls |
| `dmd_upsampling_strategy_ablation.py` | Matched reconstruction-strategy comparison |
| `classical_interpolation_baselines.py` | Linear interpolation and observation-tuned smoothing comparisons |
| `rerun_complex_revision.py` | Reproducible affected-result regeneration and control preservation |
| `revision_tv_comparison.py` | TV/POD+TV comparison through 50% noise |
| `revision_nonoracle_tv.py` | Matched modest-noise non-oracle TV comparison |
| `revision_weak_comparison.py`, `weak_integral_library.py` | Matched compact-test integral regression with raw, linear, EDMD, TV and POD controls |
| `validate_revision_weak.py` | Independent quadrature, weak identity and clean-recovery checks |
| `make_revision_weak_table.py` | Appendix D table from the complete weak comparison |
| `regenerate_presentation_assets.py` | Portable presentation-only regeneration from saved records |
| `validate_complex_propagation.py` | Propagation correctness and fitted-map diagnostics |
| `validate_revision_tv_outputs.py` | TV solver/output validation and shared-arm consistency |
| `make_revision1_figures_tables.py`, `sync_revision_outputs.py` | Manuscript tables, figures and statistics |

## Numerical protocols

Main POD ranks are 8 for Burgers, 5 for Fisher–KPP, and 4 for advection–diffusion. The separate front-type Fisher experiment uses rank 2. The modest-noise non-oracle study uses Burgers rank 6 and Fisher rank 2, base time step 0.01 and 48 spatial points. Main PDE STLSQ uses 11 thresholds; non-oracle and sensitivity PDE studies use 20. Explicit CLI settings override presets.

Burgers/Fisher optDMD uses PyDMD with zero bagging trials. The retained standalone advection–diffusion optDMD comparator fits real exponentials and is explicitly described as restricted; it is not a general complex-rate optDMD implementation. No new algorithm is silently substituted into those historical control rows.

`README_REVISION_EXPERIMENT.md` details TV tuning and the higher-noise design. `README_NONORACLE_TV.md` specifies the frozen optional experiment and observation-only selection. `README_REVISION1_REPORTING.md` defines uncertainty and paired comparisons. `README_WEAK_COMPARISON.md` gives the fully weak library, fixed physical test windows, exact commands and limitations of the targeted experiment. The latter explicitly changes one nuisance PDE term to enable complete derivative transfer; every weak-form method shares that library.

## Optional full regeneration of historical controls

`requirements.txt` records the supplied full-suite dependency set; it is separate from the environment actually used for the focused revision. To rerun original comparison methods as well, create a separate environment with those dependencies, then use:

```bash
bash run_publication_benchmarks.sh publication results_full
python classical_interpolation_baselines.py --preset publication --outdir results_full/classical_interpolation_publication
```

These optional independent reruns are not needed to use the submitted results. Original historical runtime versions were not fully recorded, so bitwise agreement of retained controls is not guaranteed. The optional `--include-gp` classical extension is unreported and is not part of the manuscript comparison. `--resume` in the original benchmark scripts assumes unchanged settings; use a fresh output directory after changing any configuration.

## License and citation

The code retains the supplied MIT license. Cite the accompanying paper by Pongpisit Thanasutives and Yoshinobu Kawahara when using the research package.

# Matched non-oracle comparison with TV differentiation

This is one focused extension of the manuscript's existing 1%-noise non-oracle demonstration. The protocol was frozen before evaluating any outcomes in `results/revision_nonoracle_tv/protocol_frozen.json`. It adds regularized-differentiation comparators without expanding the noise/system grid or tuning against the true equations.

## Exact reproduction

From `code_repository`, after installing `requirements_revision.txt`:

```bash
python -m pip install kneed==0.8.6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python revision_nonoracle_tv.py
```

The required `kneed` version matters: on the archived candidate records, version 0.8.6 reproduces all 40 saved supports and thresholds, whereas the dependency-free fallback produces two different Fisher–KPP selections. The new script requires `kneed` to be importable instead of silently changing the selector.

To rebuild the summary/table from completed outputs:

```bash
python revision_nonoracle_tv.py --summarize-only
```

## Fixed matched settings

- Systems: Van der Pol, periodic Burgers, and the existing steep-front Fisher–KPP case.
- Noise: 1% additive Gaussian using the original componentwise ODE/global PDE scaling.
- Seeds: 0–4, with the original record seed formula.
- Base time step: 0.01 for all systems; original horizons 20 for Van der Pol and 2 for the PDEs.
- Sparse factors: 8 for Van der Pol, 4 for the PDEs; these factors divide the full horizons, so all observations and the q=5 dense grids are uniform and retain both endpoints.
- PDE spatial grid: 48 points; POD ranks 6 for Burgers and 2 for Fisher–KPP; RBF dictionary has at most 20 centres.
- ODE arms: raw finite differences, polynomial EDMD, and TV differentiation.
- PDE arms: raw finite differences, POD-EDMD-RBF, raw-field TV, and POD+TV using the identical POD rank.
- Every method sees the same noisy record. EDMD uses the corrected common reconstruction implementation. TV uses its integrated denoised state together with its regularized derivative.
- TV tuning: the same nine-point regularization grid, three observation-only interior folds, training-only scaling/POD, and per-channel 1e-5 primal–dual-gap tolerance as the higher-noise experiment.
- Discovery: original polynomial/spectral libraries and STLSQ threshold grids (10 for ODEs and 20 for PDEs), original 5,000-row PDE regression cap and record RNG convention.
- Selection: original equation-wise EBIC/Pareto elbow, EBIC gamma 0.5 and Kneedle sensitivity 1. No clean measurements, true support, true coefficient or ground-truth score enters candidate selection. Models are selected separately within each method; EBIC values from different methods are not compared.

The design comprises 15 noisy datasets, 55 preprocessing/method fits and 70 equation-level selections, because Van der Pol has two target equations. This is an operational comparison in one modest-noise setting, not a high-noise robustness guarantee or a formal model-selection consistency result. Reconstructed samples are correlated; this experiment retains the existing EBIC heuristic rather than asserting that dense reconstructed rows are independent measurements.

## Outputs and safeguards

- `method_records.csv`: all method fits, status, timing, row counts and selected TV tuning values.
- `candidates.csv` and `pareto_front.csv`: complete candidate threshold paths and the support-size/EBIC curves, with ground-truth diagnostics appended only after selection.
- `selected_by_equation.csv`: each selected equation, coefficients, threshold, selected support, actual selection rule, exact-support and exact-size indicators, and final accuracy.
- `selected_coefficients.csv`: selected and true coefficients by named feature.
- `summary.csv`: all methods, valid/expected counts, exact-support and exact-size counts, selected-support-size range and median, support F1 and coefficient error.
- `tv_cv_diagnostics.csv`: all TV fold/penalty diagnostics.
- `failures.csv`: any failed candidate or method evaluation; failures are not silently discarded.
- `run_manifest.json`: fixed protocol, environment versions, source hashes and timing.
- `validation.json`: uniform-grid and endpoint assertions, identical observation pairing, source immutability during execution, and model-selection checks.
- `revision_nonoracle_tv_table.tex`: reproducible manuscript table.
- `shared_arm_concordance.json`: all 40 overlapping raw/EDMD equation selections agree exactly with the independently regenerated original non-oracle experiment in thresholds, selected size, F1, coefficient error, RSS, BIC, EBIC and selection rule.

The selector is called with candidate data that contain no ground-truth columns. A metamorphic test adds deliberately nonsensical truth-diagnostic columns and verifies that the chosen model is unchanged. This validates absence of truth use in model selection; it is separate from evaluating whether the selected model is correct.

## Completed outcomes

All 55 method fits and 70 equation selections completed with no failed candidates. All methods recover the exact Van der Pol and Burgers supports in all five seeds. Polynomial EDMD has the lowest Van der Pol coefficient error; TV and POD+TV have lower Burgers coefficient errors than POD-EDMD-RBF. For the Fisher–KPP front, exact support is recovered in 0/5 raw-FD, 2/5 TV, 5/5 POD-EDMD-RBF and 5/5 POD+TV records. The corresponding median coefficient errors are 0.058688, 0.061225, 0.010782 and 0.007928. Raw FD selects the correct support size in three Fisher seeds but never the correct terms. The full selected-size ranges and accuracy values are retained in `summary.csv` and the manuscript table.

All 675 TV CV solves and 25 final TV fits meet the 1e-5 gap tolerance; their maximum relative gaps are 9.98755e-6 and 9.92268e-6, respectively. No selected TV penalty is on either grid boundary. These outcomes support conditional utility, including a positive Van der Pol coefficient result, and show that the successful Fisher spatial preprocessing is not unique to a learned Koopman operator. No higher-noise operational claim follows from this 1% setting.

The manifest retains the executed source hashes and separately records final package hashes. The only post-run differences are a documentation-only PDE comparator docstring correction and presentation-only changes adding a selected-support-size column and an existing-table label alias. Neither changes the numerical experiment. No user-machine rerun is required for the included results.

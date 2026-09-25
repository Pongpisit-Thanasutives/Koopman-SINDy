# Reviewer-2 targeted extension

See revision_tv_comparison.py for the complete protocol and exact command.

- `raw_results.csv`: every method/seed/noise/sampling record, including failures.
- `coefficients.csv`: complete coefficient vectors and true coefficients.
- `tv_cv_diagnostics.csv`: every fold and regularizer solve; no ground truth enters CV.
- `summary_by_noise.csv`: ten records per system/method/noise (two spacings, five seeds); SE averages the two spacings within seed before computing SE over seeds.
- `summary_all_noise.csv`: 40 records per system and method.
- `tv_tuning_summary.csv`: selected regularizer ranges and grid-boundary counts.
- `run_manifest.json`: versions, fixed configuration and command.

TV regularizes the interval derivative and uses the resulting integrated state in the discovery library. POD+TV uses the same fixed spatial rank as POD-EDMD-RBF, with each CV fold's spatial basis fitted only to training observations. The main Fisher-KPP rank is 5, as recorded in the archived main results. All methods use the original libraries and downstream oracle threshold grids. Added 20% and 50% noise are stress tests, without a claim of universal robustness.

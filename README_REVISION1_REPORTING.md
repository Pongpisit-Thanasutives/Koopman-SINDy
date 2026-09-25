# Reviewer 1: figures and statistical reporting

Run from the repository root, preferably in a disposable working copy because reporting regenerates derived files and provenance:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python make_revision1_figures_tables.py
```

The script reads the current main ODE, Burgers/Fisher–KPP, advection–diffusion, and q/r-sensitivity raw CSVs after the corrected experiments are complete. It does **not** rerun model-discovery benchmarks. It regenerates two fixed illustrative interpolations only (seed 0, 3% noise, temporal sparse factor 16, upsampling factor 5), using the same corrected benchmark functions as the reported experiments. SHA256 fingerprints of the raw inputs and reconstruction code are saved in `results/revision1_reporting/reporting_manifest.json`.

Outputs below use the default `generated_assets/` directory. Pass `--latex-dir /path/to/working_copy/latex_source` to direct manuscript assets elsewhere; relative paths are interpreted from the invoking working directory. Numerical summaries and reporting manifests remain under this repository's `results/`.

Outputs:

- `generated_assets/tables/revision_ode_system.tex` and `revision_pde_system.tex`: corrected means and medians, with valid-record counts and standard errors.
- `generated_assets/tables/revision_paired_wins.tex`: strict improvement percentages and matched denominators by system.
- `generated_assets/figures/revision_interpolated_data.{pdf,png}`: noisy observations, inserted points, reset values, and clean reference data for Van der Pol and a Burgers time trace.
- `generated_assets/figures/revision_ode_noise_sparsity.{pdf,png}` and `revision_pde_noise_sparsity.{pdf,png}`: arithmetic mean coefficient error with one sample standard deviation over the other grid axis and seeds.
- Matching figure-insertion fragments are in `generated_assets/tables/revision_*_figure.tex`.
- `generated_assets/tables/revision_pod_attribution.tex`: matched raw-FD, POD-only (`q=1`), and POD-EDMD-RBF (`q=5`) controls at the fixed main rank for each PDE. The representative sensitivity slice uses 3% noise, sparse factor eight, `nx=64`, base `dt=0.01`, `T=2`, 12 RBF centres, and five paired seeds. It is a matched attribution comparison within this slice, not a claim that every main-grid setting is identical.
- Unrounded table statistics, matched win counts, curve statistics, invalid-record metadata, and representative-interpolation values/settings are in `results/revision1_reporting`.
- `pod_controls_paired_differences.csv` reports paired `q=5 minus q=1` mean changes, their seed standard errors, and strict win/tie counts. `pod_controls_matched_raw.csv` contains every contributing matched observation record.

After all experiment summaries have been regenerated, run `python sync_revision_outputs.py` (or supply the same `--latex-dir` destination) to copy the corrected appendix and Pareto figures and regenerate the non-oracle, strategy-ablation, and high-noise-TV table fragments. This command reports every interpolation strategy, including global rollout, and distinguishes correct support size from recovery of the correct active terms.

## Standard error and dispersion

For `N` valid records and `G` seed groups, the reported mean is the original record-weighted mean. The seed-cluster standard error is

```text
SE = sqrt[G/(G-1) * sum_g {sum_(i in g) (y_i - mean(y))}^2] / N.
```

For balanced grids, this equals the sample standard deviation of seed-level grid means divided by `sqrt(G)`. It quantifies variability across seed repetitions on the fixed benchmark grid. It is not an uncertainty estimate over arbitrary new systems or regimes. The few missing records retain their original available-record weighting. Figures instead show ordinary sample standard deviation across pooled valid records, which includes regime heterogeneity and is not a standard error or confidence interval.

No finite outlier is discarded. Figure axes are linear from zero to 0.01 and logarithmic above 0.01; the lower mean-minus-SD boundary is truncated at zero. `main_largest_errors.csv` identifies the largest current valid error for each method/system. Historical EDMD error values are not reused after propagation is corrected.

Two archived optDMD records are invalid due to SVD errors: Burgers (10% noise, sparse factor 32, seed 0) and Fisher–KPP (3% noise, sparse factor 16, seed 3). Each corresponding main table row has 159 valid records out of 160. Other main method/system rows are complete. Win rates use only valid method/baseline pairs, matched by system, noise, sparse factor, and seed; ties are not wins.

## Validation

The script rejects duplicate evaluation keys and requires finite metrics for valid records. The promoted POD comparison retains identical complete seed triples for all three controls. Table values are recomputed from current raw records; `main_table_complex_correction_changes.csv` compares them with the prior reporting snapshot when that archive is available. The complete manuscript determines final figure placement and page numbering. Generated PDF/PNG figures and table layouts are inspected again after corrected outputs are synchronized.

## Second-revision presentation regeneration

From any working directory, use:

```sh
python /path/to/koopman_sindy/regenerate_presentation_assets.py
```

This rebuilds the current tables and figures from saved numerical results; it does not integrate systems, draw new noise, tune parameters, fit models, or rerun experiments. It retains the saved representative interpolation illustration. Every protected raw result, coefficient/CV record, and numerical execution manifest is checked for identical SHA256 before and after generation. `results/revision1_reporting/presentation_generation_manifest.json` records current reporting code and generated manuscript-asset hashes separately from the preserved experimental provenance. Appendix prose/table fragments are written into `generated_assets/appendices` by default. Use `--latex-dir` to change the asset destination. TV prose in a manuscript's `sections/` directory is edited separately and is never overwritten by this command. The command refreshes derived reporting files in the repository even when a custom asset destination is supplied; use a disposable working copy to retain the delivered reports unchanged.

The TV display now uses all 720 saved method records. Table 7 (`tab:tv_high_noise`) reports 5–10% and 20–50% noise bands, each with twenty records per method (two noise levels, two spacings, five seeds). F1 and practical Score are means with standard errors over five seed means; coefficient error is the record median. Figure 6 (`tv_noise_coefficient_score.pdf`) resolves all four noise levels with separate coefficient-error and Score panels. `revision_tv_reporting.py` additionally saves matched comparisons by noise, by noise and spacing, and by band, as well as every paired record and the separate 1%-noise non-oracle evidence. The complete fixed grid is retained; no experiment or record is selected to favor a method.

The display label for the no-preprocessing temporal finite-difference arm is now **Raw FD**, including main and appendix tables. This is a presentation alias; archived raw method identifiers remain `baseline`. Weak-form regression, when reported separately, uses its own explicit raw-state label because that downstream method does not use temporal finite differences.

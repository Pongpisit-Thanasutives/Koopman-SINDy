# Final code audit and revision provenance

This document describes the final complex-propagation revision. It supersedes the earlier editorial-stage audit in which the main EDMD results had been retained. All reported outputs affected by the propagation/timestamp corrections have now been regenerated; unaffected comparison records were preserved. The supplied original package was retained unchanged outside this revision, and earlier raw results are retained in explicitly named archive directories.

The later author-requested refinement adds a fixed-window weak-form comparison and revises presentation without changing the saved numerical evidence below. See `README_WEAK_COMPARISON.md` for that experiment, `results/revision1_reporting/presentation_generation_manifest.json` for presentation-only changes, and `local_checking/REVISION_NOTES.md` in the separately delivered full submission package for the editorial changes.

## Standalone release scope

The post-submission GitHub release changes reporting-path handling, the full-package validation preflight, and standalone documentation only. Manuscript assets default to `generated_assets/`, with explicit destinations supported by the reporting commands. Saved numerical results, execution manifests, and numerical implementations are preserved; the release preservation record is `RELEASE_VALIDATION.json`.

The validation reports under `results/final_validation/` describe the frozen submitted research package. Their document checks and source fingerprints require that complete original package and are not a test certificate for this modified code-only release. Standalone implementation checks and reporting commands are listed in `README.md`; they can refresh diagnostic and derived output files and should be run in a disposable copy when retaining exact delivered provenance matters.

## Final numerical correction

The reported ODE EDMD and PDE POD-EDMD implementations now use `koopman_propagation.py`.

1. **Complex evolution is retained until physical readout.** Earlier code repeatedly applied the real part of a fractional-step matrix. In general, `Re(K^(1/q))^q` does not equal the fitted map K. The corrected implementation advances the complex observable state and takes its real physical-coordinate block only when recording a snapshot. Ordinary conjugate eigenpairs often already produce real fractional powers; the correction is not based on a claim that every complex mode was previously corrupted.
2. **Matrix functions avoid an eigenvector inverse.** A complex Schur decomposition gives K=Q T Q*. Schur diagonal values with imaginary magnitude no larger than `100 * machine_epsilon * max(1, abs(lambda))` are placed exactly on the real axis. Eigenvalues with magnitude below `1e-12` are replaced by positive `1e-12`. This explicitly defines a slightly regularized matrix `K_reg = Q T_reg Q*`. Fractional powers of `T_reg` use SciPy's complex fractional matrix function and are transformed by the unitary Q; integer powers use `matrix_power`. Negative-real eigenvalues use the selected principal scalar argument convention. Spectral regularization counts and perturbation sizes are recorded in the validation diagnostics. Non-finite input matrices or computed powers raise explicit errors rather than silently producing a successful record.
3. **Time steps are treated consistently.** The original observation arrays and nominal dense output grids are preserved. When appending the terminal observation creates a shorter final interval, that unequal-lag pair is excluded from the discrete-map fit. Each query is evaluated from its preceding observation using its actual elapsed time divided by the nominal observation spacing. Only roundoff-level equality can advance the anchor; the former half-substep reset to a genuinely later observation is removed. The dense grid can still stop just before the final observation, as in the supplied experiments; no new dense endpoint or baseline derivative convention is silently introduced.
4. **Local-reset design is retained.** The method still resets to observed state coordinates or their POD projections. Predicted interval endpoints need not equal the following observation, so reset jumps remain. The correction does not establish a uniquely identifiable continuous-time Koopman generator, remove dictionary/model bias or temporal aliasing, or guarantee better discovery accuracy. Analytic model derivatives were not substituted for the prescribed finite-difference discovery targets.

The correction covers the reported EDMD/POD-EDMD methods, their validation, and their reported reconstruction-strategy variants. Optional unreported state-DMD/forward-backward/TLS/Hankel helpers retain their legacy implementations; they are not represented as validated by this correction.

## Meaningful validation

`python validate_complex_propagation.py` passed six tests:

- preservation of phase and the one-step map for a negative-real eigenvalue;
- the correct fractional evolution of a rotation with conjugate eigenpairs;
- a defective Jordan block, which cannot be handled by an invertible eigenvector basis;
- a shortened final observation interval evaluated from the correct preceding anchor;
- explicit near-zero spectral regularization;
- explicit rejection of non-finite matrices.

The script also records eight representative fitted-map diagnostics: Lorenz–63, Van der Pol, Burgers and Fisher–KPP, each clean and at 3% noise. They use seed 0, base dt=0.005, q=5, ODE sparse factor 32, PDE sparse factor 8, and main PDE ranks 8/5. Operator and state-weighted defects distinguish a large matrix norm discrepancy from its effect on observed physical coordinates. These are numerical consistency diagnostics, not additional claims of improved equation-recovery performance. See `results/propagation_validation/`.

## Regenerated evidence and preserved controls

`rerun_complex_revision.py` is the executable specification. It provides named stages and saves a final manifest. The complete command is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python rerun_complex_revision.py
```

This driver uses the delivered baseline/control files and reruns the affected methods. The complete revision shell runner additionally regenerates the TV, non-oracle TV, and targeted weak-form studies and all reporting assets; see `README.md`.

| Output | Final records | Revision action |
|---|---:|---|
| Main ODE | 1200 | Reran all 800 polynomial/RBF EDMD records; retained 400 raw baselines |
| Main Burgers/Fisher–KPP | 960 | Reran 320 POD-EDMD records; retained 320 raw and 320 optDMD records |
| Main advection–diffusion | 180 | Reran 60 POD-EDMD records; retained 60 raw and 60 restricted-optDMD records |
| Appendix A q/r sensitivity | 545 | Regenerated the complete paired grid, including rank 5 and exact q=1 controls |
| Appendix C ODE | 1600 | Reused the 400 corrected polynomial-EDMD records from the identical main ODE grid; retained the other 1200 records |
| Appendix C PDE | 1920 | Reran all 480 observation-validated POD-EDMD records; retained the other 1440 records |
| Original non-oracle study | 40 selected equation records | Regenerated candidates, complexity–EBIC curves and selections from 30 trajectory fits, using kneed 0.8.6 |
| Fisher–KPP front sensitivity | 10 | Reran 5 POD-EDMD records; validated and retained the 5 original raw baselines |
| Interpolation-strategy ablation | 600 | Regenerated all assisted strategies with matched fitted dictionaries; retained 120 original raw-baseline records |
| TV/high-noise revision | 720 | Regenerated separately using the corrected core |
| Non-oracle TV extension | 70 selected equation records | Completed separately using the corrected core and the frozen matched protocol |

Every newly computed assisted record succeeded. The main PDE raw file still contains the two original optDMD numerical failures, one per system; these were not caused by this revision and are not hidden. Thus its successful-record count is 958, whereas the other benchmark grids in the table have no failures. No additional local experiment is required to use the delivered revised manuscript.

Affected coefficient examples were regenerated or replaced along with the raw metrics. Both main ODE/PDE example files and the classical ODE examples have their affected methods replaced. The classical PDE example file contains the corrected POD-EDMD examples; additional ADE/front coefficient examples are also supplied. Summary CSVs, best-case CSVs, win rates, appendix fragments and figures derive from the final records.

## Exact settings and random-number matching

- Main ODE uses dt=0.005, sparse factors 8/16/32/64, noise 0/0.01/0.03/0.05/0.10, ten seeds, q=5, and up to 40 RBF centers.
- Main Burgers/Fisher uses nx=64, dt=0.005, sparse factors 4/8/16/32, the same five noise levels, eight seeds, q=5, up to 30 RBF centers, and ranks **8 and 5**, respectively. Rank 5 is established by the supplied raw records; the earlier manuscript/default rank 2 was a provenance error.
- Separate main advection–diffusion uses nx=48, dt=0.01, sparse factors 4/8/16, noise 0.01/0.03/0.05/0.10, five seeds, q=5, rank 4 and 12 RBF centers. Its original method loop consumed an optDMD regression-row subsample before choosing RBF centers. The targeted rerun replays that exact RNG draw, preserving the original RBF choices without recomputing the unaffected optDMD comparator.
- The documented original non-oracle command uses the quick preset: Van der Pol dt=0.01 and sparse factor 8; Burgers/Fisher nx=48, dt=0.01 and sparse factor 4; 1% noise, five seeds, q=5, ranks 6/2, 20 RBF centers and at most 5000 regression rows. Fisher uses the front initial condition. The corresponding raw/EDMD arms agree exactly with all 40 shared equation records in the non-oracle TV extension; its separate concordance file records the validation.
- Fisher front sensitivity uses nx=48, dt=0.01, sparse factor 8, noise 0.01, rank 2, 20 RBF centers and maximum 5000 rows. These previously unrecorded details were recovered by reproducing the supplied baseline and assisted records with the original implementation; the final driver checks all five baseline coefficient errors before replacing assisted records.
- Appendix A uses dt=0.01, nx=64 for PDEs, 3% noise, ODE sparse factor 16, PDE factor 8, seeds 0–4, 12 RBF centers, maximum 10000 regression rows, q in 1/3/5/7 and rank in 1/2/4/5/6/8/10/12. Stable integer system codes replace process-randomized Python hashes, and each system/seed shares the same noisy observations across q/r. ODE q=1 returns the observations exactly; PDE q=1 performs only POD projection at observed times.
- The strategy study uses 10% noise, ODE sparse factor 64, PDE factor 32, twenty seeds, dt=0.005, nx=64, rank 8 and 40 RBF centers. Its observation indices are uniformly subsampled and cropped, **without appending an off-grid terminal observation**. Consequently every residual-correction anchor lies on the evaluated dense grid. An explicit guard now rejects off-grid use of that helper. RBF strategies share the same post-noise RNG state so they use the same centers and fitted dictionary, rather than silently changing the model between reconstruction strategies.

## Appendix C validation correction

In addition to phase/timestamp consistency, rank validation was corrected. The original validation removed interior observations but passed the resulting nonuniform times to a uniform-step reset routine. That could fit across an inappropriate doubled lag and evaluate a held-out point using a later anchor.

The final three-fold validation fits POD and RBF features using only training observations; fits the map only from training pairs separated by one original observation interval; and predicts each held-out observation from its preceding training anchor at the true elapsed fraction. All required folds must succeed for a candidate rank to receive a finite score. The rank candidates remain 1/2/3/4/6/8. No clean trajectory, true support, coefficient error or downstream oracle score is used to tune the rank.

The final standalone `results/appendix_c_corrected_pod/` is synchronized with the 480 complex-propagation records in `results/complex_reruns/appendix_c_pod/`. It is not the earlier intermediate checkpoint. The obsolete reporting `--merge-classical` option is disabled; the rerun driver performs the controlled merge.

## Preserved controls and numerical drift

Copied main/raw/classical controls match the archived records exactly in the final CSVs. The original non-oracle baseline was recomputed with the specified selector: support, threshold and F1 agree exactly; the maximum coefficient-error difference from the supplied record is about 1.05e-10. This is recorded explicitly rather than described as bitwise agreement.

Recomputing the strategy baselines in the revision environment produced a maximum coefficient-error difference of about 1.73e-5 in Fisher–KPP, with unchanged F1. The final table retains the original unaffected baseline records. The complete new execution and a paired drift CSV are retained under `results/complex_reruns/`, so this preservation is fully inspectable. The assisted strategies all use the same prescribed noisy data, seeds and revised numerical environment.

## Reproducibility and selection details

The final revision uses Python 3.12.14 with the versions in `requirements_revision.txt`. Kneed 0.8.6 is required for the specified original non-oracle selector; the absent-library fallback can choose different knees and is not used for the final revision. Tabulate 0.9.0 supports the existing strategy/model-selection Markdown reporters. The earlier `revision_audit_environment.json` documents the initial editorial-stage environment, where these packages were absent; the **final** environment and source/output hashes are in `complex_revision_manifest.json` and the separate TV manifests.

Three isolated core runs finished before a documentation-only correction to the restricted real-exponent comparator's docstring. Their execution hashes and the final package hashes are preserved separately in `complex_reruns/isolated_execution_source_provenance.json`. The strategy's post-run off-grid validity guard likewise has a separate execution/final hash note; it changes no reported numerical branch. Execution hashes are not silently rewritten to pretend that post-run documentation/validation guards were present earlier.

Explicit ODE/PDE CLI arguments now override presets. `--resume` in the original benchmark scripts still assumes identical settings, because its case key does not encode every hyperparameter; use a fresh output directory after changing a configuration. The focused driver's `--reuse-completed` option is only for completed isolated runs from the same corrected source/settings.

The reported classical comparison has four methods. GP smoothing is an optional, unreported `--include-gp` extension. Main Burgers/Fisher optDMD uses PyDMD with zero bagging trials; the standalone ADE control is a retained **restricted real-exponent** fit, not general complex-rate optDMD. Travelling Fourier modes are oscillatory, so these two comparators are explicitly distinguished.

The ODE polynomial library contains every monomial of total degree 0 through 3 once: 20 features for Lorenz and 10 for Van der Pol. RBF lifts contain the constant, linear coordinates and Gaussian RBFs; centers are sampled from observed states/POD coordinates without replacement. Their inverse width is the reciprocal of the median strictly positive squared center distance, with a numerical floor. STLSQ solves use normalized library columns but threshold and return physical-unit coefficients; its equivalent physical-coordinate ridge penalty is alpha times the squared norm of the column-norm-scaled coefficients.

Oracle selection scans all thresholds, including different thresholds giving the same model size. It compares candidates by score, then F1, then lower coefficient error; an exact remaining tie retains the first ascending-grid threshold. The main PDE/ADE/classical/strategy grid has 11 values. Non-oracle, q/r sensitivity and the front study use the expanded 20-value PDE grid.

In the non-oracle candidate scan, a repeated exact support is skipped unless its RSS improves. An improved candidate is appended; the earlier row need not be deleted because final selection retains the lowest EBIC per support size. The elbow acts on that complexity–EBIC curve. There is no additional dominance-pruning pass and no access to true support or true coefficients for selection.

The main results and revised controls support system-dependent conclusions. Correct propagation is necessary for internal consistency; it is not presented as a guarantee of superior accuracy or a reason to conceal settings where raw data, POD-only denoising or TV/spline preprocessing perform better.

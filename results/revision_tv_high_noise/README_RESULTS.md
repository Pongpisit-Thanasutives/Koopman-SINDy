# TV comparison: complete saved regime grid

Reporting can be regenerated without numerical experiments, from any working directory:

```sh
python /path/to/code_repository/revision_tv_reporting.py
```

The archived experiment has 720 successful method records: five systems, four noise levels (5%,10%,20%,50%), two sampling spacings per system, five seeds, three ODE arms and four PDE arms. Every arm sees the same noisy observations for each matched record. All downstream STLSQ thresholds use the original oracle diagnostic rule.

- `raw_results.csv`, `coefficients.csv`, `tv_cv_diagnostics.csv`, and `run_manifest.json` are execution artifacts and are not rewritten by reporting.
- `summary_by_noise.csv`: ten records per system/method/noise.
- `summary_by_noise_band.csv`: all records separated into 5–10% and20–50% bands; twenty records per system/method/band.
- `summary_by_noise_spacing.csv`: five seeds for each individual sampling/noise condition.
- Means are record-level means. SEs first average the fixed noise/sampling conditions within each seed and then take the standard deviation of five seed averages divided by sqrt(5). Coefficient-error medians are taken over records.
- `paired_by_noise*.csv`: strict matched EDMD-minus-comparator mean differences, seed SEs, wins, ties, sample counts, and seed-mean direction counts. Smaller error is better; larger F1/Score is better. Ties are not wins. Exact-support counts are in the summary CSVs.
- `tv_noise_coefficient_score.pdf/png`: each system's median coefficient error (left, log scale) and mean Score with seed SE (right), displaying all four noise levels; dotted dividers separate the two table bands. EDMD denotes polynomial EDMD for ODEs and POD-EDMD-RBF for PDEs.
- `tv_high_noise_support_f1.pdf/png`: companion support-recovery figure.
- `artifact_generation_manifest.json` identifies reporting code, input hashes, output hashes, and generation environment separately from the preserved execution provenance.

POD+TV and POD-EDMD-RBF use the same fixed rank. TV penalties use noisy holdouts only, with POD refitted in each training fold. The complete results establish conditional performance differences, not universal high-noise robustness or complete PDE recovery.

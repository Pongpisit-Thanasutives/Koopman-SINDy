# Reporting changes after complex propagation correction

All current tables and figure data were regenerated after the final corrected raw-data signal. No additional discovery experiments were run by the reporting scripts. The two representative interpolation illustrations were reconstructed with the corrected benchmark functions using their stated seed, noise, grid, and dictionaries.

## Claims requiring updated wording

- Main ODE results: polynomial EDMD retains the largest mean support F1 and practical Score for both systems; it does not have the lowest median coefficient error for Van der Pol. EDMD-RBF has median error 0.012276 versus 0.015795 for polynomial EDMD (raw 0.106755). Avoid an unqualified claim that polynomial EDMD is best by coefficient accuracy on both ODEs.
- Lorenz polynomial EDMD median error changes from 0.526120 to 0.545940. The old finite error 41.750558 is not a result of the corrected implementation and must not be described as a current outlier. Current largest errors are listed below.
- Matched fixed-rank POD controls show that most of the representative Burgers and advection–diffusion gain over raw differentiation is already present after POD-only denoising. Additional EDMD temporal reconstruction is not uniformly beneficial. This comparison does not separate learned temporal dynamics from time-grid densification alone.
- In the fixed-rank control, Burgers median error rises from 0.217892 (POD only) to 0.225424 (POD+EDMD); Fisher–KPP has a small mixed effect (median 0.037132 to 0.039053, but mean error decreases slightly); advection–diffusion improves modestly from median 0.038968 to 0.036053. F1 is unchanged in all three controls. Paired differences, not independent-method standard errors, quantify the uncertainty of incremental effects below.
- Stress-condition strategy ablation: residual correction improves ODE median error and mean Score over local reset for both polynomial and RBF EDMD. Local RBF has the larger support F1. PDE local reset, global rollout and two-sided interpolation are numerically near-tied at displayed precision, and all are worse than the raw baseline on the displayed aggregate metrics. Remove the prior claim that two-sided interpolation offers a meaningful PDE improvement. The generated table now includes all global-rollout rows.
- The non-oracle selected supports are unchanged: assisted methods recover exact supports in all five seeds per equation. Corrected Burgers assisted median coefficient error is 0.018629 (was 0.019580); Fisher–KPP is 0.010782 (was 0.010501). The raw Fisher–KPP selected size ranges from 2 to 5 and recovers zero exact supports.
- Fisher-front oracle sensitivity: corrected POD-EDMD F1/median error/Score is 0.920/0.018/0.898; use its current summary for unrounded values.
- Appendix A uses base dt=0.01 for ODEs as well as PDEs. POD-control PDEs have nx=64, RBF centres=12. These differ from some main settings and are now stated explicitly. The best-score (q,r) points remain advection–diffusion (3,4), Burgers (1,5), and Fisher–KPP (3,10), but cross-rank optima do not isolate the temporal contribution.
- Appendix C's ODE polynomial method still has the best mean Score and median coefficient error among the four methods in that appendix (RBF EDMD is not included there). In the PDE comparison, tuned splines retain the highest pooled F1 and Score; POD-EDMD retains the lowest pooled median coefficient error. Per-system conclusions are generated directly from the corrected data.

## Main-table old versus corrected values

| system              | method       |   f1_mean_before |   f1_mean_corrected |   coef_error_median_before |   coef_error_median_corrected |   score_mean_before |   score_mean_corrected |
|:--------------------|:-------------|-----------------:|--------------------:|---------------------------:|------------------------------:|--------------------:|-----------------------:|
| lorenz63            | baseline     |         0.724680 |            0.724680 |                   0.911171 |                      0.911171 |            0.476956 |               0.476956 |
| lorenz63            | edmd_poly3   |         0.737449 |            0.737649 |                   0.526120 |                      0.545940 |            0.561494 |               0.548909 |
| lorenz63            | edmd_rbf     |         0.705130 |            0.724449 |                   0.931703 |                      0.885420 |            0.518318 |               0.526317 |
| vanderpol_mu2       | baseline     |         0.962893 |            0.962893 |                   0.106755 |                      0.106755 |            0.840205 |               0.840205 |
| vanderpol_mu2       | edmd_poly3   |         0.998730 |            0.999286 |                   0.015418 |                      0.015795 |            0.976581 |               0.977149 |
| vanderpol_mu2       | edmd_rbf     |         0.982894 |            0.982902 |                   0.019394 |                      0.012276 |            0.935557 |               0.944643 |
| burgers             | baseline     |         0.730208 |            0.730208 |                   0.297639 |                      0.297639 |            0.579963 |               0.579963 |
| burgers             | pydmd_optdmd |         0.786024 |            0.786024 |                   0.179443 |                      0.179443 |            0.688419 |               0.688419 |
| burgers             | pod_edmd_rbf |         0.774861 |            0.777361 |                   0.216544 |                      0.215269 |            0.657558 |               0.663199 |
| fisher_kpp          | baseline     |         0.836981 |            0.836981 |                   0.038533 |                      0.038533 |            0.809902 |               0.809902 |
| fisher_kpp          | pydmd_optdmd |         0.846002 |            0.846002 |                   0.038765 |                      0.038765 |            0.821593 |               0.821593 |
| fisher_kpp          | pod_edmd_rbf |         0.826450 |            0.828117 |                   0.038713 |                      0.038689 |            0.784529 |               0.785556 |
| advection_diffusion | baseline     |         0.655556 |            0.655556 |                   0.165805 |                      0.165805 |            0.548801 |               0.548801 |
| advection_diffusion | optdmd       |         0.666667 |            0.666667 |                   0.028640 |                      0.028640 |            0.632517 |               0.632517 |
| advection_diffusion | pod_edmd_rbf |         0.702778 |            0.703889 |                   0.038936 |                      0.044226 |            0.656184 |               0.659158 |

## Current main largest coefficient errors

| system              | method       |   sparse_factor |    noise |   seed |   coef_error |
|:--------------------|:-------------|----------------:|---------:|-------:|-------------:|
| burgers             | pydmd_optdmd |              32 | 0.050000 |      2 |   743.203378 |
| lorenz63            | edmd_poly3   |              64 | 0.100000 |      5 |     3.987279 |
| lorenz63            | edmd_rbf     |              64 | 0.030000 |      9 |     3.231118 |
| fisher_kpp          | pod_edmd_rbf |               4 | 0.100000 |      0 |     2.932091 |
| lorenz63            | baseline     |              32 | 0.010000 |      9 |     2.134593 |
| burgers             | pod_edmd_rbf |              16 | 0.100000 |      1 |     0.986671 |
| vanderpol_mu2       | edmd_rbf     |              16 | 0.100000 |      3 |     0.955789 |
| vanderpol_mu2       | baseline     |              64 | 0.030000 |      6 |     0.815949 |
| burgers             | baseline     |              16 | 0.100000 |      7 |     0.781243 |
| advection_diffusion | pod_edmd_rbf |              16 | 0.100000 |      4 |     0.674632 |
| fisher_kpp          | baseline     |               4 | 0.100000 |      3 |     0.651172 |
| advection_diffusion | baseline     |               8 | 0.100000 |      0 |     0.579171 |
| vanderpol_mu2       | edmd_poly3   |              32 | 0.100000 |      9 |     0.316856 |
| advection_diffusion | optdmd       |              16 | 0.100000 |      0 |     0.293201 |
| fisher_kpp          | pydmd_optdmd |              32 | 0.100000 |      6 |     0.075711 |

## Corrected main paired win rates

| system              | method       |   paired_n |   coefficient_wins |   coefficient_win_percent |   score_wins |   score_win_percent |
|:--------------------|:-------------|-----------:|-------------------:|--------------------------:|-------------:|--------------------:|
| lorenz63            | edmd_poly3   |        200 |                119 |                 59.500000 |          114 |           57.000000 |
| lorenz63            | edmd_rbf     |        200 |                105 |                 52.500000 |          113 |           56.500000 |
| vanderpol_mu2       | edmd_poly3   |        200 |                192 |                 96.000000 |          192 |           96.000000 |
| vanderpol_mu2       | edmd_rbf     |        200 |                191 |                 95.500000 |          188 |           94.000000 |
| burgers             | pydmd_optdmd |        159 |                140 |                 88.050314 |          140 |           88.050314 |
| burgers             | pod_edmd_rbf |        160 |                139 |                 86.875000 |          139 |           86.875000 |
| fisher_kpp          | pydmd_optdmd |        159 |                 84 |                 52.830189 |           84 |           52.830189 |
| fisher_kpp          | pod_edmd_rbf |        160 |                 65 |                 40.625000 |           65 |           40.625000 |
| advection_diffusion | optdmd       |         60 |                 50 |                 83.333333 |           50 |           83.333333 |
| advection_diffusion | pod_edmd_rbf |         60 |                 59 |                 98.333333 |           58 |           96.666667 |

## Fixed-rank matched POD controls

| system              | control   |   n_paired |   f1_mean |    f1_se |   coef_error_median |   score_mean |   score_se |
|:--------------------|:----------|-----------:|----------:|---------:|--------------------:|-------------:|-----------:|
| burgers             | raw       |          5 |  0.666667 | 0.000000 |            0.289903 |     0.516375 |   0.001541 |
| burgers             | pod_only  |          5 |  0.666667 | 0.000000 |            0.217892 |     0.547411 |   0.001535 |
| burgers             | pod_edmd  |          5 |  0.666667 | 0.000000 |            0.225424 |     0.537541 |   0.008279 |
| fisher_kpp          | raw       |          5 |  0.800000 | 0.000000 |            0.036742 |     0.768592 |   0.002245 |
| fisher_kpp          | pod_only  |          5 |  0.800000 | 0.000000 |            0.037132 |     0.768477 |   0.002207 |
| fisher_kpp          | pod_edmd  |          5 |  0.800000 | 0.000000 |            0.039053 |     0.769589 |   0.003054 |
| advection_diffusion | raw       |          5 |  0.666667 | 0.000000 |            0.167736 |     0.570976 |   0.001598 |
| advection_diffusion | pod_only  |          5 |  0.666667 | 0.000000 |            0.038968 |     0.642045 |   0.000624 |
| advection_diffusion | pod_edmd  |          5 |  0.666667 | 0.000000 |            0.036053 |     0.643382 |   0.000549 |

## Paired q5 minus q1 changes

| system              |   fixed_pod_rank |   n_paired |   delta_f1_mean |   delta_f1_se |   delta_coef_error_mean |   delta_coef_error_se |   delta_score_mean |   delta_score_se |   coefficient_wins |   coefficient_ties |   score_wins |   score_ties |
|:--------------------|-----------------:|-----------:|----------------:|--------------:|------------------------:|----------------------:|-------------------:|-----------------:|-------------------:|-------------------:|-------------:|-------------:|
| burgers             |                8 |          5 |        0.000000 |      0.000000 |                0.023549 |              0.019734 |          -0.009870 |         0.008185 |                  1 |                  0 |            1 |            0 |
| fisher_kpp          |                5 |          5 |        0.000000 |      0.000000 |               -0.001474 |              0.001807 |           0.001112 |         0.001357 |                  3 |                  0 |            3 |            0 |
| advection_diffusion |                4 |          5 |        0.000000 |      0.000000 |               -0.002159 |              0.000546 |           0.001337 |         0.000338 |                  5 |                  0 |            5 |            0 |

## Corrected strategy summary

| setting   | method       | strategy           |   n_valid |   f1_mean |   coef_error_median |   score_mean |
|:----------|:-------------|:-------------------|----------:|----------:|--------------------:|-------------:|
| ODE       | edmd_poly3   | local_reset        |        40 |  0.749407 |            0.681949 |     0.564747 |
| ODE       | edmd_poly3   | global_rollout     |        40 |  0.596258 |            1.091565 |     0.262014 |
| ODE       | edmd_poly3   | residual_corrected |        40 |  0.767779 |            0.576469 |     0.567116 |
| ODE       | edmd_poly3   | two_sided          |        40 |  0.719115 |            0.799190 |     0.490665 |
| ODE       | edmd_rbf     | local_reset        |        40 |  0.745072 |            0.794096 |     0.526420 |
| ODE       | edmd_rbf     | global_rollout     |        40 |  0.568172 |            1.032335 |     0.272507 |
| ODE       | edmd_rbf     | residual_corrected |        40 |  0.736659 |            0.571318 |     0.547339 |
| ODE       | edmd_rbf     | two_sided          |        40 |  0.657451 |            0.955852 |     0.416478 |
| PDE       | pod_edmd_rbf | local_reset        |        40 |  0.711111 |            0.455770 |     0.558091 |
| PDE       | pod_edmd_rbf | global_rollout     |        40 |  0.711111 |            0.455761 |     0.558090 |
| PDE       | pod_edmd_rbf | residual_corrected |        40 |  0.681905 |            0.612592 |     0.524836 |
| PDE       | pod_edmd_rbf | two_sided          |        40 |  0.711111 |            0.455770 |     0.558091 |
| ODE       | edmd_poly3   | baseline           |        40 |  0.804933 |            0.703141 |     0.522310 |
| ODE       | edmd_rbf     | baseline           |        40 |  0.804933 |            0.703141 |     0.522310 |
| PDE       | pod_edmd_rbf | baseline           |        40 |  0.733333 |            0.409196 |     0.571386 |

## Corrected non-oracle selection

| setting   | system        | equation   | method       |   n |   true_k |   selected_k_min |   selected_k_max |   selected_k_median |   exact_k_count |   exact_support_count |   f1_mean |   coef_error_median |
|:----------|:--------------|:-----------|:-------------|----:|---------:|-----------------:|-----------------:|--------------------:|----------------:|----------------------:|----------:|--------------------:|
| ODE       | vanderpol_mu2 | dx/dt      | baseline     |   5 |        1 |                1 |                1 |            1.000000 |               5 |                     5 |  1.000000 |            0.005301 |
| ODE       | vanderpol_mu2 | dx/dt      | edmd_poly3   |   5 |        1 |                1 |                1 |            1.000000 |               5 |                     5 |  1.000000 |            0.001008 |
| ODE       | vanderpol_mu2 | dy/dt      | baseline     |   5 |        3 |                3 |                3 |            3.000000 |               5 |                     5 |  1.000000 |            0.031306 |
| ODE       | vanderpol_mu2 | dy/dt      | edmd_poly3   |   5 |        3 |                3 |                3 |            3.000000 |               5 |                     5 |  1.000000 |            0.004531 |
| PDE       | burgers       | u_t        | baseline     |   5 |        2 |                2 |                2 |            2.000000 |               5 |                     5 |  1.000000 |            0.046075 |
| PDE       | burgers       | u_t        | pod_edmd_rbf |   5 |        2 |                2 |                2 |            2.000000 |               5 |                     5 |  1.000000 |            0.018629 |
| PDE       | fisher_kpp    | u_t        | baseline     |   5 |        3 |                2 |                5 |            3.000000 |               3 |                     0 |  0.660000 |            0.058688 |
| PDE       | fisher_kpp    | u_t        | pod_edmd_rbf |   5 |        3 |                3 |                3 |            3.000000 |               5 |                     5 |  1.000000 |            0.010782 |

## Reporting validation

The corrected interpolation and regime plots were visually inspected. The three new POD-control, full-strategy, and non-oracle table fragments compiled at the journal text width without overfull boxes and were visually inspected. Final manuscript pagination remains controlled by the root manuscript build. `reporting_manifest.json` fingerprints all contributing raw data and reporting/reconstruction code.

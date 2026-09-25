# Findings from the frozen weak-form experiment

All 60 planned noisy records and all 300 preprocessing-method results completed successfully. Each system has six fixed noise/sampling conditions and five seeds; nothing was selected for reporting based on the direction of the results. There is no general improvement of weak-SINDy identification from EDMD preprocessing in this experiment.

Van der Pol shows a limited favorable regime: EDMD improves mean Score relative to raw, linear interpolation and TV at 5% noise for both spacings, and at 10% noise for the denser spacing. At the coarser 5% setting, median coefficient error is 0.0292 (EDMD), 0.0507 (raw), 0.0507 (TV), and 0.1377 (linear); exact support is recovered in 5/5 EDMD records versus 4/5 raw and TV records. However, EDMD loses mean Score at the coarse 10% condition and both 20% conditions. Across all 30 records its median coefficient error is smaller than raw, but mean coefficient error is larger (0.3167 versus 0.2281), support F1 drops from 0.9398 to 0.8727, and Score drops from 0.8060 to 0.7440. This distinction prevents a favorable median from masking a less reliable high-noise result.

For Burgers, weak fitting already works well from raw observations. POD-EDMD provides no pooled advantage over raw weak fitting, linear interpolation, POD-only or either TV control. Raw versus POD-EDMD mean F1 is 0.9333 versus 0.9044, median coefficient error 0.01382 versus 0.01592, and Score 0.91753 versus 0.88628. Their exact support counts are 24/30 and 21/30. POD-only Score is 0.89788, slightly above POD-EDMD. EDMD beats raw Score in 9/30 matched records, whereas Van der Pol has 15/30 wins. These are descriptive paired counts, not significance tests.

The results support compatibility of local learned reconstruction with an integral downstream fit and a limited low-noise ODE gain. They do not extend the paper's method into a generally noise-robust weak formulation, and they do not establish superiority over a complete WSINDy implementation. Weak integration and TV can already reduce effects of differentiation noise; in this study additional learned reconstruction introduces biases that can outweigh its benefit. The last sentence is an interpretation, not a separate causal experiment. The linear and POD controls help distinguish generic densification and spatial projection from the combined EDMD preprocessing effect.

The PDE library is conservative and replaces the main study's nuisance u*u_xx by (u²)_xx in every arm. Thus these are matched comparisons among preprocessing choices inside the same weak library, not a controlled direct comparison of weak and strong formulations.

## All-condition pooled results

| system        | method      | method_label           |   support_f1_mean |   coef_error_median |   practical_score_mean |   exact_support_rate |   state_error_median |   n_ok |
|:--------------|:------------|:-----------------------|------------------:|--------------------:|-----------------------:|---------------------:|---------------------:|-------:|
| burgers       | edmd_weak   | EDMD + weak            |           0.90444 |             0.01592 |                0.88628 |              0.7     |              0.09214 |     30 |
| burgers       | linear_weak | Linear q5 + weak       |           0.93333 |             0.01317 |                0.91761 |              0.8     |              0.08291 |     30 |
| burgers       | pod_tv_weak | POD + TV states + weak |           0.92667 |             0.0111  |                0.91071 |              0.76667 |              0.0547  |     30 |
| burgers       | pod_weak    | POD only + weak        |           0.91556 |             0.0152  |                0.89788 |              0.73333 |              0.07595 |     30 |
| burgers       | raw_weak    | Raw + weak             |           0.93333 |             0.01382 |                0.91753 |              0.8     |              0.10038 |     30 |
| burgers       | tv_weak     | TV states + weak       |           0.92222 |             0.01075 |                0.90662 |              0.76667 |              0.05527 |     30 |
| vanderpol_mu2 | edmd_weak   | EDMD + weak            |           0.87265 |             0.04965 |                0.74397 |              0.56667 |              0.09652 |     30 |
| vanderpol_mu2 | linear_weak | Linear q5 + weak       |           0.91577 |             0.13484 |                0.76739 |              0.66667 |              0.09039 |     30 |
| vanderpol_mu2 | raw_weak    | Raw + weak             |           0.93984 |             0.09294 |                0.80599 |              0.7     |              0.09867 |     30 |
| vanderpol_mu2 | tv_weak     | TV states + weak       |           0.94213 |             0.07438 |                0.81346 |              0.73333 |              0.07018 |     30 |

## All twelve noise/sampling regimes

Mean Score across five seeds; full F1, coefficient-error and uncertainty records are in `summary_by_condition.csv`.

| system        |   obs_dt |   noise |   raw_score |   linear_score |   edmd_score |   tv_score |   edmd_minus_raw |
|:--------------|---------:|--------:|------------:|---------------:|-------------:|-----------:|-----------------:|
| burgers       |     0.04 |    0.05 |     0.99472 |        0.99471 |      0.99474 |    0.99474 |          2e-05   |
| burgers       |     0.04 |    0.1  |     0.99166 |        0.99157 |      0.99166 |    0.99165 |         -0       |
| burgers       |     0.04 |    0.2  |     0.98272 |        0.98275 |      0.93669 |    0.91525 |         -0.04603 |
| burgers       |     0.16 |    0.05 |     0.99275 |        0.99297 |      0.99252 |    0.9923  |         -0.00023 |
| burgers       |     0.16 |    0.1  |     0.90979 |        0.91002 |      0.76952 |    0.91233 |         -0.14027 |
| burgers       |     0.16 |    0.2  |     0.63355 |        0.63362 |      0.63254 |    0.63346 |         -0.00101 |
| vanderpol_mu2 |     0.08 |    0.05 |     0.98087 |        0.97878 |      0.98358 |    0.98024 |          0.00272 |
| vanderpol_mu2 |     0.08 |    0.1  |     0.84682 |        0.86619 |      0.94959 |    0.93969 |          0.10278 |
| vanderpol_mu2 |     0.08 |    0.2  |     0.63345 |        0.74937 |      0.55179 |    0.59023 |         -0.08166 |
| vanderpol_mu2 |     0.32 |    0.05 |     0.90037 |        0.71848 |      0.97713 |    0.90034 |          0.07675 |
| vanderpol_mu2 |     0.32 |    0.1  |     0.89734 |        0.77763 |      0.64502 |    0.89737 |         -0.25232 |
| vanderpol_mu2 |     0.32 |    0.2  |     0.57713 |        0.51386 |      0.35671 |    0.57289 |         -0.22041 |

## Paired comparisons over all records

Differences are EDMD minus the named control; positive Score/F1 differences favor EDMD, negative coefficient-error differences favor EDMD.

| system        | control     |   mean_delta_f1 |   median_delta_error |   mean_delta_score |   edmd_score_wins |   score_ties |   n_pairs |
|:--------------|:------------|----------------:|---------------------:|-------------------:|------------------:|-------------:|----------:|
| burgers       | linear_weak |        -0.02889 |              0.00023 |           -0.03133 |                10 |            0 |        30 |
| burgers       | pod_tv_weak |        -0.02222 |              0.0004  |           -0.02443 |                11 |            0 |        30 |
| burgers       | pod_weak    |        -0.01111 |              0.0001  |           -0.0116  |                14 |            0 |        30 |
| burgers       | raw_weak    |        -0.02889 |              0.00025 |           -0.03125 |                 9 |            0 |        30 |
| burgers       | tv_weak     |        -0.01778 |              0.00044 |           -0.02034 |                13 |            0 |        30 |
| vanderpol_mu2 | linear_weak |        -0.04311 |             -0.00364 |           -0.02341 |                16 |            0 |        30 |
| vanderpol_mu2 | raw_weak    |        -0.06719 |             -0.00527 |           -0.06202 |                15 |            0 |        30 |
| vanderpol_mu2 | tv_weak     |        -0.06947 |             -0.0026  |           -0.06949 |                16 |            0 |        30 |

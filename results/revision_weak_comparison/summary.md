# Focused weak-SINDy comparison

Oracle sparsity thresholds; all planned noise/sampling/seed combinations retained. The fully weak conservative PDE library changes one nuisance term relative to the main strong-form study. These are within-library preprocessing comparisons, not a complete WSINDy benchmark.

| system        | method      | method_label           |   support_f1_mean |   coef_error_median |   practical_score_mean |   exact_support_rate |   state_error_median |   n_ok |
|:--------------|:------------|:-----------------------|------------------:|--------------------:|-----------------------:|---------------------:|---------------------:|-------:|
| burgers       | edmd_weak   | EDMD + weak            |          0.904444 |           0.0159152 |               0.886279 |             0.7      |            0.0921447 |     30 |
| burgers       | linear_weak | Linear q5 + weak       |          0.933333 |           0.0131722 |               0.917609 |             0.8      |            0.0829111 |     30 |
| burgers       | pod_tv_weak | POD + TV states + weak |          0.926667 |           0.0110999 |               0.910705 |             0.766667 |            0.0546996 |     30 |
| burgers       | pod_weak    | POD only + weak        |          0.915556 |           0.0151991 |               0.897878 |             0.733333 |            0.0759538 |     30 |
| burgers       | raw_weak    | Raw + weak             |          0.933333 |           0.0138242 |               0.917531 |             0.8      |            0.10038   |     30 |
| burgers       | tv_weak     | TV states + weak       |          0.922222 |           0.0107479 |               0.906622 |             0.766667 |            0.0552673 |     30 |
| vanderpol_mu2 | edmd_weak   | EDMD + weak            |          0.872653 |           0.0496538 |               0.74397  |             0.566667 |            0.0965239 |     30 |
| vanderpol_mu2 | linear_weak | Linear q5 + weak       |          0.915767 |           0.134839  |               0.767385 |             0.666667 |            0.0903894 |     30 |
| vanderpol_mu2 | raw_weak    | Raw + weak             |          0.939841 |           0.0929355 |               0.805995 |             0.7      |            0.0986658 |     30 |
| vanderpol_mu2 | tv_weak     | TV states + weak       |          0.942126 |           0.0743827 |               0.81346  |             0.733333 |            0.0701788 |     30 |

See summary_by_condition.csv for all conditions and paired_summary.csv for matched contrasts.

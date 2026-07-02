# ODE benchmark tables

## Overall comparison

| Method | Mean support F1 | Median coefficient error | Mean practical score |
|---|---:|---:|---:|
| Baseline | 0.844 | 0.308 | 0.659 |
| EDMD-polynomial | 0.868 | 0.032 | 0.769 |
| Linear interpolation | 0.862 | 0.127 | 0.715 |
| Tuned smoothing spline | 0.863 | 0.047 | 0.743 |

## Median coefficient error by noise level

| Noise | Baseline | EDMD-polynomial | Linear interpolation | Tuned smoothing spline |
|---:|---:|---:|---:|---:|
| 0.00 | 0.234 | 0.009 | 0.088 | 0.019 |
| 0.01 | 0.232 | 0.011 | 0.087 | 0.026 |
| 0.03 | 0.205 | 0.021 | 0.121 | 0.037 |
| 0.05 | 0.286 | 0.039 | 0.118 | 0.057 |
| 0.10 | 0.385 | 0.087 | 0.190 | 0.143 |

## Median coefficient error by sparse factor

| Sparse factor | Baseline | EDMD-polynomial | Linear interpolation | Tuned smoothing spline |
|---:|---:|---:|---:|---:|
| 8 | 0.098 | 0.010 | 0.022 | 0.016 |
| 16 | 0.222 | 0.018 | 0.109 | 0.030 |
| 32 | 0.587 | 0.324 | 0.397 | 0.376 |
| 64 | 0.912 | 0.565 | 0.880 | 0.932 |
# PDE benchmark tables

## Overall comparison

| Method | Mean support F1 | Median coefficient error | Mean practical score |
|---|---:|---:|---:|
| Baseline | 0.754 | 0.047 | 0.661 |
| POD-EDMD-RBF | 0.775 | 0.042 | 0.713 |
| Linear interpolation | 0.757 | 0.051 | 0.675 |
| Tuned smoothing spline | 0.777 | 0.045 | 0.726 |

## Median coefficient error by noise level

| Noise | Baseline | POD-EDMD-RBF | Linear interpolation | Tuned smoothing spline |
|---:|---:|---:|---:|---:|
| 0.00 | 0.002 | 0.000 | 0.000 | 0.000 |
| 0.01 | 0.041 | 0.025 | 0.042 | 0.042 |
| 0.03 | 0.168 | 0.046 | 0.118 | 0.047 |
| 0.05 | 0.387 | 0.097 | 0.276 | 0.060 |
| 0.10 | 0.680 | 0.310 | 0.574 | 0.169 |

## Median coefficient error by sparse factor

| Sparse factor | Baseline | POD-EDMD-RBF | Linear interpolation | Tuned smoothing spline |
|---:|---:|---:|---:|---:|
| 4 | 0.051 | 0.029 | 0.072 | 0.041 |
| 8 | 0.046 | 0.036 | 0.049 | 0.043 |
| 16 | 0.048 | 0.042 | 0.047 | 0.044 |
| 32 | 0.046 | 0.102 | 0.049 | 0.047 |
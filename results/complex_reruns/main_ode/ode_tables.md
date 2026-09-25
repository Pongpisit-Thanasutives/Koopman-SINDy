# ODE benchmark tables

## Overall comparison

| Method | Mean support F1 | Median coefficient error | Mean practical score |
|---|---:|---:|---:|
| EDMD-polynomial | 0.868 | 0.030 | 0.763 |
| EDMD-RBF | 0.854 | 0.038 | 0.735 |

## Median coefficient error by noise level

| Noise | EDMD-polynomial | EDMD-RBF |
|---:|---:|---:|
| 0.00 | 0.007 | 0.005 |
| 0.01 | 0.010 | 0.009 |
| 0.03 | 0.020 | 0.023 |
| 0.05 | 0.039 | 0.047 |
| 0.10 | 0.087 | 0.274 |

## Median coefficient error by sparse factor

| Sparse factor | EDMD-polynomial | EDMD-RBF |
|---:|---:|---:|
| 8 | 0.010 | 0.009 |
| 16 | 0.016 | 0.021 |
| 32 | 0.314 | 0.311 |
| 64 | 0.566 | 0.889 |
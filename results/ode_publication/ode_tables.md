# ODE benchmark tables

## Overall comparison

| Method | Mean support F1 | Median coefficient error | Mean practical score |
|---|---:|---:|---:|
| Baseline | 0.844 | 0.308 | 0.659 |
| EDMD-polynomial | 0.868 | 0.032 | 0.769 |
| EDMD-RBF | 0.844 | 0.059 | 0.727 |

## Median coefficient error by noise level

| Noise | Baseline | EDMD-polynomial | EDMD-RBF |
|---:|---:|---:|---:|
| 0.00 | 0.234 | 0.009 | 0.007 |
| 0.01 | 0.232 | 0.011 | 0.012 |
| 0.03 | 0.205 | 0.021 | 0.028 |
| 0.05 | 0.286 | 0.039 | 0.063 |
| 0.10 | 0.385 | 0.087 | 0.232 |

## Median coefficient error by sparse factor

| Sparse factor | Baseline | EDMD-polynomial | EDMD-RBF |
|---:|---:|---:|---:|
| 8 | 0.098 | 0.010 | 0.009 |
| 16 | 0.222 | 0.018 | 0.021 |
| 32 | 0.587 | 0.324 | 0.403 |
| 64 | 0.912 | 0.565 | 0.926 |
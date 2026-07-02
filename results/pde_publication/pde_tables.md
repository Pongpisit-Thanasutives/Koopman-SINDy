# PDE benchmark tables

## Overall comparison

| Method | Mean support F1 | Median coefficient error | Mean practical score |
|---|---:|---:|---:|
| Baseline | 0.784 | 0.044 | 0.695 |
| PyDMD optimized DMD | 0.816 | 0.042 | 0.755 |
| POD-EDMD-RBF | 0.801 | 0.044 | 0.721 |

## Median coefficient error by noise level

| Noise | Baseline | PyDMD optimized DMD | POD-EDMD-RBF |
|---:|---:|---:|---:|
| 0.00 | 0.001 | 0.002 | 0.002 |
| 0.01 | 0.105 | 0.038 | 0.040 |
| 0.03 | 0.162 | 0.164 | 0.111 |
| 0.05 | 0.240 | 0.119 | 0.215 |
| 0.10 | 0.673 | 0.076 | 0.573 |

## Median coefficient error by sparse factor

| Sparse factor | Baseline | PyDMD optimized DMD | POD-EDMD-RBF |
|---:|---:|---:|---:|
| 4 | 0.044 | 0.042 | 0.043 |
| 8 | 0.043 | 0.041 | 0.043 |
| 16 | 0.044 | 0.041 | 0.044 |
| 32 | 0.041 | 0.045 | 0.052 |
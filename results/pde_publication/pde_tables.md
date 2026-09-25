# PDE benchmark tables

## Overall comparison

| Method | Mean support F1 | Median coefficient error | Mean practical score |
|---|---:|---:|---:|
| Baseline | 0.784 | 0.044 | 0.695 |
| optDMD | 0.816 | 0.042 | 0.755 |
| POD-EDMD-RBF | 0.803 | 0.043 | 0.724 |

## Median coefficient error by noise level

| Noise | Baseline | optDMD | POD-EDMD-RBF |
|---:|---:|---:|---:|
| 0.00 | 0.001 | 0.002 | 0.001 |
| 0.01 | 0.105 | 0.038 | 0.040 |
| 0.03 | 0.162 | 0.164 | 0.133 |
| 0.05 | 0.240 | 0.119 | 0.153 |
| 0.10 | 0.673 | 0.076 | 0.564 |

## Median coefficient error by sparse factor

| Sparse factor | Baseline | optDMD | POD-EDMD-RBF |
|---:|---:|---:|---:|
| 4 | 0.044 | 0.042 | 0.042 |
| 8 | 0.043 | 0.041 | 0.042 |
| 16 | 0.044 | 0.041 | 0.043 |
| 32 | 0.041 | 0.045 | 0.069 |
# Appendix C: classical non-dynamical interpolation techniques

## ODE classical interpolation comparison

| System | Method | q | Mean F1 | Median coeff. err. | Mean score | n |
|---|---|---:|---:|---:|---:|---:|
| Lorenz--63 | Baseline | -- | 0.725 | 0.911 | 0.477 | 200 |
| Lorenz--63 | Linear interpolation | 5 | 0.727 | 0.911 | 0.493 | 200 |
| Lorenz--63 | Tuned smoothing spline | 5 | 0.730 | 0.936 | 0.530 | 200 |
| Lorenz--63 | EDMD-polynomial | 5 | 0.737 | 0.526 | 0.561 | 200 |
| Van der Pol | Baseline | -- | 0.963 | 0.107 | 0.840 | 200 |
| Van der Pol | Linear interpolation | 5 | 0.998 | 0.049 | 0.937 | 200 |
| Van der Pol | Tuned smoothing spline | 5 | 0.995 | 0.027 | 0.957 | 200 |
| Van der Pol | EDMD-polynomial | 5 | 0.999 | 0.015 | 0.977 | 200 |

## PDE classical interpolation comparison

| System | Method | q | r | Mean F1 | Median coeff. err. | Mean score | n |
|---|---|---:|---:|---:|---:|---:|---:|
| Burgers | Baseline | -- | -- | 0.730 | 0.298 | 0.580 | 160 |
| Burgers | Linear interpolation | 5 | -- | 0.732 | 0.252 | 0.598 | 160 |
| Burgers | Tuned smoothing spline | 5 | -- | 0.752 | 0.165 | 0.661 | 160 |
| Burgers | POD-EDMD-RBF | 5 | 4 | 0.788 | 0.192 | 0.692 | 160 |
| Fisher--KPP | Baseline | -- | -- | 0.837 | 0.039 | 0.810 | 160 |
| Fisher--KPP | Linear interpolation | 5 | -- | 0.839 | 0.039 | 0.812 | 160 |
| Fisher--KPP | Tuned smoothing spline | 5 | -- | 0.857 | 0.042 | 0.831 | 160 |
| Fisher--KPP | POD-EDMD-RBF | 5 | 2 | 0.851 | 0.036 | 0.825 | 160 |
| Advection--diffusion | Baseline | -- | -- | 0.695 | 0.168 | 0.593 | 160 |
| Advection--diffusion | Linear interpolation | 5 | -- | 0.702 | 0.118 | 0.615 | 160 |
| Advection--diffusion | Tuned smoothing spline | 5 | -- | 0.722 | 0.024 | 0.687 | 160 |
| Advection--diffusion | POD-EDMD-RBF | 5 | 4 | 0.734 | 0.027 | 0.695 | 160 |

## Win rate versus no-upsampling baseline

| setting   | method              | method_label           |   paired_cases |   support_f1_win_rate |   coef_error_win_rate |   practical_score_win_rate |
|:----------|:--------------------|:-----------------------|---------------:|----------------------:|----------------------:|---------------------------:|
| ODE       | edmd_poly3          | EDMD-polynomial        |            400 |                 0.295 |                 0.807 |                      0.802 |
| ODE       | linear_interp       | Linear interpolation   |            400 |                 0.312 |                 0.752 |                      0.752 |
| ODE       | smoothing_spline_cv | Tuned smoothing spline |            400 |                 0.312 |                 0.752 |                      0.693 |
| PDE       | linear_interp       | Linear interpolation   |            480 |                 0.044 |                 0.827 |                      0.804 |
| PDE       | pod_edmd_rbf        | POD-EDMD-RBF           |            480 |                 0.173 |                 0.835 |                      0.823 |
| PDE       | smoothing_spline_cv | Tuned smoothing spline |            480 |                 0.138 |                 0.802 |                      0.790 |

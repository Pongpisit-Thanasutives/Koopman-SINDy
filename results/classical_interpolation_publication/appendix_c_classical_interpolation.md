# Appendix C: classical non-dynamical interpolation techniques

## ODE classical interpolation comparison

| System | Method | q | Mean F1 | Median coeff. err. | Mean score | n |
|---|---|---:|---:|---:|---:|---:|
| Lorenz--63 | Raw FD | -- | 0.725 | 0.911 | 0.477 | 200 |
| Lorenz--63 | Linear interpolation | 5 | 0.727 | 0.911 | 0.493 | 200 |
| Lorenz--63 | Tuned smoothing spline | 5 | 0.730 | 0.936 | 0.530 | 200 |
| Lorenz--63 | EDMD-polynomial | 5 | 0.738 | 0.546 | 0.549 | 200 |
| Van der Pol | Raw FD | -- | 0.963 | 0.107 | 0.840 | 200 |
| Van der Pol | Linear interpolation | 5 | 0.998 | 0.049 | 0.937 | 200 |
| Van der Pol | Tuned smoothing spline | 5 | 0.995 | 0.027 | 0.957 | 200 |
| Van der Pol | EDMD-polynomial | 5 | 0.999 | 0.016 | 0.977 | 200 |

## PDE classical interpolation comparison

| System | Method | q | r | Mean F1 | Median coeff. err. | Mean score | n |
|---|---|---:|---:|---:|---:|---:|---:|
| Burgers | Raw FD | -- | -- | 0.730 | 0.298 | 0.580 | 160 |
| Burgers | Linear interpolation | 5 | -- | 0.732 | 0.252 | 0.598 | 160 |
| Burgers | Tuned smoothing spline | 5 | -- | 0.752 | 0.165 | 0.661 | 160 |
| Burgers | POD-EDMD-RBF | 5 | 4 | 0.771 | 0.197 | 0.672 | 160 |
| Fisher--KPP | Raw FD | -- | -- | 0.837 | 0.039 | 0.810 | 160 |
| Fisher--KPP | Linear interpolation | 5 | -- | 0.839 | 0.039 | 0.812 | 160 |
| Fisher--KPP | Tuned smoothing spline | 5 | -- | 0.857 | 0.042 | 0.831 | 160 |
| Fisher--KPP | POD-EDMD-RBF | 5 | 2 | 0.835 | 0.037 | 0.789 | 160 |
| Advection--diffusion | Raw FD | -- | -- | 0.695 | 0.168 | 0.593 | 160 |
| Advection--diffusion | Linear interpolation | 5 | -- | 0.702 | 0.118 | 0.615 | 160 |
| Advection--diffusion | Tuned smoothing spline | 5 | -- | 0.722 | 0.024 | 0.687 | 160 |
| Advection--diffusion | POD-EDMD-RBF | 5 | 4 | 0.718 | 0.027 | 0.678 | 160 |

## Win rate versus no-upsampling baseline

| Setting | Method | Paired cases | F1 win rate | Coefficient-error win rate | Score win rate |
|---|---|---:|---:|---:|---:|
| ODE | EDMD-polynomial | 400 | 0.280 | 0.777 | 0.765 |
| ODE | Linear interpolation | 400 | 0.312 | 0.752 | 0.752 |
| ODE | Tuned smoothing spline | 400 | 0.312 | 0.752 | 0.693 |
| PDE | Linear interpolation | 480 | 0.044 | 0.827 | 0.804 |
| PDE | POD-EDMD-RBF | 480 | 0.175 | 0.779 | 0.767 |
| PDE | Tuned smoothing spline | 480 | 0.138 | 0.802 | 0.790 |

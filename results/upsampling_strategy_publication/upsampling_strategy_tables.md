# DMD/EDMD upsampling strategy ablation

Stress setting: noise=0.1, ODE sparse factor=64, PDE sparse factor=32, seeds=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19.

## Overall by strategy

| setting   | method       | strategy           |   mean_support_f1 |   median_coef_error |   mean_practical_score |   n_ok |
|:----------|:-------------|:-------------------|------------------:|--------------------:|-----------------------:|-------:|
| ODE       | edmd_poly3   | baseline           |             0.805 |               0.703 |                  0.522 |     40 |
| ODE       | edmd_poly3   | local_reset        |             0.760 |               0.681 |                  0.569 |     40 |
| ODE       | edmd_poly3   | global_rollout     |             0.592 |               1.111 |                  0.258 |     40 |
| ODE       | edmd_poly3   | residual_corrected |             0.767 |               0.575 |                  0.566 |     40 |
| ODE       | edmd_poly3   | two_sided          |             0.719 |               0.799 |                  0.491 |     40 |
| ODE       | edmd_rbf     | baseline           |             0.805 |               0.703 |                  0.522 |     40 |
| ODE       | edmd_rbf     | local_reset        |             0.715 |               0.849 |                  0.509 |     40 |
| ODE       | edmd_rbf     | global_rollout     |             0.580 |               1.040 |                  0.274 |     40 |
| ODE       | edmd_rbf     | residual_corrected |             0.749 |               0.571 |                  0.555 |     40 |
| ODE       | edmd_rbf     | two_sided          |             0.691 |               0.983 |                  0.447 |     40 |
| PDE       | pod_edmd_rbf | baseline           |             0.733 |               0.409 |                  0.571 |     40 |
| PDE       | pod_edmd_rbf | local_reset        |             0.707 |               0.457 |                  0.555 |     40 |
| PDE       | pod_edmd_rbf | global_rollout     |             0.707 |               0.454 |                  0.556 |     40 |
| PDE       | pod_edmd_rbf | residual_corrected |             0.680 |               0.613 |                  0.522 |     40 |
| PDE       | pod_edmd_rbf | two_sided          |             0.711 |               0.450 |                  0.558 |     40 |

## By system

| setting   | system        | method       | strategy           |   mean_support_f1 |   median_coef_error |   mean_practical_score |
|:----------|:--------------|:-------------|:-------------------|------------------:|--------------------:|-----------------------:|
| ODE       | lorenz63      | edmd_poly3   | baseline           |             0.615 |               1.029 |                  0.303 |
| ODE       | lorenz63      | edmd_poly3   | global_rollout     |             0.439 |               1.820 |                  0.135 |
| ODE       | lorenz63      | edmd_poly3   | local_reset        |             0.547 |               1.131 |                  0.244 |
| ODE       | lorenz63      | edmd_poly3   | residual_corrected |             0.545 |               1.215 |                  0.241 |
| ODE       | lorenz63      | edmd_poly3   | two_sided          |             0.510 |               1.563 |                  0.193 |
| ODE       | lorenz63      | edmd_rbf     | baseline           |             0.615 |               1.029 |                  0.303 |
| ODE       | lorenz63      | edmd_rbf     | global_rollout     |             0.462 |               1.353 |                  0.191 |
| ODE       | lorenz63      | edmd_rbf     | local_reset        |             0.481 |               1.360 |                  0.205 |
| ODE       | lorenz63      | edmd_rbf     | residual_corrected |             0.509 |               1.317 |                  0.219 |
| ODE       | lorenz63      | edmd_rbf     | two_sided          |             0.495 |               1.151 |                  0.210 |
| ODE       | vanderpol_mu2 | edmd_poly3   | baseline           |             0.994 |               0.346 |                  0.742 |
| ODE       | vanderpol_mu2 | edmd_poly3   | global_rollout     |             0.745 |               0.972 |                  0.381 |
| ODE       | vanderpol_mu2 | edmd_poly3   | local_reset        |             0.972 |               0.080 |                  0.895 |
| ODE       | vanderpol_mu2 | edmd_poly3   | residual_corrected |             0.989 |               0.113 |                  0.891 |
| ODE       | vanderpol_mu2 | edmd_poly3   | two_sided          |             0.928 |               0.150 |                  0.789 |
| ODE       | vanderpol_mu2 | edmd_rbf     | baseline           |             0.994 |               0.346 |                  0.742 |
| ODE       | vanderpol_mu2 | edmd_rbf     | global_rollout     |             0.699 |               0.938 |                  0.357 |
| ODE       | vanderpol_mu2 | edmd_rbf     | local_reset        |             0.949 |               0.148 |                  0.812 |
| ODE       | vanderpol_mu2 | edmd_rbf     | residual_corrected |             0.989 |               0.113 |                  0.891 |
| ODE       | vanderpol_mu2 | edmd_rbf     | two_sided          |             0.887 |               0.364 |                  0.683 |
| PDE       | burgers       | pod_edmd_rbf | baseline           |             0.667 |               0.746 |                  0.381 |
| PDE       | burgers       | pod_edmd_rbf | global_rollout     |             0.614 |               0.712 |                  0.357 |
| PDE       | burgers       | pod_edmd_rbf | local_reset        |             0.614 |               0.716 |                  0.356 |
| PDE       | burgers       | pod_edmd_rbf | residual_corrected |             0.595 |               0.745 |                  0.340 |
| PDE       | burgers       | pod_edmd_rbf | two_sided          |             0.622 |               0.703 |                  0.361 |
| PDE       | fisher_kpp    | pod_edmd_rbf | baseline           |             0.800 |               0.053 |                  0.762 |
| PDE       | fisher_kpp    | pod_edmd_rbf | global_rollout     |             0.800 |               0.043 |                  0.755 |
| PDE       | fisher_kpp    | pod_edmd_rbf | local_reset        |             0.800 |               0.045 |                  0.755 |
| PDE       | fisher_kpp    | pod_edmd_rbf | residual_corrected |             0.766 |               0.064 |                  0.704 |
| PDE       | fisher_kpp    | pod_edmd_rbf | two_sided          |             0.800 |               0.049 |                  0.755 |

# Complex-propagation strategy ablation

Archived unaffected baselines are retained; paired assisted strategies use identical noisy observations and RBF centers.

| setting   | method       | strategy           |   mean_support_f1 |   median_coef_error |   mean_practical_score |   n_ok |
|:----------|:-------------|:-------------------|------------------:|--------------------:|-----------------------:|-------:|
| ODE       | edmd_poly3   | baseline           |             0.805 |               0.703 |                  0.522 |     40 |
| ODE       | edmd_poly3   | local_reset        |             0.749 |               0.682 |                  0.565 |     40 |
| ODE       | edmd_poly3   | global_rollout     |             0.596 |               1.092 |                  0.262 |     40 |
| ODE       | edmd_poly3   | residual_corrected |             0.768 |               0.576 |                  0.567 |     40 |
| ODE       | edmd_poly3   | two_sided          |             0.719 |               0.799 |                  0.491 |     40 |
| ODE       | edmd_rbf     | baseline           |             0.805 |               0.703 |                  0.522 |     40 |
| ODE       | edmd_rbf     | local_reset        |             0.745 |               0.794 |                  0.526 |     40 |
| ODE       | edmd_rbf     | global_rollout     |             0.568 |               1.032 |                  0.273 |     40 |
| ODE       | edmd_rbf     | residual_corrected |             0.737 |               0.571 |                  0.547 |     40 |
| ODE       | edmd_rbf     | two_sided          |             0.657 |               0.956 |                  0.416 |     40 |
| PDE       | pod_edmd_rbf | baseline           |             0.733 |               0.409 |                  0.571 |     40 |
| PDE       | pod_edmd_rbf | local_reset        |             0.711 |               0.456 |                  0.558 |     40 |
| PDE       | pod_edmd_rbf | global_rollout     |             0.711 |               0.456 |                  0.558 |     40 |
| PDE       | pod_edmd_rbf | residual_corrected |             0.682 |               0.613 |                  0.525 |     40 |
| PDE       | pod_edmd_rbf | two_sided          |             0.711 |               0.456 |                  0.558 |     40 |

| setting   | system        | method       | strategy           |   mean_support_f1 |   median_coef_error |   mean_practical_score |
|:----------|:--------------|:-------------|:-------------------|------------------:|--------------------:|-----------------------:|
| ODE       | lorenz63      | edmd_poly3   | baseline           |             0.615 |               1.029 |                  0.303 |
| ODE       | lorenz63      | edmd_poly3   | global_rollout     |             0.448 |               1.904 |                  0.143 |
| ODE       | lorenz63      | edmd_poly3   | local_reset        |             0.527 |               1.107 |                  0.235 |
| ODE       | lorenz63      | edmd_poly3   | residual_corrected |             0.547 |               1.087 |                  0.243 |
| ODE       | lorenz63      | edmd_poly3   | two_sided          |             0.510 |               1.563 |                  0.193 |
| ODE       | lorenz63      | edmd_rbf     | baseline           |             0.615 |               1.029 |                  0.303 |
| ODE       | lorenz63      | edmd_rbf     | global_rollout     |             0.449 |               1.251 |                  0.193 |
| ODE       | lorenz63      | edmd_rbf     | local_reset        |             0.541 |               1.332 |                  0.232 |
| ODE       | lorenz63      | edmd_rbf     | residual_corrected |             0.484 |               1.336 |                  0.204 |
| ODE       | lorenz63      | edmd_rbf     | two_sided          |             0.495 |               1.151 |                  0.210 |
| ODE       | vanderpol_mu2 | edmd_poly3   | baseline           |             0.994 |               0.346 |                  0.742 |
| ODE       | vanderpol_mu2 | edmd_poly3   | global_rollout     |             0.745 |               0.972 |                  0.381 |
| ODE       | vanderpol_mu2 | edmd_poly3   | local_reset        |             0.972 |               0.080 |                  0.895 |
| ODE       | vanderpol_mu2 | edmd_poly3   | residual_corrected |             0.989 |               0.113 |                  0.891 |
| ODE       | vanderpol_mu2 | edmd_poly3   | two_sided          |             0.928 |               0.150 |                  0.789 |
| ODE       | vanderpol_mu2 | edmd_rbf     | baseline           |             0.994 |               0.346 |                  0.742 |
| ODE       | vanderpol_mu2 | edmd_rbf     | global_rollout     |             0.688 |               0.919 |                  0.352 |
| ODE       | vanderpol_mu2 | edmd_rbf     | local_reset        |             0.950 |               0.121 |                  0.821 |
| ODE       | vanderpol_mu2 | edmd_rbf     | residual_corrected |             0.989 |               0.113 |                  0.891 |
| ODE       | vanderpol_mu2 | edmd_rbf     | two_sided          |             0.819 |               0.367 |                  0.623 |
| PDE       | burgers       | pod_edmd_rbf | baseline           |             0.667 |               0.746 |                  0.381 |
| PDE       | burgers       | pod_edmd_rbf | global_rollout     |             0.622 |               0.703 |                  0.361 |
| PDE       | burgers       | pod_edmd_rbf | local_reset        |             0.622 |               0.703 |                  0.361 |
| PDE       | burgers       | pod_edmd_rbf | residual_corrected |             0.587 |               0.750 |                  0.334 |
| PDE       | burgers       | pod_edmd_rbf | two_sided          |             0.622 |               0.703 |                  0.361 |
| PDE       | fisher_kpp    | pod_edmd_rbf | baseline           |             0.800 |               0.053 |                  0.762 |
| PDE       | fisher_kpp    | pod_edmd_rbf | global_rollout     |             0.800 |               0.048 |                  0.755 |
| PDE       | fisher_kpp    | pod_edmd_rbf | local_reset        |             0.800 |               0.048 |                  0.755 |
| PDE       | fisher_kpp    | pod_edmd_rbf | residual_corrected |             0.777 |               0.059 |                  0.716 |
| PDE       | fisher_kpp    | pod_edmd_rbf | two_sided          |             0.800 |               0.048 |                  0.755 |

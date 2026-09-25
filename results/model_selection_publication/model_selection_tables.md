# Equation-wise EBIC/elbow model selection

| Setting | System | Equation | Method | True k | Selected k (median) | Selected k (mode) | Exact-k rate | Mean F1 | Median coefficient error | n |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ODE | vanderpol_mu2 | dx/dt | Baseline | 1 | 1.0 | 1 | 100.0% | 1.000 | 0.005 | 5 |
| ODE | vanderpol_mu2 | dx/dt | EDMD-polynomial | 1 | 1.0 | 1 | 100.0% | 1.000 | 0.001 | 5 |
| ODE | vanderpol_mu2 | dy/dt | Baseline | 3 | 3.0 | 3 | 100.0% | 1.000 | 0.031 | 5 |
| ODE | vanderpol_mu2 | dy/dt | EDMD-polynomial | 3 | 3.0 | 3 | 100.0% | 1.000 | 0.005 | 5 |
| PDE | burgers | u_t | Baseline | 2 | 2.0 | 2 | 100.0% | 1.000 | 0.046 | 5 |
| PDE | burgers | u_t | POD-EDMD-RBF | 2 | 2.0 | 2 | 100.0% | 1.000 | 0.019 | 5 |
| PDE | fisher_kpp | u_t | Baseline | 3 | 3.0 | 3 | 60.0% | 0.660 | 0.059 | 5 |
| PDE | fisher_kpp | u_t | POD-EDMD-RBF | 3 | 3.0 | 3 | 100.0% | 1.000 | 0.011 | 5 |

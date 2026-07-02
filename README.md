# Koopman/DMD-assisted upsampling for sparse equation discovery

Code, figures, and benchmark outputs for the paper on Koopman/DMD-assisted
upsampling. The preprocessing step reconstructs a denser trajectory from sparse,
noisy samples before SINDy/PDE-FIND estimates derivatives and fits sparse
regression models. The manuscript source and compiled PDF are included at the
package root.

## What's here

- `*.py` — benchmark and figure scripts (see [Scripts](#scripts)).
- `results/` — generated CSV outputs, one subdirectory per experiment.
- `figures/` — generated PNG figures used in the manuscript.
- `requirements.txt` — pinned, import-tested dependencies.
- `LICENSE` — MIT.

The bundled `results/` and `figures/` were produced by the commands below and are
tracked so the paper's numbers can be inspected without rerunning anything.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`SciencePlots` is included for figure styling; the figure scripts fall back to a
plain matplotlib serif style if it is missing.

## Reproduce

Every command writes into `results/` and/or `figures/`. The shipped outputs were
generated with exactly these commands, so a fresh run overwrites them in place.

**Full benchmark suite** (ODE, PDE, advection–diffusion, non-oracle model
selection, and the interpolation-strategy ablation):

```bash
bash run_publication_benchmarks.sh            # defaults: preset=publication, outroot=results
```

To regenerate everything in the paper, also run the Appendix A study and the
figures (next two sections). Or run any single piece on its own:

**Advection–diffusion benchmark + Figure 2:**

```bash
python koopman_sindy_advection_diffusion_benchmark.py --outdir results/advection_diffusion_benchmark
```

**Non-oracle EBIC/Pareto model selection + Figure 3:**

```bash
python koopman_sindy_model_selection_experiment.py \
  --preset quick --seeds 0,1,2,3,4 --noise 0.01 \
  --ode-sparse-factor 8 --pde-sparse-factor 4 \
  --ode-systems vanderpol_mu2 --pde-systems burgers,fisher_kpp \
  --fisher-ic front --fisher-rank 2 --max-rows 5000 \
  --outdir results/model_selection_publication
```

**Appendix A — upsampling-factor `q` / POD-rank `r` sensitivity** (includes the
`q=1` POD-only denoising control):

```bash
python koopman_sindy_qr_sensitivity.py --preset appendix                               # full run
python koopman_sindy_qr_sensitivity.py --preset smoke --outdir results/qr_sensitivity_smoke  # quick check
```

This also writes the Appendix A LaTeX fragment to
`results/qr_sensitivity/appendix_a_qr_sensitivity.tex`.


**Appendix C — classical non-dynamical interpolation techniques:**

```bash
python classical_interpolation_baselines.py --preset publication \
  --outdir results/classical_interpolation_publication

# Quick check:
python classical_interpolation_baselines.py --preset quick \
  --outdir results/classical_interpolation_smoke
```

This writes CSV summaries and the LaTeX fragment
`results/classical_interpolation_publication/appendix_c_classical_interpolation.tex`.
The comparison uses four methods per setting:

- ODE: Baseline, Linear interpolation, Tuned smoothing spline, EDMD-polynomial.
- PDE: Baseline, Linear interpolation, Tuned smoothing spline, POD-EDMD-RBF.

The interpolation factor `q` is fixed for all non-baseline methods. The
smoothing-spline parameter and the POD rank `r` for POD-EDMD-RBF are selected by
deterministic interior holdout validation on sparse noisy observations only. The
raw CSVs record the selected values and validation errors.

**Manuscript figures only:**

```bash
python make_manuscript_figures.py
```

## Scripts

- `koopman_sindy_ode_benchmark.py` — ODE benchmark (Lorenz–63, Van der Pol).
- `koopman_sindy_pde_benchmark.py` — PDE benchmark infrastructure (Burgers, Fisher–KPP, advection–diffusion).
- `koopman_sindy_advection_diffusion_benchmark.py` — advection–diffusion benchmark and visualisation.
- `koopman_sindy_model_selection_experiment.py` — equation-wise EBIC/Pareto (non-oracle) model selection.
- `dmd_upsampling_strategy_ablation.py` — interpolation-strategy ablation (Appendix B).
- `koopman_sindy_qr_sensitivity.py` — Appendix A `q`/`r` sensitivity study.
- `classical_interpolation_baselines.py` — Appendix C classical interpolation/smoothing techniques.
- `make_manuscript_figures.py` — regenerates the manuscript figures.
- `run_publication_benchmarks.sh` — one-command script for the main publication outputs.

## Key result files

- `results/ode_publication/ode_by_system.csv`
- `results/pde_publication/pde_by_system.csv`
- `results/advection_diffusion_benchmark/advection_diffusion_summary.csv`
- `results/fisher_kpp_front_sensitivity/fisher_kpp_front_summary.csv`
- `results/model_selection_publication/model_selection_selected_by_equation.csv`
- `results/upsampling_strategy_publication/upsampling_strategy_summary.csv`
- `results/classical_interpolation_publication/classical_interpolation_by_system.csv`

Accuracy is reported per system, not as aggregate ODE/PDE means; each table
caption states the systems and settings used to produce it.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

If you use this code, please cite the accompanying paper (Thanasutives &
Kawahara). Update this section with the final reference once available.

#!/usr/bin/env bash
set -euo pipefail

PRESET="${1:-publication}"
OUTROOT="${2:-results}"
mkdir -p "$OUTROOT"

echo "Running ODE benchmark ($PRESET)..."
python koopman_sindy_ode_benchmark.py \
  --preset "$PRESET" \
  --outdir "$OUTROOT/ode_publication" \
  --resume

echo "Running PDE benchmark ($PRESET) with system-specific POD ranks..."
python koopman_sindy_pde_benchmark.py \
  --preset "$PRESET" \
  --outdir "$OUTROOT/pde_publication" \
  --rank-mode system \
  --burgers-rank 8 \
  --fisher-rank 5 \
  --systems burgers,fisher_kpp \
  --resume

echo "Running advection-diffusion PDE benchmark..."
python koopman_sindy_advection_diffusion_benchmark.py --outdir "$OUTROOT/advection_diffusion_benchmark"

echo "Running equation-wise EBIC/Pareto model selection (moderate non-oracle setting)..."
python koopman_sindy_model_selection_experiment.py \
  --preset quick \
  --seeds 0,1,2,3,4 \
  --noise 0.01 \
  --ode-sparse-factor 8 \
  --pde-sparse-factor 4 \
  --ode-systems vanderpol_mu2 \
  --pde-systems burgers,fisher_kpp \
  --fisher-ic front \
  --fisher-rank 2 \
  --max-rows 5000 \
  --outdir "$OUTROOT/model_selection_publication"

echo "Running interpolation-strategy ablation..."
if [ "$PRESET" = "quick" ]; then
  python dmd_upsampling_strategy_ablation.py \
    --outdir "$OUTROOT/upsampling_strategy_publication" \
    --seeds 0,1 \
    --noise 0.10 \
    --ode-sparse-factor 64 \
    --pde-sparse-factor 32 \
    --nx 48 \
    --rank 6 \
    --rbf-centers 20
else
  python dmd_upsampling_strategy_ablation.py \
    --outdir "$OUTROOT/upsampling_strategy_publication" \
    --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --noise 0.10 \
    --ode-sparse-factor 64 \
    --pde-sparse-factor 32 \
    --nx 64 \
    --rank 8 \
    --rbf-centers 40
fi

echo "All publication runs complete. Outputs are in: $OUTROOT"

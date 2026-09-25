#!/usr/bin/env bash
set -euo pipefail

PRESET="${1:-publication}"
OUTDIR="${2:-results/classical_interpolation_${PRESET}}"

python classical_interpolation_baselines.py \
  --preset "${PRESET}" \
  --outdir "${OUTDIR}" \
  --resume

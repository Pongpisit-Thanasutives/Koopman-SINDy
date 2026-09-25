#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Recompute affected EDMD evidence; preserve the supplied unaffected controls.
python validate_complex_propagation.py
python rerun_complex_revision.py
python revision_tv_comparison.py --jobs "${1:-4}" --outdir results/revision_tv_high_noise
python revision_nonoracle_tv.py
python validate_revision_weak.py
python revision_weak_comparison.py --jobs "${1:-4}"
python make_revision_weak_table.py
# Rebuild all manuscript assets from the completed records.
python regenerate_presentation_assets.py

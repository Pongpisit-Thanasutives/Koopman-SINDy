#!/usr/bin/env python3
"""Rebuild numerical tables/figures from saved results, without experiments.

Portable: python /path/to/code_repository/regenerate_presentation_assets.py
Assets are written to generated_assets/ unless --latex-dir selects a destination.
The optional weak-form experiment has a separate reporting workflow.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import pandas as pd

ROOT=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latex-dir',type=Path,default=ROOT/'generated_assets',
                        help='Asset destination; relative paths use the invoking working directory.')
    args=parser.parse_args()
    # Resolve caller-relative paths before legacy plotters switch to the repository.
    latex_dir=args.latex_dir.expanduser().resolve()
    saved_interpolation=ROOT/'figures/revision_interpolated_data.pdf'
    if not saved_interpolation.is_file():
        parser.error(f'Regeneration requires the saved illustration: {saved_interpolation}')
    os.chdir(ROOT)  # Legacy appendix plotters use repository-relative destinations.
    protected=[p for p in (ROOT/'results').rglob('*') if p.is_file()
               and 'revision_weak' not in str(p)
               and (p.name in ['run_manifest.json','complex_revision_manifest.json','protocol_frozen.json',
                               'coefficients.csv','tv_cv_diagnostics.csv','selected_by_equation.csv']
                    or p.name.endswith('raw_results.csv') or p.name.endswith('_raw.csv'))]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    subprocess.run([sys.executable,str(ROOT/'revision_tv_reporting.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'classical_interpolation_baselines.py'),'--skip-run'],check=True)
    import koopman_sindy_qr_sensitivity as qr
    folder=ROOT/'results/qr_sensitivity'
    raw=pd.read_csv(folder/'qr_sensitivity_raw.csv')
    summary=pd.read_csv(folder/'qr_sensitivity_summary.csv')
    # Archived Appendix preset; no integration, noise generation or model fitting.
    args=SimpleNamespace(noise=.03,ode_sparse_factor=16,pde_sparse_factor=8,
        seeds=sorted(raw.seed.unique()),ode_dt=.01,pde_dt=.01,pde_n=64,rbf_centers=12)
    qr.plot_ode_q(summary,str(folder));qr.plot_pde_qr(summary,str(folder))
    qr.write_appendix_fragment(summary,str(folder),args)
    subprocess.run([sys.executable,str(ROOT/'make_revision1_figures_tables.py'),
                    '--skip-interpolation','--latex-dir',str(latex_dir)],check=True)
    subprocess.run([sys.executable,str(ROOT/'sync_revision_outputs.py'),
                    '--latex-dir',str(latex_dir)],check=True)
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected},'Execution artifacts changed.'
    # Refresh main reporting provenance after sync has produced its derived summaries.
    import make_revision1_figures_tables as mainreport
    mainreport.reporting_manifest(mainreport.read_raw(),folder/'qr_sensitivity_raw.csv',
        ROOT/'results/revision1_reporting',False)
    sources=['regenerate_presentation_assets.py','revision_tv_reporting.py','revision_tv_comparison.py',
        'make_revision1_figures_tables.py','sync_revision_outputs.py','koopman_sindy_qr_sensitivity.py',
        'classical_interpolation_baselines.py']
    output=ROOT/'results/revision1_reporting/presentation_generation_manifest.json'
    manifest=dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        purpose='Presentation-only regeneration from saved numerical results; execution manifests preserved',
        portable_command='python /path/to/code_repository/regenerate_presentation_assets.py',
        protected_execution_sha256=before,reporting_source_sha256={n:sha(ROOT/n) for n in sources},
        manuscript_asset_path_base='repository parent; ../ components permit external destinations',
        manuscript_asset_sha256={os.path.relpath(p,ROOT.parent):sha(p) for foldername in ['tables','figures','appendices']
            for p in sorted((latex_dir/foldername).glob('*')) if p.is_file()})
    output.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Presentation artifacts regenerated; {len(before)} execution files unchanged.')

if __name__=='__main__':
    main()

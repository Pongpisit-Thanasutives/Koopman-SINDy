#!/usr/bin/env python3
"""Check the completed TV experiment's sampling, convergence and provenance."""
import ast
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import revision_tv_comparison as study
from revision_tv_solver import smooth_tv

ROOT=Path(__file__).resolve().parent
out=ROOT/'results/revision_tv_high_noise'
raw=pd.read_csv(out/'raw_results.csv')
cv=pd.read_csv(out/'tv_cv_diagnostics.csv')
tv=raw[raw.method.isin(['tv_diff','pod_tv_diff'])]
assert len(raw)==720 and raw.status.eq('ok').all()
assert len(cv)==8640 and len(tv)==320
assert cv.solver_converged.all() and tv.tv_final_converged.all()
assert cv.solver_max_relative_gap.max()<=1e-5
assert tv.tv_final_max_relative_gap.max()<=1e-5
assert pd.read_csv(out/'failed_records.csv').empty
checks=[]
for name,cfg in study.CONFIG.items():
    _,times,y,_=study.dataset(name)
    for factor in cfg['factors']:
        ind=np.arange(0,len(times),factor);t=times[ind];values=y[ind]
        dense=np.linspace(t[0],t[-1],5*(len(t)-1)+1)
        assert np.allclose(np.diff(t),np.diff(t)[0],rtol=1e-11,atol=1e-12)
        assert dense[0]==t[0] and dense[-1]==t[-1]
        assert np.allclose(dense[::5],t,rtol=0,atol=1e-14)
        _,dz,_=smooth_tv(values,t,0.)
        fd=(study.ode.central_difference if cfg['kind']=='ode' else study.pde.temporal_derivative)(values,t)[1]
        error=float(np.max(np.abs(dz[1:-1]-fd)))
        assert error<1e-10
        checks.append(dict(system=name,sparse_factor=factor,observed_endpoint=float(t[-1]),
                           uniform_observed_grid=True,exact_dense_endpoints=True,dense_anchors_present=True,
                           max_zero_tv_vs_central_derivative_difference=error))
manifest=json.loads((out/'run_manifest.json').read_text())
final_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in manifest['source_sha256']}
recorded=manifest.get('final_package_source_sha256',manifest['source_sha256'])
reporting=json.loads((out/'artifact_generation_manifest.json').read_text())
# Execution provenance is immutable. The v2 revision changes only the saved-
# data reporting delegation and default output directory in the runner.
# Verify that separately; do not overwrite the historical execution hashes.
for name,expected in reporting['preserved_execution_sha256'].items():
    assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected, f'Changed execution artifact: {name}'
for name,expected in reporting['reporting_source_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, f'Changed reporting source: {name}'
changed=[name for name in final_hashes if final_hashes[name]!=recorded[name]]
assert set(changed)<= {'revision_tv_comparison.py'}, f'Unaccounted numerical source changes: {changed}'
if changed:
    audit=json.loads((ROOT/'results/final_validation/tv_reporting_change_audit.json').read_text())
    assert audit['executed_source_sha256']==manifest['source_sha256']['revision_tv_comparison.py']
    assert audit['current_source_sha256']==final_hashes['revision_tv_comparison.py']
    tree=ast.parse((ROOT/'revision_tv_comparison.py').read_text())
    functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    for name,expected in audit['unchanged_numeric_function_ast_sha256'].items():
        actual=hashlib.sha256(ast.dump(functions[name],include_attributes=False).encode()).hexdigest()
        assert actual==expected, f'Changed numerical routine: {name}'
    tree.body=[node for node in tree.body if not (isinstance(node,ast.FunctionDef) and node.name in ('main','summarize'))]
    actual=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    assert actual==audit['unchanged_module_excluding_main_and_summarize_ast_sha256'], 'Changed numerical configuration'
report=dict(checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),sampling_checks=checks,
            n_method_records=len(raw),n_cv_solves=len(cv),n_final_tv_fits=len(tv),
            max_cv_relative_gap=float(cv.solver_max_relative_gap.max()),
            max_final_relative_gap=float(tv.tv_final_max_relative_gap.max()),
            max_masked_stationarity=float(cv.solver_masked_dual_stationarity_max.max()),
            max_dual_box_violation=float(cv.solver_dual_box_violation.max()),
            zero_regularizer_fits=int(tv.tv_grid_lower.sum()),upper_grid_fits=int(tv.tv_grid_upper.sum()),
            numerical_source_hashes_match_execution_or_documented_final_package=True,
            preserved_execution_artifact_hashes_match=True,
            current_reporting_source_hashes_match=True,
            documented_reporting_only_source_changes=changed)
(out/'validation_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='sampling_checks'},indent=2))

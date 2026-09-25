#!/usr/bin/env python3
"""Frozen matched 1%-noise non-oracle TV extension of the existing demonstration.

No truth is passed into model selection. TV regularization is selected only from
noisy observation holdouts; the same original EBIC/Pareto elbow selects every
method's equation. Default run: 15 datasets, 55 method fits, 70 equation selections.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python revision_nonoracle_tv.py
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback
import warnings
import numpy as np
import pandas as pd
import scipy
import kneed  # Required: do not silently replace the archived Kneedle selector.
import koopman_sindy_ode_benchmark as ode
import koopman_sindy_pde_benchmark as pde
import koopman_sindy_model_selection_experiment as selection
from revision_tv_comparison import tv_cv,LAMBDAS

PROTOCOL={
    'noise':.01,'seeds':[0,1,2,3,4],'q':5,'tv_tolerance':1e-5,
    'ode_dt':.01,'ode_sparse_factor':8,'ode_system':'vanderpol_mu2',
    'pde_dt':.01,'pde_sparse_factor':4,'nx':48,'pde_max_rows':5000,
    'burgers_rank':6,'fisher_rank':2,'fisher_ic':'front','rbf_centers':20,
    'ebic_gamma':.5,'knee_sensitivity':1.,'selection_rule':'elbow_ebic',
    'ode_thresholds':selection.ODE_THRESHOLDS.tolist(),
    'pde_thresholds':selection.PDE_THRESHOLDS.tolist(),
    'tv_grid':LAMBDAS.tolist(),
    'sampling':'uniform observed intervals and endpoint-inclusive q5 grids; all factors divide horizons',
    'arms_ode':['baseline','edmd_poly3','tv_diff'],
    'arms_pde':['baseline','pod_edmd_rbf','tv_diff','pod_tv_diff'],
    'row_policy':'Original 5000-row PDE cap and record RNG convention; TV/raw grids fall below cap',
    'purpose':'Focused matched extension of existing non-oracle 1%-noise demonstration; no additional regimes',
}
LABELS={'baseline':'Raw FD','edmd_poly3':'Polynomial EDMD','pod_edmd_rbf':'POD-EDMD-RBF',
        'tv_diff':'TV','pod_tv_diff':'POD+TV'}
SOURCES=['revision_nonoracle_tv.py','revision_tv_comparison.py','revision_tv_solver.py',
         'koopman_sindy_ode_benchmark.py','koopman_sindy_pde_benchmark.py',
         'koopman_sindy_model_selection_experiment.py','koopman_propagation.py']


def source_hashes():
    base=Path(__file__).resolve().parent
    return {n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in SOURCES if (base/n).exists()}


def candidate_path(theta,y,metadata,thresholds,cfg):
    """Fit threshold path and score residuals, without truth or post-selection metrics."""
    rows=[];failures=[];seen={};n,p=theta.shape
    for i,threshold in enumerate(thresholds):
        try:
            xi=ode.stlsq(theta,y[:,None],threshold=float(threshold)).ravel() if metadata['setting']=='ODE' else pde.stlsq(theta,y,threshold=float(threshold))
            active=tuple(np.flatnonzero(np.abs(xi)>1e-10).tolist())
            rss=float(np.sum((theta@xi-y)**2))
            if not np.isfinite(rss) or not np.isfinite(xi).all():raise FloatingPointError('nonfinite sparse-regression output')
            if active in seen and seen[active]<=rss:continue
            seen[active]=rss
            bic,ebic=selection.bic_ebic_from_rss(rss,n,len(active),p,cfg.ebic_gamma)
            rows.append(dict(**metadata,candidate_id=i,threshold=float(threshold),support_size=len(active),
                             rss=rss,bic=bic,ebic=ebic,active_terms=';'.join(map(str,active)),
                             coefficients=json.dumps(xi.tolist()),n_regression_rows=n,n_library_terms=p))
        except Exception as exc:
            failures.append(dict(**metadata,threshold=float(threshold),status=f'fail:{type(exc).__name__}: {exc}'))
    return rows,failures


def build_table(summary,out):
    systems=[('vanderpol_mu2','dx/dt',r'Van der Pol $\dot{x}$'),('vanderpol_mu2','dy/dt',r'Van der Pol $\dot{y}$'),
             ('burgers','u_t','Burgers'),('fisher_kpp','u_t','Fisher--KPP (front)')]
    lines=[r'\begin{table}[!htbp]',r'\centering',
           r'\TBL{\caption{Matched non-oracle comparison at 1\% noise using five seeds per equation and method. Selected $k$ gives the support-size range across seeds. Exact support counts require the correct terms, not only the correct size. F1 is the mean and coefficient error is the median after the same EBIC/Pareto selection. TV strength is selected from noisy observations only. Boldface marks the best displayed value within each equation.\label{tab:nonoracle_tv}\label{tab:model_selection}}}',
           r'{\begin{tabular}{@{}llcccc@{}}\toprule',
           r'\TCH{System/equation} & \TCH{Method} & \TCH{Selected $k$} & \TCH{Exact support} & \TCH{F1} & \TCH{Coeff. err.} \\ \midrule']
    for system,equation,label in systems:
        d=summary[(summary.system==system)&(summary.equation==equation)].set_index('method')
        methods=PROTOCOL['arms_ode'] if system=='vanderpol_mu2' else PROTOCOL['arms_pde']
        for j,method in enumerate(methods):
            if method not in d.index:continue
            r=d.loc[method]
            k=str(int(r.selected_k_min)) if r.selected_k_min==r.selected_k_max else f'{int(r.selected_k_min)}--{int(r.selected_k_max)}'
            exact=f'{int(r.exact_support_count)}/{int(r.n_expected)}'
            if r.exact_support_count==d.exact_support_count.max():
                exact=r'$\mathbf{'+exact+'}$'
            cells=[exact]
            for column,maximize in [('support_f1_mean',True),('coef_error_median',False)]:
                cell=f'{r[column]:.3f}'
                best=d[column].max() if maximize else d[column].min()
                if cell==f'{best:.3f}':cell=r'$\mathbf{'+cell+'}$'
                cells.append(cell)
            lines.append((f'\\multirow{{{len(methods)}}}{{*}}{{{label}}}' if j==0 else '')+
                         f' & {LABELS[method]} & {k} & '+' & '.join(cells)+r' \\')
        if system!='fisher_kpp':lines.append(r'\midrule')
    lines += [r'\botrule\end{tabular}}',r'\end{table}']
    (out/'revision_nonoracle_tv_table.tex').write_text('\n'.join(lines)+'\n')


def summarize(out):
    selected=pd.read_csv(out/'selected_by_equation.csv')
    summary=selected.groupby(['setting','system','equation','method','method_label'],as_index=False).agg(
        n_valid=('seed','size'),true_support_size=('true_support_size','first'),
        exact_support_count=('exact_support','sum'),exact_k_count=('exact_k','sum'),
        selected_k_min=('support_size','min'),selected_k_max=('support_size','max'),selected_k_median=('support_size','median'),
        support_f1_mean=('support_f1','mean'),coef_error_median=('coef_error','median'))
    summary['n_expected']=len(PROTOCOL['seeds']);summary.to_csv(out/'summary.csv',index=False)
    build_table(summary,out)
    return summary


def run(out):
    before=source_hashes();started=time.perf_counter()
    manifest=dict(protocol=PROTOCOL,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
                  kneed=kneed.__version__,source_sha256=before,
                  selection_inputs='Only candidate support sizes, RSS/BIC/EBIC and data-fitted coefficients; truth constructed after model selection')
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cfg=selection.SelectionConfig(.5,1.,'elbow_ebic')
    datasets=[('ODE',ode.vanderpol_system(),8,0),('PDE',pde.burgers_system(48),4,6),
              ('PDE',pde.fisher_kpp_system(48,ic_variant='front'),4,2)]
    candidate_rows=[];selected_rows=[];front_rows=[];cv_rows=[];metadata_rows=[];failure_rows=[];coefficient_rows=[]
    selection_checks=[];timing_checks=[]
    for setting,sysobj,factor,rank in datasets:
        module=ode if setting=='ODE' else pde
        times,clean=(ode.integrate_system if setting=='ODE' else pde.integrate_pde)(sysobj,.01)
        inds=np.arange(0,len(times),factor);t=times[inds];observed_clean=clean[inds]
        dense=np.linspace(t[0],t[-1],5*(len(t)-1)+1)
        assert t[-1]==times[-1] and dense[0]==t[0] and dense[-1]==t[-1]
        assert np.allclose(np.diff(t),np.diff(t)[0],rtol=1e-11,atol=1e-12)
        assert np.allclose(dense[::5],t,rtol=0,atol=1e-14)
        timing_checks.append(dict(system=sysobj.name,n_observed=len(t),n_dense=len(dense),observed_endpoint=float(t[-1]),
                                  uniform_observed=True,dense_anchors_present=True,exact_grid_endpoints=True))
        methods=PROTOCOL['arms_ode'] if setting=='ODE' else PROTOCOL['arms_pde']
        for seed in PROTOCOL['seeds']:
            seedkey=seed+1000*factor+100000*int(PROTOCOL['noise']*1000)
            reference_obs=None
            for method in methods:
                metadata=dict(setting=setting,system=sysobj.name,method=method,method_label=LABELS[method],seed=seed,noise=.01,sparse_factor=factor)
                tick=time.perf_counter();info={}
                try:
                    rng=np.random.default_rng(seedkey)
                    observations=module.add_noise(observed_clean,.01,rng)
                    if reference_obs is None:reference_obs=observations.copy()
                    assert np.array_equal(observations,reference_obs)
                    if method=='baseline':
                        state,derivative=(ode.central_difference if setting=='ODE' else pde.temporal_derivative)(observations,t)
                    elif method=='edmd_poly3':
                        reconstruction=ode.edmd_reconstruct(observations,t,dense,kind='poly',degree=3,var_names=sysobj.var_names,rng=rng)
                        state,derivative=ode.central_difference(reconstruction,dense)
                    elif method=='pod_edmd_rbf':
                        reconstruction=pde.pod_edmd_reconstruct(observations,t,dense,rank,'rbf',rng,20)
                        state,derivative=pde.temporal_derivative(reconstruction,dense)
                    else:
                        reconstruction,dy,info,cv=tv_cv(observations,t,setting=='ODE',rank if method=='pod_tv_diff' else 0,1e-5)
                        state,derivative=reconstruction[1:-1],dy[1:-1]
                        cv_rows.extend([{**metadata,**v} for v in cv])
                    if setting=='ODE':
                        theta,names=ode.polynomial_library(state,sysobj.var_names,degree=3)
                        targets=[(f'd{var}/dt',derivative[:,j],j) for j,var in enumerate(sysobj.var_names)]
                    else:
                        kvec=pde.periodic_wavenumbers(48,sysobj.params['L'])
                        theta,names=pde.pde_library(state,kvec);y=derivative.ravel()
                        if len(y)>5000:
                            keep=rng.choice(len(y),5000,replace=False);theta,y=theta[keep],y[keep]
                        targets=[('u_t',y,0)]
                    for equation,y,j in targets:
                        mm={**metadata,'equation':equation}
                        candidates,failures=candidate_path(theta,y,mm,selection.ODE_THRESHOLDS if setting=='ODE' else selection.PDE_THRESHOLDS,cfg)
                        failure_rows.extend(failures)
                        if not candidates:raise RuntimeError('No finite regression candidates')
                        c=pd.DataFrame(candidates)
                        assert not any(n in c for n in ['true_support_size','support_f1','coef_error','exact_support','true_coefficients'])
                        chosen,front=selection.select_models(c,cfg)
                        if len(chosen)!=1:raise RuntimeError(f'Expected one selected equation, obtained{len(chosen)}')
                        # Metamorphic safeguard: adding nonsensical truth labels cannot alter chosen model.
                        poison=c.assign(true_support_size=-999,support_f1=np.nan,coef_error=-1e200,exact_support=-1)
                        again,_=selection.select_models(poison,cfg)
                        assert chosen.candidate_id.iloc[0]==again.candidate_id.iloc[0]
                        selection_checks.append(dict(**mm,truth_absent_from_selection=True,poisoned_diagnostics_invariant=True))
                        # Access ground truth only AFTER selection is complete.
                        truth=sysobj.true_coefficients(names)
                        truth=truth[:,j] if setting=='ODE' else truth
                        def evaluate(row):
                            r=dict(row);xi=np.asarray(json.loads(r['coefficients']))
                            r.update(true_support_size=int(np.count_nonzero(np.abs(truth)>1e-10)),
                                     true_active_terms=';'.join(map(str,np.flatnonzero(np.abs(truth)>1e-10))),
                                     support_f1=selection.support_f1_vec(xi,truth),coef_error=selection.coeferr_vec(xi,truth),
                                     exact_support=int(np.array_equal(np.abs(xi)>1e-10,np.abs(truth)>1e-10)),
                                     exact_k=int(r['support_size']==np.count_nonzero(np.abs(truth)>1e-10)))
                            return r
                        candidate_rows.extend([evaluate(r) for r in c.to_dict('records')])
                        front_rows.extend([evaluate(r) for r in front.to_dict('records')])
                        selected=evaluate(chosen.iloc[0].to_dict());selected_rows.append(selected)
                        selected_coef=np.asarray(json.loads(selected['coefficients']))
                        for name,val,true in zip(names,selected_coef,truth):
                            coefficient_rows.append(dict(**mm,feature=name,coefficient=float(val),true_coefficient=float(true)))
                    metadata_rows.append(dict(**metadata,status='ok',n_observed=len(t),pod_rank=rank if method in ['pod_edmd_rbf','pod_tv_diff'] else 0,
                                              n_regression_rows=len(theta),upsample=5 if method in ['edmd_poly3','pod_edmd_rbf'] else 1,
                                              wall_seconds=time.perf_counter()-tick,**info))
                except Exception as exc:
                    failure_rows.append(dict(**metadata,status=f'fail:{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
                    metadata_rows.append(dict(**metadata,status=f'fail:{type(exc).__name__}: {exc}',wall_seconds=time.perf_counter()-tick))
            print(f'{sysobj.name} seed{seed} complete',flush=True)
    frames={'candidates.csv':candidate_rows,'selected_by_equation.csv':selected_rows,'pareto_front.csv':front_rows,
            'tv_cv_diagnostics.csv':cv_rows,'method_records.csv':metadata_rows,'selected_coefficients.csv':coefficient_rows}
    for name,rows in frames.items():pd.DataFrame(rows).to_csv(out/name,index=False)
    pd.DataFrame(failure_rows,columns=None if failure_rows else ['setting','system','method','seed','status']).to_csv(out/'failures.csv',index=False)
    after=source_hashes();assert before==after,'Core source changed during execution; rerun with a frozen core.'
    validation=dict(timing_checks=timing_checks,selection_checks=selection_checks,
                    n_expected_methods=55,n_valid_methods=sum(r['status']=='ok' for r in metadata_rows),
                    n_expected_selected_equations=70,n_selected_equations=len(selected_rows),
                    n_candidate_failures=len(failure_rows),source_unchanged_during_run=True)
    if cv_rows:validation['max_tv_cv_gap']=float(max(r['solver_max_relative_gap'] for r in cv_rows))
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    manifest.update(elapsed_seconds=time.perf_counter()-started,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary=summarize(out);print(summary.to_string(index=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir',default='results/revision_nonoracle_tv')
    ap.add_argument('--freeze-only',action='store_true')
    ap.add_argument('--summarize-only',action='store_true')
    args=ap.parse_args();out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:summarize(out);return
    frozen=out/'protocol_frozen.json'
    if frozen.exists():
        assert json.loads(frozen.read_text())['protocol']==PROTOCOL,'Frozen protocol differs; choose a new output directory.'
    else:
        frozen.write_text(json.dumps(dict(protocol=PROTOCOL,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2)+'\n')
    if args.freeze_only:
        print('Protocol frozen; no experiments executed.');return
    run(out)

if __name__=='__main__':main()

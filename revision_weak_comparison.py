#!/usr/bin/env python3
"""Small frozen weak-SINDy diagnostic; consult revision_weak_protocol.json.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python revision_weak_comparison.py --jobs 4
Reports: python revision_weak_comparison.py --summarize-only
No PySINDy/PyWSINDy dependency: this is an independently implemented fixed-test
diagnostic, not a replication of either complete package or its model selector.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
import pandas as pd
import scipy
import koopman_sindy_ode_benchmark as ode
import koopman_sindy_pde_benchmark as pde
from revision_tv_comparison import dataset, tv_cv, ODE_THRESHOLDS, PDE_THRESHOLDS
from weak_integral_library import ode_weak, pde_weak

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'revision_weak_protocol.json'
CONFIG=json.loads(PROTOCOL.read_text())
KEYS=['system','sparse_factor','noise','seed']
LABELS={'raw_weak':'Raw + weak','linear_weak':'Linear q5 + weak','edmd_weak':'EDMD + weak',
        'tv_weak':'TV states + weak','pod_weak':'POD only + weak','pod_tv_weak':'POD + TV states + weak'}
SOURCES=['revision_weak_comparison.py','weak_integral_library.py','revision_weak_protocol.json',
         'revision_tv_comparison.py','revision_tv_solver.py','koopman_sindy_ode_benchmark.py',
         'koopman_sindy_pde_benchmark.py','koopman_propagation.py']


def hashes():
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCES}


def weak_library(state,t,sysobj,cfg):
    centers=cfg['time_halfwidth']+cfg['time_center_stride']*np.arange(cfg['n_time_centers'])
    if sysobj.name=='vanderpol_mu2':
        return ode_weak(state,t,sysobj.var_names,centers,cfg['time_halfwidth'])
    length=sysobj.params['L']; xc=np.arange(cfg['n_space_centers'])*length/cfg['n_space_centers']
    return pde_weak(state,t,sysobj.x,length,centers,cfg['time_halfwidth'],xc,length/4)


def oracle(theta,target,sysobj):
    is_ode=sysobj.name=='vanderpol_mu2'; module=ode if is_ode else pde
    names=(ode.polynomial_library(np.zeros((2,2)),sysobj.var_names,degree=3)[1] if is_ode
           else ['1','u','u^2','u_x','u*u_x','u^2*u_x','u_xx','(u^2)_xx'])
    truth=sysobj.true_coefficients(names);best=None
    for threshold in ODE_THRESHOLDS if is_ode else PDE_THRESHOLDS:
        coef=module.stlsq(theta,target,threshold=float(threshold))
        f1=module.support_f1(coef,truth);error=module.coefficient_error(coef,truth)
        key=(f1/(1+error),f1,-error)
        if best is None or key>best[0]:best=(key,coef,float(threshold))
    key,coef,threshold=best
    return dict(practical_score=key[0],support_f1=key[1],coef_error=-key[2],threshold=threshold,
                exact_support=int(np.array_equal(abs(coef)>1e-10,abs(truth)>1e-10)),
                weak_truth_residual=float(np.linalg.norm(theta@truth-target)/max(np.linalg.norm(target),1e-15)),
                weak_fit_residual=float(np.linalg.norm(theta@coef-target)/max(np.linalg.norm(target),1e-15)),
                weak_rows=theta.shape[0],library_columns=theta.shape[1]),coef,truth,names


def run_record(task):
    name,factor,noise,seed=task;cfg=CONFIG['systems'][name];is_ode=name=='vanderpol_mu2'
    sysobj,t,clean,truth_interp=dataset(name)
    idx=np.arange(0,len(t),factor);idx=idx[t[idx]<=cfg['horizon']+1e-10]
    tobs=t[idx];uclean=clean[idx]
    assert np.isclose(tobs[-1],cfg['horizon'])
    rng=np.random.default_rng(seed+1000*factor+100000*int(noise*1000))
    noisy=(ode.add_noise if is_ode else pde.add_noise)(uclean,noise,rng)
    # EDMD receives the state of the same RNG after the noise draw, exactly as
    # the original benchmark. Other methods never consume this generator.
    data_hash=hashlib.sha256(noisy.tobytes()).hexdigest()
    tnew=np.linspace(tobs[0],tobs[-1],5*(len(tobs)-1)+1)
    rows=[];coefrows=[];cvrows=[]
    for method in cfg['methods']:
        started=time.perf_counter();details={}
        row=dict(system=name,method=method,method_label=LABELS[method],sparse_factor=factor,
                 noise=noise,seed=seed,observation_sha256=data_hash,n_observed=len(tobs),
                 obs_dt=float(tobs[1]-tobs[0]),observed_t_end=float(tobs[-1]),
                 pod_rank=cfg.get('pod_rank',0) if method in ('edmd_weak','pod_weak','pod_tv_weak') else 0,
                 upsample=5 if method in ('edmd_weak','linear_weak') else 1)
        try:
            teval=tobs
            if method=='raw_weak': state=noisy
            elif method=='linear_weak':
                teval=tnew;state=np.column_stack([np.interp(tnew,tobs,col) for col in noisy.T])
            elif method=='edmd_weak':
                teval=tnew
                if is_ode:
                    state=ode.edmd_reconstruct(noisy,tobs,tnew,kind='poly',degree=3,var_names=sysobj.var_names,rng=rng)
                else:
                    state=pde.pod_edmd_reconstruct(noisy,tobs,tnew,cfg['pod_rank'],'rbf',rng,cfg['rbf_centers'])
            elif method=='pod_weak':
                mu,modes,z=pde.pod_fit(noisy,cfg['pod_rank']);state=pde.pod_reconstruct(mu,modes,z)
            else:
                rank=cfg['pod_rank'] if method=='pod_tv_weak' else 0
                state,_unused_derivative,details,cv=tv_cv(noisy,tobs,is_ode,rank,1e-5)
                for c in cv:cvrows.append({**{k:row[k] for k in KEYS},'method':method,**c})
            if not np.isfinite(state).all():raise FloatingPointError('nonfinite reconstructed states')
            theta,target,names=weak_library(state,teval,sysobj,cfg)
            result,coef,truth,names_fit=oracle(theta,target,sysobj)
            assert names==names_fit
            result['state_error']=float(np.linalg.norm(state-truth_interp(teval))/np.linalg.norm(truth_interp(teval)))
            result['n_state_samples']=len(teval)
            row.update(result);row.update(details);row['status']='ok'
            for i,feature in enumerate(names):
                for eq in range(coef.shape[1] if is_ode else 1):
                    coefrows.append({**{k:row[k] for k in KEYS},'method':method,'feature':feature,'equation':eq,
                                     'coef':float(coef[i,eq] if is_ode else coef[i]),
                                     'true_coef':float(truth[i,eq] if is_ode else truth[i])})
        except Exception as exc:
            row['status']=f'fail: {type(exc).__name__}: {exc}';row['traceback']=traceback.format_exc()
        row['wall_seconds']=time.perf_counter()-started;rows.append(row)
    return rows,coefrows,cvrows


def summarize(out):
    raw=pd.read_csv(out/'raw_results.csv');ok=raw[raw.status=='ok']
    keys=['system','method','method_label','sparse_factor','obs_dt','noise']
    cond=ok.groupby(keys,as_index=False).agg(
        support_f1_mean=('support_f1','mean'),support_f1_se=('support_f1','sem'),
        coef_error_median=('coef_error','median'),coef_error_mean=('coef_error','mean'),coef_error_se=('coef_error','sem'),
        practical_score_mean=('practical_score','mean'),practical_score_se=('practical_score','sem'),
        exact_support_rate=('exact_support','mean'),state_error_median=('state_error','median'),
        weak_truth_residual_median=('weak_truth_residual','median'),n_ok=('status','size'))
    counts=raw.groupby(keys,as_index=False).agg(n_planned=('status','size'))
    counts.merge(cond,how='left',on=keys).to_csv(out/'summary_by_condition.csv',index=False)
    aggregate=ok.groupby(['system','method','method_label'],as_index=False).agg(
        support_f1_mean=('support_f1','mean'),coef_error_median=('coef_error','median'),
        practical_score_mean=('practical_score','mean'),exact_support_rate=('exact_support','mean'),
        state_error_median=('state_error','median'),n_ok=('status','size'))
    aggregate.to_csv(out/'summary_all_conditions.csv',index=False)
    paired=[]
    for name,group in ok.groupby('system'):
        for comparator in ['raw_weak','linear_weak','tv_weak','pod_weak','pod_tv_weak']:
            left=group[group.method=='edmd_weak'];right=group[group.method==comparator]
            merge=left.merge(right,on=KEYS,suffixes=('_edmd','_control'))
            for _,r in merge.iterrows():
                paired.append({**{k:r[k] for k in KEYS},'control':comparator,
                               'delta_f1':r.support_f1_edmd-r.support_f1_control,
                               'delta_error':r.coef_error_edmd-r.coef_error_control,
                               'delta_score':r.practical_score_edmd-r.practical_score_control,
                               'edmd_score_win':int(r.practical_score_edmd>r.practical_score_control+1e-12),
                               'edmd_score_tie':int(abs(r.practical_score_edmd-r.practical_score_control)<=1e-12)})
    pairs=pd.DataFrame(paired);pairs.to_csv(out/'paired_differences.csv',index=False)
    pairs.groupby(['system','control'],as_index=False).agg(
        mean_delta_f1=('delta_f1','mean'),median_delta_error=('delta_error','median'),
        mean_delta_score=('delta_score','mean'),edmd_score_wins=('edmd_score_win','sum'),
        score_ties=('edmd_score_tie','sum'),n_pairs=('seed','size')).to_csv(out/'paired_summary.csv',index=False)
    # A seed is the independent replicate. Pool the six conditions for each
    # seed before forming pooled SEs; do not treat overlapping windows as trials.
    seed=ok.groupby(['system','method','seed'],as_index=False)[['support_f1','practical_score','coef_error']].mean()
    seed.to_csv(out/'pooled_seed_means.csv',index=False)
    seed.groupby(['system','method'],as_index=False).agg(
        support_f1_mean=('support_f1','mean'),support_f1_se=('support_f1','sem'),
        practical_score_mean=('practical_score','mean'),practical_score_se=('practical_score','sem'),
        coef_error_mean=('coef_error','mean'),coef_error_se=('coef_error','sem')).to_csv(out/'pooled_seed_summary.csv',index=False)
    (out/'summary.md').write_text('# Focused weak-SINDy comparison\n\n'
        'Oracle sparsity thresholds; all planned noise/sampling/seed combinations retained. '
        'The fully weak conservative PDE library changes one nuisance term relative to the main strong-form study. '
        'These are within-library preprocessing comparisons, not a complete WSINDy benchmark.\n\n'
        +aggregate.to_markdown(index=False)+'\n\nSee summary_by_condition.csv for all conditions and paired_summary.csv for matched contrasts.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--outdir',type=Path,default=ROOT/'results/revision_weak_comparison')
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--seeds',default='0,1,2,3,4')
    parser.add_argument('--noise',default='.05,.10,.20')
    parser.add_argument('--systems',default='vanderpol_mu2,burgers')
    args=parser.parse_args();out=args.outdir;out.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:summarize(out);return
    tasks=[(name,factor,noise,seed) for name in args.systems.split(',')
           for factor in CONFIG['systems'][name]['factors'] for noise in map(float,args.noise.split(','))
           for seed in map(int,args.seeds.split(','))]
    start=time.perf_counter();source_hashes=hashes()
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
                  source_sha256=source_hashes,protocol=CONFIG,tasks=tasks,n_tasks=len(tasks),
                  n_expected_method_rows=sum(len(CONFIG['systems'][t[0]]['methods']) for t in tasks))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows=[];coefs=[];cv=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(run_record,t):t for t in tasks}
        for done,f in enumerate(as_completed(futures),1):
            r,c,v=f.result();rows.extend(r);coefs.extend(c);cv.extend(v)
            print(f'{done}/{len(tasks)} {futures[f]}: {sum(x["status"]=="ok" for x in r)}/{len(r)} methods',flush=True)
    sort=KEYS+['method']
    pd.DataFrame(rows).sort_values(sort).to_csv(out/'raw_results.csv',index=False)
    pd.DataFrame(coefs).sort_values(sort+['equation','feature']).to_csv(out/'coefficients.csv',index=False)
    pd.DataFrame(cv).sort_values(sort+['fold','lam']).to_csv(out/'tv_cv.csv',index=False)
    if hashes()!=source_hashes:raise RuntimeError('Source changed during experiment; do not report this run')
    manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-start,
                    n_method_rows=len(rows),n_failed=sum(r['status']!='ok' for r in rows),sources_unchanged=True)
    manifest['output_sha256']={f:hashlib.sha256((out/f).read_bytes()).hexdigest()
                               for f in ['raw_results.csv','coefficients.csv','tv_cv.csv']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');summarize(out)


if __name__=='__main__':main()

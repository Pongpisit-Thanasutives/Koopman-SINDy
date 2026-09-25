#!/usr/bin/env python3
"""Targeted reviewer-2 comparison of TV differentiation and EDMD, 5--50% noise.

Every preprocessing hyperparameter is fixed in advance except TV regularization,
which is selected exclusively using held-out noisy observations.  The downstream
STLSQ threshold is oracle selected, exactly as in the original diagnostic study.
This is a finite-difference/TV comparison, not a WSINDy implementation.

Full run (200 noisy records, 720 method records):
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python revision_tv_comparison.py --jobs 4
Smoke: add --seeds 0 --noise 0.10 --systems lorenz63,burgers
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import hashlib
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import CubicSpline
import koopman_sindy_ode_benchmark as ode
import koopman_sindy_pde_benchmark as pde
from revision_tv_solver import smooth_tv

LAMBDAS = np.array([0., 1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1000.])
ODE_THRESHOLDS = np.array([0., 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.])
PDE_THRESHOLDS = np.array([0., 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
CONFIG = {
 'lorenz63': dict(kind='ode', dt=.005, factors=[16,64], rank=0),
 'vanderpol_mu2': dict(kind='ode', dt=.005, factors=[16,64], rank=0),
 'burgers': dict(kind='pde', dt=.005, factors=[8,32], rank=8, nx=64, centers=30),
 'fisher_kpp': dict(kind='pde', dt=.005, factors=[8,32], rank=5, nx=64, centers=30),
 'advection_diffusion': dict(kind='pde', dt=.01, factors=[4,16], rank=4, nx=48, centers=12),
}
CACHE = {}
LABELS={'baseline':'Raw FD', 'edmd_poly3':'Polynomial EDMD', 'pod_edmd_rbf':'POD-EDMD-RBF',
        'tv_diff':'TV differentiation', 'pod_tv_diff':'POD + TV differentiation'}


def dataset(name):
    if name in CACHE:
        return CACHE[name]
    cfg=CONFIG[name]
    if name=='lorenz63': sysobj=ode.lorenz_system()
    elif name=='vanderpol_mu2': sysobj=ode.vanderpol_system()
    elif name=='burgers': sysobj=pde.burgers_system(n=cfg['nx'])
    elif name=='fisher_kpp': sysobj=pde.fisher_kpp_system(n=cfg['nx'])
    elif name=='advection_diffusion': sysobj=pde.advection_diffusion_system(n=cfg['nx'])
    else: raise ValueError(name)
    t,u=(ode.integrate_system if cfg['kind']=='ode' else pde.integrate_pde)(sysobj,cfg['dt'])
    CACHE[name]=(sysobj,t,u,CubicSpline(t,u,axis=0))
    return CACHE[name]


def prep_tv(y,t,mask,is_ode,pod_rank):
    """Fit centering, scaling and (if used) POD from training observations only."""
    train=np.flatnonzero(mask)
    if pod_rank:
        mu,modes,lat=pde.pod_fit(y[train],pod_rank)
        full=np.zeros((len(t),lat.shape[1])); full[train]=lat
        scale=max(float(np.std(lat)),1e-12)
        normalized=full/scale
        return normalized, (mu,modes,scale)
    mu=y[train].mean(axis=0)
    scale=np.maximum(np.std(y[train],axis=0),1e-12) if is_ode else max(float(np.std(y[train])),1e-12)
    normalized=np.zeros_like(y);normalized[train]=(y[train]-mu)/scale
    return normalized,(mu,None,scale)


def unprep(z,dz,params):
    mu,modes,scale=params
    if modes is None: return z*scale+mu,dz*scale
    return (z*scale)@modes.T+mu,(dz*scale)@modes.T


def tv_cv(y,t,is_ode,pod_rank,tol):
    """Three deterministic interior temporal folds; raw PDE CV uses 8 positions.

    POD is refitted on training snapshots in each fold, so the spatial basis
    cannot leak held-out observations into regularizer selection.
    """
    if not is_ode and not pod_rank:
        col=np.unique(np.round(np.linspace(0,y.shape[1]-1,8)).astype(int))
        ycv=y[:,col]
    else: ycv=y
    splits=ode._cv_splits_from_observations(len(t),3)
    scores=np.zeros(len(LAMBDAS)); records=[]
    for fold,(train,val) in enumerate(splits):
        mask=np.zeros(len(t),dtype=bool);mask[train]=True
        yn,params=prep_tv(ycv,t,mask,is_ode,pod_rank)
        norm=np.maximum(np.var(ycv[train],axis=0),1e-12) if is_ode else max(float(np.var(ycv[train])),1e-12)
        for i,lam in enumerate(LAMBDAS):
            z,dz,diag=smooth_tv(yn,t,float(lam),mask=mask,tol=tol,max_iter=20000)
            z,_=unprep(z,dz,params)
            score=float(np.mean((z[val]-ycv[val])**2/norm))
            scores[i]+=score/len(splits)
            records.append(dict(fold=fold,lam=float(lam),cv_mse=score,**{f'solver_{k}':v for k,v in diag.items()}))
    # The first minimizer selects the smaller penalty in exact numerical ties.
    best=min(range(len(LAMBDAS)),key=lambda i:(scores[i],LAMBDAS[i]))
    lam=float(LAMBDAS[best]);mask=np.ones(len(t),dtype=bool)
    yn,params=prep_tv(y,t,mask,is_ode,pod_rank)
    z,dz,diag=smooth_tv(yn,t,lam,tol=tol,max_iter=20000)
    z,dz=unprep(z,dz,params)
    info=dict(tv_lambda=lam,tv_cv_mse=float(scores[best]),tv_grid_edge=int(best in (0,len(LAMBDAS)-1)),
              tv_grid_lower=int(best==0),tv_grid_upper=int(best==len(LAMBDAS)-1),tv_cv_folds=len(splits),
              tv_cv_spatial_points=ycv.shape[1],**{f'tv_final_{k}':v for k,v in diag.items()})
    return z,dz,info,records


def fit_oracle(states,derivative,sysobj,cfg):
    if cfg['kind']=='ode':
        theta,names=ode.polynomial_library(states,sysobj.var_names,degree=3)
        target=derivative;module=ode;thresholds=ODE_THRESHOLDS
    else:
        k=pde.periodic_wavenumbers(states.shape[1],sysobj.params['L'])
        theta,names=pde.pde_library(states,k);target=derivative.ravel()
        module=pde;thresholds=PDE_THRESHOLDS
    truth=sysobj.true_coefficients(names)
    best=None
    for thr in thresholds:
        coef=module.stlsq(theta,target,threshold=float(thr))
        f1=module.support_f1(coef,truth);cerr=module.coefficient_error(coef,truth)
        score=f1/(1+cerr);key=(score,f1,-cerr)
        if best is None or key>best[0]:best=(key,coef,float(thr))
    key,coef,thr=best
    return dict(support_f1=key[1],coef_error=-key[2],practical_score=key[0],threshold=thr,
                exact_support=int(np.array_equal(np.abs(coef)>1e-10,np.abs(truth)>1e-10))),coef,truth,names


def run_record(task):
    name,factor,noise,seed,tol=task
    started=time.perf_counter();cfg=CONFIG[name];is_ode=cfg['kind']=='ode'
    sysobj,t,clean,truth_interp=dataset(name)
    idx=np.arange(0,len(t),factor)
    # All methods receive the same genuinely uniform observed grid.
    # Do not append a shortened terminal interval to a fixed-step EDMD fit.
    tobs=t[idx];uclean=clean[idx]
    seedkey=seed+1000*factor+100000*int(noise*1000)
    rng=np.random.default_rng(seedkey)
    y=(ode.add_noise if is_ode else pde.add_noise)(uclean,noise,rng)
    # All methods see the identical record; EDMD's RNG follows the noise draw.
    methods=['baseline','edmd_poly3' if is_ode else 'pod_edmd_rbf','tv_diff']
    if not is_ode:methods.append('pod_tv_diff')
    rows=[];coefs=[];cvrows=[]
    for method in methods:
        timer=time.perf_counter();details={}
        row=dict(system=name,method=method,method_label=LABELS[method],sparse_factor=factor,
                 obs_dt=float(tobs[1]-tobs[0]),noise=noise,seed=seed,n_observed=len(tobs),observed_t_end=float(tobs[-1]),
                 base_t_end=float(t[-1]),cropped_time=float(t[-1]-tobs[-1]),
                 pod_rank=cfg['rank'] if method in ('pod_edmd_rbf','pod_tv_diff') else 0,
                 upsample=5 if method in ('edmd_poly3','pod_edmd_rbf') else 1)
        try:
            if method=='baseline':
                state,dy=(ode.central_difference if is_ode else pde.temporal_derivative)(y,tobs);teval=tobs[1:-1]
            elif method=='edmd_poly3':
                tnew=np.linspace(tobs[0],tobs[-1],5*(len(tobs)-1)+1)
                rec=ode.edmd_reconstruct(y,tobs,tnew,kind='poly',degree=3,var_names=sysobj.var_names,rng=rng)
                state,dy=ode.central_difference(rec,tnew);teval=tnew[1:-1]
            elif method=='pod_edmd_rbf':
                tnew=np.linspace(tobs[0],tobs[-1],5*(len(tobs)-1)+1)
                rec=pde.pod_edmd_reconstruct(y,tobs,tnew,cfg['rank'],'rbf',rng,cfg['centers'])
                state,dy=pde.temporal_derivative(rec,tnew);teval=tnew[1:-1]
            else:
                rec,drec,details,cv=tv_cv(y,tobs,is_ode,cfg['rank'] if method=='pod_tv_diff' else 0,tol)
                state,dy=rec[1:-1],drec[1:-1];teval=tobs[1:-1]
                for r in cv:cvrows.append({**{k:row[k] for k in ['system','method','sparse_factor','noise','seed']},**r})
            if not (np.isfinite(state).all() and np.isfinite(dy).all()):raise FloatingPointError('nonfinite reconstruction')
            result,coef,truth,names=fit_oracle(state,dy,sysobj,cfg)
            true_states=truth_interp(teval)
            true_deriv=np.asarray([sysobj.rhs(tt,uu) for tt,uu in zip(teval,true_states)])
            result['derivative_error']=float(np.linalg.norm(dy-true_deriv)/np.linalg.norm(true_deriv))
            result['state_error']=float(np.linalg.norm(state-true_states)/np.linalg.norm(true_states))
            row.update(result);row.update(details);row['status']='ok'
            for ii,feature in enumerate(names):
                for jj in range(coef.shape[1] if coef.ndim==2 else 1):
                    coefs.append({**{k:row[k] for k in ['system','method','sparse_factor','noise','seed']},
                                  'feature':feature,'equation':jj,'coef':float(coef[ii,jj] if coef.ndim==2 else coef[ii]),
                                  'true_coef':float(truth[ii,jj] if truth.ndim==2 else truth[ii])})
        except Exception as exc:
            row['status']=f'fail: {type(exc).__name__}: {exc}'
            row['traceback']=traceback.format_exc()
        row['wall_seconds']=time.perf_counter()-timer;rows.append(row)
    return rows,coefs,cvrows,time.perf_counter()-started


def summarize(outdir):
    # Reporting is isolated from numerical fitting and preserves execution provenance.
    from revision_tv_reporting import summarize as report_saved_results
    report_saved_results(outdir)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir',default=str(Path(__file__).resolve().parent/'results/revision_tv_high_noise'))
    ap.add_argument('--jobs',type=int,default=4)
    ap.add_argument('--systems',default=','.join(CONFIG))
    ap.add_argument('--noise',default='0.05,0.10,0.20,0.50')
    ap.add_argument('--seeds',default='0,1,2,3,4')
    ap.add_argument('--tol',type=float,default=1e-5)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--summarize-only',action='store_true')
    args=ap.parse_args();out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:summarize(out);return
    rows=[];coefs=[];cvrows=[];completed=set()
    if args.resume and (out/'raw_results.csv').exists():
        rows=pd.read_csv(out/'raw_results.csv').to_dict('records')
        coefs=pd.read_csv(out/'coefficients.csv').to_dict('records')
        cvrows=pd.read_csv(out/'tv_cv_diagnostics.csv').to_dict('records')
        # Resume only fully successful records, rerun all methods on incomplete records.
        d=pd.DataFrame(rows)
        for key,g in d.groupby(['system','sparse_factor','noise','seed']):
            if (g.status=='ok').all() and len(g)==(3 if CONFIG[key[0]]['kind']=='ode' else 4):completed.add(key)
        rows=[r for r in rows if (r['system'],r['sparse_factor'],r['noise'],r['seed']) in completed]
        coefs=[r for r in coefs if (r['system'],r['sparse_factor'],r['noise'],r['seed']) in completed]
        cvrows=[r for r in cvrows if (r['system'],r['sparse_factor'],r['noise'],r['seed']) in completed]
    tasks=[(name,s,n,k,args.tol) for name in args.systems.split(',') for s in CONFIG[name]['factors']
           for n in map(float,args.noise.split(',')) for k in map(int,args.seeds.split(','))
           if (name,s,n,k) not in completed]
    manifest=dict(command=' '.join(sys.argv),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                  pandas=pd.__version__,config=CONFIG,lambdas=LAMBDAS.tolist(),tv_tolerance=args.tol,
                  seeds=list(map(int,args.seeds.split(','))),noise=list(map(float,args.noise.split(','))),
                  downstream_selection='oracle STLSQ threshold grid; ground truth excluded from all preprocessing',
                  sampling='uniform sparse observations only; omit shortened final interval for every method; dense grid includes exact observed endpoints',
                  tv_cv='3 deterministic interior folds; endpoints in training; raw PDE uses 8 equally spaced positions; POD refit in each fold',
                  started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  source_sha256={name:hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest() for name in ['revision_tv_comparison.py','revision_tv_solver.py','koopman_sindy_ode_benchmark.py','koopman_sindy_pde_benchmark.py','koopman_propagation.py']})
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    begin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(run_record,t):t for t in tasks}
        for i,fut in enumerate(as_completed(futures),1):
            rr,cc,vv,elapsed=fut.result();rows.extend(rr);coefs.extend(cc);cvrows.extend(vv)
            for r in rr:
                if r['status']!='ok':print('FAIL',r['system'],r['sparse_factor'],r['noise'],r['seed'],r['method'],r['status'],flush=True)
            if i%5==0 or i==len(tasks):
                pd.DataFrame(rows).to_csv(out/'raw_results.csv',index=False)
                pd.DataFrame(coefs).to_csv(out/'coefficients.csv',index=False)
                pd.DataFrame(cvrows).to_csv(out/'tv_cv_diagnostics.csv',index=False)
                print(f'{i}/{len(tasks)} records; {len(rows)} method rows; elapsed {time.perf_counter()-begin:.1f}s; latest {futures[fut][:4]} ({elapsed:.1f}s)',flush=True)
    manifest['elapsed_seconds']=time.perf_counter()-begin
    manifest['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2));summarize(out)

if __name__=='__main__':main()

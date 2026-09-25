#!/usr/bin/env python3
"""Focused correctness validation, not an equation-recovery accuracy benchmark."""
from pathlib import Path
import argparse,json,platform
import numpy as np
import pandas as pd
from scipy.linalg import expm
import koopman_sindy_ode_benchmark as o
import koopman_sindy_pde_benchmark as p
from koopman_propagation import FractionalEvolution,nominal_observation_pairs,local_observable_reconstruction


def run(outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);checks=[]
    def check(name,error,tol=1e-9):
        assert np.isfinite(error) and error<tol,(name,error,tol)
        checks.append(dict(test=name,error=float(error),tolerance=tol,passed=True))
    M=np.array([[-1.]])
    B=FractionalEvolution(M).power(.2)
    check('negative-real eigenvalue retains phase',np.linalg.norm(np.linalg.matrix_power(B,5)-M))
    assert abs(B.imag[0,0])>.5
    A=np.array([[0.,-2.],[2.,0.]])
    check('conjugate-pair rotation',np.linalg.norm(FractionalEvolution(expm(.3*A)).power(.2)-expm(.06*A)))
    M=np.array([[1.,1.],[0.,1.]])
    check('defective Jordan block',np.linalg.norm(FractionalEvolution(M).power(.2)-np.array([[1.,.2],[0.,1.]])))
    t=np.array([0.,.5,1.,1.2]);dt,pairs=nominal_observation_pairs(t)
    assert dt==.5 and pairs.tolist()==[True,True,False]
    q=np.array([0.,.1,.5,.8,1.,1.1,1.19,1.2]);v=np.exp(.2*t)[:,None]
    result=local_observable_reconstruction(np.array([[np.exp(.1)]]),v,t,q,[0],dt)
    check('short final interval and preceding anchor',np.max(np.abs(result[:,0]-np.exp(.2*q))))
    E=FractionalEvolution(np.diag([0.,.9]));assert E.regularized_eigenvalues==1
    check('explicit spectral floor',np.linalg.norm(np.linalg.matrix_power(E.power(.2),5)-E.regularized_matrix))
    try:FractionalEvolution(np.array([[np.nan]]))
    except FloatingPointError:check('nonfinite input rejected',0.)
    else:raise AssertionError('nonfinite matrix accepted')
    rows=[]
    for noise in [0.,.03]:
        for typ,sy,rank,sf in [('ODE',o.lorenz_system(),0,32),('ODE',o.vanderpol_system(),0,32),('PDE',p.burgers_system(),8,8),('PDE',p.fisher_kpp_system(),5,8)]:
            mod=o if typ=='ODE' else p
            t,Y=o.integrate_system(sy,.005) if typ=='ODE' else p.integrate_pde(sy,.005)
            ix=np.arange(0,len(t),sf)
            if ix[-1]!=len(t)-1:ix=np.r_[ix,len(t)-1]
            rng=np.random.default_rng(1000*sf+100000*int(noise*1000));X=mod.add_noise(Y[ix],noise,rng)
            if typ=='ODE':
                Phi,n=o.polynomial_library(X,sy.var_names,3);state=[n.index(v) for v in sy.var_names]
            else:
                _,_,Z=p.pod_fit(X,rank);Phi=p.rbf_features(Z,p.rbf_fit(Z,30,rng));state=list(range(1,rank+1))
            dt,pairs=nominal_observation_pairs(t[ix]);left,right=Phi[:-1][pairs],Phi[1:][pairs]
            K=np.linalg.solve(left.T@left+1e-8*np.eye(Phi.shape[1]),left.T@right)
            E=FractionalEvolution(K);B=E.power(.2);Kr=E.regularized_matrix
            R=np.linalg.matrix_power(B.real,5);C=np.linalg.matrix_power(B,5)
            ns=max(np.linalg.norm(right[:,state]),1e-12);nk=max(np.linalg.norm(Kr),1e-12)
            rows.append(dict(system=sy.name,noise=noise,seed=0,sparse_factor=sf,pod_rank=rank,q=5,
                fitted_pairs=int(pairs.sum()),excluded_irregular_pairs=int((~pairs).sum()),
                imaginary_substep_relative_norm=float(np.linalg.norm(B.imag)/np.linalg.norm(B)),
                real_projected_roundtrip_defect=float(np.linalg.norm(R-Kr)/nk),complex_roundtrip_defect=float(np.linalg.norm(C-Kr)/nk),
                real_projected_state_roundtrip_defect=float(np.linalg.norm((left@(R-Kr))[:,state])/ns),
                complex_state_roundtrip_defect=float(np.linalg.norm((left@(C-Kr))[:,state])/ns),
                regularized_eigenvalues=E.regularized_eigenvalues,spectral_regularization_relative_change=E.regularization_relative_change))
    df=pd.DataFrame(rows);df.to_csv(out/'representative_propagation_diagnostics.csv',index=False)
    (out/'correctness_checks.json').write_text(json.dumps(dict(python=platform.python_version(),checks=checks),indent=2)+'\n')
    lines=['# Complex propagation validation','',f'All {len(checks)} correctness checks passed.','',
      'The checks cover a negative-real eigenvalue, a real rotation with conjugate eigenpairs, a defective Jordan block, an irregular final observation interval, spectral regularization, and non-finite input rejection.','',
      'Complex Schur evaluation avoids an eigenvector inverse. Schur diagonal values whose imaginary part is at most 100 times machine epsilon times max(1,abs(lambda)) are made exactly real; eigenvalues of magnitude below 1e-12 are replaced by positive 1e-12. Fractional powers use the principal complex branch. The reported round trips are relative to the explicitly regularized matrix.','',
      'The eight diagnostic fits use seed0, base dt0.005, q5, ODE sparse factor32 and PDE factor8, Burgers/Fisher ranks8/5, and clean or3% noisy observations. They establish consistency with the fitted map, not better equation recovery.','',
      '| System | Noise | Real-step round trip | Complex round trip | Real-step state defect | Complex state defect |','|---|---:|---:|---:|---:|---:|']
    for r in df.itertuples():lines.append(f'| {r.system} | {r.noise:.2f} | {r.real_projected_roundtrip_defect:.3g} | {r.complex_roundtrip_defect:.3g} | {r.real_projected_state_roundtrip_defect:.3g} | {r.complex_state_roundtrip_defect:.3g} |')
    lines+=['','Ordinary conjugate eigenpairs often already yield real fractional powers; projection is harmless there. Negative-real eigenvalues lie on the branch cut. The branch convention does not uniquely identify a continuous-time generator. Local-reset jumps, finite-dictionary bias, aliasing and noise remain.','',
      'Observation arrays and nominal dense output grids are unchanged. Map fitting excludes unequal-lag final transitions. Queries use actual elapsed time from their preceding anchor.']
    (out/'PROPAGATION_VALIDATION.md').write_text('\n'.join(lines)+'\n')
    print(f'{len(checks)} checks passed; {len(df)} representative diagnostics written to {out}')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--outdir',default='results/propagation_validation');run(ap.parse_args().outdir)

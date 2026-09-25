"""Deterministic analytical, leakage, independent-QP, and runtime validation."""
from pathlib import Path
import json
import sys
import time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from revision_tv_solver import smooth_tv
rng=np.random.default_rng(20260924)
report={'seed':20260924,'checks':[]}

def record(name, **kw):
    report['checks'].append(dict(name=name,**kw))
    print(name,kw,flush=True)

# Nullspace exactness includes physical derivative units and irregular times.
t=np.r_[0,np.cumsum(rng.uniform(.02,.06,50))]
y=np.column_stack([np.full(t.size,3.5),1.4-2.3*t])
mask=np.arange(t.size)%3!=1
for lam in [1e-4,1,100]:
    z,dz,diag=smooth_tv(y,t,lam,mask=mask)
    assert np.max(abs(z-y))<1e-10
    assert np.max(abs(dz-np.array([0,-2.3])))<1e-9
    record('constant_linear',lam=lam,max_state_error=float(np.max(abs(z-y))),max_derivative_error=float(np.max(abs(dz-np.array([0,-2.3])))))

# Exact zero-penalty data fit; masked response changes must have NO influence.
t=np.linspace(0,2*np.pi,101)
y=np.sin(t)
z,dz,diag=smooth_tv(y,t,0)
assert np.max(abs(z-y))<1e-12
assert np.max(abs(dz-np.cos(t)))<7e-4
record('sine_zero_penalty',max_derivative_error=float(np.max(abs(dz-np.cos(t)))))
mask=np.arange(t.size)%3!=1
y1=y+.04*rng.standard_normal(len(y)); y2=y1.copy(); y2[~mask]=np.nan
z1,d1,a=smooth_tv(y1,t,.1,mask=mask,tol=1e-6)
z2,d2,b=smooth_tv(y2,t,.1,mask=mask,tol=1e-6)
assert np.array_equal(z1,z2) and np.array_equal(d1,d2)
record('heldout_response_exclusion',bitwise_identical=True,relative_gap=a['max_relative_gap'])

# Independent epigraph QP solved by SLSQP, full and masked irregular grids.
for masked in [False,True]:
    n=17
    t=np.r_[0,np.cumsum(rng.uniform(.7,1.3,n-1))]
    y=np.sin(t/3)+.1*rng.standard_normal(n)
    mask=np.arange(n)%3!=1 if masked else np.ones(n,dtype=bool)
    lam=.37; tau=(t-t[0])/np.median(np.diff(t)); h=np.diff(tau)
    B=np.zeros((n-2,n))
    for k in range(n-2): B[k,k:k+3]=[1/h[k],-1/h[k]-1/h[k+1],1/h[k+1]]
    z,dz,diag=smooth_tv(y,t,lam,mask=mask,tol=1e-8,max_iter=20000)
    m=n-2
    C=np.vstack([np.c_[-B,np.eye(m)],np.c_[B,np.eye(m)]])
    def obj(v): return .5*np.sum((v[:n][mask]-y[mask])**2)+lam*np.sum(v[n:])
    def jac(v): return np.r_[np.where(mask,v[:n]-y,0),np.full(m,lam)]
    result=minimize(obj,np.r_[z,abs(B@z)],jac=jac,method='SLSQP',constraints=[{'type':'ineq','fun':lambda v:C@v,'jac':lambda v:C}],options={'ftol':1e-12,'maxiter':2000})
    assert result.success, result.message
    objgap=obj(np.r_[z,abs(B@z)])-result.fun
    assert objgap<2e-7
    assert diag['dual_box_violation']==0
    assert diag['masked_dual_stationarity_max']<1e-10
    record('independent_quadratic_program',masked=masked,objective_difference=float(objgap),max_state_difference=float(np.max(abs(z-result.x[:n]))),diagnostics=diag)

# Main experiment dimensions, all penalties, three masks, simultaneous channels.
for n,d in [(51,64),(251,3)]:
    t=np.r_[0,np.cumsum(rng.uniform(.8,1.2,n-1))]
    t=t/t[-1]*6
    y=np.sin(t[:,None]+np.arange(d)[None,:]/3)+.05*rng.standard_normal((n,d))
    for fold in range(3):
        mask=np.arange(n)%3!=fold; mask[[0,-1]]=True
        for lam in [0,1e-4,1e-3,1e-2,.1,1,10,100,1000]:
            begin=time.perf_counter()
            z,dz,diag=smooth_tv(y,t,lam,mask=mask,tol=1e-5,max_iter=20000)
            assert diag['max_relative_gap']<=1e-5
            assert diag['dual_box_violation']==0
            assert diag['masked_dual_stationarity_max']<1e-8
            record('representative_runtime',n=n,d=d,fold=fold,lam=lam,seconds=time.perf_counter()-begin,iterations=diag['iterations'],relative_gap=diag['max_relative_gap'],masked_stationarity=diag['masked_dual_stationarity_max'])

# Explicit failure handling: impossible iteration budget may not silently pass.
try:
    smooth_tv(y,t,1,mask=mask,tol=1e-12,max_iter=1)
except RuntimeError:
    record('nonconvergence_raises',passed=True)
else:
    raise AssertionError('Expected explicit nonconvergence error')
report['all_passed']=True
Path(__file__).with_name('validation_report_publication.json').write_text(json.dumps(report,indent=2)+'\n')

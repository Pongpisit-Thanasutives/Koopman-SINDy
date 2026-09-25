"""Fixed compact-test integral libraries; no differentiation of observed data.

The quadrature is exact for the piecewise-linear interpolant of each *nodal
feature*, separately, against polynomial test derivatives. This is not exact
integration of a nonlinear function of a piecewise-linear state. In space-time
the interpolant is a tensor product (bilinear). This follows the quadrature
principle documented by PySINDy's WeakPDELibrary, with deterministic fixed
physical supports instead of randomly drawn, grid-shrunk supports.
"""
from functools import lru_cache
import numpy as np
from numpy.polynomial import Polynomial
import koopman_sindy_ode_benchmark as ode

PDE_NAMES = ['1', 'u', 'u^2', 'u_x', 'u*u_x', 'u^2*u_x', 'u_xx', '(u^2)_xx']


def integral_weights(grid, center, halfwidth, derivative=0, p=4):
    """Weights for integral f(s) d^derivative[(1-((s-c)/H)^2)^p]/ds^d.

    Test function is zero outside [c-H,c+H]. Grid cells that intersect the
    support are clipped analytically, so no moving of test supports is needed.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError('grid must be a strictly increasing vector')
    if halfwidth <= 0 or derivative < 0 or p <= derivative:
        raise ValueError('require halfwidth>0 and p>derivative>=0')
    tol = 1e-10 * max(1., abs(grid[-1]), halfwidth)
    if center-halfwidth < grid[0]-tol or center+halfwidth > grid[-1]+tol:
        raise ValueError('test support must lie inside the grid')
    s = (grid-center)/halfwidth
    a = np.maximum(s[:-1], -1.); b = np.minimum(s[1:], 1.)
    selected = b > a
    poly = (Polynomial([1., 0., -1.])**p).deriv(derivative)
    antideriv = poly.integ(); firstmoment = (poly*Polynomial([0., 1.])).integ()
    i0 = np.zeros(len(grid)-1); i1 = i0.copy()
    i0[selected] = antideriv(b[selected])-antideriv(a[selected])
    i1[selected] = firstmoment(b[selected])-firstmoment(a[selected])
    ds = np.diff(s)
    left = (s[1:]*i0-i1)/ds
    right = (i1-s[:-1]*i0)/ds
    w = np.zeros(len(grid)); w[:-1] += left; w[1:] += right
    return w * halfwidth**(1-derivative)


def periodic_weights(x, length, center, halfwidth, derivative=0, p=4):
    """Periodic copy of a compact test, including windows crossing x=0."""
    x=np.asarray(x); n=len(x)
    if halfwidth >= length/2 or x[0] != 0:
        raise ValueError('require x[0]=0 and halfwidth < period/2')
    extended=np.concatenate([x-length,x,x+length])
    return integral_weights(extended,center,halfwidth,derivative,p).reshape(3,n).sum(axis=0)


@lru_cache(maxsize=64)
def _time_weights(grid_tuple, centers_tuple, halfwidth):
    t=np.asarray(grid_tuple)
    return tuple(np.stack([integral_weights(t,c,halfwidth,d) for c in centers_tuple]) for d in [0,1])


def ode_weak(states,t,var_names,centers,halfwidth):
    theta,names=ode.polynomial_library(states,var_names,degree=3)
    w0,w1=_time_weights(tuple(t),tuple(centers),halfwidth)
    # A common physical normalization changes neither relative feature scale
    # nor regression coefficients; it merely keeps row magnitudes moderate.
    norm=float(w0[0].sum())
    return w0@theta/norm, -w1@states/norm, names


def pde_weak(states,t,x,length,time_centers,time_halfwidth,space_centers,space_halfwidth):
    """Fully weak conservative 8-term library, with explicit nuisance change.

    u*u_x and u^2*u_x use fluxes u^2/2 and u^3/3. The final column is
    (u^2)_xx, NOT u*u_xx; the latter cannot remove all data derivatives.
    """
    wt,dt=_time_weights(tuple(t),tuple(time_centers),time_halfwidth)
    wx=[np.stack([periodic_weights(x,length,c,space_halfwidth,d) for c in space_centers]) for d in [0,1,2]]
    norm=float(wt[0].sum()*wx[0][0].sum())
    def product(field,d=0): return (wt@field@wx[d].T).ravel()/norm
    theta=np.column_stack([product(np.ones_like(states)),product(states),product(states**2),
                           -product(states,1),-.5*product(states**2,1),
                           -product(states**3,1)/3.,product(states,2),product(states**2,2)])
    target=-(dt@states@wx[0].T).ravel()/norm
    return theta,target,list(PDE_NAMES)

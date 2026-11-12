"""Validate finite-bath analytic Jacobians against actual finite differences."""
import numpy as np
import pytest

from rubycgw.cluster_ed_gw import (
    _real_bath_hybridization_jacobian,bath_hybridization,
    BathParameters,fit_finite_bath,
)
from rubycgw.cluster_ed_gw_covariant import (
    _complex_bath_residual_jacobian,fit_finite_bath_complex,
)


def _fd_jacobian(fun,x,step=1.5e-7):
    result=[]
    for k in range(len(x)):
        xp=x.copy();xm=x.copy()
        xp[k]+=step;xm[k]-=step
        result.append((fun(xp)-fun(xm))/(2*step))
    return np.column_stack(result)


@pytest.mark.parametrize("seed",(11,43))
def test_complex_bath_analytic_jacobian_matches_finite_difference(seed):
    rng=np.random.default_rng(seed)
    norb,nbath=3,2
    mu=.18
    w=np.asarray([.5,1.5,3.])
    eps=np.asarray([-.3,.4])
    hyb=rng.normal(size=(norb,nbath))*.3+1j*rng.normal(size=(norb,nbath))*.3
    weights=np.asarray([1.,.5,.25])
    x=np.r_[eps,hyb.real.ravel(),hyb.imag.ravel()]
    nvc=norb*nbath
    def unpack(x):
        e=x[:nbath]
        v=x[nbath:nbath+nvc].reshape(norb,nbath)+1j*x[nbath+nvc:].reshape(norb,nbath)
        return e,v
    def fn(x):
        e,v=unpack(x)
        model=bath_hybridization(w,mu,e,v)*weights[:,None,None]
        return np.r_[model.real.ravel(),model.imag.ravel()]
    actual=_complex_bath_residual_jacobian(w,mu,eps,hyb,weights)
    expected=_fd_jacobian(fn,x)
    np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=4e-7)


@pytest.mark.parametrize("metric",["delta","g0"])
def test_real_bath_analytic_jacobian_matches_fd(metric):
    rng=np.random.default_rng(18)
    norb,nbath=3,2
    mu=.1;w=np.array([.4,1.1,2.8]);eps=np.array([-.45,.35])
    v=rng.normal(size=(norb,nbath))*.25
    weights=np.array([1.,.4,.2])
    H=np.asarray([[.1,.02,0.],[.02,-.08,.03],[0.,.03,.17]])
    x=np.r_[eps,v.ravel()]
    eye=np.eye(norb)
    def fn(x):
        model=bath_hybridization(w,mu,x[:nbath],x[nbath:].reshape(norb,nbath))
        if metric=="g0":
            model=np.linalg.inv(
                (1j*w[:,None,None]+mu)*eye[None]-H[None]-model
            )
        model*=weights[:,None,None]
        return np.r_[model.real.ravel(),model.imag.ravel()]
    J=_real_bath_hybridization_jacobian(w,mu,eps,v)
    if metric=="g0":
        model=bath_hybridization(w,mu,eps,v)
        G0=np.linalg.inv((1j*w[:,None,None]+mu)*eye[None]-H[None]-model)
        J=np.einsum("wab,wbcj,wcd->wadj",G0,J,G0,optimize=True)
    J*=weights[:,None,None,None]
    actual=np.r_[J.real.reshape(-1,len(x)),J.imag.reshape(-1,len(x))]
    expected=_fd_jacobian(fn,x)
    np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=4e-7)


@pytest.mark.parametrize("complex_bath",[False,True])
def test_bath_fits_with_analytic_and_numerical_jacobian_agree(complex_bath):
    mu=.17
    w=np.asarray([-.95,-.55,-.15,.15,.55,.95,1.35,1.95])
    e=np.asarray([-.35,.55])
    v=np.array([[.4,.1],[.2,-.35]],complex)
    if complex_bath:
        v+=1j*np.asarray([[.13,-.02],[.07,.15]])
    target=bath_hybridization(w,mu,e,v)
    start=BathParameters(
        energies=e+np.array([.04,-.03]),
        couplings=v+.008,
        fit_error=1.,nfev=0,
    )
    fitter=fit_finite_bath_complex if complex_bath else fit_finite_bath
    output=[]
    for jac in ("analytic","2-point"):
        out=fitter(
            target,w,mu,nbath=2,nfit=4,max_nfev=180,
            initial=start,jacobian=jac,xtol=1e-10,
        )
        output.append(out)
        assert out.fit_error<2e-5
    a,b=output
    np.testing.assert_allclose(
        bath_hybridization(w,mu,a.energies,a.couplings),
        bath_hybridization(w,mu,b.energies,b.couplings),
        atol=1e-5,rtol=1e-5
    )

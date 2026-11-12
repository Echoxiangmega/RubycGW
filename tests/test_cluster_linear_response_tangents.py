"""Independent central-difference checks of analytic ED and bath tangents.

Finite differences here are TEST ORACLES only, not used in the analytic
solver. The production impurity response uses a single unperturbed ED.
"""
from __future__ import annotations

import numpy as np
import pytest

from rubycgw.impurity_ed import FiniteBathImpurityED
from rubycgw.impurity_linear_response import impurity_green_tangent
from rubycgw.cluster_ed_gw import BathParameters,bath_hybridization
from rubycgw.cluster_bath_linear_response import (
    complex_bath_fit_tangent,bath_delta_tangent
)


@pytest.mark.parametrize("complex_direction",[False,True])
def test_impurity_analytic_lehmann_tangent_against_full_ed_finite_difference(
    complex_direction
):
    h=np.array([
        [.15,.22,.13],
        [.22,-.17,-.09],
        [.13,-.09,.36],
    ],complex)
    D=np.array([
        [.19,.16+.07j,.06],
        [.16-.07j,-.05,-.11],
        [.06,-.11,.13]
    ],complex)
    if not complex_direction:
        D=D.real.astype(complex)
    interaction=((0,1,.37),)
    mu=.12;T=.45
    omega=(2*np.arange(-3,3)+1)*np.pi*T
    impurity=FiniteBathImpurityED(h,interaction,correlated_orbitals=(0,1))
    impurity.diagonalize()
    analytic,diag=impurity_green_tangent(
        impurity,omega,mu,T,D,discard_weight_tol=0.
    )
    eps=1e-5
    results=[]
    for sign in (-1.,1.):
        current=FiniteBathImpurityED(h+sign*eps*D,interaction,
                                    correlated_orbitals=(0,1))
        current.diagonalize()
        g,_=current.green_iomega(1j*omega,mu,T,
                                 discard_weight_tol=0.)
        results.append(g)
    fd=(results[1]-results[0])/(2*eps)
    np.testing.assert_allclose(analytic,fd,rtol=2.e-5,atol=2.e-6)
    assert diag.source_states>0
    assert np.all(np.isfinite(analytic))


def test_bath_stationary_hessian_implicit_response_exact_representable_shift():
    w=(2*np.arange(-6,6)+1)*np.pi*.28
    mu=.1
    eps=np.array([-.43,.67])
    v=np.array([[.39+.08j,.1-.07j],[.21-.16j,-.34+.11j]])
    bath=BathParameters(eps,v,0.,0)
    target=bath_hybridization(w,mu,eps,v)
    dxi=np.array([.025,-.031])
    dv=np.array([[.018+.013j,-.008+.007j],
                 [.014-.006j,.006-.017j]])
    dtarget=bath_delta_tangent(w,mu,bath,dxi,dv)
    got_xi,got_v,diag=complex_bath_fit_tangent(
        target,dtarget,w,mu,bath,nfit=5,
        stationarity_tol=1e-7
    )
    predicted=bath_delta_tangent(w,mu,bath,got_xi,got_v)
    np.testing.assert_allclose(predicted,dtarget,atol=2e-7,rtol=1e-5)
    assert diag.linearized_residual<1e-6
    assert not diag.active_bounds


def test_bath_hessian_rejects_nonstationary_reference():
    w=(2*np.arange(-4,4)+1)*np.pi*.3
    eps=np.array([-.3])
    v=np.array([[.4+.15j],[.3-.12j]])
    bath=BathParameters(eps,v,0.,0)
    target=np.zeros((len(w),2,2),complex)
    with pytest.raises(RuntimeError,match="not stationary"):
        complex_bath_fit_tangent(
            target,np.zeros_like(target),w,.1,bath,nfit=3,
            stationarity_tol=1e-10
        )


def test_bath_exact_hessian_nonzero_residual_vs_refitted_finite_difference():
    from rubycgw.cluster_ed_gw_covariant import fit_finite_bath_complex
    rng=np.random.default_rng(311)
    T=.21
    w=(2*np.arange(-6,6)+1)*np.pi*T
    mu=.09
    eps=np.array([-.28])
    v=np.array([[.42+.06j],[.16-.13j]])
    model=bath_hybridization(w,mu,eps,v)
    # Add an out-of-bath-manifold target with a nonzero least-squares
    # residual. If one drops the r*Hessian term, the response is wrong.
    perturb=.016*(rng.normal(size=model.shape)+1j*
                   rng.normal(size=model.shape))
    target=model+perturb
    fitted=fit_finite_bath_complex(
        target,w,mu,nbath=1,nfit=4,
        max_nfev=250,xtol=1e-11,
        initial=BathParameters(eps,v,0.,0),
    )
    delta=.023*(rng.normal(size=model.shape)+1j*
                 rng.normal(size=model.shape))
    dx,dv,diag=complex_bath_fit_tangent(
        target,delta,w,mu,fitted,nfit=4,
        stationarity_tol=5e-5,svd_cutoff=1e-11,
    )
    predicted=bath_delta_tangent(w,mu,fitted,dx,dv)
    eps_step=2e-4
    vals=[]
    for sign in (-1.,1.):
        refit=fit_finite_bath_complex(
            target+sign*eps_step*delta,w,mu,
            nbath=1,nfit=4,max_nfev=250,
            xtol=1e-11,initial=fitted
        )
        vals.append(bath_hybridization(
            w,mu,refit.energies,refit.couplings
        ))
    fd=(vals[1]-vals[0])/(2*eps_step)
    np.testing.assert_allclose(predicted,fd,atol=2e-4,rtol=4e-2)
    assert diag.stationarity<5e-5
    assert diag.linearized_residual<5e-5

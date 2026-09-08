import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.sox_diagnostic import (
    free_green_tau_direction,
    solve_free_mu,
    sox_response_coefficient_free,
    sox_self_energy_tau,
    sox_vertex_tau,
)


def _free_tau(h, mu, tau, T):
    e, U = np.linalg.eigh(0.5*(h+h.conj().T))
    xi=e-mu; beta=1/T
    f=1/(np.exp(np.clip(beta*xi,-700,700))+1)
    gp=-np.exp(-xi*tau)*(1-f)
    gm=np.exp(xi*tau)*f
    return (U*gp[None,:])@U.conj().T, (U*gm[None,:])@U.conj().T


def test_sox_vertex_is_directional_derivative_of_sox_self_energy():
    h=np.array([[.1,.2+.03j,0],[.2-.03j,-.2,.15],[0,.15,.3]],complex)
    K=np.array([[.2,.1j,.04],[-.1j,-.1,.02j],[.04,-.02j,-.1]],complex)
    v=np.array([[0,.7,.4],[.7,0,.2],[.4,.2,0]],complex)
    T=.23;mu=.04;tau=1.37
    gp,gm,xp,xm=free_green_tau_direction(h,mu,K,tau,T)
    analytic=sox_vertex_tau(gp,gm,xp,xm,v)
    eps=2e-6
    gp1,gm1=_free_tau(h+eps*K,mu,tau,T)
    gp0,gm0=_free_tau(h-eps*K,mu,tau,T)
    numeric=(sox_self_energy_tau(gp1,gm1,v)-sox_self_energy_tau(gp0,gm0,v))/(2*eps)
    np.testing.assert_allclose(analytic,numeric,rtol=2e-8,atol=2e-9)


def test_12site_bare_sox_current_coefficient_is_negative_and_quadrature_converged():
    exact=ExactSmallRubyThermal(2,1,RubyParameters(V=0.,ti=.4,t1=.2,t2=.2))
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=24,nOmega=4,T=.08)
    mu=solve_free_mu(exact.h0,6.,grid.T)
    for ch in ('z_same','z_opposite'):
        K=exact.pseudospin_operator(ch,(0.,0.))
        b48,_=sox_response_coefficient_free(exact.h0,mu,exact.Vunit,K,grid,n_quad=48)
        b96,_=sox_response_coefficient_free(exact.h0,mu,exact.Vunit,K,grid,n_quad=96)
        assert abs(b96.imag)<1e-10
        assert b96.real<0.
        np.testing.assert_allclose(b48,b96,rtol=2e-7,atol=2e-8)

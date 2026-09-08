import numpy as np

from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


def test_noninteracting_exact_green_matches_one_body_resolvent():
    T = 0.2
    exact = ExactSmallRubyThermal(1, 1, RubyParameters(V=0.0))
    exact.diagonalize(0.0)
    mu = exact.solve_mu(3.0, T)
    omega = (2 * np.arange(-3, 4) + 1) * np.pi * T
    G, sel = exact.green_iomega(
        1j * omega,
        mu,
        T,
        discard_weight_tol=0.0,
    )
    eye = np.eye(exact.n_sites, dtype=complex)
    ref = np.stack([
        np.linalg.inv((1j * w + mu) * eye - exact.h0)
        for w in omega
    ])
    assert sel.discarded_weight < 1e-14
    assert abs(sel.average_particles - 3.0) < 1e-10
    assert np.max(np.abs(G - ref)) < 2e-10


def test_two_by_one_pseudospin_harmonics_are_hermitian_at_gamma_and_m1():
    exact = ExactSmallRubyThermal(2, 1, RubyParameters(V=1.0))
    for q in ((0.0, 0.0), (0.5, 0.0)):
        for ch in ("x_even", "z_same", "z_opposite"):
            K = exact.pseudospin_operator(ch, q=q)
            assert K.shape == (12, 12)
            assert np.max(np.abs(K - K.conj().T)) < 1e-12


def test_interaction_matrix_contains_only_six_bonds_per_primitive_cell():
    exact = ExactSmallRubyThermal(2, 1, RubyParameters(V=1.0))
    # 2 cells x 6 intra-triangle undirected bonds.  Vunit is symmetric, so
    # the number of nonzero matrix entries is twice the number of bonds.
    assert len(exact.interaction_pairs) == 12
    assert np.count_nonzero(np.abs(exact.Vunit) > 1e-14) == 24


def test_static_matrix_matches_scalar_lehmann_and_operator_linearity():
    exact = ExactSmallRubyThermal(1, 1, RubyParameters(V=.3))
    exact.diagonalize(.3)
    T=.2;mu=exact.solve_mu(3.,T)
    ops=np.array([exact.pseudospin_operator(ch) for ch in ('x_even','z_same','z_opposite')])
    chi,means=exact.static_susceptibility_matrix(ops,mu,T)
    for i,op in enumerate(ops):
        _,scalar,mean,_=exact.correlation_tau(op,np.array([0.,1./T]),mu,T,discard_weight_tol=0.)
        np.testing.assert_allclose(chi[i,i],scalar,atol=1e-10)
        np.testing.assert_allclose(means[i],mean,atol=1e-12)
    weights=np.array([.3,.7,-.4])
    _,scalar,_,_=exact.correlation_tau(np.einsum('a,aij->ij',weights,ops),np.array([0.,1./T]),mu,T,discard_weight_tol=0.)
    np.testing.assert_allclose(weights@chi@weights,scalar,atol=1e-10)
    assert np.linalg.eigvalsh(chi).min() > -1e-10


def test_static_matrix_free_fermions_and_degenerate_levels():
    from rubycgw.response_tail import build_tail_reference,reference_response_infinite
    from rubycgw.grids import MatsubaraGrid
    exact=ExactSmallRubyThermal(1,1,RubyParameters(V=0.,ti=0.,t1=0.,t2=0.))
    exact.diagonalize(0.)
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=4,nOmega=1,T=.2)
    ops=np.array([np.eye(6),exact.pseudospin_operator('z_same')])
    chi,_=exact.static_susceptibility_matrix(ops,0.,grid.T)
    ref=build_tail_reference(exact.h0[None,None],0.,np.zeros((6,6)),grid)
    expected=np.array([[-np.trace(left@reference_response_infinite(ref,right,grid)[0,0]).real for right in ops] for left in ops])
    np.testing.assert_allclose(chi,expected,atol=1e-11)


def test_paired_thermal_truncation_preserves_kms_and_static_response():
    exact=ExactSmallRubyThermal(1,1,RubyParameters(V=.7))
    exact.diagonalize(.7)
    T=.06;mu=exact.solve_mu(3.,T)
    K=exact.pseudospin_operator('x_even')
    tau=np.linspace(0.,1./T,31)
    truncated,chi,_,sel=exact.correlation_tau(K,tau,mu,T,discard_weight_tol=1e-8)
    full,chi_full,_,_=exact.correlation_tau(K,tau,mu,T,discard_weight_tol=0.)
    assert sum(map(len,sel.kept_indices)) < 2**exact.n_sites
    np.testing.assert_allclose(truncated,truncated[::-1],atol=1e-12)
    np.testing.assert_allclose(truncated,full,atol=1e-7)
    np.testing.assert_allclose(chi,chi_full,atol=1e-6)

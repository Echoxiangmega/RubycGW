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

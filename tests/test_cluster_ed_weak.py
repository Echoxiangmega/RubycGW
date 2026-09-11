import numpy as np

from benchmark_cluster_ed_weak_chi import _choose_ed_method
from rubycgw.gf2 import compute_gf2_nonhartree, second_order_bare_interaction
from rubycgw.cluster_ed_weak_fast import canonical_weak_solver
from rubycgw.cluster_ed_weak_covariant import solve_cluster_source_warm_weak
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.sox_covariant import SOXOptions
from run_cluster_ed_weak import _resolved_mesh


class _Args:
    Lx = 3
    Ly = 2
    nk1 = None
    nk2 = None


class _EDArgs:
    ed_mode = "auto"


def test_weak_solver_names():
    assert canonical_weak_solver("gw") == "gw"
    assert canonical_weak_solver("GF2") == "gf2"
    assert canonical_weak_solver("second-order") == "gf2"


def test_optional_kmesh_defaults_to_finite_torus_labels():
    a = _Args()
    assert _resolved_mesh(a) == (3, 2)
    a.nk1 = 8
    assert _resolved_mesh(a) == (8, 2)
    a.nk2 = 6
    assert _resolved_mesh(a) == (8, 6)


def test_second_order_bare_interaction_is_v_plus_vpv():
    V = np.asarray([[[[2.0, 0.3], [0.3, 1.0]]]], dtype=complex)
    P = np.asarray([[[[[0.2, -0.1], [0.05, 0.4]]]]], dtype=complex)
    got = second_order_bare_interaction(P, V)
    expected = V[None] + V[None] @ P @ V[None]
    np.testing.assert_allclose(got, expected, atol=1e-14, rtol=1e-14)


def test_gf2_nonhartree_vanishes_at_zero_interaction():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=2, nOmega=1, T=0.2)
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0)
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_interaction(grid.qmesh(), params)
    eye = np.eye(6, dtype=complex)
    G = np.linalg.inv(
        (1j * grid.omega[:, None, None, None, None]) * eye[None, None, None]
        - h0[None]
    )
    sigma, _, W2, sigma_f, sigma_sox = compute_gf2_nonhartree(
        G,
        Vq,
        h0,
        0.0,
        np.zeros((6, 6), dtype=complex),
        grid,
        backend="direct",
        sox_opts=SOXOptions(n_quad=16, tail_complete=False),
    )
    np.testing.assert_allclose(Vq, 0.0, atol=1e-14)
    np.testing.assert_allclose(W2, 0.0, atol=1e-14)
    np.testing.assert_allclose(sigma_f, 0.0, atol=1e-14)
    np.testing.assert_allclose(sigma_sox, 0.0, atol=1e-14)
    np.testing.assert_allclose(sigma, 0.0, atol=1e-14)


def test_ed_auto_skips_when_kmesh_differs_from_finite_torus():
    assert _choose_ed_method(_EDArgs(), 2, 1, 2.0, False) == "none"


def test_generic_source_solver_symbol_is_callable():
    assert callable(solve_cluster_source_warm_weak)

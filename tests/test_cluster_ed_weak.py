import numpy as np

from rubycgw.gf2 import second_order_bare_interaction
from rubycgw.cluster_ed_weak_fast import canonical_weak_solver
from run_cluster_ed_weak import _resolved_mesh


class _Args:
    Lx = 3
    Ly = 2
    nk1 = None
    nk2 = None


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

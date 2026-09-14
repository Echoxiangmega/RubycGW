import numpy as np

from rubycgw.model import RubyParameters, build_interaction
from vprime_study.model import (
    INTERTRIANGLE_BONDS,
    VPrimeParameters,
    build_vprime_interaction,
    vprime_cluster_interactions,
    vprime_cluster_matrix,
)


def test_vprime_zero_reduces_to_baseline_interaction():
    q = np.asarray([
        [0.0, 0.0],
        [0.25, 0.0],
        [0.0, 0.5],
        [0.25, 0.5],
    ])
    p0 = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=1.2)
    pp = VPrimeParameters(ti=0.4, t1=0.2, t2=0.2, V=1.2, Vprime=0.0)
    assert np.allclose(build_vprime_interaction(q, pp), build_interaction(q, p0))


def test_vprime_is_on_all_six_intertriangle_hopping_bonds():
    assert len(INTERTRIANGLE_BONDS) == 6
    assert set(INTERTRIANGLE_BONDS) == {
        (1, 4, (0, 0)),
        (5, 0, (0, -1)),
        (2, 3, (-1, 0)),
        (3, 1, (0, -1)),
        (2, 5, (0, 0)),
        (0, 4, (-1, 0)),
    }


def test_q0_projection_matches_cluster_matrix_and_preserves_attraction():
    p = VPrimeParameters(V=1.2, Vprime=-0.05)
    q0 = np.zeros((1, 1, 2), dtype=float)
    vq0 = build_vprime_interaction(q0, p)[0, 0]
    vc = vprime_cluster_matrix(p)
    assert np.allclose(vq0, vc)
    interactions = vprime_cluster_interactions(p)
    assert len(interactions) == 12
    attractive = [u for _i, _j, u in interactions if u < 0]
    assert len(attractive) == 6
    assert np.allclose(attractive, -0.05)


def test_intercell_vprime_makes_lattice_interaction_q_dependent():
    p = VPrimeParameters(V=1.2, Vprime=-0.05)
    q = np.asarray([[0.0, 0.0], [0.5, 0.0]])
    vq = build_vprime_interaction(q, p)
    assert not np.allclose(vq[0], vq[1])

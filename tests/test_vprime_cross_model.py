import numpy as np

from vprime_study.model import VPrimeParameters, build_vprime_interaction
from vprime_study.cross_model import (
    CROSS_INTERTRIANGLE_BONDS,
    REFERENCE_CROSS_BONDS,
    REFERENCE_STRAIGHT_BONDS,
    VPrimeCrossParameters,
    build_vprime_vcross_interaction,
    reference_pair_effective_couplings,
    vprime_vcross_cluster_interactions,
    vprime_vcross_cluster_matrix,
)


def test_vcross_zero_reduces_to_existing_vprime_model():
    q = np.asarray([
        [0.0, 0.0],
        [0.25, 0.0],
        [0.0, 0.5],
        [0.25, 0.5],
    ])
    p_old = VPrimeParameters(V=1.8, Vprime=-0.1)
    p_new = VPrimeCrossParameters(V=1.8, Vprime=-0.1, Vcross=0.0)
    assert np.allclose(
        build_vprime_vcross_interaction(q, p_new),
        build_vprime_interaction(q, p_old),
    )


def test_reference_pair_crosses_only_the_two_nearby_links():
    assert set(REFERENCE_STRAIGHT_BONDS) == {
        (1, 4, (0, 0)),
        (2, 5, (0, 0)),
    }
    assert set(REFERENCE_CROSS_BONDS) == {
        (1, 5, (0, 0)),
        (2, 4, (0, 0)),
    }
    # In particular the far same-cell A0-B0 link is not introduced.
    assert (0, 3, (0, 0)) not in CROSS_INTERTRIANGLE_BONDS


def test_vcross_bonds_cover_three_triangle_pair_orientations():
    assert len(CROSS_INTERTRIANGLE_BONDS) == 6
    assert set(CROSS_INTERTRIANGLE_BONDS) == {
        (1, 5, (0, 0)),
        (2, 4, (0, 0)),
        (0, 3, (0, 1)),
        (1, 5, (0, 1)),
        (2, 4, (-1, 0)),
        (0, 3, (-1, 0)),
    }


def test_q0_projection_sums_repeated_crossed_pairs():
    p = VPrimeCrossParameters(V=1.8, Vprime=-0.1, Vcross=-0.05)
    q0 = np.zeros((1, 1, 2), dtype=float)
    vq0 = build_vprime_vcross_interaction(q0, p)[0, 0]
    vc = vprime_vcross_cluster_matrix(p)
    assert np.allclose(vq0, vc)

    interactions = {(i, j): u for i, j, u in vprime_vcross_cluster_interactions(p)}
    assert np.isclose(interactions[(0, 3)], -0.10)
    assert np.isclose(interactions[(1, 5)], -0.10)
    assert np.isclose(interactions[(2, 4)], -0.10)
    # Six intra + six straight + three q=0-projected cross pair types = all 15 pairs.
    assert len(interactions) == 15


def test_intercell_vcross_makes_lattice_interaction_q_dependent():
    p = VPrimeCrossParameters(V=1.8, Vprime=0.0, Vcross=-0.05)
    q = np.asarray([[0.0, 0.0], [0.5, 0.0]])
    vq = build_vprime_vcross_interaction(q, p)
    assert not np.allclose(vq[0], vq[1])


def test_reference_effective_couplings_match_derived_formula():
    jn, jm, jz = reference_pair_effective_couplings(
        t=0.2, V=1.8, Vprime=-0.1, Vcross=-0.05
    )
    s = 0.2**2 / 1.8
    assert np.isclose(jn, 5.0 * s / 9.0 + (-0.1 - 0.05) / 18.0)
    assert np.isclose(jm, -s / 3.0 + (0.1 - 0.05) / 6.0)
    assert np.isclose(jz, -s / 3.0)
    assert abs(jz) > abs(jn)
    assert abs(jz) > abs(jm)

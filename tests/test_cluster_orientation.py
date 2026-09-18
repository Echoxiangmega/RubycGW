import numpy as np

from rubycgw.cluster_orientation import (
    build_oriented_lattice_fields,
    gauge_transform_lattice,
    intracell_block,
    orientation_b_shift,
    rotate_local_between_orientations,
    rotate_solution_between_orientations,
    transform_between_orientations,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.cluster_ed_gw import cluster_interaction_matrix
from rubycgw.c3_constraint import C3_ORBITAL_PERM
from rubycgw.model import build_h0
from rubycgw.models.ruby import physical_pair_cluster_interactions
from vprime_study.cross_model import (
    VPrimeCrossParameters,
    build_vprime_vcross_interaction,
)


def _assert_only_pairs(h, pairs, *, atol=1e-12):
    h = np.asarray(h)
    expected = {tuple(sorted(p)) for p in pairs}
    actual = set()
    for a in range(3):
        for b in range(3, 6):
            if abs(h[a, b]) > atol:
                actual.add((a, b))
    assert actual == expected


def test_orientation_shifts_are_the_three_neighbor_gauges():
    assert np.array_equal(orientation_b_shift(0), [0, 0])
    assert np.array_equal(orientation_b_shift(1), [0, 1])
    assert np.array_equal(orientation_b_shift(2), [-1, 0])


def test_gauge_transform_roundtrip():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(4, 3, 3, 6, 6)) + 1j * rng.normal(size=(4, 3, 3, 6, 6))
    y = gauge_transform_lattice(x, (0, 1))
    z = gauge_transform_lattice(y, (0, -1))
    assert np.max(np.abs(z - x)) < 1e-12


def test_transform_between_orientations_roundtrip():
    rng = np.random.default_rng(456)
    x = rng.normal(size=(2, 3, 3, 6, 6)) + 1j * rng.normal(size=(2, 3, 3, 6, 6))
    y = transform_between_orientations(x, 0, 2)
    z = transform_between_orientations(y, 2, 0)
    assert np.max(np.abs(z - x)) < 1e-12


def test_three_orientations_internalize_three_expected_ab_pairs():
    grid = MatsubaraGrid(nk1=6, nk2=6, nw=2, nOmega=1, T=0.1)
    params = VPrimeCrossParameters(ti=0.4, t1=0.21, t2=0.17, V=1.2, Vprime=-0.1, Vcross=-0.05)
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_vprime_vcross_interaction(grid.qmesh(), params)

    expected = {
        0: ((1, 4), (2, 5)),
        1: ((0, 5), (1, 3)),
        2: ((0, 4), (2, 3)),
    }
    for r in (0, 1, 2):
        hr, _ = build_oriented_lattice_fields(h0, Vq, r)
        hc = intracell_block(hr)
        _assert_only_pairs(hc, expected[r])


def test_physical_pair_cluster_equals_oriented_q_average_without_collapse():
    grid = MatsubaraGrid(nk1=6, nk2=6, nw=2, nOmega=1, T=0.1)
    params = VPrimeCrossParameters(
        ti=0.4, t1=0.2, t2=0.2, V=1.8, Vprime=-0.1, Vcross=-0.07
    )
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_vprime_vcross_interaction(grid.qmesh(), params)
    for r in (0, 1, 2):
        _, vr = build_oriented_lattice_fields(h0, Vq, r)
        local_exact = np.mean(vr, axis=(0, 1))
        local_pair = cluster_interaction_matrix(
            physical_pair_cluster_interactions(params, r)
        )
        assert np.max(np.abs(local_pair - local_exact)) < 1e-12

        # Every selected crossed bond keeps the bare Vcross; no 2*Vcross fold.
        terms = physical_pair_cluster_interactions(params, r)
        cross = [u for i, j, u in terms if i < 3 <= j and abs(u - params.Vcross) < 1e-14]
        assert len(cross) == 2


def test_orientation_change_is_unitary_at_each_k_and_q0_interaction_unchanged():
    grid = MatsubaraGrid(nk1=4, nk2=4, nw=2, nOmega=1, T=0.1)
    params = VPrimeCrossParameters(ti=0.4, t1=0.2, t2=0.2, V=1.8, Vprime=-0.1, Vcross=-0.07)
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_vprime_vcross_interaction(grid.qmesh(), params)

    eig0 = np.linalg.eigvalsh(h0)
    for r in (0, 1, 2):
        hr, vr = build_oriented_lattice_fields(h0, Vq, r)
        assert np.max(np.abs(np.linalg.eigvalsh(hr) - eig0)) < 1e-12
        assert np.max(np.abs(vr[0, 0] - Vq[0, 0])) < 1e-12


def test_physical_c3_maps_one_oriented_hamiltonian_to_the_next_frame():
    grid = MatsubaraGrid(nk1=6, nk2=6, nw=2, nOmega=1, T=0.1)
    params = VPrimeCrossParameters(ti=0.4, t1=0.2, t2=0.2, V=1.8, Vprime=-0.1, Vcross=-0.07)
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_vprime_vcross_interaction(grid.qmesh(), params)
    hs = [build_oriented_lattice_fields(h0, Vq, r)[0] for r in (0, 1, 2)]
    for src in (0, 1, 2):
        dst = (src + 1) % 3
        mapped = rotate_solution_between_orientations(hs[src], src, dst)
        assert np.max(np.abs(mapped - hs[dst])) < 1e-12


def test_c3_related_local_cluster_matrix_remains_local_in_target_frame():
    rng = np.random.default_rng(789)
    x = rng.normal(size=(3, 6, 6)) + 1j * rng.normal(size=(3, 6, 6))
    for src in (0, 1, 2):
        for dst in (0, 1, 2):
            y = rotate_local_between_orientations(
                x, src, dst, nk1=3, nk2=3
            )
            assert y.shape == x.shape


def test_diagonal_translation_invariant_density_is_gauge_invariant():
    n = np.asarray([0.34, 0.33, 0.33, 0.32, 0.35, 0.33])
    x = np.zeros((3, 3, 6, 6), dtype=complex)
    x[..., np.arange(6), np.arange(6)] = n
    for r in (0, 1, 2):
        y = gauge_transform_lattice(x, orientation_b_shift(r))
        assert np.max(np.abs(y - x)) < 1e-12


def _sorted_density_terms(terms):
    return sorted(
        (min(int(i), int(j)), max(int(i), int(j)), round(float(u), 14))
        for i, j, u in terms
    )


def _rotate_density_terms_once(terms):
    p = np.asarray(C3_ORBITAL_PERM, dtype=int)
    return tuple((int(p[i]), int(p[j]), float(u)) for i, j, u in terms)


def test_physical_pair_interactions_form_exact_c3_orbit():
    params = VPrimeCrossParameters(
        ti=0.4, t1=0.2, t2=0.2, V=1.8, Vprime=-0.1, Vcross=-0.07
    )
    terms = [physical_pair_cluster_interactions(params, r) for r in (0, 1, 2)]
    assert _sorted_density_terms(_rotate_density_terms_once(terms[0])) == _sorted_density_terms(terms[1])
    assert _sorted_density_terms(_rotate_density_terms_once(terms[1])) == _sorted_density_terms(terms[2])
    assert _sorted_density_terms(_rotate_density_terms_once(terms[2])) == _sorted_density_terms(terms[0])


def test_h0_and_full_interaction_are_c3_related_on_2x2_mesh():
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=2, nOmega=1, T=0.1)
    params = VPrimeCrossParameters(
        ti=0.4, t1=0.2, t2=0.2, V=1.8, Vprime=-0.1, Vcross=-0.07
    )
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_vprime_vcross_interaction(grid.qmesh(), params)
    oriented = [build_oriented_lattice_fields(h0, Vq, r) for r in (0, 1, 2)]
    for src in (0, 1, 2):
        dst = (src + 1) % 3
        hm = rotate_solution_between_orientations(oriented[src][0], src, dst)
        vm = rotate_solution_between_orientations(oriented[src][1], src, dst)
        assert np.max(np.abs(hm - oriented[dst][0])) < 1e-12
        assert np.max(np.abs(vm - oriented[dst][1])) < 1e-12

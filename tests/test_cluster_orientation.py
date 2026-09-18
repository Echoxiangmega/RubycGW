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

from rubycgw.supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    hartree_self_energy_matrix,
)
from rubycgw.supercell_gw_fast import (
    _build_tail_cache,
    _solve_mu_matrix_fast,
    density_from_G_cached,
)
from rubycgw.supercell_gw_split import compute_sigma_gw_split_components

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


def test_single_scgw_map_is_orientation_gauge_covariant_on_2x2():
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=5, nOmega=3, T=0.08)
    params = VPrimeCrossParameters(
        ti=0.4, t1=0.2, t2=0.2, V=1.8, Vprime=-0.1, Vcross=-0.07
    )
    h0_base = build_h0(grid.kmesh(), params)
    Vq_base = build_vprime_vcross_interaction(grid.qmesh(), params)

    # Start from a deliberately nonsymmetric but gauge-related self-energy
    # state so the test probes covariance rather than accidental symmetry.
    rng = np.random.default_rng(20260918)
    sh0 = np.diag(rng.normal(scale=0.05, size=6)).astype(complex)
    sg0 = (
        rng.normal(scale=0.01, size=(grid.nf, 2, 2, 6, 6))
        + 1j * rng.normal(scale=0.01, size=(grid.nf, 2, 2, 6, 6))
    )
    sg0 = 0.5 * (sg0 + np.swapaxes(sg0.conj(), -1, -2))
    mu0 = 0.17

    h0s, Vqs = build_oriented_lattice_fields(h0_base, Vq_base, 0)
    G0 = np.linalg.inv(
        (1j * grid.omega[:, None, None, None, None] + mu0)
        * np.eye(6)[None, None, None, :, :]
        - h0s[None]
        - sh0[None, None, None]
        - sg0
    )
    cache0 = _build_tail_cache(h0s, sh0)
    n0 = density_from_G_cached(G0, grid, mu0, cache0)
    P0 = compute_polarization_matrix(G0, grid, backend="fft")
    W0 = compute_screened_interaction_matrix(P0, Vqs)
    St0, Sf0, Sc0, rho0 = compute_sigma_gw_split_components(
        G0, W0, Vqs, grid, h0s, mu0, sh0, backend="fft"
    )

    for r in (1, 2):
        hr, Vr = build_oriented_lattice_fields(h0_base, Vq_base, r)
        Gr = transform_between_orientations(G0, 0, r)
        sgr = transform_between_orientations(sg0, 0, r)

        cacher = _build_tail_cache(hr, sh0)
        nr = density_from_G_cached(Gr, grid, mu0, cacher)
        Pr = compute_polarization_matrix(Gr, grid, backend="fft")
        Wr = compute_screened_interaction_matrix(Pr, Vr)
        Str, Sfr, Scr, rhor = compute_sigma_gw_split_components(
            Gr, Wr, Vr, grid, hr, mu0, sh0, backend="fft"
        )

        assert np.max(np.abs(nr - n0)) < 2e-12
        assert np.max(np.abs(Pr - transform_between_orientations(P0, 0, r))) < 2e-11
        assert np.max(np.abs(Wr - transform_between_orientations(W0, 0, r))) < 2e-11
        assert np.max(np.abs(rhor - transform_between_orientations(rho0, 0, r))) < 2e-11
        assert np.max(np.abs(Sfr - transform_between_orientations(Sf0, 0, r))) < 2e-11
        assert np.max(np.abs(Scr - transform_between_orientations(Sc0, 0, r))) < 2e-11
        assert np.max(np.abs(Str - transform_between_orientations(St0, 0, r))) < 2e-11

        shr = hartree_self_energy_matrix(nr, Vr[0, 0])
        assert np.max(np.abs(shr - hartree_self_energy_matrix(n0, Vqs[0, 0]))) < 2e-12

        mur, Gmur, _, _ = _solve_mu_matrix_fast(
            hr, shr, sgr, grid, 2.0, mu0, 1e-10, 100
        )
        mu_ref, Gmu0, _, _ = _solve_mu_matrix_fast(
            h0s, hartree_self_energy_matrix(n0, Vqs[0, 0]), sg0,
            grid, 2.0, mu0, 1e-10, 100
        )
        assert abs(mur - mu_ref) < 2e-11
        assert np.max(
            np.abs(Gmur - transform_between_orientations(Gmu0, 0, r))
        ) < 2e-10

import numpy as np

from rubycgw import MatsubaraGrid, RubyParameters, build_h0, build_interaction
from rubycgw.cgw import VertexOptions, vertex_corrections_q0
from rubycgw.finite_q_cgw import (
    negative_q_index,
    q_index_from_reduced,
    q_reduced_from_index,
    solve_vertex_finite_q,
    susceptibility_matrix_finite_q,
    vertex_corrections_finite_q,
)
from rubycgw.gw import (
    build_g0_inverse,
    compute_polarization,
    compute_screened_interaction,
)
from rubycgw.pseudospin import primitive_pseudospin_vertex


def _random_complex(rng, shape, scale=1.0):
    return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def test_q_index_helpers_are_periodic_and_exact():
    grid = MatsubaraGrid(nk1=6, nk2=9, nw=2, nOmega=1, T=0.2)
    assert q_index_from_reduced((1.0 / 3.0, 1.0 / 3.0), grid) == (2, 3)
    assert q_index_from_reduced((-1.0 / 6.0, -1.0 / 9.0), grid) == (5, 8)
    assert negative_q_index((2, 3), grid) == (4, 6)
    assert np.allclose(q_reduced_from_index((2, 3), grid), (1.0 / 3.0, 1.0 / 3.0))


def test_finite_q_zero_corrections_match_production_q0():
    rng = np.random.default_rng(112233)
    grid = MatsubaraGrid(nk1=3, nk2=2, nw=3, nOmega=1, T=0.13)
    shape_g = (grid.nf, grid.nk1, grid.nk2, 6, 6)
    shape_w = (grid.nb, grid.nk1, grid.nk2, 6, 6)
    shape_v = (grid.nk1, grid.nk2, 6, 6)
    G = _random_complex(rng, shape_g, 0.12)
    W = _random_complex(rng, shape_w, 0.08)
    Vq = _random_complex(rng, shape_v, 0.05)
    Gamma = _random_complex(rng, shape_g, 0.09)

    for backend in ("direct", "fft"):
        opts = VertexOptions(
            include_hartree=True,
            include_fock=True,
            include_mt=True,
            include_al=True,
            verbose=False,
            momentum_backend=backend,
        )
        ref = vertex_corrections_q0(G, W, Vq, Gamma, grid, opts)
        got = vertex_corrections_finite_q(
            G, W, Vq, Gamma, (0, 0), grid, opts
        )
        for a, b in zip(ref, got):
            assert np.max(np.abs(a - b)) < 2e-11


def test_finite_q_fft_matches_direct_for_nonzero_external_q():
    rng = np.random.default_rng(445566)
    grid = MatsubaraGrid(nk1=3, nk2=3, nw=3, nOmega=1, T=0.11)
    shape_g = (grid.nf, grid.nk1, grid.nk2, 6, 6)
    shape_w = (grid.nb, grid.nk1, grid.nk2, 6, 6)
    shape_v = (grid.nk1, grid.nk2, 6, 6)
    G = _random_complex(rng, shape_g, 0.10)
    W = _random_complex(rng, shape_w, 0.07)
    Vq = _random_complex(rng, shape_v, 0.04)
    Gamma = _random_complex(rng, shape_g, 0.08)
    p = (1, 2)

    direct_opts = VertexOptions(
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=True,
        verbose=False,
        momentum_backend="direct",
    )
    fft_opts = VertexOptions(
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=True,
        verbose=False,
        momentum_backend="fft",
    )
    direct = vertex_corrections_finite_q(
        G, W, Vq, Gamma, p, grid, direct_opts
    )
    fft = vertex_corrections_finite_q(
        G, W, Vq, Gamma, p, grid, fft_opts
    )
    for a, b in zip(direct, fft):
        assert np.max(np.abs(a - b)) < 3e-11


def test_v_zero_finite_q_vertex_is_bare_and_matches_bubble():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=3, nk2=3, nw=5, nOmega=1, T=0.12)
    h0 = build_h0(grid.kmesh(), params)
    G0 = np.linalg.inv(build_g0_inverse(h0, grid, mu=0.0))
    Vq = build_interaction(grid.qmesh(), params)
    W = np.zeros((grid.nb, grid.nk1, grid.nk2, 6, 6), dtype=complex)
    K = primitive_pseudospin_vertex("x_even")
    p = (1, 1)

    opts = VertexOptions(
        max_iter=5,
        tol=1e-13,
        solver="gmres",
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=True,
        verbose=False,
    )
    result = solve_vertex_finite_q(G0, W, Vq, K, p, grid, opts)
    Kfield = np.broadcast_to(K, G0.shape)
    assert result.converged
    assert np.max(np.abs(result.Gamma - Kfield)) < 1e-13

    chi = susceptibility_matrix_finite_q(
        G0, K[None, ...], [result.Gamma], p, grid
    )[0, 0]
    Gp = np.roll(G0, shift=(-p[0], -p[1]), axis=(1, 2))
    pref = -(grid.T / grid.nk)
    bubble = pref * np.einsum(
        "ab,nxybc,nxycd,nxyda->",
        K,
        Gp,
        Kfield,
        G0,
        optimize=True,
    )
    assert abs(chi - bubble) < 1e-12


def test_noninteracting_diagonal_susceptibility_obeys_q_minus_q_relation():
    params = RubyParameters(V=0.0, ti=0.4, t1=0.2, t2=0.2)
    grid = MatsubaraGrid(nk1=3, nk2=3, nw=8, nOmega=1, T=0.09)
    h0 = build_h0(grid.kmesh(), params)
    G0 = np.linalg.inv(build_g0_inverse(h0, grid, mu=0.1))
    K = primitive_pseudospin_vertex("x_even")
    Kfield = np.broadcast_to(K, G0.shape).copy()
    p = (1, 0)
    pm = negative_q_index(p, grid)
    chip = susceptibility_matrix_finite_q(
        G0, K[None, ...], [Kfield], p, grid
    )[0, 0]
    chim = susceptibility_matrix_finite_q(
        G0, K[None, ...], [Kfield], pm, grid
    )[0, 0]
    assert abs(chip - np.conj(chim)) < 2e-11


def test_full_interacting_kernel_obeys_q_minus_q_for_diagonal_channel():
    """Integration-level check of finite-q AL routing on a physical G,W pair."""
    params = RubyParameters(V=0.04, ti=0.4, t1=0.2, t2=0.2)
    grid = MatsubaraGrid(nk1=3, nk2=3, nw=4, nOmega=1, T=0.12)
    h0 = build_h0(grid.kmesh(), params)
    G0 = np.linalg.inv(build_g0_inverse(h0, grid, mu=0.05))
    Vq = build_interaction(grid.qmesh(), params)
    P = compute_polarization(G0, grid, backend="fft")
    W = compute_screened_interaction(P, Vq, grid)
    K = primitive_pseudospin_vertex("x_even")
    p = (1, 0)
    pm = negative_q_index(p, grid)
    opts = VertexOptions(
        max_iter=60,
        tol=2e-10,
        solver="gmres",
        gmres_restart=10,
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=True,
        verbose=False,
        momentum_backend="fft",
    )
    vp = solve_vertex_finite_q(G0, W, Vq, K, p, grid, opts)
    vm = solve_vertex_finite_q(G0, W, Vq, K, pm, grid, opts)
    assert vp.converged
    assert vm.converged
    chip = susceptibility_matrix_finite_q(
        G0, K[None, ...], [vp.Gamma], p, grid
    )[0, 0]
    chim = susceptibility_matrix_finite_q(
        G0, K[None, ...], [vm.Gamma], pm, grid
    )[0, 0]
    assert abs(chip - np.conj(chim)) < 5e-8

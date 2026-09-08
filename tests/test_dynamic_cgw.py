import numpy as np

from rubycgw import MatsubaraGrid, RubyParameters
from rubycgw.dynamic_cgw import (
    solve_vertex_iomega,
    susceptibility_matrix_iomega,
    vertex_corrections_iomega,
)
from rubycgw.ed_cgw_benchmark import bubble_iomega
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import build_h0, build_interaction
from rubycgw.gw import build_g0_inverse, compute_polarization, compute_screened_interaction
from rubycgw.pseudospin import primitive_pseudospin_vertex
from rubycgw.supercell_cgw import SupercellVertexOptions, vertex_corrections_q0


def _random_complex(rng, shape, scale=1.0):
    return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def test_external_m_zero_matches_static_q0_kernel():
    rng = np.random.default_rng(20260908)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=3, nOmega=2, T=0.13)
    norb = 4
    G = _random_complex(rng, (grid.nf, grid.nk1, grid.nk2, norb, norb), 0.10)
    W = _random_complex(rng, (grid.nb, grid.nk1, grid.nk2, norb, norb), 0.08)
    Vq = _random_complex(rng, (grid.nk1, grid.nk2, norb, norb), 0.05)
    Gamma = _random_complex(rng, G.shape, 0.07)

    for backend in ("direct", "fft"):
        opts = SupercellVertexOptions(
            include_hartree=True,
            include_fock=True,
            include_mt=True,
            include_al=True,
            verbose=False,
            momentum_backend=backend,
        )
        ref = vertex_corrections_q0(G, W, Vq, Gamma, grid, opts)
        got = vertex_corrections_iomega(G, W, Vq, Gamma, 0, grid, opts)
        for a, b in zip(ref, got):
            assert np.max(np.abs(a - b)) < 5e-11


def test_nonzero_external_m_fft_matches_direct():
    rng = np.random.default_rng(20260909)
    grid = MatsubaraGrid(nk1=2, nk2=3, nw=4, nOmega=2, T=0.11)
    norb = 3
    G = _random_complex(rng, (grid.nf, grid.nk1, grid.nk2, norb, norb), 0.10)
    W = _random_complex(rng, (grid.nb, grid.nk1, grid.nk2, norb, norb), 0.07)
    Vq = _random_complex(rng, (grid.nk1, grid.nk2, norb, norb), 0.04)
    Gamma = _random_complex(rng, G.shape, 0.08)

    od = SupercellVertexOptions(verbose=False, momentum_backend="direct")
    of = SupercellVertexOptions(verbose=False, momentum_backend="fft")
    direct = vertex_corrections_iomega(G, W, Vq, Gamma, 1, grid, od)
    fft = vertex_corrections_iomega(G, W, Vq, Gamma, 1, grid, of)
    for a, b in zip(direct, fft):
        assert np.max(np.abs(a - b)) < 8e-11


def test_v_zero_dynamic_vertex_matches_bubble_at_every_external_frequency():
    params = RubyParameters(V=0.0, ti=0.4, t1=0.2, t2=0.2)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=5, nOmega=2, T=0.12)
    h0 = build_h0(grid.kmesh(), params)
    G0 = np.linalg.inv(build_g0_inverse(h0, grid, mu=0.0))
    Vq = build_interaction(grid.qmesh(), params)
    W = np.zeros((grid.nb, grid.nk1, grid.nk2, 6, 6), dtype=complex)
    K = primitive_pseudospin_vertex("x_even")
    bubbles = bubble_iomega(G0, K[None, ...], grid)[:, 0, 0]

    opts = SupercellVertexOptions(
        max_iter=8,
        tol=1e-13,
        solver="gmres",
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=True,
        verbose=False,
        momentum_backend="fft",
    )
    for im, m in enumerate(grid.m_values):
        res = solve_vertex_iomega(G0, W, Vq, K, int(m), grid, opts)
        assert res.converged
        chi = susceptibility_matrix_iomega(
            G0, K[None, ...], [res.Gamma], int(m), grid
        )[0, 0]
        assert abs(chi - bubbles[im]) < 2e-12


def test_interacting_diagonal_response_obeys_positive_negative_frequency_relation():
    params = RubyParameters(V=0.04, ti=0.4, t1=0.2, t2=0.2)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=5, nOmega=2, T=0.12)
    h0 = build_h0(grid.kmesh(), params)
    G0 = np.linalg.inv(build_g0_inverse(h0, grid, mu=0.05))
    Vq = build_interaction(grid.qmesh(), params)
    P = compute_polarization(G0, grid, backend="fft")
    W = compute_screened_interaction(P, Vq, grid)
    K = primitive_pseudospin_vertex("x_even")
    opts = SupercellVertexOptions(
        max_iter=80,
        tol=3e-10,
        solver="gmres",
        gmres_restart=10,
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=True,
        verbose=False,
        momentum_backend="fft",
    )
    rp = solve_vertex_iomega(G0, W, Vq, K, +1, grid, opts)
    rm = solve_vertex_iomega(G0, W, Vq, K, -1, grid, opts)
    assert rp.converged and rm.converged
    chip = susceptibility_matrix_iomega(G0, K[None, ...], [rp.Gamma], +1, grid)[0, 0]
    chim = susceptibility_matrix_iomega(G0, K[None, ...], [rm.Gamma], -1, grid)[0, 0]
    # Finite Matsubara boxes break the exact relation only by cutoff effects.
    scale = max(1.0, abs(chip), abs(chim))
    assert abs(chim - np.conj(chip)) / scale < 2e-5

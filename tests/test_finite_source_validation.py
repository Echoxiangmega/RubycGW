import numpy as np

from rubycgw.ed_cgw_benchmark import static_response_from_gammas
from rubycgw.finite_source_validation import (
    add_bilinear_source,
    bilinear_expectation,
    central_finite_difference,
    quadratic_zero_source_extrapolation,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.pseudospin import supercell_pseudospin_harmonic_vertex
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_cgw import SupercellVertexOptions, solve_vertex_q0
from rubycgw.supercell_gw import dyson_from_sigma_matrix
from rubycgw.supercell_gw_bootstrap import AndersonOptions, solve_matrix_gw_anderson


def _noninteracting_G(h0, grid, mu=0.0):
    norb = h0.shape[-1]
    sh = np.zeros((norb, norb), dtype=complex)
    sgw = np.zeros((grid.nf, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    return dyson_from_sigma_matrix(h0, grid, mu, sh, sgw)


def test_noninteracting_finite_source_has_cgw_bubble_sign_and_normalization():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=10, nOmega=1, T=0.13)
    params = RubyParameters(V=0.0, ti=0.4, t1=0.2, t2=0.2)
    h0 = build_supercell_h0(grid.kmesh(), params)
    K, _, _ = supercell_pseudospin_harmonic_vertex("z_same", "q0")
    G0 = _noninteracting_G(h0, grid, mu=0.07)
    Kfield = np.broadcast_to(K, G0.shape).copy()
    chi = static_response_from_gammas(G0, K[None, ...], [Kfield], grid)[0, 0]

    h = 2e-5
    Gp = _noninteracting_G(add_bilinear_source(h0, K, +h), grid, mu=0.07)
    Gm = _noninteracting_G(add_bilinear_source(h0, K, -h), grid, mu=0.07)
    mp = bilinear_expectation(Gp, K, grid)
    mm = bilinear_expectation(Gm, K, grid)
    fd = central_finite_difference(mp, mm, h)
    assert abs(fd - chi) < 2e-7


def test_quadratic_zero_source_extrapolation_recovers_intercept():
    h = np.array([0.04, 0.02, 0.01])
    truth = 2.3 - 0.17j
    slope = -4.2 + 0.3j
    values = truth + slope * h**2
    got, got_slope = quadratic_zero_source_extrapolation(h, values)
    assert abs(got - truth) < 1e-13
    assert abs(got_slope - slope) < 1e-11


def test_weak_interacting_cgw_matches_direct_fixed_mu_source_derivative():
    """End-to-end small-grid check of the H/F/MT/AL functional derivative."""
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=3, nOmega=1, T=0.20)
    params = RubyParameters(V=0.015, ti=0.4, t1=0.2, t2=0.2)
    h0 = build_supercell_h0(grid.kmesh(), params)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    K, _, _ = supercell_pseudospin_harmonic_vertex("z_same", "q0")

    gw_opts = GWOptions(
        mu=0.0,
        target_filling=None,
        max_iter=100,
        tol=2e-9,
        mixing=0.20,
        verbose=False,
        momentum_backend="fft",
    )
    zero = solve_matrix_gw_anderson(
        h0, Vq, grid, opts=gw_opts, initial=None, anderson=AndersonOptions()
    )
    assert zero.converged

    vopts = SupercellVertexOptions(
        max_iter=100,
        tol=2e-9,
        solver="gmres",
        gmres_restart=10,
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=True,
        verbose=False,
        momentum_backend="fft",
    )
    vertex = solve_vertex_q0(zero.G, zero.W, Vq, K, grid, opts=vopts)
    assert vertex.converged
    chi = static_response_from_gammas(
        zero.G, K[None, ...], [vertex.Gamma], grid
    )[0, 0]

    h = 7e-4
    plus = solve_matrix_gw_anderson(
        add_bilinear_source(h0, K, +h),
        Vq,
        grid,
        opts=gw_opts,
        initial=zero,
        anderson=AndersonOptions(),
    )
    minus = solve_matrix_gw_anderson(
        add_bilinear_source(h0, K, -h),
        Vq,
        grid,
        opts=gw_opts,
        initial=zero,
        anderson=AndersonOptions(),
    )
    assert plus.converged
    assert minus.converged
    fd = central_finite_difference(
        bilinear_expectation(plus.G, K, grid),
        bilinear_expectation(minus.G, K, grid),
        h,
    )
    rel = abs(fd - chi) / max(abs(chi), 1e-12)
    assert rel < 5e-3

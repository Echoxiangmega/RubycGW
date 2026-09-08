import numpy as np

from rubycgw.dynamic_cgw import DynamicVertexOptions
from rubycgw.finite_q_cgw import FiniteQVertexOptions
from rubycgw.finite_source_validation import (
    add_bilinear_source,
    bilinear_expectation,
    central_finite_difference,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.production_cgw import solve_vertex_q0_tail
from rubycgw.production_dynamic_cgw import solve_vertex_iomega_tail
from rubycgw.production_finite_q_cgw import solve_vertex_finite_q_tail
from rubycgw.pseudospin import supercell_pseudospin_harmonic_vertex
from rubycgw.response_tail import build_tail_reference
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw_bootstrap import AndersonOptions, solve_matrix_gw_anderson
from rubycgw.ed_cgw_benchmark import static_response_from_gammas


def _zero_background(V=0.04, nw=4, T=0.20):
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=nw, nOmega=1, T=T)
    params = RubyParameters(V=V, ti=0.4, t1=0.2, t2=0.2)
    h0 = build_supercell_h0(grid.kmesh(), params)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    opts = GWOptions(
        mu=0.0,
        target_filling=None,
        max_iter=180,
        tol=2e-10,
        mixing=0.18,
        verbose=False,
        momentum_backend="fft",
    )
    zero = solve_matrix_gw_anderson(
        h0, Vq, grid, opts=opts, initial=None, anderson=AndersonOptions()
    )
    assert zero.converged
    ref = build_tail_reference(h0, zero.mu, zero.Sigma_H, grid)
    return grid, params, h0, Vq, opts, zero, ref


def test_tail_static_cgw_matches_production_fixed_mu_finite_source():
    grid, params, h0, Vq, gw_opts, zero, ref = _zero_background()
    K, _, _ = supercell_pseudospin_harmonic_vertex("z_same", "q0")
    vopts = SupercellVertexOptions(
        max_iter=140,
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
    vertex = solve_vertex_q0_tail(
        zero.G, zero.W, Vq, K, grid, reference=ref, opts=vopts
    )
    assert vertex.converged
    chi = static_response_from_gammas(
        zero.G, K[None, ...], [vertex.Gamma], grid
    )[0, 0]

    h = 2.0e-4
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
    assert plus.converged and minus.converged
    fd = central_finite_difference(
        bilinear_expectation(plus.G, K, grid),
        bilinear_expectation(minus.G, K, grid),
        h,
    )
    rel = abs(fd - chi) / max(abs(chi), 1e-12)
    assert rel < 3e-4


def test_tail_dynamic_m0_reduces_to_tail_static():
    grid, params, h0, Vq, gw_opts, zero, ref = _zero_background(V=0.025, nw=3)
    K, _, _ = supercell_pseudospin_harmonic_vertex("z_opposite", "q0")
    sopts = SupercellVertexOptions(
        max_iter=120,
        tol=3e-10,
        solver="gmres",
        gmres_restart=10,
        verbose=False,
        momentum_backend="fft",
    )
    dopts = DynamicVertexOptions(**sopts.__dict__)
    a = solve_vertex_q0_tail(zero.G, zero.W, Vq, K, grid, ref, opts=sopts)
    b = solve_vertex_iomega_tail(zero.G, zero.W, Vq, K, 0, grid, ref, opts=dopts)
    assert a.converged and b.converged
    assert np.max(np.abs(a.Gamma - b.Gamma)) < 3e-8
    assert np.max(np.abs(a.Gamma_H - b.Gamma_H)) < 3e-8
    assert np.max(np.abs(a.Gamma_F - b.Gamma_F)) < 3e-8


def test_tail_finite_q_q0_reduces_to_tail_static():
    grid, params, h0, Vq, gw_opts, zero, ref = _zero_background(V=0.02, nw=3)
    K, _, _ = supercell_pseudospin_harmonic_vertex("z_same", "q0")
    sopts = SupercellVertexOptions(
        max_iter=120,
        tol=3e-10,
        solver="gmres",
        gmres_restart=10,
        verbose=False,
        momentum_backend="fft",
    )
    qopts = FiniteQVertexOptions(**sopts.__dict__)
    a = solve_vertex_q0_tail(zero.G, zero.W, Vq, K, grid, ref, opts=sopts)
    b = solve_vertex_finite_q_tail(
        zero.G, zero.W, Vq, K, (0, 0), grid, ref, opts=qopts
    )
    assert a.converged and b.converged
    assert np.max(np.abs(a.Gamma - b.Gamma)) < 3e-8

import numpy as np

from rubycgw import (
    GWOptions,
    MatsubaraGrid,
    RubyParameters,
    VertexOptions,
    build_h0,
    build_interaction,
    primitive_pseudospin_vertex,
    rebuild_primitive_fixed_point,
    solve_gw,
    solve_vertex_q0,
)
from rubycgw.gw import build_g0_inverse
from rubycgw.primitive_gw import compute_sigma_gw_split_components


def test_public_primitive_gw_uses_production_split_solver_at_v_zero():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=4, nOmega=2, T=0.1)
    opts = GWOptions(
        target_filling=2.0,
        max_iter=3,
        tol=1e-12,
        mixing=0.5,
        verbose=False,
    )
    result = solve_gw(params, grid, opts)
    assert result.converged
    assert result.final_error < 1e-12
    assert np.max(np.abs(result.Sigma_H)) < 1e-13
    assert np.max(np.abs(result.Sigma_GW)) < 1e-13

    rebuilt = rebuild_primitive_fixed_point(result, params, grid)
    assert float(rebuilt["residual"]) < 1e-12
    assert abs(np.sum(rebuilt["density"]) - 2.0) < 1e-8


def test_split_self_energy_is_static_fock_when_w_equals_v():
    params = RubyParameters(V=0.3)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=4, nOmega=2, T=0.12)
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_interaction(grid.qmesh(), params)
    G = np.linalg.inv(build_g0_inverse(h0, grid, mu=0.0))
    sigma_h = np.zeros((6, 6), dtype=complex)

    # If W is exactly its instantaneous high-frequency limit V, then W-V=0.
    W = np.broadcast_to(Vq[None, :, :, :, :],
                        (grid.nb, grid.nk1, grid.nk2, 6, 6)).copy()
    total, sigma_f, sigma_c, _ = compute_sigma_gw_split_components(
        G, W, Vq, grid, h0, 0.0, sigma_h, backend="fft"
    )

    assert np.max(np.abs(sigma_c)) < 1e-13
    assert np.max(np.abs(total - sigma_f[None, :, :, :, :])) < 1e-13


def test_modern_primitive_cgw_gmres_and_legacy_vq0_input_at_v_zero():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=5, nOmega=2, T=0.1)
    h0 = build_h0(grid.kmesh(), params)
    G0 = np.linalg.inv(build_g0_inverse(h0, grid, mu=0.0))
    Vq = build_interaction(grid.qmesh(), params)
    W = np.zeros((grid.nb, grid.nk1, grid.nk2, 6, 6), dtype=complex)
    K = primitive_pseudospin_vertex("x_even")
    opts = VertexOptions(
        solver="gmres",
        max_iter=8,
        gmres_restart=4,
        tol=1e-13,
        verbose=False,
    )

    full = solve_vertex_q0(G0, W, Vq, K, grid, opts)
    old_input = solve_vertex_q0(G0, W, Vq[0, 0], K, grid, opts)
    Kfield = np.broadcast_to(K, G0.shape)

    assert full.converged
    assert full.solver == "gmres"
    assert full.final_error < 1e-13
    assert np.max(np.abs(full.Gamma - Kfield)) < 1e-13
    assert np.max(np.abs(full.Gamma_F)) < 1e-13
    assert np.max(np.abs(old_input.Gamma - full.Gamma)) < 1e-13


def test_vertex_options_expose_production_fock_and_gmres_controls():
    opts = VertexOptions()
    assert opts.solver == "gmres"
    assert opts.include_fock
    assert opts.include_mt
    assert opts.include_al
    assert opts.gmres_restart > 0

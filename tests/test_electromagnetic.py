import numpy as np

from rubycgw.electromagnetic import (
    ElectromagneticBackground,
    build_supercell_h0_peierls,
    compare_covariant_to_finite_difference,
    finite_difference_electromagnetic_response,
    peierls_flux_vertex,
    solve_electromagnetic_response,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    dyson_from_sigma_matrix,
)
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast


def _background_from_result(params, grid, h0, Vq, result):
    return ElectromagneticBackground(
        metadata={
            "V": float(params.V),
            "primitive_filling": float(np.sum(result.density) / 3.0),
            "T": float(grid.T),
        },
        grid=grid,
        params=params,
        h0=h0,
        Vq=Vq,
        G=result.G,
        P=result.P,
        W=result.W,
        Sigma_H=result.Sigma_H,
        Sigma_GW=result.Sigma_GW,
        density=result.density,
        mu=result.mu,
    )


def test_peierls_vertex_matches_hamiltonian_central_difference():
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=2, nOmega=1, T=0.2)
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.15, V=0.1)
    delta = 1e-6

    for channel in ("A", "B", "opposite", "same"):
        hp = build_supercell_h0_peierls(grid.kmesh(), params, +delta, channel)
        hm = build_supercell_h0_peierls(grid.kmesh(), params, -delta, channel)
        fd = (hp - hm) / (2.0 * delta)
        K = peierls_flux_vertex(params, channel)
        target = np.broadcast_to(K, fd.shape)
        assert np.max(np.abs(fd - target)) < 2e-11


def test_noninteracting_covariant_response_matches_full_finite_difference():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=10, nOmega=2, T=0.25)
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.15, V=0.0)
    h0 = build_supercell_h0(grid.kmesh(), params)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    sigma_h = np.zeros((18, 18), dtype=complex)
    sigma_gw = np.zeros((grid.nf, 1, 1, 18, 18), dtype=complex)
    mu = 0.07
    G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
    P = compute_polarization_matrix(G, grid, backend="direct")
    W = compute_screened_interaction_matrix(P, Vq)
    density = 0.5 + (grid.T / grid.nk) * np.sum(
        np.diagonal(G, axis1=-2, axis2=-1), axis=(0, 1, 2)
    ).real

    background = ElectromagneticBackground(
        metadata={"V": 0.0, "primitive_filling": float(np.sum(density) / 3.0), "T": grid.T},
        grid=grid,
        params=params,
        h0=h0,
        Vq=Vq,
        G=G,
        P=P,
        W=W,
        Sigma_H=sigma_h,
        Sigma_GW=sigma_gw,
        density=density,
        mu=mu,
    )
    vopts = SupercellVertexOptions(
        max_iter=20,
        tol=1e-12,
        solver="gmres",
        gmres_restart=4,
        verbose=False,
        momentum_backend="direct",
    )
    analytic = solve_electromagnetic_response(
        background, "same", fixed_filling=False, vertex_options=vopts
    )
    fd_opts = GWOptions(
        mu=mu,
        target_filling=None,
        max_iter=10,
        tol=1e-12,
        mixing=0.5,
        mixing_method="linear",
        verbose=False,
        momentum_backend="direct",
    )
    fd = finite_difference_electromagnetic_response(
        background,
        "same",
        2e-5,
        fixed_filling=False,
        gw_options=fd_opts,
    )
    metrics = compare_covariant_to_finite_difference(analytic, fd)

    assert analytic.vertex_converged
    assert metrics["G"]["rel_max"] < 2e-8
    assert metrics["Gamma"]["rel_max"] < 2e-10


def test_interacting_covariant_response_tracks_self_consistent_finite_difference():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=20, nOmega=3, T=0.30)
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.15, V=0.02)
    h0 = build_supercell_h0(grid.kmesh(), params)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    opts = GWOptions(
        mu=0.05,
        target_filling=None,
        max_iter=120,
        tol=3e-10,
        mixing=0.35,
        mixing_method="pulay",
        pulay_history=6,
        pulay_start=3,
        mu_tol=1e-10,
        mu_max_iter=40,
        verbose=False,
        momentum_backend="direct",
    )
    gw = solve_matrix_gw_fast(h0, Vq, grid, opts=opts)
    assert gw.converged

    background = _background_from_result(params, grid, h0, Vq, gw)
    vopts = SupercellVertexOptions(
        max_iter=80,
        tol=2e-10,
        solver="gmres",
        gmres_restart=8,
        verbose=False,
        momentum_backend="direct",
    )
    analytic = solve_electromagnetic_response(
        background, "opposite", fixed_filling=False, vertex_options=vopts
    )
    assert analytic.vertex_converged

    fd = finite_difference_electromagnetic_response(
        background,
        "opposite",
        2e-4,
        fixed_filling=False,
        gw_options=opts,
    )
    metrics = compare_covariant_to_finite_difference(analytic, fd)

    assert metrics["G"]["rel_max"] < 3e-2
    assert metrics["Sigma_H"]["rel_max"] < 3e-2
    assert metrics["Sigma_GW"]["rel_max"] < 8e-2
    assert metrics["W"]["rel_max"] < 5e-2

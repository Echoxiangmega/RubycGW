import numpy as np

from rubycgw.ed_cgw_benchmark import static_response_from_gammas
from rubycgw.finite_box_gw import (
    density_from_G_finite_box,
    one_body_density_matrix_finite_box,
    solve_matrix_gw_finite_box,
)
from rubycgw.finite_source_validation import (
    add_bilinear_source,
    bilinear_expectation,
    central_finite_difference,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.pseudospin import supercell_pseudospin_harmonic_vertex
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_cgw import SupercellVertexOptions, solve_vertex_q0


def test_finite_box_density_matches_rho_diagonal_average():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=2, nOmega=1, T=0.2)
    rng = np.random.default_rng(3)
    G = rng.normal(size=(grid.nf, 1, 1, 4, 4)) + 1j * rng.normal(
        size=(grid.nf, 1, 1, 4, 4)
    )
    # Enforce the +/- Matsubara Hermiticity needed by an equilibrium Green function.
    G = 0.5 * (G + np.swapaxes(G[::-1].conj(), -1, -2))
    density = density_from_G_finite_box(G, grid)
    rho = one_body_density_matrix_finite_box(G, grid)
    assert np.max(np.abs(density - np.diagonal(rho[0, 0]).real)) < 1e-12


def test_finite_box_scgw_derivative_matches_static_cgw():
    """Exact discrete-map check: same finite box on both sides of FDT."""
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=3, nOmega=1, T=0.22)
    params = RubyParameters(V=0.025, ti=0.4, t1=0.2, t2=0.2)
    h0 = build_supercell_h0(grid.kmesh(), params)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    K, _, _ = supercell_pseudospin_harmonic_vertex("z_same", "q0")

    gopts = GWOptions(
        mu=0.03,
        target_filling=None,
        max_iter=180,
        tol=2e-10,
        mixing=0.20,
        mixing_method="linear",
        verbose=False,
        momentum_backend="fft",
    )
    zero = solve_matrix_gw_finite_box(h0, Vq, grid, opts=gopts)
    assert zero.converged

    vopts = SupercellVertexOptions(
        max_iter=120,
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
    vertex = solve_vertex_q0(zero.G, zero.W, Vq, K, grid, opts=vopts)
    assert vertex.converged
    chi = static_response_from_gammas(
        zero.G, K[None, ...], [vertex.Gamma], grid
    )[0, 0]

    h = 2e-4
    plus = solve_matrix_gw_finite_box(
        add_bilinear_source(h0, K, +h), Vq, grid, opts=gopts, initial=zero
    )
    minus = solve_matrix_gw_finite_box(
        add_bilinear_source(h0, K, -h), Vq, grid, opts=gopts, initial=zero
    )
    assert plus.converged and minus.converged
    fd = central_finite_difference(
        bilinear_expectation(plus.G, K, grid),
        bilinear_expectation(minus.G, K, grid),
        h,
    )
    rel = abs(fd - chi) / max(abs(chi), 1e-12)
    assert rel < 2e-4

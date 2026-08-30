import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_gw import dyson_from_sigma_matrix
from rubycgw.supercell_gw_split import (
    compute_sigma_gw_split_components,
    compute_static_fock_matrix_direct,
    compute_static_fock_matrix_fft,
    one_body_density_matrix_tail,
)


def test_static_fock_fft_matches_direct():
    rng = np.random.default_rng(123)
    grid = MatsubaraGrid(nk1=2, nk2=3, nw=2, nOmega=1, T=0.1)
    norb = 4
    x = rng.normal(size=(grid.nk1, grid.nk2, norb, norb)) + 1j * rng.normal(
        size=(grid.nk1, grid.nk2, norb, norb)
    )
    rho = 0.5 * (x + np.swapaxes(x.conj(), -1, -2))
    y = rng.normal(size=(grid.nk1, grid.nk2, norb, norb)) + 1j * rng.normal(
        size=(grid.nk1, grid.nk2, norb, norb)
    )
    Vq = 0.5 * (y + np.swapaxes(y.conj(), -1, -2))

    direct = compute_static_fock_matrix_direct(rho, Vq, grid)
    fft = compute_static_fock_matrix_fft(rho, Vq, grid)
    assert np.max(np.abs(direct - fft)) < 1e-12


def test_tail_density_matrix_reproduces_noninteracting_density():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=2, nk2=1, nw=5, nOmega=1, T=0.1)
    h0 = build_supercell_h0(grid.kmesh(), params)
    sigma_h = np.zeros((18, 18), dtype=complex)
    sigma_gw = np.zeros((grid.nf, grid.nk1, grid.nk2, 18, 18), dtype=complex)
    mu = 0.13
    G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)

    rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    evals, evecs = np.linalg.eigh(h0)
    occ = 1.0 / (np.exp((evals - mu) / grid.T) + 1.0)
    exact = np.einsum("xyaj,xyj,xybj->xyab", evecs, occ, evecs.conj(), optimize=True)
    assert np.max(np.abs(rho - exact)) < 1e-12


def test_split_map_has_static_fock_plus_dynamic_wc():
    rng = np.random.default_rng(7)
    params = RubyParameters(V=0.2)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=3, nOmega=2, T=0.1)
    h0 = build_supercell_h0(grid.kmesh(), params)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    sigma_h = np.zeros((18, 18), dtype=complex)
    sigma_gw = np.zeros((grid.nf, 1, 1, 18, 18), dtype=complex)
    G = dyson_from_sigma_matrix(h0, grid, 0.0, sigma_h, sigma_gw)
    W = np.broadcast_to(Vq[None], (grid.nb,) + Vq.shape).copy()
    W += 1e-3 * (
        rng.normal(size=W.shape) + 1j * rng.normal(size=W.shape)
    )

    total, sigma_f, sigma_c, _ = compute_sigma_gw_split_components(
        G, W, Vq, grid, h0, 0.0, sigma_h, backend="fft"
    )
    assert total.shape == G.shape
    assert sigma_f.shape == (1, 1, 18, 18)
    assert sigma_c.shape == G.shape
    assert np.max(np.abs(total - sigma_c - sigma_f[None])) < 1e-12

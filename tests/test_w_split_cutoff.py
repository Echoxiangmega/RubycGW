import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.supercell_gw import compute_sigma_gw_matrix


def test_sigma_map_is_linear_under_w_equals_v_plus_wc():
    rng = np.random.default_rng(1234)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=2, nOmega=1, T=0.1)
    norb = 3
    G = rng.normal(size=(grid.nf, 1, 1, norb, norb)) + 1j * rng.normal(
        size=(grid.nf, 1, 1, norb, norb)
    )
    Vq = rng.normal(size=(1, 1, norb, norb))
    Vinst = np.broadcast_to(Vq[None, :, :, :, :], (grid.nb, 1, 1, norb, norb))
    Wc = rng.normal(size=Vinst.shape) + 1j * rng.normal(size=Vinst.shape)
    W = Vinst + Wc

    full = compute_sigma_gw_matrix(G, W, grid, backend="fft")
    bare = compute_sigma_gw_matrix(G, Vinst, grid, backend="fft")
    corr = compute_sigma_gw_matrix(G, Wc, grid, backend="fft")

    assert np.max(np.abs(full - bare - corr)) < 1e-12

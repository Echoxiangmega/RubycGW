import numpy as np

from rubycgw import MatsubaraGrid
from rubycgw.gw import (
    compute_polarization_direct,
    compute_polarization_fft,
    compute_sigma_gw_direct,
    compute_sigma_gw_fft,
)
from rubycgw.cgw import gamma_mt_q0, gamma_al_q0


def _random_complex(rng, shape, scale=1.0):
    return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def test_fft_polarization_matches_direct():
    rng = np.random.default_rng(1234)
    grid = MatsubaraGrid(nk1=3, nk2=2, nw=3, nOmega=1, T=0.17)
    shape_g = (grid.nf, grid.nk1, grid.nk2, 6, 6)
    G = _random_complex(rng, shape_g, scale=0.2)

    direct = compute_polarization_direct(G, grid)
    fft = compute_polarization_fft(G, grid)

    assert np.max(np.abs(direct - fft)) < 1e-11


def test_fft_sigma_matches_direct():
    rng = np.random.default_rng(5678)
    grid = MatsubaraGrid(nk1=3, nk2=2, nw=3, nOmega=1, T=0.13)
    shape_g = (grid.nf, grid.nk1, grid.nk2, 6, 6)
    shape_w = (grid.nb, grid.nk1, grid.nk2, 6, 6)
    G = _random_complex(rng, shape_g, scale=0.2)
    W = _random_complex(rng, shape_w, scale=0.1)

    direct = compute_sigma_gw_direct(G, W, grid)
    fft = compute_sigma_gw_fft(G, W, grid)

    assert np.max(np.abs(direct - fft)) < 1e-11


def test_fft_mt_and_al_match_direct():
    rng = np.random.default_rng(9012)
    grid = MatsubaraGrid(nk1=3, nk2=2, nw=3, nOmega=1, T=0.11)
    shape_g = (grid.nf, grid.nk1, grid.nk2, 6, 6)
    shape_w = (grid.nb, grid.nk1, grid.nk2, 6, 6)
    G = _random_complex(rng, shape_g, scale=0.15)
    W = _random_complex(rng, shape_w, scale=0.08)
    Gamma = _random_complex(rng, shape_g, scale=0.12)

    mt_direct = gamma_mt_q0(G, W, Gamma, grid, backend="direct")
    mt_fft = gamma_mt_q0(G, W, Gamma, grid, backend="fft")
    assert np.max(np.abs(mt_direct - mt_fft)) < 1e-11

    al1_direct, al2_direct = gamma_al_q0(G, W, Gamma, grid, backend="direct")
    al1_fft, al2_fft = gamma_al_q0(G, W, Gamma, grid, backend="fft")
    assert np.max(np.abs(al1_direct - al1_fft)) < 1e-11
    assert np.max(np.abs(al2_direct - al2_fft)) < 1e-11

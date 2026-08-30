import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.supercell_gw_periodic_pulay import _gw_residual_mode_diagnostics


def test_uniform_real_identity_residual_is_fully_scalar():
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=3, nOmega=1, T=0.1)
    norb = 4
    c = 2.5e-5
    res = np.zeros((grid.nf, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    res += c * np.eye(norb)[None, None, None, :, :]

    d = _gw_residual_mode_diagnostics(res, grid)
    assert np.isclose(d["raw_max"], c)
    assert d["projected_max"] < 1e-14
    assert np.isclose(d["scalar_shift"], c)
    assert np.isclose(d["scalar_fraction"], 1.0)


def test_edge_residual_is_identified():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=4, nOmega=1, T=0.1)
    norb = 3
    res = np.zeros((grid.nf, 1, 1, norb, norb), dtype=complex)
    iedge = int(np.argmax(np.abs(grid.omega)))
    res[iedge, 0, 0, 1, 2] = 7e-5 + 2e-5j

    d = _gw_residual_mode_diagnostics(res, grid)
    assert np.isclose(d["edge_max"], abs(7e-5 + 2e-5j))
    assert d["max_iw"] == iedge
    assert d["max_a"] == 1
    assert d["max_b"] == 2
    assert d["scalar_fraction"] == 0.0

import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_hf import solve_supercell_hf_seed


def test_supercell_hf_seed_v0_is_noninteracting():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=3, nOmega=1, T=0.2)
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0)
    h0 = build_supercell_h0(grid.kmesh(), params)
    Vq = build_supercell_interaction(grid.qmesh(), params)

    result = solve_supercell_hf_seed(
        h0,
        Vq,
        grid,
        target_filling=6.0,
        max_iter=20,
        tol=1e-12,
        mixing=0.5,
        mu_tol=1e-12,
    )

    assert result.converged
    assert abs(np.sum(result.density) - 6.0) < 1e-10
    assert np.max(np.abs(result.Sigma_H)) < 1e-13
    assert np.max(np.abs(result.Sigma_F)) < 1e-13
    assert result.seed.Sigma_GW.shape == (
        grid.nf,
        grid.nk1,
        grid.nk2,
        18,
        18,
    )


def test_supercell_hf_seed_static_fock_is_hermitian_and_broadcast():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=3, nOmega=1, T=0.2)
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.05)
    h0 = build_supercell_h0(grid.kmesh(), params)
    Vq = build_supercell_interaction(grid.qmesh(), params)

    result = solve_supercell_hf_seed(
        h0,
        Vq,
        grid,
        target_filling=6.0,
        max_iter=250,
        tol=1e-9,
        mixing=0.3,
        mu_tol=1e-12,
    )

    assert result.converged
    assert abs(np.sum(result.density) - 6.0) < 1e-9
    assert np.max(
        np.abs(result.Sigma_F - np.swapaxes(result.Sigma_F.conj(), -1, -2))
    ) < 1e-11
    for n in range(1, grid.nf):
        assert np.max(np.abs(result.seed.Sigma_GW[n] - result.seed.Sigma_GW[0])) < 1e-13

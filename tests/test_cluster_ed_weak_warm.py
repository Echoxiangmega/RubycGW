from pathlib import Path

import numpy as np

from rubycgw.cluster_ed_weak_warm import (
    _initial_from_warm,
    load_cluster_ed_warm_start,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions, GWResult
from run_cluster_ed_weak_warm import _pop_warm_start


def _background(grid, sigma_value=2.0):
    dyn = np.full((grid.nf, grid.nk1, grid.nk2, 6, 6), sigma_value, dtype=complex)
    bos = np.zeros((grid.nb, grid.nk1, grid.nk2, 6, 6), dtype=complex)
    return GWResult(
        G=np.zeros_like(dyn),
        W=bos.copy(),
        P=bos.copy(),
        Sigma_H=np.zeros((6, 6), dtype=complex),
        Sigma_GW=dyn,
        mu=0.0,
        density=np.zeros(6),
        converged=True,
        iterations=1,
        final_error=0.0,
        mixing_method="pulay",
        min_screening_singular_value=1.0,
        min_screening_m=0,
        min_screening_Omega=0.0,
        min_screening_q1=0.0,
        min_screening_q2=0.0,
        min_screening_mode=np.zeros(6, dtype=complex),
        min_density_mode=np.zeros(6, dtype=complex),
        min_density_mode_residual=0.0,
    )


def _write_old(path: Path, old_grid: MatsubaraGrid, *, V=1.0, weak="gw"):
    sigma_imp = np.full((old_grid.nf, 6, 6), 5.0, dtype=complex)
    sigma_cweak = np.full_like(sigma_imp, 3.0)
    sigma_emb = np.full(
        (old_grid.nf, old_grid.nk1, old_grid.nk2, 6, 6), 7.0, dtype=complex
    )
    G = np.full_like(sigma_emb, 0.25)
    np.savez_compressed(
        path,
        Lx=old_grid.nk1,
        Ly=old_grid.nk2,
        nk1=old_grid.nk1,
        nk2=old_grid.nk2,
        weak_solver=weak,
        V=V,
        filling=2.0,
        T=old_grid.T,
        ti=0.4,
        t1=0.2,
        t2=0.2,
        omega=old_grid.omega,
        Sigma_H=np.eye(6) * 0.1,
        Sigma_ED_cluster=sigma_imp,
        Sigma_weak_cluster=sigma_cweak,
        Sigma_emb=sigma_emb,
        G=G,
        mu=0.12,
        bath_energies=np.linspace(-1, 1, 6),
        bath_couplings=np.ones((6, 6)) * 0.1,
        bath_fit_error=0.2,
    )


def test_warm_arg_is_removed_before_base_driver_parsing():
    argv = ["prog", "--Lx", "4", "--warm-start", "old.npz", "--V", "2"]
    got = _pop_warm_start(argv)
    assert got == Path("old.npz")
    assert argv == ["prog", "--Lx", "4", "--V", "2"]


def test_cross_mesh_transplants_cluster_correction(tmp_path):
    old_grid = MatsubaraGrid(nk1=2, nk2=1, nw=2, nOmega=1, T=0.08)
    new_grid = MatsubaraGrid(nk1=4, nk2=4, nw=2, nOmega=1, T=0.08)
    path = tmp_path / "old.npz"
    _write_old(path, old_grid)
    warm = load_cluster_ed_warm_start(
        path,
        weak_solver="gw",
        grid=new_grid,
        filling=2.0,
        T=0.08,
        ti=0.4,
        t1=0.2,
        t2=0.2,
        V=1.0,
        nbath=6,
    )
    np.testing.assert_allclose(warm.DeltaSigma_cluster, 2.0)
    bg = _background(new_grid, sigma_value=2.0)
    h0 = np.zeros((4, 4, 6, 6), dtype=complex)
    _, sigma_emb, sigma_imp, G, mu, bath, mode = _initial_from_warm(
        warm,
        bg,
        h0,
        new_grid,
        GWOptions(target_filling=None),
        1.0,
    )
    assert mode == "cluster-correction"
    np.testing.assert_allclose(sigma_emb, 4.0)
    np.testing.assert_allclose(sigma_imp, 5.0)
    assert G.shape == (new_grid.nf, 4, 4, 6, 6)
    assert mu == 0.12
    assert bath is not None


def test_same_mesh_same_v_reuses_full_saved_state(tmp_path):
    grid = MatsubaraGrid(nk1=2, nk2=1, nw=2, nOmega=1, T=0.08)
    path = tmp_path / "old.npz"
    _write_old(path, grid)
    warm = load_cluster_ed_warm_start(
        path,
        weak_solver="gw",
        grid=grid,
        filling=2.0,
        T=0.08,
        ti=0.4,
        t1=0.2,
        t2=0.2,
        V=1.0,
        nbath=6,
    )
    bg = _background(grid, sigma_value=2.0)
    h0 = np.zeros((2, 1, 6, 6), dtype=complex)
    _, sigma_emb, _, G, _, _, mode = _initial_from_warm(
        warm,
        bg,
        h0,
        grid,
        GWOptions(target_filling=None),
        1.0,
    )
    assert mode == "full-state"
    np.testing.assert_allclose(sigma_emb, 7.0)
    np.testing.assert_allclose(G, 0.25)


def test_v_continuation_uses_cluster_correction_not_full_state(tmp_path):
    grid = MatsubaraGrid(nk1=2, nk2=1, nw=2, nOmega=1, T=0.08)
    path = tmp_path / "old.npz"
    _write_old(path, grid, V=1.0)
    warm = load_cluster_ed_warm_start(
        path,
        weak_solver="gw",
        grid=grid,
        filling=2.0,
        T=0.08,
        ti=0.4,
        t1=0.2,
        t2=0.2,
        V=1.2,
        nbath=6,
    )
    bg = _background(grid, sigma_value=2.0)
    h0 = np.zeros((2, 1, 6, 6), dtype=complex)
    _, sigma_emb, _, _, _, _, mode = _initial_from_warm(
        warm,
        bg,
        h0,
        grid,
        GWOptions(target_filling=None),
        1.2,
    )
    assert mode == "cluster-correction"
    np.testing.assert_allclose(sigma_emb, 4.0)

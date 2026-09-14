from pathlib import Path

import numpy as np
import pytest

from rubycgw.cluster_restart import load_cluster_ed_gw_restart
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters


def _write_checkpoint(path: Path, grid: MatsubaraGrid, *, nbath: int = 2, V: float = 1.0):
    shape = (grid.nf, grid.nk1, grid.nk2, 6, 6)
    local = (grid.nf, 6, 6)
    np.savez_compressed(
        path,
        Lx=grid.nk1,
        Ly=grid.nk2,
        V=V,
        filling=2.0,
        T=grid.T,
        ti=0.4,
        t1=0.2,
        t2=0.2,
        omega=grid.omega,
        Omega=grid.Omega,
        G=np.ones(shape, dtype=complex),
        Sigma_H=np.eye(6, dtype=complex),
        Sigma_emb=2.0 * np.ones(shape, dtype=complex),
        Sigma_ED_cluster=3.0 * np.ones(local, dtype=complex),
        mu=0.123,
        bath_energies=np.linspace(-1.0, 1.0, nbath),
        bath_couplings=np.full((6, nbath), 0.2 + 0.1j, dtype=complex),
    )


def test_load_cluster_restart_roundtrip(tmp_path):
    grid = MatsubaraGrid(nk1=2, nk2=1, nw=5, nOmega=3, T=0.08)
    path = tmp_path / "restart.npz"
    _write_checkpoint(path, grid)
    state = load_cluster_ed_gw_restart(
        path,
        Lx=2,
        Ly=1,
        filling=2.0,
        T=0.08,
        params=RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=1.0),
        grid=grid,
        nbath=2,
    )
    assert state.G.shape == (grid.nf, 2, 1, 6, 6)
    assert state.Sigma_imp.shape == (grid.nf, 6, 6)
    assert state.bath.couplings.shape == (6, 2)
    assert np.isclose(state.mu, 0.123)
    assert state.source_path == str(path)


def test_load_cluster_restart_rejects_changed_physics(tmp_path):
    grid = MatsubaraGrid(nk1=2, nk2=1, nw=5, nOmega=3, T=0.08)
    path = tmp_path / "restart.npz"
    _write_checkpoint(path, grid)
    with pytest.raises(ValueError, match="V mismatch"):
        load_cluster_ed_gw_restart(
            path,
            Lx=2,
            Ly=1,
            filling=2.0,
            T=0.08,
            params=RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=1.1),
            grid=grid,
            nbath=2,
        )


def test_load_cluster_restart_rejects_changed_bath_size(tmp_path):
    grid = MatsubaraGrid(nk1=2, nk2=1, nw=5, nOmega=3, T=0.08)
    path = tmp_path / "restart.npz"
    _write_checkpoint(path, grid, nbath=2)
    with pytest.raises(ValueError, match="bath-size mismatch"):
        load_cluster_ed_gw_restart(
            path,
            Lx=2,
            Ly=1,
            filling=2.0,
            T=0.08,
            params=RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=1.0),
            grid=grid,
            nbath=3,
        )

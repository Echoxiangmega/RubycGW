from types import SimpleNamespace

import numpy as np
import pytest

from rubycgw.cluster_restart import ClusterEDGWRestartState, load_cluster_ed_gw_restart
from rubycgw.cluster_restart_solver import _continuation_background
from rubycgw.cluster_ed_gw import BathParameters
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters


def _write_checkpoint(path, grid, *, V=1.2, filling=2.0, nbath=2):
    gshape = (grid.nf, grid.nk1, grid.nk2, 6, 6)
    ishape = (grid.nf, 6, 6)
    np.savez_compressed(
        path,
        Lx=grid.nk1,
        Ly=grid.nk2,
        V=float(V),
        ti=0.4,
        t1=0.2,
        t2=0.2,
        filling=float(filling),
        T=float(grid.T),
        omega=np.asarray(grid.omega),
        Omega=np.asarray(grid.Omega),
        G=np.ones(gshape, dtype=complex),
        Sigma_H=np.eye(6, dtype=complex) * 0.3,
        Sigma_emb=np.ones(gshape, dtype=complex) * 0.2,
        Sigma_ED_cluster=np.ones(ishape, dtype=complex) * 0.1,
        mu=0.7,
        bath_energies=np.linspace(-1.0, 1.0, nbath),
        bath_couplings=np.ones((6, nbath), dtype=complex) * 0.05,
    )


def test_strict_restart_rejects_changed_V_but_continuation_accepts(tmp_path):
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=2, nOmega=1, T=0.08)
    path = tmp_path / "state.npz"
    _write_checkpoint(path, grid, V=1.2, nbath=2)
    target = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=1.3)

    with pytest.raises(ValueError, match="restart V mismatch"):
        load_cluster_ed_gw_restart(
            path,
            Lx=2,
            Ly=2,
            filling=2.0,
            T=0.08,
            params=target,
            grid=grid,
            nbath=2,
        )

    state = load_cluster_ed_gw_restart(
        path,
        Lx=2,
        Ly=2,
        filling=2.0,
        T=0.08,
        params=target,
        grid=grid,
        nbath=2,
        allow_interaction_change=True,
    )
    assert state.source_V == pytest.approx(1.2)
    assert state.mu == pytest.approx(0.7)
    assert state.G.shape == (grid.nf, 2, 2, 6, 6)


def test_continuation_still_rejects_one_body_or_grid_changes(tmp_path):
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=2, nOmega=1, T=0.08)
    path = tmp_path / "state.npz"
    _write_checkpoint(path, grid, V=1.2, nbath=2)
    changed_hopping = RubyParameters(ti=0.41, t1=0.2, t2=0.2, V=2.0)

    with pytest.raises(ValueError, match="restart ti mismatch"):
        load_cluster_ed_gw_restart(
            path,
            Lx=2,
            Ly=2,
            filling=2.0,
            T=0.08,
            params=changed_hopping,
            grid=grid,
            nbath=2,
            allow_interaction_change=True,
        )


def test_continuation_background_is_direct_embedded_carrier():
    grid = MatsubaraGrid(nk1=2, nk2=3, nw=2, nOmega=1, T=0.08)
    gshape = (grid.nf, grid.nk1, grid.nk2, 6, 6)
    restart = ClusterEDGWRestartState(
        G=np.ones(gshape, dtype=complex) * 3.0,
        Sigma_H=np.eye(6, dtype=complex) * 4.0,
        Sigma_emb=np.ones(gshape, dtype=complex) * 5.0,
        Sigma_imp=np.ones((grid.nf, 6, 6), dtype=complex) * 6.0,
        mu=7.0,
        bath=BathParameters(
            energies=np.asarray([-0.5, 0.5]),
            couplings=np.ones((6, 2), dtype=complex),
            fit_error=np.nan,
            nfev=0,
        ),
        source_path="old.npz",
        source_V=1.2,
    )

    bg = _continuation_background(restart, grid)
    assert bg.converged
    assert bg.iterations == 0
    assert bg.mixing_method == "parameter-continuation-seed"
    assert bg.mu == pytest.approx(7.0)
    assert np.allclose(bg.G, restart.G)
    assert np.allclose(bg.Sigma_H, restart.Sigma_H)
    assert np.allclose(bg.Sigma_GW, restart.Sigma_emb)
    assert bg.W.shape == (grid.nb, 2, 3, 6, 6)
    assert np.allclose(bg.W, 0.0)
    assert np.allclose(bg.P, 0.0)

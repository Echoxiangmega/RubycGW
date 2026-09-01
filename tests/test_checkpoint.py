from types import SimpleNamespace

import numpy as np
import pytest

from rubycgw.checkpoint import (
    checkpoint_filename,
    find_exact_compatible_checkpoint,
    find_nearest_compatible_checkpoint,
    load_supercell_checkpoint,
    read_checkpoint_metadata,
    save_supercell_checkpoint,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.supercell import NSUP


def _fake_gw(grid, mu=1.23, converged=True, final_error=1e-10):
    density = np.full(NSUP, 0.5, dtype=float)
    return SimpleNamespace(
        Sigma_H=np.eye(NSUP, dtype=complex) * 0.2,
        Sigma_GW=np.zeros(
            (grid.nf, grid.nk1, grid.nk2, NSUP, NSUP), dtype=complex
        ),
        density=density,
        mu=float(mu),
        final_error=float(final_error),
        converged=bool(converged),
    )


def test_checkpoint_roundtrip_and_V_continuation(tmp_path):
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=3, nOmega=1, T=0.1)
    params_old = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.8)
    params_new = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=1.2)
    gw = _fake_gw(grid)

    path = tmp_path / checkpoint_filename(0.8, 3.0, grid)
    save_supercell_checkpoint(path, gw, params_old, grid, 3.0, source=0.0)
    seed, meta, density = load_supercell_checkpoint(
        path, params_new, grid, 3.0
    )

    assert meta["V"] == 0.8
    assert meta["source"] == 0.0
    assert seed.mu == gw.mu
    assert seed.Sigma_H.shape == (NSUP, NSUP)
    assert seed.Sigma_GW.shape == gw.Sigma_GW.shape
    assert np.allclose(seed.Sigma_H, gw.Sigma_H)
    assert np.allclose(density, gw.density)


def test_auto_restart_picks_nearest_compatible_zero_source(tmp_path):
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=3, nOmega=1, T=0.1)
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=1.1)
    gw = _fake_gw(grid)

    for V in (0.7, 0.8, 1.0):
        p = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=V)
        path = tmp_path / checkpoint_filename(V, 3.0, grid)
        save_supercell_checkpoint(path, gw, p, grid, 3.0, source=0.0)

    chosen = find_nearest_compatible_checkpoint(
        tmp_path, 0.9, params, grid, 3.0
    )
    assert chosen is not None
    _, meta, _ = load_supercell_checkpoint(chosen, params, grid, 3.0)
    assert meta["V"] == 0.8

    chosen = find_nearest_compatible_checkpoint(
        tmp_path, 1.1, params, grid, 3.0
    )
    assert chosen is not None
    _, meta, _ = load_supercell_checkpoint(chosen, params, grid, 3.0)
    assert meta["V"] == 1.0


def test_nonconverged_checkpoint_requires_explicit_permission(tmp_path):
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=3, nOmega=1, T=0.1)
    params = RubyParameters(V=1.4)
    gw = _fake_gw(grid, converged=False, final_error=2e-5)
    path = tmp_path / checkpoint_filename(1.4, 2.0, grid)

    with pytest.raises(ValueError):
        save_supercell_checkpoint(path, gw, params, grid, 2.0)

    save_supercell_checkpoint(
        path, gw, params, grid, 2.0, allow_nonconverged=True
    )
    meta = read_checkpoint_metadata(path)
    assert meta["converged"] is False
    assert np.isclose(meta["final_error"], 2e-5)


def test_failed_checkpoint_is_same_V_only_for_automatic_restart(tmp_path):
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=3, nOmega=1, T=0.1)
    params = RubyParameters(V=1.6)

    good = _fake_gw(grid, converged=True, final_error=1e-10)
    p_good = RubyParameters(V=1.2)
    good_path = tmp_path / checkpoint_filename(1.2, 2.0, grid)
    save_supercell_checkpoint(good_path, good, p_good, grid, 2.0)

    failed = _fake_gw(grid, converged=False, final_error=3e-5)
    p_failed = RubyParameters(V=1.4)
    failed_path = tmp_path / checkpoint_filename(1.4, 2.0, grid)
    save_supercell_checkpoint(
        failed_path,
        failed,
        p_failed,
        grid,
        2.0,
        allow_nonconverged=True,
    )

    exact = find_exact_compatible_checkpoint(
        tmp_path, 1.4, params, grid, 2.0, allow_nonconverged=True
    )
    assert exact == failed_path

    # At a larger target V, ordinary continuation must ignore the failed V=1.4
    # checkpoint and fall back to the most recent converged state.
    nearest = find_nearest_compatible_checkpoint(
        tmp_path, 1.6, params, grid, 2.0
    )
    assert nearest == good_path

import json
from types import SimpleNamespace

import numpy as np

from rubycgw.checkpoint import checkpoint_filename, save_supercell_checkpoint
from rubycgw.checkpoint_temperature import (
    estimate_sigma_infinity,
    interpolate_sigma_gw_temperature,
    write_temperature_interpolated_checkpoint,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.supercell import NSUP


def _omega(nw, T):
    n = np.arange(-nw, nw, dtype=float)
    return (2.0 * n + 1.0) * np.pi * T


def test_temperature_interpolation_preserves_static_self_energy():
    nw = 4
    sigma_static = np.eye(NSUP, dtype=complex) * 0.37
    sigma = np.broadcast_to(
        sigma_static,
        (2 * nw, 2, 2, NSUP, NSUP),
    ).copy()

    out = interpolate_sigma_gw_temperature(
        sigma,
        source_T=0.05,
        target_T=0.08,
        target_nw=nw,
        tail_pairs=2,
    )

    assert out.shape == sigma.shape
    assert np.allclose(out, sigma_static[None, None, None, :, :], atol=1e-12)
    assert np.allclose(estimate_sigma_infinity(sigma, tail_pairs=2), sigma_static)


def test_temperature_interpolation_uses_inverse_frequency_tail():
    nw = 4
    source_T = 0.05
    target_T = 0.10
    omega_old = _omega(nw, source_T)
    omega_new = _omega(nw, target_T)

    sigma = np.zeros((2 * nw, 1, 1, NSUP, NSUP), dtype=complex)
    sigma_inf = 0.4
    amp = 0.3
    sigma[:, 0, 0, 0, 0] = sigma_inf + amp / (1j * omega_old)
    sigma[:, 0, 0, 1, 1] = sigma_inf

    out = interpolate_sigma_gw_temperature(
        sigma,
        source_T=source_T,
        target_T=target_T,
        target_nw=nw,
        tail_pairs=2,
    )

    # The outer target frequencies lie outside the source window, where the
    # warm-start continuation is explicitly proportional to 1/omega.
    assert np.allclose(
        out[-1, 0, 0, 0, 0],
        sigma_inf + amp / (1j * omega_new[-1]),
        atol=5e-3,
    )
    assert np.allclose(
        out[0, 0, 0, 0, 0],
        sigma_inf + amp / (1j * omega_new[0]),
        atol=5e-3,
    )
    assert np.allclose(
        out[:, 0, 0, 1, 1],
        sigma_inf,
        atol=1e-12,
    )


def test_write_temperature_checkpoint_is_target_T_warm_start(tmp_path):
    old_grid = MatsubaraGrid(nk1=2, nk2=2, nw=3, nOmega=1, T=0.05)
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=1.73)
    sigma_gw = np.zeros(
        (old_grid.nf, old_grid.nk1, old_grid.nk2, NSUP, NSUP),
        dtype=complex,
    )
    sigma_gw[..., 0, 0] = 0.25

    gw = SimpleNamespace(
        Sigma_H=np.eye(NSUP, dtype=complex) * 0.2,
        Sigma_GW=sigma_gw,
        density=np.full(NSUP, 0.5, dtype=float),
        mu=3.7,
        final_error=5e-7,
        converged=True,
    )
    source = tmp_path / checkpoint_filename(1.73, 3.0, old_grid)
    save_supercell_checkpoint(source, gw, params, old_grid, 3.0, source=0.0)

    output, meta = write_temperature_interpolated_checkpoint(
        source,
        target_T=0.06,
    )

    assert output.exists()
    assert meta["T"] == 0.06
    assert meta["nw"] == old_grid.nw
    assert meta["nk1"] == old_grid.nk1
    assert meta["nk2"] == old_grid.nk2
    assert meta["converged"] is False
    assert meta["final_error"] == 1.0
    assert meta["temperature_interpolated_warm_start"] is True

    with np.load(output, allow_pickle=False) as data:
        stored_meta = json.loads(str(data["metadata_json"].item()))
        stored_h = np.asarray(data["Sigma_H"])
        stored_gw = np.asarray(data["Sigma_GW"])
        stored_density = np.asarray(data["density"])

    assert stored_meta["T"] == 0.06
    assert stored_gw.shape == sigma_gw.shape
    assert np.allclose(stored_h, gw.Sigma_H)
    assert np.allclose(stored_density, gw.density)

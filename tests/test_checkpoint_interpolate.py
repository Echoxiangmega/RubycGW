import json
from types import SimpleNamespace

import numpy as np

from rubycgw.checkpoint import checkpoint_filename, save_supercell_checkpoint
from rubycgw.checkpoint_interpolate import (
    interpolate_sigma_gw_kmesh,
    write_interpolated_checkpoint,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.supercell import NSUP


def test_fourier_interpolation_reproduces_representable_mode():
    nf = 2
    nk1 = nk2 = 3
    k1 = np.arange(nk1, dtype=float) / nk1
    k2 = np.arange(nk2, dtype=float) / nk2
    a, b = np.meshgrid(k1, k2, indexing="ij")

    sigma = np.zeros((nf, nk1, nk2, NSUP, NSUP), dtype=complex)
    for n in range(nf):
        values = (n + 1.0) * np.exp(2j * np.pi * (a - b))
        sigma[n, :, :, 0, 1] = values

    out = interpolate_sigma_gw_kmesh(sigma, 4, 4)
    u = np.arange(4, dtype=float) / 4.0
    x, y = np.meshgrid(u, u, indexing="ij")
    expected = np.exp(2j * np.pi * (x - y))

    assert out.shape == (nf, 4, 4, NSUP, NSUP)
    assert np.allclose(out[0, :, :, 0, 1], expected, atol=1e-12)
    assert np.allclose(out[1, :, :, 0, 1], 2.0 * expected, atol=1e-12)


def test_write_interpolated_checkpoint_is_target_mesh_warm_start(tmp_path):
    old_grid = MatsubaraGrid(nk1=3, nk2=3, nw=2, nOmega=1, T=0.1)
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=1.73)

    sigma_gw = np.zeros(
        (old_grid.nf, old_grid.nk1, old_grid.nk2, NSUP, NSUP),
        dtype=complex,
    )
    k1 = np.arange(old_grid.nk1, dtype=float) / old_grid.nk1
    k2 = np.arange(old_grid.nk2, dtype=float) / old_grid.nk2
    a, b = np.meshgrid(k1, k2, indexing="ij")
    sigma_gw[:, :, :, 2, 3] = np.exp(2j * np.pi * (a + b))[None, :, :]

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

    output, meta = write_interpolated_checkpoint(source, 4, 4)
    assert output.exists()
    assert meta["nk1"] == 4
    assert meta["nk2"] == 4
    assert meta["converged"] is False
    assert meta["final_error"] == 1.0
    assert meta["interpolated_warm_start"] is True

    with np.load(output, allow_pickle=False) as data:
        stored_meta = json.loads(str(data["metadata_json"].item()))
        stored_h = np.asarray(data["Sigma_H"])
        stored_gw = np.asarray(data["Sigma_GW"])
        stored_density = np.asarray(data["density"])

    assert stored_meta["nk1"] == 4
    assert stored_meta["nk2"] == 4
    assert stored_gw.shape == (old_grid.nf, 4, 4, NSUP, NSUP)
    assert np.allclose(stored_h, gw.Sigma_H)
    assert np.allclose(stored_density, gw.density)

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from rubycgw.cluster_ed_gw import BathParameters
from rubycgw.cluster_restart import ClusterEDGWRestartState
from rubycgw.cluster_restart_solver import solve_cluster_ed_gw_fast_restarted


@dataclass
class _Background:
    G: np.ndarray
    Sigma_H: np.ndarray
    Sigma_GW: np.ndarray
    mu: float
    converged: bool = True
    final_error: float = 1e-10


def test_restart_injects_dynamic_state_and_bath(monkeypatch):
    from rubycgw import cluster_ed_gw_fast as fast

    shape = (3, 1, 1, 6, 6)
    local = (3, 6, 6)
    bath = BathParameters(
        energies=np.array([-0.5, 0.5]),
        couplings=np.full((6, 2), 0.2 + 0.1j),
        fit_error=np.nan,
        nfev=0,
    )
    restart = ClusterEDGWRestartState(
        G=np.full(shape, 2.0 + 0.0j),
        Sigma_H=np.full((6, 6), 3.0 + 0.0j),
        Sigma_emb=np.full(shape, 4.0 + 0.0j),
        Sigma_imp=np.full(local, 5.0 + 0.0j),
        mu=0.37,
        bath=bath,
        source_path="old.npz",
    )
    physical_background = _Background(
        G=np.zeros(shape, dtype=complex),
        Sigma_H=np.zeros((6, 6), dtype=complex),
        Sigma_GW=np.zeros(shape, dtype=complex),
        mu=0.0,
    )

    original_cluster = lambda *a, **k: (
        np.full(local, 11.0 + 0.0j),
        "unused1",
        "unused2",
    )

    def original_fit(*args, **kwargs):
        return kwargs.get("initial")

    monkeypatch.setattr(fast, "cluster_gw_self_energy", original_cluster)
    monkeypatch.setattr(fast, "fit_finite_bath", original_fit)

    seen = {}

    def fake_solver(h0, Vq, params, grid, *, gw_opts, embed_opts, background):
        seen["background"] = background
        seen["first_sigma_imp"] = fast.cluster_gw_self_energy(None)[0]
        seen["second_sigma_cgw"] = fast.cluster_gw_self_energy(None)[0]
        seen["first_bath"] = fast.fit_finite_bath(initial=None)
        return SimpleNamespace(background=background)

    monkeypatch.setattr(fast, "solve_cluster_ed_gw_fast", fake_solver)

    opts = SimpleNamespace(verbose=False)
    result = solve_cluster_ed_gw_fast_restarted(
        None,
        None,
        None,
        None,
        gw_opts=SimpleNamespace(),
        embed_opts=opts,
        restart=restart,
        background=physical_background,
    )

    assert np.allclose(seen["background"].G, restart.G)
    assert np.allclose(seen["background"].Sigma_H, restart.Sigma_H)
    assert np.allclose(seen["background"].Sigma_GW, restart.Sigma_emb)
    assert np.isclose(seen["background"].mu, restart.mu)
    assert np.allclose(seen["first_sigma_imp"], restart.Sigma_imp)
    assert np.allclose(seen["second_sigma_cgw"], 11.0)
    assert seen["first_bath"] is restart.bath
    assert result.background is physical_background

    # The wrapper must restore the production maps even after the call.
    assert fast.cluster_gw_self_energy is original_cluster
    assert fast.fit_finite_bath is original_fit

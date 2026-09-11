import numpy as np

from rubycgw.cluster_ed_gw_fast import (
    ClusterEDGWFastOptions,
    _pack_dynamic,
    _unpack_dynamic,
    solve_cluster_ed_gw_fast,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters, build_h0, build_interaction


def test_pack_dynamic_roundtrip_balances_impurity_block():
    emb = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2).astype(complex)
    imp = (10 + np.arange(2 * 2 * 2, dtype=float)).reshape(2, 2, 2).astype(complex)
    packed = _pack_dynamic(emb, imp, nk=3)
    emb2, imp2 = _unpack_dynamic(packed, emb.shape, imp.shape, nk=3)
    assert np.max(np.abs(emb2 - emb)) < 1e-14
    assert np.max(np.abs(imp2 - imp)) < 1e-14


def test_fast_zero_interaction_embedding_reduces_to_lattice_gw():
    p = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=2, nk2=1, nw=3, nOmega=1, T=0.16)
    h0 = build_h0(grid.kmesh(), p)
    Vq = build_interaction(grid.qmesh(), p)
    gw_opts = GWOptions(
        target_filling=2.0,
        max_iter=5,
        tol=1e-10,
        mixing=0.5,
        mixing_method="linear",
        verbose=False,
        momentum_backend="direct",
    )
    eopts = ClusterEDGWFastOptions(
        max_iter=3,
        tol=1e-8,
        mixing=0.5,
        mixing_method="pulay",
        pulay_start=2,
        impurity_mixing=1.0,
        nbath=2,
        bath_fit_nfreq=2,
        bath_fit_max_nfev=20,
        discard_weight_tol=0.0,
        verbose=False,
    )
    out = solve_cluster_ed_gw_fast(
        h0, Vq, p, grid, gw_opts=gw_opts, embed_opts=eopts
    )
    den = np.linalg.norm(out.background.G.ravel())
    assert np.linalg.norm((out.G - out.background.G).ravel()) / den < 5e-8
    assert np.max(np.abs(out.Sigma_ED_cluster)) < 5e-8
    assert out.residual_history.ndim == 1
    assert len(out.residual_history) == out.iterations

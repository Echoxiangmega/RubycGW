import numpy as np

from rubycgw.api import (
    BackgroundConfig,
    EffectiveEDConfig,
    GridConfig,
    RubyModel,
    run_effective_ed,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters, build_interaction
from rubycgw.models import (
    ExtendedRubyParameters,
    build_extended_interaction,
    extended_cluster_interactions,
)
from vprime_study.cross_model import VPrimeCrossParameters


def test_extended_model_reduces_to_baseline_interaction():
    grid = MatsubaraGrid(nk1=3, nk2=3, nw=2, nOmega=1, T=0.1)
    base = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.7)
    ext = ExtendedRubyParameters(
        ti=0.4, t1=0.2, t2=0.2, V=0.7, Vprime=0.0, Vcross=0.0
    )
    assert np.max(
        np.abs(build_interaction(grid.qmesh(), base) - build_extended_interaction(grid.qmesh(), ext))
    ) < 1e-12


def test_historical_vcross_parameters_are_canonical_alias():
    assert VPrimeCrossParameters is ExtendedRubyParameters


def test_crossed_cluster_projection_sums_repeated_pairs():
    p = ExtendedRubyParameters(V=0.0, Vprime=0.0, Vcross=-0.07)
    couplings = {(a, b): u for a, b, u in extended_cluster_interactions(p)}
    assert np.isclose(couplings[(0, 3)], -0.14)
    assert np.isclose(couplings[(1, 5)], -0.14)
    assert np.isclose(couplings[(2, 4)], -0.14)


def test_public_configs_are_composable():
    cfg = BackgroundConfig(grid=GridConfig(Lx=2, Ly=2, nw=4, nOmega=2, T=0.08))
    grid = cfg.grid.build()
    assert grid.nk1 == 2
    assert grid.nk2 == 2
    assert np.isclose(grid.T, 0.08)


def test_effective_ed_workflow_small_torus():
    model = RubyModel(ti=0.4, t1=0.2, t2=0.2, V=1.8, Vprime=-0.1, Vcross=-0.05)
    result = run_effective_ed(model, EffectiveEDConfig(Lx=1, Ly=1, nev=2))
    assert result.nspin == 2
    assert result.structure_factor.shape == (1, 6, 6)
    assert np.isfinite(result.ground_energy)

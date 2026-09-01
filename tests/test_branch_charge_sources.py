import numpy as np

import search_supercell_branches as branch_search
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.supercell import (
    NSUP,
    add_charge_source,
    build_supercell_h0,
    charge_order_diagnostics,
    charge_source_pattern,
)


def test_merged_default_branch_set():
    assert branch_search.DEFAULT_BRANCHES == (
        "normal",
        "co",
        "intra",
        "ab",
        "same",
        "opposite",
    )


def test_q0_charge_source_patterns_repeat_primitive_motif():
    expected = {
        "intra": np.array([1.0, -0.5, -0.5, 1.0, -0.5, -0.5]),
        "ab": np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]),
    }
    for channel, p6 in expected.items():
        pattern = charge_source_pattern(channel).reshape(3, 6)
        assert pattern.shape == (3, 6)
        assert np.max(np.abs(np.mean(pattern, axis=0) - p6)) < 1e-14
        assert np.max(np.abs(pattern - p6[None, :])) < 1e-14
        assert abs(np.mean(pattern)) < 1e-14
        assert abs(np.max(np.abs(pattern)) - 1.0) < 1e-14


def test_joint_intra_source_targets_both_triangles_equally():
    n0 = 0.4
    amp = 0.07

    density = n0 + amp * charge_source_pattern("intra")
    d = charge_order_diagnostics(density)
    expected = 1.5 * amp
    assert d["Delta_Q"] < 1e-13
    assert abs(d["Delta_A"] - expected) < 1e-13
    assert abs(d["Delta_B"] - expected) < 1e-13
    assert abs(d["Delta_intra"] - expected) < 1e-13
    assert abs(d["Delta_AB"]) < 1e-13


def test_ab_source_targets_only_triangle_imbalance():
    n0 = 0.4
    amp = 0.07
    density = n0 + amp * charge_source_pattern("ab")
    d = charge_order_diagnostics(density)
    assert d["Delta_Q"] < 1e-13
    assert d["Delta_intra"] < 1e-13
    assert d["Delta_A"] < 1e-13
    assert d["Delta_B"] < 1e-13
    assert abs(d["Delta_AB"] - 2.0 * amp) < 1e-13


def test_legacy_one_triangle_patterns_remain_available_for_diagnostics():
    pa = charge_source_pattern("intra-a").reshape(3, 6)
    pb = charge_source_pattern("intra-b").reshape(3, 6)
    assert np.allclose(pa[0], [1.0, -0.5, -0.5, 0.0, 0.0, 0.0])
    assert np.allclose(pb[0], [0.0, 0.0, 0.0, 1.0, -0.5, -0.5])


def test_add_charge_source_only_changes_diagonal():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=2, nOmega=1, T=0.1)
    h0 = build_supercell_h0(grid.kmesh(), params)
    h = 0.123

    for channel in ("co", "intra", "ab"):
        sourced = add_charge_source(h0, h, channel)
        delta = sourced - h0
        pattern = charge_source_pattern(channel)
        expected = np.zeros_like(delta)
        idx = np.diag_indices(NSUP)
        expected[..., idx[0], idx[1]] = -h * pattern
        assert np.max(np.abs(delta - expected)) < 1e-13
        assert np.max(np.abs(sourced - np.swapaxes(sourced.conj(), -1, -2))) < 1e-13


def test_branch_h0_uses_charge_source_patterns():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=2, nOmega=1, T=0.1)
    base = build_supercell_h0(grid.kmesh(), params)
    h = 0.05

    for branch in branch_search.CHARGE_BRANCHES:
        got = branch_search._branch_h0(branch, h, base, params, grid)
        ref = add_charge_source(base, h, branch)
        assert np.max(np.abs(got - ref)) < 1e-14


def test_classification_merges_delta_A_and_delta_B():
    currents = {"same_q0": 0.0j, "opposite_q0": 0.0j}
    charge = {
        "Delta_Q": 0.0,
        "Delta_intra": 2.0e-3,
        "Delta_A": 2.2e-3,
        "Delta_B": 1.8e-3,
        "Delta_AB": 0.0,
    }
    assert branch_search._classify(charge, currents, 1e-6) == "intra-CO"

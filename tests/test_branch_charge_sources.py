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


def test_expanded_default_branch_set():
    assert branch_search.DEFAULT_BRANCHES == (
        "normal",
        "co",
        "intra-a",
        "intra-b",
        "ab",
        "same",
        "opposite",
    )


def test_q0_charge_source_patterns_repeat_primitive_motif():
    expected = {
        "intra-a": np.array([1.0, -0.5, -0.5, 0.0, 0.0, 0.0]),
        "intra-b": np.array([0.0, 0.0, 0.0, 1.0, -0.5, -0.5]),
        "ab": np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]),
    }
    for channel, p6 in expected.items():
        pattern = charge_source_pattern(channel).reshape(3, 6)
        assert pattern.shape == (3, 6)
        assert np.max(np.abs(np.mean(pattern, axis=0) - p6)) < 1e-14
        assert np.max(np.abs(pattern - p6[None, :])) < 1e-14
        assert abs(np.mean(pattern)) < 1e-14
        assert abs(np.max(np.abs(pattern)) - 1.0) < 1e-14


def test_charge_source_patterns_target_expected_diagnostics():
    n0 = 0.4
    amp = 0.07

    density_a = n0 + amp * charge_source_pattern("intra-a")
    da = charge_order_diagnostics(density_a)
    assert da["Delta_Q"] < 1e-13
    assert abs(da["Delta_A"] - 1.5 * amp) < 1e-13
    assert da["Delta_B"] < 1e-13
    assert abs(da["Delta_AB"]) < 1e-13

    density_b = n0 + amp * charge_source_pattern("intra-b")
    db = charge_order_diagnostics(density_b)
    assert db["Delta_Q"] < 1e-13
    assert db["Delta_A"] < 1e-13
    assert abs(db["Delta_B"] - 1.5 * amp) < 1e-13
    assert abs(db["Delta_AB"]) < 1e-13

    density_ab = n0 + amp * charge_source_pattern("ab")
    dab = charge_order_diagnostics(density_ab)
    assert dab["Delta_Q"] < 1e-13
    assert dab["Delta_A"] < 1e-13
    assert dab["Delta_B"] < 1e-13
    assert abs(dab["Delta_AB"] - 2.0 * amp) < 1e-13


def test_add_charge_source_only_changes_diagonal():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=2, nOmega=1, T=0.1)
    h0 = build_supercell_h0(grid.kmesh(), params)
    h = 0.123

    for channel in ("co", "intra-a", "intra-b", "ab"):
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

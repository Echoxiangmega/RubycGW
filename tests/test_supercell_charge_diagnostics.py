import numpy as np

from rubycgw.supercell import charge_order_diagnostics, charge_order_parameter


def test_q0_intra_triangle_charge_order_can_have_zero_selected_phi():
    # Same primitive-cell density in all three sectors: no period-3 translation
    # breaking. Triangle A nevertheless has a clear q=0 internal charge order.
    one_cell = np.array([0.7, 0.4, 0.4, 0.5, 0.5, 0.5], dtype=float)
    density = np.tile(one_cell, 3)
    d = charge_order_diagnostics(density)

    assert abs(charge_order_parameter(density)) < 1e-13
    assert abs(d["Phi"]) < 1e-13
    assert d["Delta_Q"] < 1e-13
    assert d["Delta_translation_rms"] < 1e-13
    assert abs(d["Delta_A"] - 0.3) < 1e-13
    assert d["Delta_B"] < 1e-13
    assert abs(d["Delta_intra"] - 0.3 / np.sqrt(2.0)) < 1e-13
    assert abs(d["Delta_AB"] - (0.5 - 0.5)) < 1e-13
    assert np.allclose(d["n_q0"], one_cell, atol=1e-13)


def test_equal_A_B_intra_order_has_same_combined_amplitude():
    one_cell = np.array([0.7, 0.4, 0.4, 0.8, 0.5, 0.5], dtype=float)
    density = np.tile(one_cell, 3)
    d = charge_order_diagnostics(density)

    assert abs(d["Delta_A"] - 0.3) < 1e-13
    assert abs(d["Delta_B"] - 0.3) < 1e-13
    assert abs(d["Delta_intra"] - 0.3) < 1e-13


def test_generic_period3_charge_order_is_detected_even_when_phi_is_zero():
    # A sector-only modulation has a Q form factor proportional to (1,1,1,1,1,1),
    # which is orthogonal to the selected Phi form factor. Phi therefore misses
    # it, while the generic Delta_Q diagnostic must remain finite.
    A = 0.12
    phases = np.cos(2.0 * np.pi * np.arange(3) / 3.0)
    density = np.stack([0.5 + A * phases[s] * np.ones(6) for s in range(3)])
    d = charge_order_diagnostics(density.reshape(-1))

    assert abs(d["Phi"]) < 1e-13
    assert d["Delta_Q"] > 1e-3
    assert d["Delta_translation_rms"] > 1e-3
    assert abs(d["Delta_Q"] / np.sqrt(3.0) - d["Delta_translation_rms"]) < 1e-13
    assert d["Delta_intra"] < 1e-13
    assert d["Delta_A"] < 1e-13
    assert d["Delta_B"] < 1e-13
    assert abs(d["Delta_AB"]) < 1e-13


def test_uniform_density_has_no_charge_order_in_any_diagnostic():
    density = np.full(18, 0.37)
    d = charge_order_diagnostics(density)

    assert abs(d["Phi"]) < 1e-13
    assert d["Delta_Q"] < 1e-13
    assert d["Delta_translation_rms"] < 1e-13
    assert d["Delta_intra"] < 1e-13
    assert d["Delta_A"] < 1e-13
    assert d["Delta_B"] < 1e-13
    assert abs(d["Delta_AB"]) < 1e-13

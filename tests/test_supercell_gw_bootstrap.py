from rubycgw.supercell_gw_bootstrap import (
    AndersonOptions,
    _bootstrap_values,
    _safe_periodic_options,
)


def test_bootstrap_schedule_stays_below_target():
    assert _bootstrap_values(0.70) == [0.10, 0.25, 0.50, 0.60, 0.65]
    assert _bootstrap_values(0.60) == [0.10, 0.25, 0.50]
    assert _bootstrap_values(0.10) == []


def test_safe_periodic_options_are_local_and_conservative():
    safe = _safe_periodic_options(AndersonOptions())
    assert safe.history == 3
    assert safe.beta <= 0.25
    assert safe.step_cap <= 2.0
    assert safe.gw_beta <= 0.08
    assert safe.recovery_gw_beta <= 0.04
    assert safe.pulay_enter_gw <= 0.25
    assert safe.recovery_steps >= 5

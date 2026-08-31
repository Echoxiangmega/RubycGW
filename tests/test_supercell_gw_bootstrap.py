import numpy as np

from rubycgw.gw import GWResult
from rubycgw.supercell_gw_bootstrap import (
    AndersonOptions,
    _bootstrap_values,
    _choose_polished_result,
    _needs_finish_polish,
    _safe_periodic_options,
)


def _fake_result(error: float, converged: bool, iterations: int = 10) -> GWResult:
    z = np.zeros((1,), dtype=complex)
    return GWResult(
        G=z,
        W=z,
        P=z,
        Sigma_H=z,
        Sigma_GW=z,
        mu=0.0,
        density=np.zeros(1, dtype=float),
        converged=bool(converged),
        iterations=int(iterations),
        final_error=float(error),
        mixing_method="test",
        min_screening_singular_value=1.0,
        min_screening_m=0,
        min_screening_Omega=0.0,
        min_screening_q1=0.0,
        min_screening_q2=0.0,
        min_screening_mode=z,
        min_density_mode=z,
        min_density_mode_residual=0.0,
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


def test_near_tolerance_failure_requests_polish():
    tol = 1e-8
    assert _needs_finish_polish(_fake_result(1.008e-8, False), tol)
    assert _needs_finish_polish(_fake_result(9.0e-8, False), tol)
    assert not _needs_finish_polish(_fake_result(1.1e-7, False), tol)
    assert not _needs_finish_polish(_fake_result(0.9e-8, True), tol)


def test_polished_result_is_judged_by_original_tolerance():
    tol = 1e-8
    base = _fake_result(1.008e-8, False, iterations=250)
    trial = _fake_result(7.5e-9, False, iterations=3)
    best = _choose_polished_result(base, trial, tol)

    assert best.converged is True
    assert best.final_error == trial.final_error
    assert best.iterations == 253


def test_polish_does_not_fake_convergence_when_still_above_tol():
    tol = 1e-8
    base = _fake_result(1.008e-8, False, iterations=250)
    trial = _fake_result(1.004e-8, False, iterations=4)
    best = _choose_polished_result(base, trial, tol)

    assert best.converged is False
    assert best.final_error == trial.final_error
    assert best.iterations == 254

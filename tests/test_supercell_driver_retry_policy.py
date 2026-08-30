from types import SimpleNamespace

from run_supercell_gw import (
    _fallback_schedule,
    _gw_options,
    _near_converged,
    _primary_attempt,
)


def _args(**overrides):
    data = dict(
        no_anderson=False,
        anderson_beta=0.70,
        gw_mixing=0.20,
        gw_retry_mixings=[0.10],
        no_gw_pulay=False,
        gw_pulay_mixing=0.70,
        near_converged_threshold=1e-3,
        gw_max_iter=1000,
        gw_pulay_history=6,
        gw_pulay_start=3,
        gw_pulay_regularization=1e-10,
        mu_tol=1e-8,
        mu_max_iter=40,
        verbose_iterations=False,
        momentum_backend="fft",
    )
    data.update(overrides)
    return SimpleNamespace(**data)


def test_near_converged_threshold_is_explicit():
    args = _args()
    assert _near_converged(1e-5, args)
    assert _near_converged(1e-3, args)
    assert not _near_converged(1.01e-3, args)


def test_primary_and_fallback_schedule_are_separate():
    args = _args()
    assert _primary_attempt(args) == ("anderson", 0.70)
    assert _fallback_schedule(args) == [
        ("linear", 0.20),
        ("linear", 0.10),
        ("pulay", 0.70),
    ]


def test_gw_options_accepts_separate_iteration_budget():
    args = _args()
    opts = _gw_options(
        args,
        target_supercell=9.0,
        mu=1.2,
        method="linear",
        mixing=0.1,
        tol=1e-6,
        max_iter=150,
    )
    assert opts.max_iter == 150
    assert opts.tol == 1e-6
    assert opts.target_filling == 9.0

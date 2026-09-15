from types import SimpleNamespace

import pytest

from research import validate_cluster_ed_gw_jf as validator


def test_legacy_single_rank_is_preserved():
    args = SimpleNamespace(bath_rank=24, bath_ranks=None)
    ranks = validator._requested_bath_ranks(args)
    assert ranks == [24]
    assert validator._build_max_rank(ranks) == 24


def test_multi_rank_builds_only_largest_when_full_not_requested():
    args = SimpleNamespace(bath_rank=24, bath_ranks=[12, 24, 48, 24])
    ranks = validator._requested_bath_ranks(args)
    assert ranks == [12, 24, 48]
    assert validator._build_max_rank(ranks) == 48


def test_zero_rank_requests_one_full_build_then_truncation():
    args = SimpleNamespace(bath_rank=24, bath_ranks=[24, 48, 0])
    ranks = validator._requested_bath_ranks(args)
    assert ranks == [24, 48, 0]
    assert validator._build_max_rank(ranks) is None


def test_legacy_single_fd_step_is_preserved():
    args = SimpleNamespace(bath_fd_step=2e-4, bath_fd_steps=None)
    assert validator._requested_bath_fd_steps(args) == [2e-4]


def test_multi_fd_steps_preserve_order_and_drop_duplicates():
    args = SimpleNamespace(
        bath_fd_step=2e-4,
        bath_fd_steps=[4e-4, 2e-4, 1e-4, 2e-4],
    )
    assert validator._requested_bath_fd_steps(args) == [4e-4, 2e-4, 1e-4]


@pytest.mark.parametrize("step", [0.0, -1e-4, float("inf"), float("nan")])
def test_invalid_fd_steps_are_rejected(step):
    args = SimpleNamespace(bath_fd_step=2e-4, bath_fd_steps=[step])
    with pytest.raises(ValueError, match="finite and positive"):
        validator._requested_bath_fd_steps(args)

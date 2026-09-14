from types import SimpleNamespace

import validate_cluster_ed_gw_jf as validator


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

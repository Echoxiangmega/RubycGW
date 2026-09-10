import numpy as np

from rubycgw.fierz_channel_gw import build_channel_definition, channel_static_self_energy
from rubycgw.fierz_mixed import (
    FierzWeights,
    build_weighted_nbj_definition,
    soft_mode_sector_fractions,
    validate_weights,
    weights_from_lambda,
)


def _pairs():
    return ((0, 1), (1, 2), (2, 0))


def test_weight_family_preserves_first_order_hf():
    rng = np.random.default_rng(12)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    rho = 0.5 * (A + A.conj().T)
    V = 0.73
    ref = build_channel_definition(_pairs(), 3, V, "density")
    sigma_ref, _, _ = channel_static_self_energy(rho, ref)

    for lam in (0.0, 0.2, 0.5, 0.8, 1.0):
        weighted = build_weighted_nbj_definition(
            _pairs(), 3, V, weights_from_lambda(lam)
        )
        sigma, _, _ = channel_static_self_energy(rho, weighted)
        np.testing.assert_allclose(sigma, sigma_ref, rtol=1e-12, atol=1e-12)


def test_weighted_endpoint_channel_counts_and_couplings():
    V = 0.4
    d = build_weighted_nbj_definition(_pairs(), 3, V, weights_from_lambda(1.0))
    assert len(d.labels) == 3
    assert all(x.startswith("n") for x in d.labels)

    bj = build_weighted_nbj_definition(_pairs(), 3, V, weights_from_lambda(0.0))
    assert len(bj.labels) == 6
    assert sum(x.startswith("B(") for x in bj.labels) == 3
    assert sum(x.startswith("J(") for x in bj.labels) == 3
    np.testing.assert_allclose(np.diag(bj.coupling), -0.5 * V)

    mixed = build_weighted_nbj_definition(_pairs(), 3, V, weights_from_lambda(0.5))
    assert len(mixed.labels) == 9
    # Density block carries lambda_n V on each interacting pair.
    assert abs(mixed.coupling[0, 1] - 0.5 * V) < 1e-15
    # B and J blocks each carry -lambda_{B/J} V = -V/4.
    np.testing.assert_allclose(np.diag(mixed.coupling)[3:], -0.25 * V)


def test_soft_mode_sector_fractions_sum_to_one():
    definition = build_weighted_nbj_definition(
        _pairs(), 3, 0.2, weights_from_lambda(0.5)
    )
    z = np.arange(1, len(definition.labels) + 1, dtype=float).astype(complex)
    frac = soft_mode_sector_fractions(definition, z)
    assert abs(frac["n"] + frac["B"] + frac["J"] - 1.0) < 1e-14


def test_weight_validation_rejects_non_fierz_sum():
    try:
        validate_weights(FierzWeights(0.4, 0.4, 0.4))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid Fierz weights were accepted")

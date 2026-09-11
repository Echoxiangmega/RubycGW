import numpy as np

from rubycgw.fierz_channel_gamma import build_channel_gamma_screening
from rubycgw.fierz_channel_gw import ChannelDefinition


def test_channel_gamma_screening_recovers_rpa_identity():
    g = np.asarray([[0.42, 0.07], [0.07, -0.31]], dtype=complex)
    P = np.asarray([[0.13, -0.025], [-0.025, 0.09]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    W_rpa = np.linalg.solve(eye - g @ P, g)
    chi_rpa = -np.linalg.solve(eye - P @ g, P)
    definition = ChannelDefinition(
        mode="test",
        vertices=np.zeros((2, 1, 1), dtype=complex),
        coupling=g,
        labels=("a", "b"),
        interaction_pairs=(),
    )

    W, P_gamma, fallback, identity_error = build_channel_gamma_screening(
        definition, chi_rpa[None, ...]
    )

    assert not np.any(fallback)
    assert np.allclose(P_gamma[0], P, rtol=1e-12, atol=1e-12)
    assert np.allclose(W[0], W_rpa, rtol=1e-12, atol=1e-12)
    assert identity_error < 1e-12

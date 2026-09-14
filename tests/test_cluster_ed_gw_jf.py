import numpy as np

from rubycgw.cluster_ed_gw import BathParameters, bath_hybridization
from rubycgw.cluster_ed_gw_jf import (
    _bath_from_theta,
    _bath_theta,
    _complex_delta_derivatives,
    _pack_complex,
    _unpack_complex,
)


def test_real_pack_roundtrip():
    rng = np.random.default_rng(123)
    z = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    x = _pack_complex(z)
    assert x.dtype.kind == "f"
    assert np.allclose(_unpack_complex(x, z.shape), z)


def test_bath_theta_roundtrip_complex():
    bath = BathParameters(
        energies=np.array([-0.7, 0.4]),
        couplings=np.array(
            [
                [0.3 + 0.1j, -0.2 + 0.05j],
                [0.12 - 0.08j, 0.22 + 0.13j],
            ],
            dtype=complex,
        ),
        fit_error=0.0,
        nfev=0,
    )
    theta = _bath_theta(bath)
    rebuilt = _bath_from_theta(theta, 2, 2)
    assert np.allclose(rebuilt.energies, bath.energies)
    assert np.allclose(rebuilt.couplings, bath.couplings)


def test_complex_delta_derivatives_match_centered_difference():
    rng = np.random.default_rng(7)
    norb = 3
    nbath = 2
    omega = np.array([0.21, 0.63, 1.05])
    mu = 0.17
    bath = BathParameters(
        energies=np.array([-0.55, 0.72]),
        couplings=(
            0.25 * rng.normal(size=(norb, nbath))
            + 0.18j * rng.normal(size=(norb, nbath))
        ),
        fit_error=0.0,
        nfev=0,
    )
    theta = _bath_theta(bath)
    analytic = list(_complex_delta_derivatives(omega, mu, bath))
    assert len(analytic) == nbath + 2 * norb * nbath

    h = 2.0e-7
    for j, deriv in enumerate(analytic):
        direction = np.zeros_like(theta)
        direction[j] = 1.0
        bp = _bath_from_theta(theta + h * direction, norb, nbath)
        bm = _bath_from_theta(theta - h * direction, norb, nbath)
        dp = bath_hybridization(omega, mu, bp.energies, bp.couplings)
        dm = bath_hybridization(omega, mu, bm.energies, bm.couplings)
        numeric = (dp - dm) / (2.0 * h)
        assert np.allclose(deriv, numeric, rtol=3e-6, atol=3e-8)

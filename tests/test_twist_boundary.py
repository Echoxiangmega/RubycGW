import numpy as np

from benchmark_twist_averaged_finite_source import build_twisted_cluster_h0
from rubycgw.model import RubyParameters, build_h0
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


def _allowed_k(L1, L2, theta):
    phi = np.asarray(theta, dtype=float) / (2.0 * np.pi)
    pts = []
    for m1 in range(L1):
        for m2 in range(L2):
            pts.append([
                ((m1 + phi[0]) / float(L1)) % 1.0,
                ((m2 + phi[1]) / float(L2)) % 1.0,
            ])
    return np.asarray(pts, dtype=float)


def test_twist_zero_matches_periodic_torus():
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0)
    exact = ExactSmallRubyThermal(2, 1, params)
    twisted = build_twisted_cluster_h0(2, 1, params, (0.0, 0.0))
    assert np.max(np.abs(twisted - exact.h0)) < 1e-12


def test_twisted_torus_spectrum_matches_shifted_primitive_momenta():
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0)
    L1, L2 = 2, 1
    theta = np.asarray([0.37, -1.12])
    hcluster = build_twisted_cluster_h0(L1, L2, params, theta)
    cluster_e = np.sort(np.linalg.eigvalsh(hcluster))

    kpts = _allowed_k(L1, L2, theta)
    hk = build_h0(kpts, params)
    bloch_e = np.sort(np.linalg.eigvalsh(hk).reshape(-1))

    assert np.max(np.abs(cluster_e - bloch_e)) < 1e-11

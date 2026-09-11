import numpy as np

from analyze_cluster_ed_gw_self_energy import (
    decompose_kernel,
    dyson_kernel,
    k_to_realspace,
    realspace_to_k,
)


def test_realspace_k_roundtrip_on_two_by_one_torus():
    rng = np.random.default_rng(123)
    Gk = rng.normal(size=(5, 2, 1, 6, 6)) + 1j * rng.normal(size=(5, 2, 1, 6, 6))
    Gr = k_to_realspace(Gk, 2, 1)
    recovered = realspace_to_k(Gr, 2, 1)
    assert np.max(np.abs(recovered - Gk)) < 1e-12


def test_dyson_kernel_vanishes_for_noninteracting_green_function():
    omega = np.array([-0.7, 0.7])
    h0 = np.zeros((2, 1, 6, 6), dtype=complex)
    h0[0, 0] = np.diag(np.linspace(-0.3, 0.2, 6))
    h0[1, 0] = np.diag(np.linspace(-0.1, 0.4, 6))
    eye = np.eye(6)
    G = np.empty((2, 2, 1, 6, 6), dtype=complex)
    for n, w in enumerate(omega):
        for k in range(2):
            G[n, k, 0] = np.linalg.inv(1j * w * eye - h0[k, 0])
    K = dyson_kernel(G, h0, omega)
    assert np.max(np.abs(K)) < 1e-12


def test_local_nonlocal_decomposition_has_zero_k_average_nonlocal():
    rng = np.random.default_rng(321)
    K = rng.normal(size=(4, 3, 2, 6, 6)) + 1j * rng.normal(size=(4, 3, 2, 6, 6))
    local, nonlocal = decompose_kernel(K)
    assert np.max(np.abs(np.mean(nonlocal, axis=(1, 2)))) < 1e-13
    assert np.max(np.abs(K - local[:, None, None] - nonlocal)) < 1e-13

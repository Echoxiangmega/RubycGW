import numpy as np

from rubycgw.ed_green_compare import (
    cluster_local_green,
    primitive_local_green,
    relative_green_error,
)


def test_primitive_and_cluster_local_green_helpers():
    nf, nk1, nk2, norb = 3, 2, 2, 2
    base = np.arange(nf * nk1 * nk2 * norb * norb, dtype=float).reshape(
        nf, nk1, nk2, norb, norb
    )
    Gk = base + 1j * (0.1 * base)
    loc = primitive_local_green(Gk)
    np.testing.assert_allclose(loc, np.mean(Gk, axis=(1, 2)))

    Gc = np.zeros((nf, 2 * norb, 2 * norb), dtype=complex)
    Gc[:, :norb, :norb] = loc
    Gc[:, norb:, norb:] = 3.0 * loc
    np.testing.assert_allclose(cluster_local_green(Gc, 2, norb=norb), 2.0 * loc)


def test_relative_green_error_reference_and_scaling():
    rng = np.random.default_rng(4)
    G = rng.normal(size=(5, 3, 3)) + 1j * rng.normal(size=(5, 3, 3))
    assert relative_green_error(G, G) < 1e-14
    np.testing.assert_allclose(relative_green_error(1.1 * G, G), 0.1, rtol=1e-12, atol=1e-12)

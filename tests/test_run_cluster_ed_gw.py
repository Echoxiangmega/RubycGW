import numpy as np

from run_cluster_ed_gw import _lattice_to_realspace


def test_lattice_to_realspace_constant_k_is_cell_local():
    Gk = np.zeros((3, 2, 1, 6, 6), dtype=complex)
    base = np.diag(np.arange(1, 7, dtype=float))
    Gk[:, 0, 0] = base
    Gk[:, 1, 0] = base
    Gr = _lattice_to_realspace(Gk, 2, 1)
    assert np.max(np.abs(Gr[:, :6, :6] - base)) < 1e-13
    assert np.max(np.abs(Gr[:, 6:, 6:] - base)) < 1e-13
    assert np.max(np.abs(Gr[:, :6, 6:])) < 1e-13
    assert np.max(np.abs(Gr[:, 6:, :6])) < 1e-13

import numpy as np

from rubycgw.cluster_ed_gw_c3_constrained import _constrain_hartree


def test_constrain_hartree_handles_readonly_diag_view():
    sigma_h = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(complex)
    out = _constrain_hartree(sigma_h)
    assert np.allclose(np.diag(out)[:3], 2.0)
    assert np.allclose(np.diag(out)[3:], 5.0)
    assert np.allclose(out - np.diag(np.diag(out)), 0.0)

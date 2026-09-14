import numpy as np

import rubycgw.gw as gw
from rubycgw import cluster_ed_gw_covariant
from rubycgw import cluster_ed_gw_fast
from rubycgw import cluster_ed_weak_covariant
from rubycgw import cluster_ed_weak_fast
from rubycgw import cluster_ed_weak_warm
from rubycgw import fierz_channel_gw
from rubycgw import gf2
from rubycgw import gw_dynamic_sosex
from rubycgw import gw_sox
from rubycgw import gw_ssosex
from rubycgw import supercell_gw
from rubycgw import supercell_gw_fast
from rubycgw import supercell_hf
from rubycgw.pulay_accel import scale_invariant_pulay_coefficients
from rubycgw.supercell_gw_periodic_pulay import _gw_pulay_coefficients


def _shared_history(scale=1.0):
    zero = np.zeros((2,), dtype=complex)
    rh = [
        np.array([1.0, 0.2]),
        np.array([0.55, -0.10]),
        np.array([0.18, 0.07]),
    ]
    rg = [
        np.array([0.4, -0.3, 0.1]),
        np.array([0.15, -0.22, 0.05]),
        np.array([0.04, -0.08, 0.03]),
    ]
    return [
        (zero.copy(), zero.copy(), scale * a, scale * b)
        for a, b in zip(rh, rg)
    ]


def test_scale_invariant_pulay_is_package_default():
    # Importing any rubycgw submodule executes package __init__, which installs
    # the corrected coefficient solve.  No special launcher should be required.
    assert gw._pulay_coefficients is scale_invariant_pulay_coefficients

    c1 = gw._pulay_coefficients(_shared_history(1.0), 1e-7)
    c2 = gw._pulay_coefficients(_shared_history(1e-8), 1e-7)
    assert np.allclose(c1, c2, rtol=1e-11, atol=1e-12)


def test_all_shared_pulay_users_resolve_corrected_coefficients():
    # These production paths all import gw._mixed_self_energies.  The function
    # resolves _pulay_coefficients in the gw module at call time, so verify that
    # every imported alias reaches the corrected package-wide implementation.
    modules = (
        cluster_ed_gw_covariant,
        cluster_ed_gw_fast,
        cluster_ed_weak_covariant,
        cluster_ed_weak_fast,
        cluster_ed_weak_warm,
        fierz_channel_gw,
        gf2,
        gw_dynamic_sosex,
        gw_sox,
        gw_ssosex,
        supercell_gw,
        supercell_gw_fast,
        supercell_hf,
    )
    for module in modules:
        mixer = module._mixed_self_energies
        assert mixer.__module__ == "rubycgw.gw"
        assert (
            mixer.__globals__["_pulay_coefficients"]
            is scale_invariant_pulay_coefficients
        )


def _periodic_history(scale=1.0):
    outputs = [
        np.array([0.1, -0.2, 0.3]),
        np.array([0.2, -0.1, 0.1]),
        np.array([0.4, 0.0, -0.2]),
    ]
    residuals = [
        np.array([1.0, 0.2, -0.1]),
        np.array([0.4, -0.3, 0.2]),
        np.array([0.1, -0.15, 0.05]),
    ]
    return [
        (out.astype(complex), scale * res.astype(complex))
        for out, res in zip(outputs, residuals)
    ]


def test_periodic_supercell_pulay_is_already_scale_invariant():
    # The dedicated 18-site periodic-Pulay solver has its own DIIS coefficient
    # implementation.  Keep an explicit regression test so it cannot re-acquire
    # the same absolute-regularization floor in a later refactor.
    c1 = _gw_pulay_coefficients(_periodic_history(1.0), 1e-7)
    c2 = _gw_pulay_coefficients(_periodic_history(1e-9), 1e-7)
    assert np.allclose(c1, c2, rtol=1e-11, atol=1e-12)
    assert np.isclose(np.sum(c1), 1.0, atol=1e-12)

from pathlib import Path

import numpy as np

from rubycgw.ed_benchmark_cache import (
    ed_cache_path,
    load_ed_cache,
    save_ed_cache,
)


def _signature():
    return {
        "L1": 2,
        "L2": 1,
        "V": 1.0,
        "filling": 2.0,
        "T": 0.08,
        "ti": 0.4,
        "t1": 0.2,
        "t2": 0.2,
        "nw": 3,
        "channels": ["z_same", "z_opposite"],
    }


def test_ed_cache_roundtrip_and_signature_invalidation(tmp_path: Path):
    sig = _signature()
    omega = np.array([-0.3, 0.1, 0.5])
    G = np.arange(27, dtype=float).reshape(3, 3, 3).astype(complex)
    G += 0.2j * G
    chi = np.array([[1.2, 0.1], [0.1, 0.9]])

    path = save_ed_cache(tmp_path, sig, omega, 0.123, G, chi)
    assert path == ed_cache_path(tmp_path, sig)
    assert path.exists()

    out = load_ed_cache(tmp_path, sig, omega, nsite=3)
    assert out is not None
    assert out["path"] == path
    assert out["mu_ed"] == 0.123
    np.testing.assert_array_equal(out["G_ed"], G)
    np.testing.assert_array_equal(out["chi_ed"], chi)

    changed = dict(sig)
    changed["V"] = 0.9
    assert load_ed_cache(tmp_path, changed, omega, nsite=3) is None
    assert load_ed_cache(tmp_path, sig, omega + 1e-12, nsite=3) is None
    assert load_ed_cache(tmp_path, sig, omega, nsite=4) is None

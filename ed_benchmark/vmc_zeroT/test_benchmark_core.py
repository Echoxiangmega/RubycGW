#!/usr/bin/env python3
"""Tests that do not require NetKet; run from ed_benchmark/vmc_zeroT."""
from pathlib import Path
import sys
import numpy as np
from scipy.sparse.linalg import eigsh

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NATIVE = REPO / "ed_benchmark" / "native_ed"
sys.path.insert(0, str(NATIVE))

import geometry
import core
import observables

from benchmark_observables import benchmark_observables, ruby_star_formfactor
from vmc_ansatz import minimum_image_shells, one_body_seed


def main():
    g = geometry.RubyPBC("ring3", t=1.0, t1=0.5, t2=0.5)
    assert g.Ns == 18 and len(g.edges) == 36
    assert np.array_equal(g.M, np.array([[1, 1], [-1, 2]], dtype=int))
    assert list(g.reps) == [(0, 0), (1, 0), (1, 1)]
    assert g.point_maps()["C3"] is not None
    strip = geometry.RubyPBC("3x1", t=1.0, t1=0.5, t2=0.5)
    assert np.array_equal(strip.M, np.array([[3, 0], [0, 1]], dtype=int))
    assert strip.point_maps()["C3"] is None
    f = ruby_star_formfactor(g)
    assert len(f) == 36 and abs(f.mean()) < 1e-12 and np.max(abs(f)) == 1.0
    d, mats, counts = minimum_image_shells(g, n_shells=4)
    assert len(d) == 4 and mats.shape == (4, 18, 18) and np.all(counts > 0)
    for seed in ("normal", "sd", "isd", "lc_same", "lc_opposite"):
        M = one_body_seed(g, 6, seed, 0.08, f)
        assert M.shape == (18, 6)
        assert np.linalg.norm(M.conj().T @ M - np.eye(6)) < 1e-10

    H = core.FixedHamiltonian(g, 6, 1.0, method="sparse")
    e, u = eigsh(H.operator.real, k=6, which="SA", tol=1e-11)
    idx = np.argsort(e); e, u = e[idx], u[:, idx]
    assert abs(e[0] - (-7.9938493221581854)) < 1e-9
    manifold = u[:, (e - e[0]) < 1e-8]
    assert manifold.shape[1] >= 2
    o = benchmark_observables(g, H.states, manifold, observables)
    assert abs(o.lc_same_mean) < 1e-10 and abs(o.lc_opposite_mean) < 1e-10
    for x in o.as_dict().values():
        assert np.isfinite(x)
    print("benchmark core tests: PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Regression tests for ordinary Hex18 finite-torus symmetry projection."""
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

from symmetry_projection import (
    build_ordinary_space_group,
    characterize_unique_state,
    project_to_characters,
    projected_sector_residual,
    commutator_error,
)


def main():
    g = geometry.RubyPBC("ring3", t=1.0, t1=0.5, t2=-0.5)
    H = core.FixedHamiltonian(g, 6, 2.0, method="sparse")
    group = build_ordinary_space_group(g, H.states)
    assert len(group) == 18, f"expected 18 ordinary space-group operations, got {len(group)}"

    err, name = commutator_error(group, H.operator, n_tests=1)
    assert err < 1e-10, (name, err)

    op = H.operator.real if np.max(np.abs(np.imag(H.operator.data))) < 1e-14 else H.operator
    e, u = eigsh(op, k=4, which="SA", tol=1e-11)
    order = np.argsort(e); e, u = e[order], u[:, order]
    assert e[1] - e[0] > 1e-8, "test point must have a unique ED ground state"
    psi0 = u[:, 0] / np.linalg.norm(u[:, 0])
    chars = characterize_unique_state(group, psi0, tol=1e-8)

    p0, w0 = project_to_characters(group, chars, psi0)
    assert abs(w0 - 1.0) < 1e-10
    assert abs(abs(np.vdot(psi0, p0)) - 1.0) < 1e-10
    assert projected_sector_residual(group, chars, p0) < 1e-10

    rng = np.random.default_rng(20261118)
    x = rng.normal(size=len(H.states)) + 1j * rng.normal(size=len(H.states))
    x /= np.linalg.norm(x)
    px, w = project_to_characters(group, chars, x)
    assert px is not None and 0.0 < w < 1.0
    assert projected_sector_residual(group, chars, px) < 1e-10

    print("symmetry projection tests: PASS")
    print("group_order =", len(group), "commutator_error =", err)
    print("ED E0 =", e[0], "random target-sector weight =", w)


if __name__ == "__main__":
    main()

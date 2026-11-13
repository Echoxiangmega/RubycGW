#!/usr/bin/env python3
"""Prove that the pictured periodic 18-site hexagon and RubycGW hex18 are the SAME torus.

The two primitive conventions differ by a unimodular basis transformation
and a uniform orbital origin shift on triangle B. Checks the full hopping
matrix and all 18 density-interaction links (not just the energy spectrum).

Run from the repository root:
    python research/verify_hex18_equivalence.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

if __package__ in (None, ''):
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from rubycgw.model import RubyParameters, _base_bonds
from rubycgw.models.ruby import INTRATRIANGLE_BONDS
from rubycgw.supercell import (
    SUPERCELL_MATRIX, SUPERCELL_REPRESENTATIVES,
    build_supercell_h0, primitive_cell_to_supercell,
)
from scan_true_hex18_v import RubyTorus, PRIMITIVE_BONDS

# r_drawing = U*r_native + shift[orbital].
U = np.array([[0, 1], [-1, 0]], dtype=int)
ORBITAL_SHIFT = np.array([[0, 0]] * 3 + [[-1, -1]] * 3, dtype=int)


def check(t=1., t1=.5, t2=.5):
    pictured = RubyTorus('true_hex', t=t, t1=t1, t2=t2)
    native = RubyParameters(ti=t, t1=t1, t2=t2, V=1.)

    for (a, b, delta_local, kind), (i, j, delta_native, hop) in zip(
            PRIMITIVE_BONDS, _base_bonds(native)):
        assert (a, b) == (i, j)
        delta = U @ np.asarray(delta_native, dtype=int) + ORBITAL_SHIFT[j] - ORBITAL_SHIFT[i]
        assert np.array_equal(delta, delta_local), (
            (a, b, delta_local, kind), (i, j, delta_native, hop), delta
        )
    assert np.array_equal(U @ SUPERCELL_MATRIX, pictured.M)

    # Map all 18 native sites to pictured labels via the quotient lattice.
    P = np.zeros((18, 18), dtype=float)
    for sec, r in enumerate(SUPERCELL_REPRESENTATIVES):
        for orb in range(6):
            target = U @ r + ORBITAL_SHIFT[orb]
            matches = []
            for loc, rep in enumerate(pictured.reps):
                winding = np.linalg.solve(pictured.M, target - rep)
                if np.allclose(winding, np.rint(winding), atol=1e-12):
                    matches.append(loc)
            assert len(matches) == 1, (sec, orb, matches)
            P[6 * matches[0] + orb, 6 * sec + orb] = 1.
    assert np.all(P.sum(axis=0) == 1.) and np.all(P.sum(axis=1) == 1.)

    H_native = build_supercell_h0(np.zeros((1, 2)), native)[0]
    H_pictured = pictured.one_body()
    err = float(np.max(np.abs(H_pictured - P @ H_native @ P.T)))
    assert err < 1e-12, f'one-body matrix error={err}'

    native_pairs = []
    for sec, r in enumerate(SUPERCELL_REPRESENTATIVES):
        for i, j, delta in INTRATRIANGLE_BONDS:
            sec2, _ = primitive_cell_to_supercell(r + np.asarray(delta, dtype=int))
            native_pairs.append(tuple(sorted((6 * sec + i, 6 * sec2 + j))))
    perm = np.argmax(P, axis=0)
    mapped = sorted(tuple(sorted((int(perm[i]), int(perm[j])))) for i, j in native_pairs)
    local = sorted(tuple(sorted((b.i, b.j))) for b in pictured.bonds if b.kind == 't')
    assert mapped == local

    spec_err = float(np.max(np.abs(
        np.linalg.eigvalsh(H_native) - np.linalg.eigvalsh(H_pictured))))
    print('PASS: pictured true_hex is the same finite torus as existing RubycGW hex18')
    print('  coordinate map U:', U.tolist())
    print('  native supercell matrix:', SUPERCELL_MATRIX.tolist())
    print('  drawing supercell matrix:', pictured.M.tolist())
    print('  max one-body matrix error:', err)
    print('  max one-body eigenvalue error:', spec_err)
    print('  exactly matched density-interaction bonds:', len(local))


if __name__ == '__main__':
    check()

#!/usr/bin/env python3
"""Regression for a degenerate Hex18 ED ground multiplet and VAP sector split."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NATIVE = REPO / "ed_benchmark" / "native_ed"
sys.path.insert(0, str(NATIVE))

import geometry
from run_benchmark import solve_ed
from symmetry_projection import (
    build_translation_c3_group,
    commutator_error,
    decompose_ground_manifold_abelian,
    project_to_characters,
)


def main():
    g = geometry.RubyPBC("ring3", t=1.0, t1=0.5, t2=-0.5)
    H, evals, gs, row = solve_ed(g, 3, 2.5, k=12, degeneracy_tol=1e-8)
    if row.get("ground_manifold_truncated"):
        raise RuntimeError("Increase k in this regression; ground manifold is truncated")
    assert gs.shape[1] > 1, (
        "Reference point was expected to have a degenerate ground multiplet; "
        f"got dimension {gs.shape[1]}"
    )

    group = build_translation_c3_group(g, H.states)
    err, name = commutator_error(group, H.operator, n_tests=2)
    assert err < 1e-10, (name, err)

    sectors = decompose_ground_manifold_abelian(group, gs, tol=1e-7)
    assert sectors
    assert sum(int(s["dimension"]) for s in sectors) == gs.shape[1]

    print("degenerate ground-sector test: PASS")
    print("E0 =", float(evals[0]))
    print("ground_dim =", gs.shape[1])
    print("T x C3 group_order =", len(group))
    print("sector_count =", len(sectors))
    for i, sec in enumerate(sectors):
        phases = [round(float(np.angle(z)/np.pi), 6) for z in sec["characters"]]
        print(
            f"  sector {i}: dim={sec['dimension']} "
            f"residual={sec['residual']:.3e} phases/pi={phases}"
        )

    # Orthogonal sector projectors must resolve the complete ground space.
    for j in range(gs.shape[1]):
        psi = gs[:, j]
        w = 0.0
        for sec in sectors:
            _, ws = project_to_characters(group, sec["characters"], psi)
            w += ws
        assert abs(w-1.0) < 1e-8, (j, w)


if __name__ == "__main__":
    main()

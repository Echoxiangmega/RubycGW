import numpy as np
import pytest

from rubycgw.c3_constraint import (
    c3_lattice_residual,
    density_c3_spread,
    project_density_c3,
    project_lattice_c3,
    rotate_lattice_c3,
)
from rubycgw.cluster_ed_gw_c3_constrained import solve_cluster_ed_gw_fast_c3_constrained
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters, build_h0


def test_c3_rotation_cubes_to_identity():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(2, 3, 3, 6, 6)) + 1j * rng.normal(
        size=(2, 3, 3, 6, 6)
    )
    y = rotate_lattice_c3(rotate_lattice_c3(rotate_lattice_c3(x)))
    assert np.max(np.abs(y - x)) < 1e-12


def test_ruby_h0_is_exactly_c3_covariant_for_independent_t1_t2_orbits():
    grid = MatsubaraGrid(nk1=3, nk2=3, nw=2, nOmega=1, T=0.1)
    h0 = build_h0(grid.kmesh(), RubyParameters(ti=0.4, t1=0.2, t2=0.17, V=1.0))
    assert c3_lattice_residual(h0) < 1e-12
    assert np.max(np.abs(project_lattice_c3(h0) - h0)) < 1e-12


def test_c3_projector_rejects_rectangular_mesh():
    x = np.zeros((2, 3, 4, 6, 6), dtype=complex)
    with pytest.raises(ValueError, match="square reciprocal mesh"):
        project_lattice_c3(x)


def test_projected_random_field_is_c3_invariant():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 3, 3, 6, 6)) + 1j * rng.normal(
        size=(4, 3, 3, 6, 6)
    )
    y = project_lattice_c3(x)
    assert c3_lattice_residual(y) < 2e-12


def test_density_constraint_equalizes_each_triangle_and_preserves_total():
    d = np.asarray([0.34, 0.33, 0.32, 0.31, 0.35, 0.36])
    p = project_density_c3(d)
    assert np.allclose(p[:3], np.mean(d[:3]))
    assert np.allclose(p[3:], np.mean(d[3:]))
    assert np.isclose(np.sum(p), np.sum(d))
    assert density_c3_spread(p) < 1e-15


def test_c3_solver_symbol_is_importable():
    assert callable(solve_cluster_ed_gw_fast_c3_constrained)

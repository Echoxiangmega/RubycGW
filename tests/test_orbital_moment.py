import numpy as np

from rubycgw.model import RubyParameters
from rubycgw.orbital_moment import (
    moment_code_to_muB,
    phase_bond_current,
    plaquette_moment_from_loop_current,
    signed_polygon_area,
    supercell_site_positions,
    triangle_oriented_indices,
    triangle_signed_area,
)


def test_triangle_eta_orientations_have_opposite_geometric_handedness():
    area_a = triangle_signed_area(0, "A")
    area_b = triangle_signed_area(0, "B")
    assert area_a * area_b < 0.0
    assert np.isclose(abs(area_a), abs(area_b), rtol=0.0, atol=1e-14)


def test_triangle_area_is_translation_invariant_across_supercell_sectors():
    for tri in ("A", "B"):
        areas = np.array([triangle_signed_area(s, tri) for s in range(3)])
        assert np.allclose(areas, areas[0], rtol=0.0, atol=1e-14)


def test_uniform_closed_loop_current_gives_current_times_signed_area():
    positions = supercell_site_positions()
    loop_current = 0.37
    for tri in ("A", "B"):
        inds = triangle_oriented_indices(0, tri)
        pts = positions[list(inds)]
        area = signed_polygon_area(pts)

        # Direct discrete m=(1/2) sum I_ij (r_i x r_j)_z.
        direct = 0.0
        for ie in range(3):
            ri = pts[ie]
            rj = pts[(ie + 1) % 3]
            direct += 0.5 * loop_current * (
                ri[0] * rj[1] - ri[1] * rj[0]
            )
        projected = plaquette_moment_from_loop_current(loop_current, area)
        assert np.isclose(direct, projected, rtol=0.0, atol=1e-14)


def test_phase_bond_current_matches_dH_dphi_convention():
    # rho_ab=<c_b^dagger c_a>.  For t=0.4 and
    # <c_0^dagger c_1>=rho[1,0]=+0.25 i,
    # <dH/dphi_01>=-2 Im[t rho[1,0]]=-0.2.
    rho = np.zeros((2, 2), dtype=complex)
    rho[1, 0] = 0.25j
    rho[0, 1] = -0.25j
    assert np.isclose(
        phase_bond_current(rho, 0.4, 0, 1),
        -0.2,
        rtol=0.0,
        atol=1e-14,
    )


def test_physical_moment_conversion_is_linear_in_energy_area_and_degeneracy():
    base = float(
        moment_code_to_muB(
            1.0,
            energy_unit_ev=1.0,
            lattice_constant_angstrom=1.0,
            spin_degeneracy=1.0,
        )
    )
    scaled = float(
        moment_code_to_muB(
            1.0,
            energy_unit_ev=2.0,
            lattice_constant_angstrom=3.0,
            spin_degeneracy=4.0,
        )
    )
    assert base > 0.0
    assert np.isclose(scaled / base, 2.0 * 3.0**2 * 4.0)


def test_default_ruby_triangle_hopping_is_nonzero():
    # Guard the model assumption used by the plaquette-current postprocessor.
    params = RubyParameters()
    assert params.ti != 0.0

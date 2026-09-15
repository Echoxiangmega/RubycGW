import numpy as np

from vprime_study.effective_pseudospin_ed import (
    allowed_qmesh,
    build_sector_hamiltonian,
    effective_couplings,
    parity_basis,
    solve_effective_pseudospin_ed,
    triangle_center_bonds,
)


def test_triangle_center_bond_count_and_orientations():
    bonds = triangle_center_bonds(2, 3)
    assert len(bonds) == 3 * 2 * 3
    assert {g for _i, _j, g in bonds} == {0, 1, 2}


def test_sector_hamiltonian_is_hermitian():
    H = build_sector_hamiltonian(2, 1, 0.13, -0.07, -0.11, parity=0).matrix
    delta = H - H.getH()
    assert delta.nnz == 0 or np.max(np.abs(delta.data)) < 1e-12


def test_parity_basis_is_half_for_even_nspin():
    assert parity_basis(8, 0).size == 128
    assert parity_basis(8, 1).size == 128


def test_effective_couplings_match_vcross_formula():
    t, V, vp, vx = 0.2, 1.8, -0.1, -0.05
    s = t * t / V
    Jn, Jm, Jz = effective_couplings(t=t, V=V, Vprime=vp, Vcross=vx)
    assert np.isclose(Jn, 5*s/9 + (vp+vx)/18)
    assert np.isclose(Jm, -s/3 + (-vp+vx)/6)
    assert np.isclose(Jz, -s/3)


def test_pure_current_ising_limit_has_exact_energy_and_order():
    Jz = -0.2
    r = solve_effective_pseudospin_ed(
        Lx=2, Ly=1, Jn=0.0, Jm=0.0, Jz=Jz,
        nev=4, tol=1e-12, degeneracy_tol=1e-10,
    )
    ncell = 2
    assert np.isclose(r.ground_energy, 3*ncell*Jz)
    gamma = int(np.argmin(np.linalg.norm(r.q_centered, axis=1)))
    assert np.isclose(r.z_same[gamma] / r.nspin, 1.0)


def test_qmesh_contains_K_on_3x3_but_not_M():
    _raw, q = allowed_qmesh(3, 3)
    assert np.any(np.linalg.norm(q - np.array([1/3, 1/3]), axis=1) < 1e-12)
    assert not np.any(np.isclose(np.abs(q[:, 0]), 0.5))

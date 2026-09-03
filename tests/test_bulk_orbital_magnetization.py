import numpy as np

from rubycgw.bulk_orbital_magnetization import (
    bulk_orbital_magnetization_from_arrays,
    moment_code_to_muB,
    spectral_cartesian_covariant_derivatives,
    supercell_h0_cartesian_derivatives,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.orbital_moment import supercell_site_positions, supercell_vectors_cart
from rubycgw.supercell import NSUP, build_supercell_h0, supercell_hoppings
from rubycgw.supercell_gw import dyson_from_sigma_matrix


def _reduced_to_cartesian_k(kred):
    T1, T2 = supercell_vectors_cart()
    rows = np.vstack([T1, T2])
    return np.linalg.solve(rows, 2.0 * np.pi * np.asarray(kred, dtype=float))


def _physical_gauge_h0(kcart, params):
    positions = supercell_site_positions()
    T1, T2 = supercell_vectors_cart()
    out = np.zeros((NSUP, NSUP), dtype=complex)
    for I, J, S, amp in supercell_hoppings(params):
        S = np.asarray(S, dtype=float)
        d = S[0] * T1 + S[1] * T2 + positions[J] - positions[I]
        out[I, J] += complex(amp) * np.exp(1j * np.dot(kcart, d))
    return 0.5 * (out + out.conj().T)


def test_h0_embedding_derivative_matches_physical_gauge_finite_difference():
    params = RubyParameters(ti=0.4, t1=0.23, t2=-0.17, V=0.0)
    kred = np.array([0.217, 0.371])
    Dx, Dy = supercell_h0_cartesian_derivatives(kred[None, :], params)
    Dx = Dx[0]
    Dy = Dy[0]

    positions = supercell_site_positions()
    kcart = _reduced_to_cartesian_k(kred)
    U = np.diag(np.exp(1j * positions @ kcart))
    Dx_phys = U.conj().T @ Dx @ U
    Dy_phys = U.conj().T @ Dy @ U

    h = 1.0e-6
    ex = np.array([1.0, 0.0])
    ey = np.array([0.0, 1.0])
    fd_x = (_physical_gauge_h0(kcart + h * ex, params) - _physical_gauge_h0(kcart - h * ex, params)) / (2.0 * h)
    fd_y = (_physical_gauge_h0(kcart + h * ey, params) - _physical_gauge_h0(kcart - h * ey, params)) / (2.0 * h)

    assert np.max(np.abs(Dx_phys - fd_x)) < 2.0e-10
    assert np.max(np.abs(Dy_phys - fd_y)) < 2.0e-10


def test_spectral_derivative_of_known_bravais_harmonic():
    nk1, nk2 = 5, 3
    k1 = np.arange(nk1, dtype=float) / nk1
    field = np.zeros((nk1, nk2, NSUP, NSUP), dtype=complex)
    for i, x in enumerate(k1):
        field[i, :, 0, 0] = np.cos(2.0 * np.pi * x)

    Dx, Dy = spectral_cartesian_covariant_derivatives(field)
    T1, _ = supercell_vectors_cart()
    expected_x = -T1[0] * np.sin(2.0 * np.pi * k1)
    expected_y = -T1[1] * np.sin(2.0 * np.pi * k1)

    assert np.max(np.abs(Dx[:, :, 0, 0] - expected_x[:, None])) < 1.0e-12
    assert np.max(np.abs(Dy[:, :, 0, 0] - expected_y[:, None])) < 1.0e-12


def test_k_independent_offdiagonal_field_keeps_embedding_derivative():
    positions = supercell_site_positions()
    field = np.zeros((3, 3, NSUP, NSUP), dtype=complex)
    field[..., 0, 1] = 0.37 - 0.11j
    Dx, Dy = spectral_cartesian_covariant_derivatives(field)

    delta = positions[1] - positions[0]
    expected_x = 1j * delta[0] * (0.37 - 0.11j)
    expected_y = 1j * delta[1] * (0.37 - 0.11j)
    assert np.allclose(Dx[..., 0, 1], expected_x, rtol=0.0, atol=1.0e-14)
    assert np.allclose(Dy[..., 0, 1], expected_y, rtol=0.0, atol=1.0e-14)


def test_local_diagonal_self_energy_has_zero_physical_k_derivative():
    rng = np.random.default_rng(7)
    diag = rng.normal(size=NSUP) + 1j * rng.normal(size=NSUP)
    field = np.zeros((4, 3, 3, NSUP, NSUP), dtype=complex)
    idx = np.diag_indices(NSUP)
    field[..., idx[0], idx[1]] = diag
    Dx, Dy = spectral_cartesian_covariant_derivatives(field)
    assert np.max(np.abs(Dx)) < 1.0e-13
    assert np.max(np.abs(Dy)) < 1.0e-13


def test_noninteracting_time_reversal_symmetric_ruby_has_zero_bulk_m():
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.17, V=0.0)
    grid = MatsubaraGrid(nk1=3, nk2=3, nw=12, nOmega=1, T=0.20)
    h0 = build_supercell_h0(grid.kmesh(), params)
    sigma_h = np.zeros((NSUP, NSUP), dtype=complex)
    sigma_gw = np.zeros((grid.nf, grid.nk1, grid.nk2, NSUP, NSUP), dtype=complex)
    mu = 0.13
    G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)

    result = bulk_orbital_magnetization_from_arrays(
        h0, G, sigma_gw, mu, grid, params
    )
    assert result.complete
    assert result.field_self_energy_status == "exact_zero_noninteracting"
    assert abs(result.main_term_code) < 2.0e-11
    assert abs(result.total_code) < 2.0e-11


def test_interacting_result_is_marked_incomplete_without_sigma_B():
    params = RubyParameters(V=0.2)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=2, nOmega=1, T=0.3)
    h0 = build_supercell_h0(grid.kmesh(), params)
    sigma = np.zeros((grid.nf, 1, 1, NSUP, NSUP), dtype=complex)
    G = dyson_from_sigma_matrix(h0, grid, 0.0, np.zeros((NSUP, NSUP)), sigma)
    result = bulk_orbital_magnetization_from_arrays(h0, G, sigma, 0.0, grid, params)
    assert not result.complete
    assert result.total_code is None
    assert result.field_self_energy_term_code is None


def test_zero_sigma_B_completes_second_term_interface():
    params = RubyParameters(V=0.2)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=2, nOmega=1, T=0.3)
    h0 = build_supercell_h0(grid.kmesh(), params)
    sigma = np.zeros((grid.nf, 1, 1, NSUP, NSUP), dtype=complex)
    G = dyson_from_sigma_matrix(h0, grid, 0.0, np.zeros((NSUP, NSUP)), sigma)
    result = bulk_orbital_magnetization_from_arrays(
        h0, G, sigma, 0.0, grid, params, sigma_b=np.zeros_like(sigma)
    )
    assert result.complete
    assert result.field_self_energy_status == "provided"
    assert result.field_self_energy_term_code == 0.0
    assert np.isclose(result.total_code, result.main_term_code)


def test_physical_conversion_scales_as_energy_area_and_degeneracy():
    base = float(moment_code_to_muB(
        1.0,
        energy_unit_ev=1.0,
        lattice_constant_angstrom=1.0,
        spin_degeneracy=1.0,
    ))
    scaled = float(moment_code_to_muB(
        1.0,
        energy_unit_ev=2.0,
        lattice_constant_angstrom=3.0,
        spin_degeneracy=4.0,
    ))
    assert base > 0.0
    assert np.isclose(scaled / base, 2.0 * 3.0**2 * 4.0)

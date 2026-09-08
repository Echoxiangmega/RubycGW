import numpy as np

from rubycgw.response_mode_diagnostics import (
    combine_complex_q_vertex,
    hermitian_leading_mode,
    mode_response,
    vertex_pole_diagnostics,
)


def test_hermitian_leading_mode_inside_subspace():
    M = np.array(
        [
            [5.0, 0.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=complex,
    )
    val, vec = hermitian_leading_mode(M, [1, 2])
    assert abs(val - 3.0) < 1e-12
    assert abs(vec[0]) < 1e-12
    assert abs(mode_response(M, vec) - 3.0) < 1e-12


def test_complex_q_vertex_matches_qc_minus_i_qs_convention():
    # One internal channel, scalar orbital matrices for clarity.
    verts = np.array([[[2.0]], [[4.0]]], dtype=complex)  # Qc, Qs
    got = combine_complex_q_vertex(verts, np.array([1.0]))
    expected = (2.0 - 4.0j) / np.sqrt(2.0)
    assert np.max(np.abs(got - expected)) < 1e-12


def test_vertex_pole_diagnostics_exact_eigenmode():
    # L Gamma = lambda Gamma, and K=(1-lambda)Gamma exactly.
    lam = 0.8
    Gamma = np.array([1.0 + 2.0j, -0.5j], dtype=complex)
    Lg = lam * Gamma
    K = Gamma - Lg
    got = vertex_pole_diagnostics(K, Gamma, Lg)
    assert abs(got.source_gain - 5.0) < 1e-12
    assert abs(got.sigma_source_upper - 0.2) < 1e-12
    assert abs(got.lambda_eff - lam) < 1e-12
    assert got.lambda_eigen_residual < 1e-12
    assert got.equation_residual_relative < 1e-12


def test_sigma_source_is_upper_bound_in_diagonal_example():
    # A=diag(0.1, 0.5), so global sigma_min=0.1.  Driving the second direction
    # gives sigma_source=0.5, correctly only an upper bound on the global value.
    Gamma = np.array([0.0, 2.0], dtype=complex)
    Lg = np.array([0.0, 1.0], dtype=complex)  # L=diag(0.9,0.5)
    K = Gamma - Lg
    got = vertex_pole_diagnostics(K, Gamma, Lg)
    assert abs(got.sigma_source_upper - 0.5) < 1e-12
    assert got.sigma_source_upper >= 0.1

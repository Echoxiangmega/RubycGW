import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.supercell import (
    NSUP,
    SUPERCELL_MATRIX,
    build_supercell_h0,
    build_supercell_interaction,
    charge_order_parameter,
    folded_primitive_eigenvalues,
    period3_real_pattern,
    primitive_cell_to_supercell,
)
from rubycgw.supercell_gw import solve_supercell_gw


def test_supercell_cell_decomposition():
    T1 = SUPERCELL_MATRIX[:, 0]
    T2 = SUPERCELL_MATRIX[:, 1]
    for x in range(-4, 5):
        for y in range(-4, 5):
            s, S = primitive_cell_to_supercell((x, y))
            reconstructed = np.array([s, 0]) + S[0] * T1 + S[1] * T2
            assert s in (0, 1, 2)
            assert np.array_equal(reconstructed, np.array([x, y]))


def test_supercell_h0_is_hermitian_and_exact_band_folding():
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0)
    k = np.array([[0.137, 0.291], [0.0, 0.0], [0.37, 0.11]])
    h = build_supercell_h0(k, params)
    assert h.shape == (3, NSUP, NSUP)
    assert np.max(np.abs(h - np.swapaxes(h.conj(), -1, -2))) < 1e-13

    esc = np.sort(np.linalg.eigvalsh(h), axis=-1)
    eref = folded_primitive_eigenvalues(k, params)
    assert np.max(np.abs(esc - eref)) < 1e-12


def test_supercell_interaction_is_hermitian_and_has_coordination_four_at_q0():
    V = 0.37
    params = RubyParameters(V=V)
    q = np.array([[0.13, 0.29], [0.0, 0.0]])
    vq = build_supercell_interaction(q, params)
    assert vq.shape == (2, NSUP, NSUP)
    assert np.max(np.abs(vq - np.swapaxes(vq.conj(), -1, -2))) < 1e-13
    assert np.allclose(np.sum(vq[1], axis=1).real, 4.0 * V, atol=1e-13)
    assert np.max(np.abs(np.sum(vq[1], axis=1).imag)) < 1e-13


def test_period3_charge_pattern_and_order_parameter_normalization():
    pattern = period3_real_pattern()
    expected = np.array([
        [1.0, -0.5, -0.5, -1.0, 0.5, 0.5],
        [-0.5, -0.5, 1.0, 0.5, 0.5, -1.0],
        [-0.5, 1.0, -0.5, 0.5, -1.0, 0.5],
    ])
    assert np.allclose(pattern.reshape(3, 6), expected, atol=1e-13)
    assert abs(np.sum(pattern)) < 1e-13

    amplitude = 0.031
    density = 0.5 + amplitude * pattern
    phi = charge_order_parameter(density)
    assert abs(phi.real - amplitude) < 1e-13
    assert abs(phi.imag) < 1e-13


def test_v_zero_supercell_gw_hits_half_filling():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=3, nOmega=1, T=0.1)
    opts = GWOptions(
        target_filling=9.0,
        max_iter=4,
        tol=1e-10,
        mixing=0.5,
        verbose=False,
    )
    result = solve_supercell_gw(params, grid, opts)
    assert result.converged
    assert result.G.shape[-2:] == (NSUP, NSUP)
    assert result.min_screening_mode.shape == (NSUP,)
    assert result.min_density_mode.shape == (NSUP,)
    assert abs(np.sum(result.density) - 9.0) < 1e-8
    assert result.final_error < opts.tol
    assert abs(result.min_screening_singular_value - 1.0) < 1e-12

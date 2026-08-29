import numpy as np

from rubycgw.model import RubyParameters, build_h0, build_interaction, eta_vertices
from rubycgw.grids import MatsubaraGrid, shift_fermion_field
from rubycgw.gw import GWOptions, build_g0_inverse, solve_gw, solve_noninteracting
from rubycgw.susceptibility import chi_eta


def test_h0_hermitian():
    params = RubyParameters()
    k = np.array([[0.137, 0.291], [0.0, 0.0]])
    h = build_h0(k, params)
    assert np.max(np.abs(h - np.swapaxes(h.conj(), -1, -2))) < 1e-13


def test_interaction_hermitian():
    params = RubyParameters(V=0.37)
    q = np.array([[0.13, 0.29], [0.0, 0.0]])
    v = build_interaction(q, params)
    assert np.max(np.abs(v - np.swapaxes(v.conj(), -1, -2))) < 1e-13


def test_eta_vertices_hermitian_and_labels():
    ka, kb, kp, km = eta_vertices()
    for k in [ka, kb, kp, km]:
        assert np.max(np.abs(k - k.conj().T)) < 1e-13
    assert np.allclose(kp, (ka + kb) / np.sqrt(2.0))
    assert np.allclose(km, (ka - kb) / np.sqrt(2.0))


def test_shift_rule_k_to_k_plus_q():
    f = np.zeros((5, 3, 4, 1, 1), dtype=float)
    for n in range(5):
        for i in range(3):
            for j in range(4):
                f[n, i, j, 0, 0] = 100*n + 10*i + j
    s = shift_fermion_field(f, 1, 2, 1)
    assert s[0, 0, 0, 0, 0] == f[1, 1, 2, 0, 0]


def test_noninteracting_solver_hits_target_filling():
    params = RubyParameters(V=0.1)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=12, nOmega=2, T=0.1)
    result = solve_noninteracting(params, grid, target_filling=2.0)
    assert abs(np.sum(result.density) - 2.0) < 1e-8


def test_noninteracting_solver_brackets_near_empty_and_full():
    """Analytic tail subtraction must permit fillings close to 0 and 6."""
    params = RubyParameters(V=3.0)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=6, nOmega=2, T=0.05)
    for target in (0.05, 5.95):
        result = solve_noninteracting(params, grid, target_filling=target)
        assert abs(np.sum(result.density) - target) < 1e-8


def test_gw_result_reports_final_error():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=6, nOmega=2, T=0.1)
    opts = GWOptions(
        target_filling=2.0,
        max_iter=8,
        tol=1e-10,
        mixing=0.5,
        verbose=False,
    )
    result = solve_gw(params, grid, opts)
    assert result.converged
    assert np.isfinite(result.final_error)
    assert result.final_error < opts.tol


def test_gw_pulay_api_and_screening_diagnostic():
    """Pulay mode is accepted and V=0 has the exact unit screening denominator."""
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=4, nOmega=1, T=0.1)
    opts = GWOptions(
        target_filling=3.0,
        max_iter=6,
        tol=1e-10,
        mixing=0.7,
        mixing_method="pulay",
        pulay_history=4,
        pulay_start=2,
        verbose=False,
    )
    result = solve_gw(params, grid, opts)
    assert result.converged
    assert result.mixing_method == "pulay"
    assert abs(result.min_screening_singular_value - 1.0) < 1e-12
    assert np.isfinite(result.min_screening_Omega)
    assert np.isfinite(result.min_screening_q1)
    assert np.isfinite(result.min_screening_q2)


def test_v_zero_bare_susceptibility_is_finite():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=8, nOmega=2, T=0.1)
    h0 = build_h0(grid.kmesh(), params)
    G0 = np.linalg.inv(build_g0_inverse(h0, grid, mu=0.0))
    _, _, kp, km = eta_vertices()
    cp = chi_eta(G0, kp, grid)
    cm = chi_eta(G0, km, grid)
    assert np.isfinite(cp.real) and np.isfinite(cp.imag)
    assert np.isfinite(cm.real) and np.isfinite(cm.imag)

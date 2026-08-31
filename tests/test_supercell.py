import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters, build_interaction
from rubycgw.supercell import (
    NSUP,
    Q_PERIOD3,
    SUPERCELL_MATRIX,
    build_supercell_h0,
    build_supercell_interaction,
    charge_order_parameter,
    folded_primitive_eigenvalues,
    period3_real_pattern,
    primitive_cell_to_supercell,
    supercell_hoppings,
    supercell_to_primitive_momenta,
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


def test_supercell_has_index_three_and_q_period3_folds_to_gamma():
    # Columns are T1=a1-a2 and T2=a1+2a2 in primitive coordinates.
    assert round(abs(np.linalg.det(SUPERCELL_MATRIX))) == 3
    phases = SUPERCELL_MATRIX.T @ Q_PERIOD3
    # Q.T1=0 and Q.T2=1, hence exp(2*pi*i*Q.Tj)=1 for both supercell translations.
    assert np.allclose(phases, np.array([0.0, 1.0]), atol=1e-14)

    # The three reciprocal cosets folded onto one supercell momentum are exactly
    # k0, k0+Q and k0+2Q modulo primitive reciprocal lattice vectors.
    ksc = np.array([0.137, 0.291])
    folded = supercell_to_primitive_momenta(ksc)
    base = np.linalg.inv(SUPERCELL_MATRIX).T @ ksc
    expected = np.stack([(base + m * Q_PERIOD3) % 1.0 for m in range(3)])
    assert np.allclose(folded, expected, atol=1e-14)


def test_supercell_h0_is_hermitian_and_exact_band_folding():
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0)
    k = np.array([[0.137, 0.291], [0.0, 0.0], [0.37, 0.11]])
    h = build_supercell_h0(k, params)
    assert h.shape == (3, NSUP, NSUP)
    assert np.max(np.abs(h - np.swapaxes(h.conj(), -1, -2))) < 1e-13

    esc = np.sort(np.linalg.eigvalsh(h), axis=-1)
    eref = folded_primitive_eigenvalues(k, params)
    assert np.max(np.abs(esc - eref)) < 1e-12


def test_supercell_h0_exact_folding_with_distinct_hoppings():
    # Use deliberately unequal/sign-different hoppings so an accidental t1/t2
    # interchange or missing bond cannot hide behind the symmetric test above.
    params = RubyParameters(ti=0.43, t1=0.27, t2=-0.16, V=0.0)
    k = np.array([[0.137, 0.291], [0.37, 0.11], [0.23, 0.41]])
    esc = np.sort(np.linalg.eigvalsh(build_supercell_h0(k, params)), axis=-1)
    eref = folded_primitive_eigenvalues(k, params)
    assert np.max(np.abs(esc - eref)) < 1e-12


def test_supercell_hopping_graph_has_exact_coordination_and_bond_count():
    params = RubyParameters(ti=0.43, t1=0.27, t2=-0.16)
    directed = supercell_hoppings(params)
    # Primitive Ruby cell has 12 undirected NN bonds = 24 directed bonds.
    # The index-three supercell must therefore contain 36 undirected = 72 directed bonds.
    assert len(directed) == 72

    degree = np.zeros(NSUP, dtype=int)
    seen = set()
    for I, J, S, _ in directed:
        degree[I] += 1
        key = (int(I), int(J), int(S[0]), int(S[1]))
        assert key not in seen
        seen.add(key)
    assert np.array_equal(degree, np.full(NSUP, 4, dtype=int))


def test_supercell_interaction_is_hermitian_and_has_coordination_four_at_q0():
    V = 0.37
    params = RubyParameters(V=V)
    q = np.array([[0.13, 0.29], [0.0, 0.0]])
    vq = build_supercell_interaction(q, params)
    assert vq.shape == (2, NSUP, NSUP)
    assert np.max(np.abs(vq - np.swapaxes(vq.conj(), -1, -2))) < 1e-13
    assert np.allclose(np.sum(vq[1], axis=1).real, 4.0 * V, atol=1e-13)
    assert np.max(np.abs(np.sum(vq[1], axis=1).imag)) < 1e-13


def test_supercell_interaction_has_exact_primitive_folding():
    V = 0.37
    params = RubyParameters(ti=0.43, t1=0.27, t2=-0.16, V=V)
    qsc = np.array([[0.13, 0.29], [0.0, 0.0], [0.31, 0.17]])
    vsc = build_supercell_interaction(qsc, params)
    esc = np.sort(np.linalg.eigvalsh(vsc), axis=-1)

    folded = supercell_to_primitive_momenta(qsc)
    vprim = build_interaction(folded, params)
    eprim = np.linalg.eigvalsh(vprim)
    eref = np.sort(eprim.reshape(qsc.shape[0], NSUP), axis=-1)
    assert np.max(np.abs(esc - eref)) < 1e-12


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

import numpy as np

from rubycgw.dynamic_cgw import (
    DynamicVertexOptions,
    _bare_vertex_field_iomega,
    susceptibility_matrix_iomega,
)
from rubycgw.fock_diagnostic import fock_bond_diagnostic
from rubycgw.grids import MatsubaraGrid
from rubycgw.production_dynamic_cgw import solve_vertex_iomega_tail
from rubycgw.response_tail import build_tail_reference


def test_bond_reduction_matches_full_frequency_solver():
    for m in (0, 1):
        _check_bond_reduction(m)


def _check_bond_reduction(m):
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=8, nOmega=2, T=.2)
    h0 = np.array([[0., .3], [.3, .1]])[None, None]
    ref = build_tail_reference(h0, .05, np.zeros((2, 2)), grid)
    V = np.array([[0., .15], [.15, 0.]])[None, None]
    K = np.array([[0, 1j], [-1j, 0]])
    r = fock_bond_diagnostic(ref.gref, V, K, m, grid, ref)
    opts = DynamicVertexOptions(
        include_hartree=False, include_fock=True,
        include_mt=False, include_al=False,
        solver='gmres', tol=1e-11, verbose=False,
    )
    sol = solve_vertex_iomega_tail(
        ref.gref, np.broadcast_to(V, (grid.nb, 1, 1, 2, 2)),
        V, K, m, grid, ref, opts,
    )
    assert sol.converged
    np.testing.assert_allclose(
        _bare_vertex_field_iomega(r['gamma'][-1], ref.gref, m, grid),
        sol.Gamma, atol=1e-10,
    )
    chi = susceptibility_matrix_iomega(
        ref.gref, K[None], [sol.Gamma], m, grid,
    )[0, 0]
    np.testing.assert_allclose(r['response'][-1], chi, atol=1e-10)
    assert np.max(r['residual']) < 1e-12

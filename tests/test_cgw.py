import numpy as np

from rubycgw import (
    RubyParameters,
    MatsubaraGrid,
    VertexOptions,
    build_h0,
    build_interaction,
    eta_vertices,
    solve_vertex_q0,
    chi_eta,
)
from rubycgw.gw import build_g0_inverse


def test_v_zero_cgw_reduces_to_bare_vertex_and_bubble():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=6, nOmega=2, T=0.1)
    h0 = build_h0(grid.kmesh(), params)
    G0 = np.linalg.inv(build_g0_inverse(h0, grid, mu=0.0))
    Vq0 = build_interaction(grid.qmesh(), params)[0, 0]
    W = np.zeros((grid.nb, grid.nk1, grid.nk2, 6, 6), dtype=complex)
    _, _, Kp, _ = eta_vertices()

    opts = VertexOptions(max_iter=3, tol=1e-13, mixing=1.0, verbose=False)
    vertex = solve_vertex_q0(G0, W, Vq0, Kp, grid, opts)
    Kfield = np.broadcast_to(Kp, G0.shape)

    assert np.max(np.abs(vertex.Gamma - Kfield)) < 1e-13
    chi_bare = chi_eta(G0, Kp, grid)
    chi_cgw = chi_eta(G0, Kp, grid, Gamma=vertex.Gamma)
    assert abs(chi_cgw - chi_bare) < 1e-12

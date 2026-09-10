import numpy as np

from rubycgw.cpt import (
    cpt_green_from_cluster,
    intercluster_hopping,
    isolated_primitive_h0,
    solve_primitive_cpt_fixed_filling,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters, build_h0
from rubycgw.pseudospin import primitive_pseudospin_vertex


def test_isolated_plus_intercluster_reconstructs_bloch_hamiltonian():
    p = RubyParameters(ti=0.4, t1=0.2, t2=0.17, V=1.0)
    kpts = np.asarray(
        [
            [[0.0, 0.0]],
            [[0.5, 0.0]],
            [[0.25, 0.5]],
        ],
        dtype=float,
    )
    hcl = isolated_primitive_h0(p)
    tinter = intercluster_hopping(kpts, p, h_cluster=hcl)
    hk = build_h0(kpts, p)
    np.testing.assert_allclose(hcl[None, None] + tinter, hk, atol=1e-13, rtol=1e-13)


def test_cpt_embedding_identity_when_intercluster_hopping_zero():
    rng = np.random.default_rng(123)
    nf = 5
    A = rng.normal(size=(nf, 6, 6)) + 1j * rng.normal(size=(nf, 6, 6))
    Gcl = np.empty_like(A)
    for n in range(nf):
        mat = A[n] + 8.0 * np.eye(6)
        Gcl[n] = np.linalg.inv(mat)
    tinter = np.zeros((2, 1, 6, 6), dtype=complex)
    G = cpt_green_from_cluster(Gcl, tinter)
    np.testing.assert_allclose(G[:, 0, 0], Gcl, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(G[:, 1, 0], Gcl, atol=1e-13, rtol=1e-13)


def test_cpt_noninteracting_limit_reconstructs_exact_lattice_green():
    p = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0)
    grid = MatsubaraGrid(nk1=2, nk2=1, nw=12, nOmega=4, T=0.08)
    K = primitive_pseudospin_vertex("z_same")
    hcell = 0.03
    out = solve_primitive_cpt_fixed_filling(
        p,
        grid,
        V=0.0,
        filling=2.0,
        source_vertex=K,
        source_per_cell=hcell,
        mu_tol=1e-9,
        mu_max_iter=80,
        discard_weight_tol=0.0,
    )
    hk = build_h0(grid.kmesh(), p) - hcell * K[None, None]
    eye = np.eye(6, dtype=complex)
    invg = (
        (1j * grid.omega[:, None, None, None, None] + out.mu)
        * eye[None, None, None]
        - hk[None]
    )
    Gexact = np.linalg.inv(invg)
    np.testing.assert_allclose(out.G, Gexact, atol=2e-10, rtol=2e-10)
    assert abs(out.filling - 2.0) < 2e-8

import numpy as np

from rubycgw.cluster_ed_gw import BathParameters, bath_hybridization
from rubycgw.cluster_ed_gw_jf import (
    BathTangentModel,
    ClusterEmbeddedJacobian,
    ClusterJFOptions,
    _bath_from_theta,
    _bath_theta,
    _complex_delta_derivatives,
    _pack_complex,
    _unpack_complex,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.pseudospin import primitive_pseudospin_vertex


def test_real_pack_roundtrip():
    rng = np.random.default_rng(123)
    z = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    x = _pack_complex(z)
    assert x.dtype.kind == "f"
    assert np.allclose(_unpack_complex(x, z.shape), z)


def test_bath_theta_roundtrip_complex():
    bath = BathParameters(
        energies=np.array([-0.7, 0.4]),
        couplings=np.array(
            [
                [0.3 + 0.1j, -0.2 + 0.05j],
                [0.12 - 0.08j, 0.22 + 0.13j],
            ],
            dtype=complex,
        ),
        fit_error=0.0,
        nfev=0,
    )
    theta = _bath_theta(bath)
    rebuilt = _bath_from_theta(theta, 2, 2)
    assert np.allclose(rebuilt.energies, bath.energies)
    assert np.allclose(rebuilt.couplings, bath.couplings)


def test_complex_delta_derivatives_match_centered_difference():
    rng = np.random.default_rng(7)
    norb = 3
    nbath = 2
    omega = np.array([0.21, 0.63, 1.05])
    mu = 0.17
    bath = BathParameters(
        energies=np.array([-0.55, 0.72]),
        couplings=(
            0.25 * rng.normal(size=(norb, nbath))
            + 0.18j * rng.normal(size=(norb, nbath))
        ),
        fit_error=0.0,
        nfev=0,
    )
    theta = _bath_theta(bath)
    analytic = list(_complex_delta_derivatives(omega, mu, bath))
    assert len(analytic) == nbath + 2 * norb * nbath

    h = 2.0e-7
    for j, deriv in enumerate(analytic):
        direction = np.zeros_like(theta)
        direction[j] = 1.0
        bp = _bath_from_theta(theta + h * direction, norb, nbath)
        bm = _bath_from_theta(theta - h * direction, norb, nbath)
        dp = bath_hybridization(omega, mu, bp.energies, bp.couplings)
        dm = bath_hybridization(omega, mu, bm.energies, bm.couplings)
        numeric = (dp - dm) / (2.0 * h)
        assert np.allclose(deriv, numeric, rtol=3e-6, atol=3e-8)


def test_zero_kernel_jf_returns_bare_vertex_at_finite_q():
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=2, nOmega=1, T=0.2)
    eye = np.eye(6, dtype=complex)
    G = np.empty((grid.nf, grid.nk1, grid.nk2, 6, 6), dtype=complex)
    for n, w in enumerate(grid.omega):
        G[n] = eye[None, None] / (1j * w - 0.3)
    Gc = np.mean(G, axis=(1, 2))
    zeros_w = np.zeros((grid.nb, grid.nk1, grid.nk2, 6, 6), dtype=complex)
    zeros_v = np.zeros((grid.nk1, grid.nk2, 6, 6), dtype=complex)
    zeros_wc = np.zeros((grid.nb, 6, 6), dtype=complex)

    pos = np.flatnonzero(grid.omega > 0)[:1]
    target_size = 2 * len(pos) * 6 * 6
    tangent = BathTangentModel(
        theta0=np.zeros(1),
        mode_vectors=np.zeros((1, 1)),
        singular_values=np.ones(1),
        Ufit=np.zeros((target_size, 1)),
        sigma_modes=np.zeros((1, grid.nf, 6, 6), dtype=complex),
        inner_matrix=np.eye(1),
        fit_indices=pos,
        fit_weights=np.ones(len(pos)),
        fit_metric="delta",
        model_g0_fit=np.broadcast_to(eye, (len(pos), 6, 6)).copy(),
        Gc_inv=np.linalg.inv(Gc),
        rank=1,
        condition_number=1.0,
        build_seconds=0.0,
        fd_step=1e-4,
    )
    opts = ClusterJFOptions(
        solver="gcrotmk",
        tol=1e-10,
        maxiter=10,
        restart=4,
        recycle_dim=2,
        include_hartree=False,
        include_fock=False,
        include_mt=False,
        include_al=False,
        verbose=False,
    )
    op = ClusterEmbeddedJacobian(
        G,
        zeros_w,
        zeros_v,
        Gc,
        zeros_wc,
        np.zeros((6, 6), dtype=complex),
        grid,
        tangent,
        opts,
    )
    K = primitive_pseudospin_vertex("Ax")
    result = op.solve(K, (1, 0))
    assert result.converged
    assert result.final_error < 1e-10
    assert np.allclose(result.Gamma, np.broadcast_to(K, G.shape), rtol=1e-10, atol=1e-10)

import numpy as np

from rubycgw.fierz_pac import FierzStateCodec
from rubycgw.grids import MatsubaraGrid
from rubycgw.pseudo_arclength import PACOptions, pseudo_arclength_step


def test_pseudo_arclength_crosses_simple_fold():
    # x^2 + p - 1 = 0 has a fold at (x,p)=(0,1) when p is used as
    # the ordinary continuation parameter.  The secant direction below points
    # through the fold toward the x<0 branch.
    def residual(x, p):
        return np.asarray([x[0] ** 2 + p - 1.0])

    out = pseudo_arclength_step(
        np.asarray([0.3]), 0.91,
        np.asarray([0.1]), 0.99,
        0.25,
        residual,
        opts=PACOptions(
            tol=1e-11, max_newton=12, fd_eps=1e-7,
            gmres_rtol=1e-10, gmres_maxiter=20, gmres_restart=5,
        ),
    )
    assert out.converged
    assert out.x[0] < 0.0
    assert abs(out.x[0] ** 2 + out.parameter - 1.0) < 1e-9


def test_fierz_state_codec_roundtrip_preserves_physical_symmetry():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=3, nOmega=1, T=0.2)
    codec = FierzStateCodec(3, grid, target=2.0)
    rng = np.random.default_rng(123)

    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    static = 0.5 * (A + A.conj().T)
    sigma = np.zeros((grid.nf, 1, 1, 3, 3), dtype=complex)
    pos = rng.normal(size=(grid.nw, 3, 3)) + 1j * rng.normal(size=(grid.nw, 3, 3))
    for k in range(grid.nw):
        sigma[grid.nw + k, 0, 0] = pos[k]
        sigma[grid.nw - 1 - k, 0, 0] = pos[k].conj().T

    x = codec.encode(static, sigma, mu=0.37)
    static2, sigma2, mu2 = codec.decode(x)
    np.testing.assert_allclose(static2, static, rtol=0, atol=2e-15)
    np.testing.assert_allclose(sigma2, sigma, rtol=0, atol=2e-15)
    assert abs(mu2 - 0.37) < 1e-15
    assert x.size == codec.size

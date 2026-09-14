import numpy as np

from rubycgw.cluster_ed_gw import BathParameters, bath_hybridization
from rubycgw.cluster_ed_gw_jf import LinearizedBathFit
from rubycgw.grids import MatsubaraGrid


def test_linearized_bath_fit_reproduces_model_tangent():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=5, nOmega=2, T=0.13)
    eps = np.array([-0.8, 0.65])
    couplings = np.array(
        [
            [0.35 + 0.10j, -0.21 + 0.04j],
            [0.08 - 0.16j, 0.27 + 0.12j],
        ],
        dtype=complex,
    )
    bath = BathParameters(eps, couplings, 0.0, 0)
    lin = LinearizedBathFit(grid.omega, 0.17, bath, nfit=5, rcond=1e-12)

    theta0 = lin.theta()
    rng = np.random.default_rng(8)
    dtheta = rng.normal(size=theta0.size)
    h = 1e-7
    bp = lin.bath_from_theta(theta0 + h * dtheta)
    bm = lin.bath_from_theta(theta0 - h * dtheta)
    dp = bath_hybridization(grid.omega, 0.17, bp.energies, bp.couplings)
    dm = bath_hybridization(grid.omega, 0.17, bm.energies, bm.couplings)
    ddelta = (dp - dm) / (2.0 * h)

    recovered = lin.parameter_direction(ddelta)
    brp = lin.bath_from_theta(theta0 + h * recovered)
    brm = lin.bath_from_theta(theta0 - h * recovered)
    rp = bath_hybridization(grid.omega, 0.17, brp.energies, brp.couplings)
    rm = bath_hybridization(grid.omega, 0.17, brm.energies, brm.couplings)
    reconstructed = (rp - rm) / (2.0 * h)

    idx = lin.fit_indices
    den = max(np.linalg.norm(ddelta[idx].ravel()), 1e-300)
    err = np.linalg.norm((reconstructed[idx] - ddelta[idx]).ravel()) / den
    assert err < 2e-6
    assert lin.diagnostics.rank <= lin.diagnostics.n_parameters

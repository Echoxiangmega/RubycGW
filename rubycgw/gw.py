"""Self-consistent GW equations for the spinless six-sublattice Ruby model.

Equation convention used here:

    G^{-1} = G0^{-1} - Sigma_H - Sigma_GW
    P_ab(Q) = int_k G_ab(k+Q) G_ba(k)
    W = V + V P W
    [Sigma_GW(k)]_ab = - int_Q G_ab(k+Q) W_ba(Q)

with int_k = T/N_k sum_{k,n}, int_Q = T/N_k sum_{q,m}.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grids import MatsubaraGrid, frequency_shift_slices, roll_spatial
from .model import NSUB, RubyParameters, build_h0, build_interaction


@dataclass
class GWOptions:
    mu: float = 0.0
    target_filling: float | None = None
    max_iter: int = 100
    tol: float = 1e-8
    mixing: float = 0.25
    mu_tol: float = 1e-10
    mu_max_iter: int = 100
    verbose: bool = True


@dataclass
class GWResult:
    G: np.ndarray
    W: np.ndarray
    P: np.ndarray
    Sigma_H: np.ndarray
    Sigma_GW: np.ndarray
    mu: float
    density: np.ndarray
    converged: bool
    iterations: int


@dataclass
class NonInteractingResult:
    G0: np.ndarray
    mu: float
    density: np.ndarray


def build_g0_inverse(h0: np.ndarray, grid: MatsubaraGrid, mu: float) -> np.ndarray:
    eye = np.eye(NSUB, dtype=complex)
    out = np.empty((grid.nf, grid.nk1, grid.nk2, NSUB, NSUB), dtype=complex)
    for iw, omega in enumerate(grid.omega):
        out[iw] = (1j * omega + mu) * eye - h0
    return out


def dyson_from_sigma(h0: np.ndarray, grid: MatsubaraGrid, mu: float,
                     sigma_h: np.ndarray, sigma_gw: np.ndarray) -> np.ndarray:
    invg = build_g0_inverse(h0, grid, mu)
    invg -= sigma_h[None, None, None, :, :]
    invg -= sigma_gw
    return np.linalg.inv(invg)


def density_from_G(G: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    """Orbital density from a symmetric Matsubara box.

    The +1/2 is the analytic contribution of the 1/(i omega_n) high-frequency
    tail that is missed by the naive symmetric finite sum.
    """
    diag = np.diagonal(G, axis1=-2, axis2=-1)
    n = 0.5 + (grid.T / grid.nk) * np.sum(diag, axis=(0, 1, 2)).real
    return n


def hartree_self_energy(density: np.ndarray, Vq0: np.ndarray) -> np.ndarray:
    sigma = np.zeros((NSUB, NSUB), dtype=complex)
    sigma[np.diag_indices(NSUB)] = Vq0 @ density
    return sigma


def compute_polarization(G: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    """Compute P without allocating a full zero-padded G(k+Q) for every Q."""
    P = np.zeros((grid.nb, grid.nk1, grid.nk2, NSUB, NSUB), dtype=complex)
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue
        Gsrc = G[src]
        Gbase = G[dst]
        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                Gq = roll_spatial(Gsrc, iq1, iq2)
                P[im, iq1, iq2] = pref * np.einsum(
                    "nxyab,nxyba->ab", Gq, Gbase, optimize=True
                )
    return P


def compute_screened_interaction(P: np.ndarray, Vq: np.ndarray,
                                 grid: MatsubaraGrid) -> np.ndarray:
    W = np.zeros_like(P)
    eye = np.eye(NSUB, dtype=complex)
    for im in range(grid.nb):
        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                V = Vq[iq1, iq2]
                W[im, iq1, iq2] = np.linalg.solve(
                    eye - V @ P[im, iq1, iq2], V
                )
    return W


def compute_sigma_gw(G: np.ndarray, W: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    """Compute Sigma_GW using only the valid Matsubara window for each Q."""
    sigma = np.zeros_like(G)
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue
        Gsrc = G[src]
        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                Gq = roll_spatial(Gsrc, iq1, iq2)
                sigma[dst] -= pref * Gq * W[im, iq1, iq2].T[None, None, None, :, :]
    return sigma


def _solve_mu(h0, sigma_h, sigma_gw, grid, target, mu0, tol, max_iter):
    def f(mu):
        G = dyson_from_sigma(h0, grid, mu, sigma_h, sigma_gw)
        return float(np.sum(density_from_G(G, grid)) - target), G

    width = 2.0
    lo, hi = mu0 - width, mu0 + width
    flo, _ = f(lo)
    fhi, _ = f(hi)
    for _ in range(30):
        if flo <= 0 <= fhi:
            break
        width *= 2.0
        lo, hi = mu0 - width, mu0 + width
        flo, _ = f(lo)
        fhi, _ = f(hi)
    else:
        raise RuntimeError("Could not bracket chemical potential for target filling")

    Gmid = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid, Gmid = f(mid)
        if abs(fmid) < tol:
            return mid, Gmid
        if fmid > 0:
            hi = mid
        else:
            lo = mid
    return mid, Gmid


def solve_noninteracting(params: RubyParameters, grid: MatsubaraGrid,
                         mu: float = 0.0,
                         target_filling: float | None = None,
                         mu_tol: float = 1e-10,
                         mu_max_iter: int = 100) -> NonInteractingResult:
    """Construct G0, optionally at the same fixed filling used by GW."""
    h0 = build_h0(grid.kmesh(), params)
    sigma_h = np.zeros((NSUB, NSUB), dtype=complex)
    sigma_gw = np.zeros((grid.nf, grid.nk1, grid.nk2, NSUB, NSUB), dtype=complex)
    if target_filling is None:
        G0 = dyson_from_sigma(h0, grid, mu, sigma_h, sigma_gw)
        mu0 = float(mu)
    else:
        mu0, G0 = _solve_mu(
            h0, sigma_h, sigma_gw, grid, target_filling,
            float(mu), mu_tol, mu_max_iter
        )
    return NonInteractingResult(G0=G0, mu=mu0, density=density_from_G(G0, grid))


def _compatible_initial(initial: GWResult | None, grid: MatsubaraGrid) -> bool:
    if initial is None:
        return False
    expected = (grid.nf, grid.nk1, grid.nk2, NSUB, NSUB)
    return initial.Sigma_GW.shape == expected and initial.Sigma_H.shape == (NSUB, NSUB)


def solve_gw(params: RubyParameters, grid: MatsubaraGrid,
             opts: GWOptions = GWOptions(),
             initial: GWResult | None = None) -> GWResult:
    """Solve self-consistent GW, optionally warm-started from a previous point.

    ``initial`` is useful for continuation scans in V, filling, temperature, or
    nOmega when the fermionic array shape is unchanged.  If the shape differs
    (for example an nk or nw convergence scan), the initial state is ignored.
    The previous self-energies and chemical potential are only initial guesses;
    the equations are always iterated to the requested tolerance for the new
    parameters.
    """
    kpts = grid.kmesh()
    qpts = grid.qmesh()
    h0 = build_h0(kpts, params)
    Vq = build_interaction(qpts, params)
    Vq0 = Vq[0, 0]

    if _compatible_initial(initial, grid):
        sigma_h = np.array(initial.Sigma_H, copy=True)
        sigma_gw = np.array(initial.Sigma_GW, copy=True)
        mu = float(initial.mu)
    else:
        sigma_h = np.zeros((NSUB, NSUB), dtype=complex)
        sigma_gw = np.zeros((grid.nf, grid.nk1, grid.nk2, NSUB, NSUB), dtype=complex)
        mu = float(opts.mu)

    G = dyson_from_sigma(h0, grid, mu, sigma_h, sigma_gw)
    W = np.zeros((grid.nb, grid.nk1, grid.nk2, NSUB, NSUB), dtype=complex)
    P = np.zeros_like(W)

    converged = False
    for it in range(1, opts.max_iter + 1):
        density = density_from_G(G, grid)
        sigma_h_new = hartree_self_energy(density, Vq0)
        P = compute_polarization(G, grid)
        W = compute_screened_interaction(P, Vq, grid)
        sigma_gw_new = compute_sigma_gw(G, W, grid)

        sigma_h = (1.0 - opts.mixing) * sigma_h + opts.mixing * sigma_h_new
        sigma_gw_mixed = (1.0 - opts.mixing) * sigma_gw + opts.mixing * sigma_gw_new

        if opts.target_filling is None:
            Gnew = dyson_from_sigma(h0, grid, mu, sigma_h, sigma_gw_mixed)
        else:
            mu, Gnew = _solve_mu(
                h0, sigma_h, sigma_gw_mixed, grid, opts.target_filling,
                mu, opts.mu_tol, opts.mu_max_iter
            )

        err = max(
            float(np.max(np.abs(Gnew - G))),
            float(np.max(np.abs(sigma_gw_mixed - sigma_gw))),
            float(np.max(np.abs(sigma_h_new - sigma_h))),
        )
        G = Gnew
        sigma_gw = sigma_gw_mixed
        density = density_from_G(G, grid)

        if opts.verbose:
            print(
                f"GW iter {it:4d}: err={err:.3e}, mu={mu:.10f}, "
                f"n={np.sum(density):.10f}"
            )
        if err < opts.tol:
            converged = True
            break

    density = density_from_G(G, grid)
    sigma_h = hartree_self_energy(density, Vq0)
    P = compute_polarization(G, grid)
    W = compute_screened_interaction(P, Vq, grid)
    sigma_gw = compute_sigma_gw(G, W, grid)

    return GWResult(
        G=G, W=W, P=P, Sigma_H=sigma_h, Sigma_GW=sigma_gw,
        mu=mu, density=density, converged=converged, iterations=it,
    )

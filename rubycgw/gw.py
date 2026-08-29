"""Self-consistent GW equations for the spinless six-sublattice Ruby model.

Equation convention used here:

    G^{-1} = G0^{-1} - Sigma_H - Sigma_GW
    P_ab(Q) = int_k G_ab(k+Q) G_ba(k)
    W = V + V P W
    [Sigma_GW(k)]_ab = - int_Q G_ab(k+Q) W_ba(Q)

with int_k = T/N_k sum_{k,n}, int_Q = T/N_k sum_{q,m}.

The default ``momentum_backend='fft'`` evaluates the periodic two-dimensional
momentum convolutions by FFT while retaining the Matsubara sums explicitly.
``momentum_backend='direct'`` preserves the transparent reference loops for
validation.

For fixed-filling calculations, densities use an analytic reference-Green-
function tail subtraction.  GW self-consistency supports both ordinary linear
mixing and a small-history Pulay/DIIS accelerator.  The convergence residual is
the *raw* fixed-point residual of the Hartree and GW self-energies, so its value
is independent of the chosen damping/mixing parameter.
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
    mixing_method: str = "linear"  # "linear" or "pulay"
    pulay_history: int = 6
    pulay_start: int = 3
    pulay_regularization: float = 1e-10
    mu_tol: float = 1e-10
    mu_max_iter: int = 100
    verbose: bool = True
    momentum_backend: str = "fft"  # "fft" or "direct"


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
    final_error: float
    mixing_method: str
    min_screening_singular_value: float
    min_screening_m: int
    min_screening_Omega: float
    min_screening_q1: float
    min_screening_q2: float


@dataclass
class NonInteractingResult:
    G0: np.ndarray
    mu: float
    density: np.ndarray


def _check_backend(backend: str) -> str:
    backend = str(backend).lower()
    if backend not in {"fft", "direct"}:
        raise ValueError("momentum_backend must be 'fft' or 'direct'")
    return backend


def _check_mixing_method(method: str) -> str:
    method = str(method).lower()
    if method not in {"linear", "pulay"}:
        raise ValueError("mixing_method must be 'linear' or 'pulay'")
    return method


def _reverse_fft_spectrum(field: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    """Return the discrete spectrum F_hat(-r) on the chosen periodic axes."""
    return np.conj(np.fft.fftn(np.conj(field), axes=axes))


def _fermi(e_minus_mu: np.ndarray, T: float) -> np.ndarray:
    """Numerically stable Fermi function f(e-mu)."""
    x = np.asarray(e_minus_mu, dtype=float) / float(T)
    out = np.empty_like(x)
    high = x > 40.0
    low = x < -40.0
    mid = ~(high | low)
    out[high] = 0.0
    out[low] = 1.0
    out[mid] = 1.0 / (np.exp(x[mid]) + 1.0)
    return out


def build_g0_inverse(h0: np.ndarray, grid: MatsubaraGrid, mu: float) -> np.ndarray:
    eye = np.eye(NSUB, dtype=complex)
    return (
        (1j * grid.omega[:, None, None, None, None] + mu)
        * eye[None, None, None, :, :]
        - h0[None, :, :, :, :]
    )


def dyson_from_sigma(h0: np.ndarray, grid: MatsubaraGrid, mu: float,
                     sigma_h: np.ndarray, sigma_gw: np.ndarray) -> np.ndarray:
    invg = build_g0_inverse(h0, grid, mu)
    invg -= sigma_h[None, None, None, :, :]
    invg -= sigma_gw
    return np.linalg.inv(invg)


def density_from_G(
    G: np.ndarray,
    grid: MatsubaraGrid,
    h0: np.ndarray | None = None,
    mu: float | None = None,
    sigma_h: np.ndarray | None = None,
) -> np.ndarray:
    """Orbital density with analytic high-frequency tail subtraction.

    If ``h0`` and ``mu`` are supplied, use the static reference

        G_ref^{-1}(k,iw) = iw + mu - h0(k) - Sigma_H

    and evaluate its infinite Matsubara sum analytically.  The finite numerical
    sum is then applied only to ``G-G_ref``.
    """
    diag = np.diagonal(G, axis1=-2, axis2=-1)

    if h0 is None or mu is None:
        return 0.5 + (grid.T / grid.nk) * np.sum(diag, axis=(0, 1, 2)).real

    if sigma_h is None:
        sigma_h = np.zeros((NSUB, NSUB), dtype=complex)

    href = h0 + sigma_h[None, None, :, :]
    href = 0.5 * (href + np.swapaxes(href.conj(), -1, -2))
    evals, evecs = np.linalg.eigh(href)
    weights = np.abs(evecs) ** 2

    occ = _fermi(evals - float(mu), grid.T)
    n_ref = np.sum(weights * occ[..., None, :], axis=(0, 1, 3)) / grid.nk

    denom = (
        1j * grid.omega[:, None, None, None]
        + float(mu)
        - evals[None, :, :, :]
    )
    gref_diag = np.einsum(
        "xyaj,nxyj->nxya", weights, 1.0 / denom, optimize=True
    )
    correction = (
        (grid.T / grid.nk)
        * np.sum(diag - gref_diag, axis=(0, 1, 2)).real
    )
    return n_ref + correction


def hartree_self_energy(density: np.ndarray, Vq0: np.ndarray) -> np.ndarray:
    sigma = np.zeros((NSUB, NSUB), dtype=complex)
    sigma[np.diag_indices(NSUB)] = Vq0 @ density
    return sigma


def compute_polarization_direct(G: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
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


def compute_polarization_fft(G: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    P = np.zeros((grid.nb, grid.nk1, grid.nk2, NSUB, NSUB), dtype=complex)
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue
        A = G[src]
        B = np.swapaxes(G[dst], -1, -2)
        Ahat = np.fft.fftn(A, axes=(1, 2))
        Bhat_minus = _reverse_fft_spectrum(B, axes=(1, 2))
        product = np.sum(Ahat * Bhat_minus, axis=0)
        P[im] = pref * np.fft.ifftn(product, axes=(0, 1))
    return P


def compute_polarization(G: np.ndarray, grid: MatsubaraGrid,
                         backend: str = "fft") -> np.ndarray:
    backend = _check_backend(backend)
    if backend == "fft":
        return compute_polarization_fft(G, grid)
    return compute_polarization_direct(G, grid)


def _screening_lhs(P: np.ndarray, Vq: np.ndarray) -> np.ndarray:
    eye = np.eye(NSUB, dtype=complex)
    Vbatch = Vq[None, :, :, :, :]
    return eye[None, None, None, :, :] - np.matmul(Vbatch, P)


def compute_screened_interaction(P: np.ndarray, Vq: np.ndarray,
                                 grid: MatsubaraGrid) -> np.ndarray:
    lhs = _screening_lhs(P, Vq)
    rhs = np.broadcast_to(Vq[None, :, :, :, :], P.shape)
    return np.linalg.solve(lhs, rhs)


def screening_diagnostic(P: np.ndarray, Vq: np.ndarray,
                         grid: MatsubaraGrid) -> tuple[float, int, float, float, float]:
    """Return the smallest singular value of I-VP and the Q where it occurs."""
    lhs = _screening_lhs(P, Vq)
    svals = np.linalg.svd(lhs, compute_uv=False)
    smin_grid = svals[..., -1]
    flat = int(np.argmin(smin_grid))
    im, iq1, iq2 = np.unravel_index(flat, smin_grid.shape)
    q = grid.qmesh()[iq1, iq2]
    return (
        float(smin_grid[im, iq1, iq2]),
        int(grid.m_values[im]),
        float(grid.Omega[im]),
        float(q[0]),
        float(q[1]),
    )


def compute_sigma_gw_direct(G: np.ndarray, W: np.ndarray,
                            grid: MatsubaraGrid) -> np.ndarray:
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


def compute_sigma_gw_fft(G: np.ndarray, W: np.ndarray,
                          grid: MatsubaraGrid) -> np.ndarray:
    sigma = np.zeros_like(G)
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue
        A = G[src]
        B = np.swapaxes(W[im], -1, -2)[None, :, :, :, :]
        Ahat = np.fft.fftn(A, axes=(1, 2))
        Bhat_minus = _reverse_fft_spectrum(B, axes=(1, 2))
        conv = np.fft.ifftn(Ahat * Bhat_minus, axes=(1, 2))
        sigma[dst] -= pref * conv
    return sigma


def compute_sigma_gw(G: np.ndarray, W: np.ndarray, grid: MatsubaraGrid,
                     backend: str = "fft") -> np.ndarray:
    backend = _check_backend(backend)
    if backend == "fft":
        return compute_sigma_gw_fft(G, W, grid)
    return compute_sigma_gw_direct(G, W, grid)


def _solve_mu(h0, sigma_h, sigma_gw, grid, target, mu0, tol, max_iter):
    def f(mu):
        G = dyson_from_sigma(h0, grid, mu, sigma_h, sigma_gw)
        density = density_from_G(G, grid, h0=h0, mu=mu, sigma_h=sigma_h)
        return float(np.sum(density) - target), G

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
        raise RuntimeError(
            "Could not bracket chemical potential for target filling; "
            f"target={target}, f(lo)={flo}, f(hi)={fhi}"
        )

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
    density = density_from_G(G0, grid, h0=h0, mu=mu0, sigma_h=sigma_h)
    return NonInteractingResult(G0=G0, mu=mu0, density=density)


def _compatible_initial(initial: GWResult | None, grid: MatsubaraGrid) -> bool:
    if initial is None:
        return False
    expected = (grid.nf, grid.nk1, grid.nk2, NSUB, NSUB)
    return initial.Sigma_GW.shape == expected and initial.Sigma_H.shape == (NSUB, NSUB)


def _residual_error(res_h: np.ndarray, res_gw: np.ndarray) -> float:
    return max(
        float(np.max(np.abs(res_h))),
        float(np.max(np.abs(res_gw))),
    )


def _residual_inner(a_h: np.ndarray, a_gw: np.ndarray,
                    b_h: np.ndarray, b_gw: np.ndarray) -> float:
    """Balanced real inner product for Pulay residuals.

    Hartree and dynamic GW blocks have very different element counts, so each
    block is normalized by its own size before the two contributions are added.
    """
    h = np.vdot(a_h, b_h).real / max(a_h.size, 1)
    g = np.vdot(a_gw, b_gw).real / max(a_gw.size, 1)
    return float(h + g)


def _pulay_coefficients(history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                        regularization: float) -> np.ndarray:
    """DIIS coefficients minimizing the residual combination with sum(c)=1."""
    m = len(history)
    B = np.empty((m + 1, m + 1), dtype=float)
    B.fill(0.0)
    for i in range(m):
        _, _, rhi, rgi = history[i]
        for j in range(i, m):
            _, _, rhj, rgj = history[j]
            val = _residual_inner(rhi, rgi, rhj, rgj)
            B[i, j] = val
            B[j, i] = val

    diag_scale = max(float(np.max(np.abs(np.diag(B[:m, :m])))), 1.0)
    B[:m, :m] += float(regularization) * diag_scale * np.eye(m)
    B[:m, m] = 1.0
    B[m, :m] = 1.0
    rhs = np.zeros(m + 1, dtype=float)
    rhs[m] = 1.0
    try:
        sol = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(B, rhs, rcond=None)[0]
    return sol[:m]


def _mixed_self_energies(
    sigma_h: np.ndarray,
    sigma_gw: np.ndarray,
    sigma_h_out: np.ndarray,
    sigma_gw_out: np.ndarray,
    opts: GWOptions,
    it: int,
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the next self-energy iterate using linear or Pulay mixing."""
    alpha = float(opts.mixing)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("GW mixing must lie in (0, 1].")

    res_h = sigma_h_out - sigma_h
    res_gw = sigma_gw_out - sigma_gw

    if opts.mixing_method == "linear":
        return sigma_h + alpha * res_h, sigma_gw + alpha * res_gw

    history.append((
        np.array(sigma_h_out, copy=True),
        np.array(sigma_gw_out, copy=True),
        np.array(res_h, copy=True),
        np.array(res_gw, copy=True),
    ))
    keep = max(int(opts.pulay_history), 2)
    if len(history) > keep:
        del history[:-keep]

    if it < int(opts.pulay_start) or len(history) < 2:
        return sigma_h + alpha * res_h, sigma_gw + alpha * res_gw

    coeff = _pulay_coefficients(history, opts.pulay_regularization)
    h_diis = np.zeros_like(sigma_h)
    gw_diis = np.zeros_like(sigma_gw)
    for c, (hout, gout, _, _) in zip(coeff, history):
        h_diis += c * hout
        gw_diis += c * gout

    # Damped DIIS: alpha=1 is standard DIIS, while smaller alpha blends the
    # extrapolated solution with the current iterate for additional stability.
    return (
        (1.0 - alpha) * sigma_h + alpha * h_diis,
        (1.0 - alpha) * sigma_gw + alpha * gw_diis,
    )


def solve_gw(params: RubyParameters, grid: MatsubaraGrid,
             opts: GWOptions = GWOptions(),
             initial: GWResult | None = None) -> GWResult:
    """Solve self-consistent GW and report fixed-point/screening diagnostics."""
    backend = _check_backend(opts.momentum_backend)
    method = _check_mixing_method(opts.mixing_method)
    if opts.pulay_history < 2:
        raise ValueError("pulay_history must be at least 2")
    if opts.pulay_start < 1:
        raise ValueError("pulay_start must be at least 1")

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
    if opts.target_filling is not None:
        mu, G = _solve_mu(
            h0, sigma_h, sigma_gw, grid, opts.target_filling,
            mu, opts.mu_tol, opts.mu_max_iter
        )

    W = np.zeros((grid.nb, grid.nk1, grid.nk2, NSUB, NSUB), dtype=complex)
    P = np.zeros_like(W)
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    converged = False
    err = float("inf")
    it = 0
    for it in range(1, opts.max_iter + 1):
        density = density_from_G(G, grid, h0=h0, mu=mu, sigma_h=sigma_h)
        sigma_h_out = hartree_self_energy(density, Vq0)
        P = compute_polarization(G, grid, backend=backend)
        W = compute_screened_interaction(P, Vq, grid)
        sigma_gw_out = compute_sigma_gw(G, W, grid, backend=backend)

        res_h = sigma_h_out - sigma_h
        res_gw = sigma_gw_out - sigma_gw
        err = _residual_error(res_h, res_gw)

        if opts.verbose:
            print(
                f"GW iter {it:4d}: residual={err:.3e}, mu={mu:.10f}, "
                f"n={np.sum(density):.10f}, method={method}, backend={backend}"
            )
        if err < opts.tol:
            converged = True
            break

        sigma_h_next, sigma_gw_next = _mixed_self_energies(
            sigma_h, sigma_gw, sigma_h_out, sigma_gw_out,
            opts, it, history,
        )

        if opts.target_filling is None:
            Gnext = dyson_from_sigma(h0, grid, mu, sigma_h_next, sigma_gw_next)
        else:
            mu, Gnext = _solve_mu(
                h0, sigma_h_next, sigma_gw_next, grid, opts.target_filling,
                mu, opts.mu_tol, opts.mu_max_iter
            )

        sigma_h = sigma_h_next
        sigma_gw = sigma_gw_next
        G = Gnext

    # Re-evaluate the map on the returned iterate.  Do not overwrite the
    # returned self-energies with the map output if the solve did not converge;
    # that would make G and Sigma inconsistent and contaminate continuation.
    density = density_from_G(G, grid, h0=h0, mu=mu, sigma_h=sigma_h)
    sigma_h_out = hartree_self_energy(density, Vq0)
    P = compute_polarization(G, grid, backend=backend)
    W = compute_screened_interaction(P, Vq, grid)
    sigma_gw_out = compute_sigma_gw(G, W, grid, backend=backend)
    err = _residual_error(sigma_h_out - sigma_h, sigma_gw_out - sigma_gw)
    converged = bool(err < opts.tol)

    smin, mmin, omin, q1min, q2min = screening_diagnostic(P, Vq, grid)

    return GWResult(
        G=G,
        W=W,
        P=P,
        Sigma_H=sigma_h,
        Sigma_GW=sigma_gw,
        mu=mu,
        density=density,
        converged=converged,
        iterations=it,
        final_error=float(err),
        mixing_method=method,
        min_screening_singular_value=smin,
        min_screening_m=mmin,
        min_screening_Omega=omin,
        min_screening_q1=q1min,
        min_screening_q2=q2min,
    )

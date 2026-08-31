"""Self-consistent static Hartree-Fock seeds for the 18-site Ruby supercell.

This module is intentionally a *seed generator* for the full split-GW solver.
It solves the finite-temperature static Hartree-Fock equations

    G_HF^{-1}(k,iw) = iw + mu - h0(k) - Sigma_H - Sigma_F(k)

at fixed filling, using the same Hartree and bare-V Fock conventions as the
active split-GW implementation.  Because the HF self-energy is static, the
self-consistency loop is evaluated directly from the eigenstates of the
Hermitian HF Hamiltonian; no Matsubara cutoff enters the HF density matrix.

The converged static Fock matrix is then broadcast over fermionic Matsubara
frequencies and returned as ``GWCheckpointSeed.Sigma_GW``.  Thus a subsequent
full GW calculation starts from a genuine self-consistent HF Green function,
not from a previous GW checkpoint or from a merely hand-edited self-energy.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .checkpoint import GWCheckpointSeed
from .grids import MatsubaraGrid
from .supercell_gw import dyson_from_sigma_matrix, hartree_self_energy_matrix
from .supercell_gw_split import compute_static_fock_matrix


@dataclass(frozen=True)
class SupercellHFResult:
    """Converged (or best reached) static HF solution used to seed GW."""

    seed: GWCheckpointSeed
    G: np.ndarray
    density: np.ndarray
    rho: np.ndarray
    Sigma_H: np.ndarray
    Sigma_F: np.ndarray
    mu: float
    converged: bool
    iterations: int
    final_error: float


def _fermi(e_minus_mu: np.ndarray, T: float) -> np.ndarray:
    x = np.asarray(e_minus_mu, dtype=float) / float(T)
    out = np.empty_like(x)
    high = x > 40.0
    low = x < -40.0
    mid = ~(high | low)
    out[high] = 0.0
    out[low] = 1.0
    out[mid] = 1.0 / (np.exp(x[mid]) + 1.0)
    return out


def _static_eigensystem(
    h0: np.ndarray,
    sigma_h: np.ndarray,
    sigma_f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    heff = (
        np.asarray(h0, dtype=complex)
        + np.asarray(sigma_h, dtype=complex)[None, None, :, :]
        + np.asarray(sigma_f, dtype=complex)
    )
    heff = 0.5 * (heff + np.swapaxes(heff.conj(), -1, -2))
    return np.linalg.eigh(heff)


def _filling_from_evals(evals: np.ndarray, mu: float, T: float) -> tuple[float, float]:
    occ = _fermi(evals - float(mu), T)
    nk = int(evals.shape[0] * evals.shape[1])
    filling = float(np.sum(occ) / nk)
    slope = float(np.sum(occ * (1.0 - occ)) / (float(T) * nk))
    return filling, slope


def _solve_mu_static(
    evals: np.ndarray,
    target: float,
    T: float,
    mu0: float,
    tol: float,
    max_iter: int,
) -> float:
    """Safeguarded Newton solve of the exact static-HF filling equation."""
    norb = int(evals.shape[-1])
    target = float(target)
    if target < -tol or target > norb + tol:
        raise ValueError(f"target filling {target} outside [0,{norb}]")

    def residual(mu: float) -> tuple[float, float]:
        n, slope = _filling_from_evals(evals, mu, T)
        return float(n - target), float(slope)

    # A temperature-aware bracket around the instantaneous HF spectrum.  It is
    # expanded if necessary, so this also remains robust for unusually large
    # Hartree shifts at strong coupling.
    pad = max(2.0, 20.0 * float(T))
    lo = float(np.min(evals) - pad)
    hi = float(np.max(evals) + pad)
    flo, _ = residual(lo)
    fhi, _ = residual(hi)
    for _ in range(30):
        if flo <= 0.0 <= fhi:
            break
        pad *= 2.0
        lo = float(np.min(evals) - pad)
        hi = float(np.max(evals) + pad)
        flo, _ = residual(lo)
        fhi, _ = residual(hi)
    else:
        raise RuntimeError(
            "Could not bracket static-HF chemical potential; "
            f"target={target}, f(lo)={flo}, f(hi)={fhi}"
        )

    x = float(np.clip(float(mu0), lo, hi))
    fx, dfx = residual(x)
    best_x, best_abs = x, abs(fx)
    for _ in range(max(int(max_iter), 1)):
        if abs(fx) < float(tol):
            return float(x)
        if fx < 0.0:
            lo = x
        else:
            hi = x

        if np.isfinite(dfx) and dfx > 1.0e-12:
            trial = x - fx / dfx
        else:
            trial = 0.5 * (lo + hi)
        if not np.isfinite(trial) or trial <= lo or trial >= hi:
            trial = 0.5 * (lo + hi)

        ftrial, dftrial = residual(float(trial))
        if abs(ftrial) < best_abs:
            best_x, best_abs = float(trial), abs(ftrial)
        x, fx, dfx = float(trial), float(ftrial), float(dftrial)

    return float(best_x)


def _rho_from_eigensystem(
    evals: np.ndarray,
    evecs: np.ndarray,
    mu: float,
    T: float,
) -> np.ndarray:
    occ = _fermi(evals - float(mu), T)
    rho = np.einsum(
        "xyaj,xyj,xybj->xyab",
        evecs,
        occ,
        evecs.conj(),
        optimize=True,
    )
    return 0.5 * (rho + np.swapaxes(rho.conj(), -1, -2))


def _density_from_rho(rho: np.ndarray) -> np.ndarray:
    diag = np.diagonal(np.asarray(rho), axis1=-2, axis2=-1).real
    return np.mean(diag, axis=(0, 1))


def _hermitize_k_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=complex)
    return 0.5 * (arr + np.swapaxes(arr.conj(), -1, -2))


def solve_supercell_hf_seed(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    target_filling: float,
    *,
    mu0: float = 0.0,
    max_iter: int = 500,
    tol: float = 1.0e-9,
    mixing: float = 0.25,
    mu_tol: float = 1.0e-12,
    mu_max_iter: int = 80,
    momentum_backend: str = "fft",
    verbose: bool = False,
) -> SupercellHFResult:
    """Solve self-consistent static HF and return a GW-compatible seed.

    Parameters
    ----------
    h0
        The *actual branch Hamiltonian*, including any temporary CO/current
        source.  Solving HF with the source already present is important: the
        HF seed then belongs to the same source-deformed Hamiltonian as the
        first full-GW point.
    Vq
        Bare density interaction in the supercell Brillouin zone.
    target_filling
        Total particles per 18-site supercell (for example ``3*n_primitive``).

    Notes
    -----
    The initial HF self-energies are exactly zero.  Consequently a zero-source
    ``normal`` solve stays in the primitive-translation/TR-symmetric subspace,
    while a finite CO or current source selects the requested broken-symmetry
    direction before the full GW continuation begins.
    """
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(h0.shape[-1])
    expected = (grid.nk1, grid.nk2, norb, norb)
    if h0.shape != expected:
        raise ValueError(f"h0 shape {h0.shape} != {expected}")
    if Vq.shape != expected:
        raise ValueError(f"Vq shape {Vq.shape} != {expected}")
    if not (0.0 < float(mixing) <= 1.0):
        raise ValueError("HF mixing must lie in (0,1]")
    if int(max_iter) < 1:
        raise ValueError("HF max_iter must be positive")
    if float(tol) <= 0.0 or float(mu_tol) <= 0.0:
        raise ValueError("HF tolerances must be positive")

    sigma_h = np.zeros((norb, norb), dtype=complex)
    sigma_f = np.zeros(expected, dtype=complex)
    mu = float(mu0)
    err = float("inf")
    converged = False
    rho = np.zeros(expected, dtype=complex)
    density = np.zeros(norb, dtype=float)

    for it in range(1, int(max_iter) + 1):
        evals, evecs = _static_eigensystem(h0, sigma_h, sigma_f)
        mu = _solve_mu_static(
            evals,
            float(target_filling),
            grid.T,
            mu,
            float(mu_tol),
            int(mu_max_iter),
        )
        rho = _rho_from_eigensystem(evals, evecs, mu, grid.T)
        density = _density_from_rho(rho)

        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
        sigma_h_out = 0.5 * (sigma_h_out + sigma_h_out.conj().T)
        sigma_f_out = compute_static_fock_matrix(
            rho, Vq, grid, backend=momentum_backend
        )
        sigma_f_out = _hermitize_k_matrix(sigma_f_out)

        err_h = float(np.max(np.abs(sigma_h_out - sigma_h)))
        err_f = float(np.max(np.abs(sigma_f_out - sigma_f)))
        err = max(err_h, err_f)

        if verbose:
            print(
                f"HF iter {it:4d}: residual={err:.3e}, mu={mu:.10f}, "
                f"n={np.sum(density):.10f}, rH={err_h:.3e}, rF={err_f:.3e}"
            )

        if err < float(tol):
            sigma_h = sigma_h_out
            sigma_f = sigma_f_out
            converged = True
            break

        alpha = float(mixing)
        sigma_h = (1.0 - alpha) * sigma_h + alpha * sigma_h_out
        sigma_f = (1.0 - alpha) * sigma_f + alpha * sigma_f_out
        sigma_h = 0.5 * (sigma_h + sigma_h.conj().T)
        sigma_f = _hermitize_k_matrix(sigma_f)
    else:
        it = int(max_iter)

    # Re-evaluate the exact static density for the final self-energies.  This is
    # also the state whose Green function is handed to the GW seed.
    evals, evecs = _static_eigensystem(h0, sigma_h, sigma_f)
    mu = _solve_mu_static(
        evals,
        float(target_filling),
        grid.T,
        mu,
        float(mu_tol),
        int(mu_max_iter),
    )
    rho = _rho_from_eigensystem(evals, evecs, mu, grid.T)
    density = _density_from_rho(rho)

    sigma_gw = np.broadcast_to(
        sigma_f[None, :, :, :, :],
        (grid.nf, grid.nk1, grid.nk2, norb, norb),
    ).copy()
    G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
    seed = GWCheckpointSeed(
        Sigma_H=np.array(sigma_h, copy=True),
        Sigma_GW=sigma_gw,
        mu=float(mu),
    )
    return SupercellHFResult(
        seed=seed,
        G=G,
        density=np.asarray(density, dtype=float),
        rho=rho,
        Sigma_H=np.array(sigma_h, copy=True),
        Sigma_F=np.array(sigma_f, copy=True),
        mu=float(mu),
        converged=bool(converged),
        iterations=int(it),
        final_error=float(err),
    )


__all__ = ["SupercellHFResult", "solve_supercell_hf_seed"]

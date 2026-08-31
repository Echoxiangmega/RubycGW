"""Self-consistent static Hartree-Fock for the 18-site Ruby supercell.

The solver uses the same Hartree and bare-interaction Fock conventions as the
static part of the active split-GW implementation, but it stops at Hartree-Fock:

    G_HF^{-1}(k,iw) = iw + mu - h0(k) - Sigma_H - Sigma_F(k).

Because the HF self-energy is static, the self-consistency loop is evaluated
from the eigenstates of the Hermitian HF Hamiltonian.  Thus the density matrix,
particle number, and HF free energy do not carry a fermionic Matsubara-cutoff
error.  A Matsubara Green function is still returned for diagnostics and for
backward compatibility with code that used HF solutions as GW seeds.

An optional previous ``SupercellHFResult`` may be supplied as ``initial``.  This
is important for source continuation: a finite CO/current source can select a
broken-symmetry HF branch, which can then be followed adiabatically as the
source is reduced to zero.

The 18-site problem can have several nearly degenerate self-consistent channels,
so plain linear mixing may oscillate or stagnate.  The default is therefore a
small-history Pulay/DIIS mixer, using the same balanced fixed-point residual
machinery as the GW solver.  Linear mixing remains available for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .checkpoint import GWCheckpointSeed
from .grids import MatsubaraGrid
from .gw import GWOptions, _check_mixing_method, _mixed_self_energies
from .supercell_gw import dyson_from_sigma_matrix, hartree_self_energy_matrix
from .supercell_gw_split import compute_static_fock_matrix


@dataclass(frozen=True)
class SupercellHFResult:
    """Converged (or best reached) static 18-site Hartree-Fock solution."""

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


@dataclass(frozen=True)
class SupercellHFFreeEnergy:
    """Finite-temperature HF thermodynamics for one self-consistent solution."""

    one_body_energy: float
    hartree_energy: float
    fock_energy: float
    entropy: float
    helmholtz_free_energy: float
    grand_potential: float
    particle_number: float
    free_energy_per_primitive_cell: float
    grand_potential_per_primitive_cell: float


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


def _initial_static_self_energies(
    initial: SupercellHFResult | None,
    expected: tuple[int, ...],
    norb: int,
    mu0: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if initial is None:
        return (
            np.zeros((norb, norb), dtype=complex),
            np.zeros(expected, dtype=complex),
            float(mu0),
        )
    sigma_h = np.asarray(initial.Sigma_H, dtype=complex)
    sigma_f = np.asarray(initial.Sigma_F, dtype=complex)
    if sigma_h.shape != (norb, norb) or sigma_f.shape != expected:
        raise ValueError("incompatible initial HF solution")
    return np.array(sigma_h, copy=True), np.array(sigma_f, copy=True), float(initial.mu)


def solve_supercell_hf(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    target_filling: float,
    *,
    mu0: float = 0.0,
    initial: SupercellHFResult | None = None,
    max_iter: int = 500,
    tol: float = 1.0e-9,
    mixing: float = 0.25,
    mixing_method: str = "pulay",
    pulay_history: int = 8,
    pulay_start: int = 4,
    pulay_regularization: float = 1.0e-10,
    mu_tol: float = 1.0e-12,
    mu_max_iter: int = 80,
    momentum_backend: str = "fft",
    verbose: bool = False,
) -> SupercellHFResult:
    """Solve the finite-temperature static HF equations at fixed filling.

    ``h0`` is the actual branch Hamiltonian, including any temporary CO/current
    source.  On the first source point, leave ``initial=None`` to start from
    zero HF self-energy.  On later source points, pass the previous HF result to
    follow that branch continuously toward zero source.

    ``mixing_method='pulay'`` is the default because competing 18-site CO/LC
    channels can make simple fixed-point iteration oscillatory.  Set
    ``mixing_method='linear'`` to reproduce the old behavior.
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
    method = _check_mixing_method(mixing_method)
    if int(pulay_history) < 2:
        raise ValueError("HF pulay_history must be at least 2")
    if int(pulay_start) < 1:
        raise ValueError("HF pulay_start must be positive")
    if float(pulay_regularization) < 0.0:
        raise ValueError("HF pulay_regularization must be nonnegative")
    if int(max_iter) < 1:
        raise ValueError("HF max_iter must be positive")
    if float(tol) <= 0.0 or float(mu_tol) <= 0.0:
        raise ValueError("HF tolerances must be positive")

    sigma_h, sigma_f, mu = _initial_static_self_energies(
        initial, expected, norb, mu0
    )
    mix_opts = GWOptions(
        mixing=float(mixing),
        mixing_method=method,
        pulay_history=int(pulay_history),
        pulay_start=int(pulay_start),
        pulay_regularization=float(pulay_regularization),
    )
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    err = float("inf")
    previous_err = float("inf")
    converged = False
    rho = np.zeros(expected, dtype=complex)
    density = np.zeros(norb, dtype=float)
    it = 0

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
                f"n={np.sum(density):.10f}, rH={err_h:.3e}, rF={err_f:.3e}, "
                f"mix={method}:{mixing:g}"
            )

        if err < float(tol):
            sigma_h = sigma_h_out
            sigma_f = sigma_f_out
            converged = True
            break

        # A badly conditioned DIIS history can occasionally extrapolate across
        # competing HF basins.  If the raw fixed-point residual jumps by more
        # than one order of magnitude, discard the history and take one damped
        # linear step before rebuilding the Pulay subspace.
        if (
            method == "pulay"
            and np.isfinite(previous_err)
            and err > 10.0 * previous_err
        ):
            history.clear()
            alpha = float(mixing)
            sigma_h_next = sigma_h + alpha * (sigma_h_out - sigma_h)
            sigma_f_next = sigma_f + alpha * (sigma_f_out - sigma_f)
        else:
            sigma_h_next, sigma_f_next = _mixed_self_energies(
                sigma_h,
                sigma_f,
                sigma_h_out,
                sigma_f_out,
                mix_opts,
                it,
                history,
            )

        sigma_h = 0.5 * (sigma_h_next + sigma_h_next.conj().T)
        sigma_f = _hermitize_k_matrix(sigma_f_next)
        previous_err = float(err)

    # Re-evaluate the exact static state and its true HF fixed-point residual.
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
    sigma_f_out = _hermitize_k_matrix(
        compute_static_fock_matrix(rho, Vq, grid, backend=momentum_backend)
    )
    err = max(
        float(np.max(np.abs(sigma_h_out - sigma_h))),
        float(np.max(np.abs(sigma_f_out - sigma_f))),
    )
    converged = bool(err < float(tol))

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
        converged=converged,
        iterations=int(it),
        final_error=float(err),
    )


def solve_supercell_hf_seed(*args, **kwargs) -> SupercellHFResult:
    """Backward-compatible alias for :func:`solve_supercell_hf`."""
    return solve_supercell_hf(*args, **kwargs)


def evaluate_supercell_hf_free_energy(
    result: SupercellHFResult,
    h0: np.ndarray,
    grid: MatsubaraGrid,
    *,
    primitive_cells_per_supercell: int = 3,
) -> SupercellHFFreeEnergy:
    """Evaluate the finite-T HF Helmholtz free energy of ``result``.

    The source, if any, is already part of ``h0``.  Therefore finite-source
    values belong to the source-deformed Hamiltonian; only the zero-source value
    is used to rank physical branches.

    The expression is evaluated directly as

        F_HF = Tr[h0 rho] + E_H + E_F - T S,
        E_H  = 1/2 sum_a n_a Sigma_H,aa,
        E_F  = 1/(2 Nk) sum_k Tr[Sigma_F(k) rho(k)].

    This avoids any Matsubara cutoff in the HF thermodynamics.
    """
    if int(primitive_cells_per_supercell) < 1:
        raise ValueError("primitive_cells_per_supercell must be positive")
    h0 = np.asarray(h0, dtype=complex)
    rho = np.asarray(result.rho, dtype=complex)
    if h0.shape != rho.shape:
        raise ValueError("h0 and HF rho shapes do not match")

    nk = float(grid.nk)
    one_body = float(
        np.real(np.einsum("xyab,xyba->", h0, rho, optimize=True)) / nk
    )
    hartree = float(
        0.5
        * np.real(
            np.dot(
                np.diag(np.asarray(result.Sigma_H, dtype=complex)),
                np.asarray(result.density, dtype=float),
            )
        )
    )
    fock = float(
        0.5
        * np.real(
            np.einsum(
                "xyab,xyba->",
                np.asarray(result.Sigma_F, dtype=complex),
                rho,
                optimize=True,
            )
        )
        / nk
    )

    evals, _ = _static_eigensystem(h0, result.Sigma_H, result.Sigma_F)
    occ = _fermi(evals - float(result.mu), grid.T)
    mask1 = occ > 0.0
    mask0 = occ < 1.0
    entropy_terms = np.zeros_like(occ, dtype=float)
    entropy_terms[mask1] += occ[mask1] * np.log(occ[mask1])
    one_minus = 1.0 - occ
    entropy_terms[mask0] += one_minus[mask0] * np.log(one_minus[mask0])
    entropy = float(-np.sum(entropy_terms) / nk)

    particle_number = float(np.sum(result.density))
    free_energy = float(one_body + hartree + fock - grid.T * entropy)
    grand_potential = float(free_energy - result.mu * particle_number)
    npc = float(primitive_cells_per_supercell)
    return SupercellHFFreeEnergy(
        one_body_energy=one_body,
        hartree_energy=hartree,
        fock_energy=fock,
        entropy=entropy,
        helmholtz_free_energy=free_energy,
        grand_potential=grand_potential,
        particle_number=particle_number,
        free_energy_per_primitive_cell=free_energy / npc,
        grand_potential_per_primitive_cell=grand_potential / npc,
    )


__all__ = [
    "SupercellHFFreeEnergy",
    "SupercellHFResult",
    "evaluate_supercell_hf_free_energy",
    "solve_supercell_hf",
    "solve_supercell_hf_seed",
]
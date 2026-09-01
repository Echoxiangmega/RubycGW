"""Covariant-GW current response on the 18-site period-three Ruby supercell.

The supercell SC-GW self-energy is evaluated as

    Sigma = Sigma_H + Sigma_F + Sigma_c,
    Sigma_c = - G * (W - V),

with the instantaneous bare interaction treated as a static Fock term.  The
functional derivative used here follows the same decomposition:

    Gamma = K + Gamma_H + Gamma_F + Gamma_MT,c + Gamma_AL1 + Gamma_AL2.

``Gamma_MT,c`` contains ``W-V``.  ``Gamma_F`` is the equal-time bare-V
exchange derivative.  The AL terms continue to contain the full screened
interaction W because delta(W-V)=delta W=W (delta P) W.

All external response momenta in this module are q_sc=0.  Primitive-cell
q=0,+Q,-Q current harmonics are nevertheless all available because
Q=(1/3,1/3) folds to q_sc=0 in the index-three supercell.

The vertex equation is linear in Gamma.  The default solver therefore treats

    (I - L) Gamma = K

as a matrix-free linear system and solves it with restarted GMRES.  This is
important beyond a response instability: the old fixed-point iteration
Gamma <- K + L Gamma fails whenever the corresponding iteration eigenvalue
crosses the unit circle even though I-L can still be nonsingular.  A legacy
linear-mixing solver is retained for diagnostics and regression tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grids import MatsubaraGrid, frequency_shift_slices, roll_spatial
from .gw import _check_backend
from .model import NSUB, eta_vertices
from .supercell import NSECTOR, NSUP
from .supercell_gw import _reverse_fft_spectrum
from .supercell_gw_split import compute_static_fock_matrix


_VERTEX_SOLVERS = ("gmres", "linear")


@dataclass(frozen=True)
class SupercellVertexOptions:
    max_iter: int = 150
    tol: float = 1e-8
    mixing: float = 0.25
    solver: str = "gmres"
    gmres_restart: int = 12
    include_hartree: bool = True
    include_fock: bool = True
    include_mt: bool = True
    include_al: bool = True
    verbose: bool = True
    momentum_backend: str = "fft"


@dataclass
class SupercellVertexResult:
    Gamma: np.ndarray
    Gamma_H: np.ndarray
    Gamma_F: np.ndarray
    Gamma_MT: np.ndarray
    Gamma_AL1: np.ndarray
    Gamma_AL2: np.ndarray
    converged: bool
    iterations: int
    final_error: float
    solver: str = "gmres"


def _check_vertex_solver(name: str) -> str:
    solver = str(name).strip().lower()
    if solver not in _VERTEX_SOLVERS:
        raise ValueError(
            f"unknown vertex solver {name!r}; expected one of {_VERTEX_SOLVERS}"
        )
    return solver


def _x_field(G: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    return np.einsum("...ab,...bc,...cd->...ad", G, Gamma, G, optimize=True)


def _hartree_from_x(
    X: np.ndarray,
    Vq0: np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    norb = int(X.shape[-1])
    xdiag = np.diagonal(X, axis1=-2, axis2=-1)
    response_density = (grid.T / grid.nk) * np.sum(xdiag, axis=(0, 1, 2))
    diag = Vq0 @ response_density
    mat = np.zeros((norb, norb), dtype=complex)
    mat[np.diag_indices(norb)] = diag
    return np.broadcast_to(mat, X.shape).copy()


def _fock_from_x(
    X: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    backend: str,
) -> np.ndarray:
    """Derivative of the static Fock self-energy.

    Since X=G Gamma G decays as 1/omega^2, its equal-time response is
    absolutely convergent and can be summed directly over the fermionic box:

        delta rho(k) = T sum_n X(k,iw_n),
        Gamma_F(k) = -(1/Nk) sum_q delta rho(k+q) o V(q)^T.
    """
    delta_rho = grid.T * np.sum(X, axis=0)
    gf_static = compute_static_fock_matrix(delta_rho, Vq, grid, backend=backend)
    return np.broadcast_to(gf_static[None, ...], X.shape).copy()


def _dynamic_corrections_direct(
    G: np.ndarray,
    W: np.ndarray,
    Wc: np.ndarray,
    X: np.ndarray,
    grid: MatsubaraGrid,
    include_mt: bool,
    include_al: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zero = np.zeros_like(X)
    gmt = zero.copy()
    gal1 = zero.copy()
    gal2 = zero.copy()
    if not include_mt and not include_al:
        return gmt, gal1, gal2

    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue

        Xsrc = X[src]
        Xbase = X[dst]
        Gsrc = G[src] if include_al else None
        Gbase = G[dst] if include_al else None

        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                Xq = roll_spatial(Xsrc, iq1, iq2)
                if include_mt:
                    WcQ = Wc[im, iq1, iq2]
                    gmt[dst] -= (
                        pref * Xq * WcQ.T[None, None, None, :, :]
                    )

                if include_al:
                    Gq = roll_spatial(Gsrc, iq1, iq2)
                    WQ = W[im, iq1, iq2]
                    L1 = pref * np.einsum(
                        "nxyef,nxyfe->ef", Xq, Gbase, optimize=True
                    )
                    L2 = pref * np.einsum(
                        "nxyef,nxyfe->ef", Gq, Xbase, optimize=True
                    )
                    M1 = WQ @ L1 @ WQ
                    M2 = WQ @ L2 @ WQ
                    gal1[dst] -= pref * Gq * M1.T[None, None, None, :, :]
                    gal2[dst] -= pref * Gq * M2.T[None, None, None, :, :]

    return gmt, gal1, gal2


def _dynamic_corrections_fft(
    G: np.ndarray,
    W: np.ndarray,
    Wc: np.ndarray,
    X: np.ndarray,
    grid: MatsubaraGrid,
    include_mt: bool,
    include_al: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zero = np.zeros_like(X)
    gmt = zero.copy()
    gal1 = zero.copy()
    gal2 = zero.copy()
    if not include_mt and not include_al:
        return gmt, gal1, gal2

    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue

        Xsrc = X[src]
        Xhat = np.fft.fftn(Xsrc, axes=(1, 2))

        if include_mt:
            WcT = np.swapaxes(Wc[im], -1, -2)[None, :, :, :, :]
            Wc_hat_minus = _reverse_fft_spectrum(WcT, axes=(1, 2))
            mt_conv = np.fft.ifftn(Xhat * Wc_hat_minus, axes=(1, 2))
            gmt[dst] -= pref * mt_conv

        if include_al:
            Xbase = X[dst]
            Gsrc = G[src]
            Gbase = G[dst]
            Ghat = np.fft.fftn(Gsrc, axes=(1, 2))

            Gbase_T = np.swapaxes(Gbase, -1, -2)
            Xbase_T = np.swapaxes(Xbase, -1, -2)
            Gbase_hat_minus = _reverse_fft_spectrum(Gbase_T, axes=(1, 2))
            Xbase_hat_minus = _reverse_fft_spectrum(Xbase_T, axes=(1, 2))

            L1_product = np.sum(Xhat * Gbase_hat_minus, axis=0)
            L1 = pref * np.fft.ifftn(L1_product, axes=(0, 1))
            L2_product = np.sum(Ghat * Xbase_hat_minus, axis=0)
            L2 = pref * np.fft.ifftn(L2_product, axes=(0, 1))

            Wm = W[im]
            M1 = np.matmul(np.matmul(Wm, L1), Wm)
            M2 = np.matmul(np.matmul(Wm, L2), Wm)

            M1T = np.swapaxes(M1, -1, -2)[None, :, :, :, :]
            M2T = np.swapaxes(M2, -1, -2)[None, :, :, :, :]
            M1hat_minus = _reverse_fft_spectrum(M1T, axes=(1, 2))
            M2hat_minus = _reverse_fft_spectrum(M2T, axes=(1, 2))
            gal1[dst] -= pref * np.fft.ifftn(
                Ghat * M1hat_minus, axes=(1, 2)
            )
            gal2[dst] -= pref * np.fft.ifftn(
                Ghat * M2hat_minus, axes=(1, 2)
            )

    return gmt, gal1, gal2


def vertex_corrections_q0(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    Gamma: np.ndarray,
    grid: MatsubaraGrid,
    opts: SupercellVertexOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    backend = _check_backend(opts.momentum_backend)
    X = _x_field(G, Gamma)
    zero = np.zeros_like(Gamma)
    gh = _hartree_from_x(X, Vq[0, 0], grid) if opts.include_hartree else zero.copy()
    gf = _fock_from_x(X, Vq, grid, backend) if opts.include_fock else zero.copy()
    Wc = W - Vq[None, :, :, :, :]
    dyn = _dynamic_corrections_fft if backend == "fft" else _dynamic_corrections_direct
    gmt, gal1, gal2 = dyn(
        G,
        W,
        Wc,
        X,
        grid,
        include_mt=opts.include_mt,
        include_al=opts.include_al,
    )
    return gh, gf, gmt, gal1, gal2


def _vertex_kernel_sum(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    Gamma: np.ndarray,
    grid: MatsubaraGrid,
    opts: SupercellVertexOptions,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    parts = vertex_corrections_q0(G, W, Vq, Gamma, grid, opts)
    total = parts[0] + parts[1] + parts[2] + parts[3] + parts[4]
    return total, parts


def _initial_gamma_field(initial_gamma: np.ndarray | None, Kfield: np.ndarray) -> np.ndarray:
    if initial_gamma is None:
        return Kfield.copy()
    arr = np.asarray(initial_gamma, dtype=complex)
    if arr.shape == Kfield.shape:
        return np.array(arr, copy=True)
    if arr.shape == Kfield.shape[-2:]:
        return np.broadcast_to(arr, Kfield.shape).copy()
    return Kfield.copy()


def _maxabs(arr: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(arr))))


def _gmres_matrix_free(
    apply_A,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float,
    max_iter: int,
    restart: int,
    verbose: bool = False,
) -> tuple[np.ndarray, bool, int, float]:
    """Restarted complex GMRES for a matrix-free operator.

    ``max_iter`` counts Krylov operator applications, not restart cycles.  The
    convergence criterion is the *actual max-norm equation residual* evaluated
    at every restart (and at an Arnoldi happy breakdown), so ``tol`` has the
    same transparent meaning as the residual printed by the driver.

    The Krylov basis is kept as arrays with the same shape as ``b``; no explicit
    dense representation of the enormous vertex operator is ever constructed.
    """
    if int(max_iter) < 1:
        raise ValueError("GMRES max_iter must be positive")
    if int(restart) < 1:
        raise ValueError("GMRES restart must be positive")
    if float(tol) <= 0.0:
        raise ValueError("GMRES tol must be positive")

    b = np.asarray(b, dtype=complex)
    x = np.asarray(x0, dtype=complex).copy()
    restart = min(int(restart), int(max_iter))
    total_it = 0

    r = b - apply_A(x)
    err = _maxabs(r)
    if verbose:
        print(f"supercell cGW GMRES initial: residual_max={err:.3e}")
    if err < float(tol):
        return x, True, total_it, err

    # Arnoldi breakdown threshold relative to the norm of each new vector.
    breakdown_eps = 100.0 * np.finfo(float).eps

    while total_it < int(max_iter):
        beta = float(np.linalg.norm(r.ravel()))
        if beta == 0.0:
            return x, True, total_it, 0.0

        m = min(restart, int(max_iter) - total_it)
        basis: list[np.ndarray] = [r / beta]
        H = np.zeros((m + 1, m), dtype=complex)
        y_last = None
        used = 0
        happy_breakdown = False

        for j in range(m):
            w = np.asarray(apply_A(basis[j]), dtype=complex).copy()
            total_it += 1

            # Twice-modified Gram-Schmidt is materially more robust for the
            # strongly non-normal vertex kernels encountered near an instability.
            for _ in range(2):
                for i in range(j + 1):
                    hij = np.vdot(basis[i].ravel(), w.ravel())
                    H[i, j] += hij
                    w -= hij * basis[i]

            hnext = float(np.linalg.norm(w.ravel()))
            H[j + 1, j] = hnext
            used = j + 1

            small_rhs = np.zeros(used + 1, dtype=complex)
            small_rhs[0] = beta
            Hused = H[: used + 1, :used]
            y_last, *_ = np.linalg.lstsq(Hused, small_rhs, rcond=None)
            est_l2 = float(np.linalg.norm(small_rhs - Hused @ y_last))
            est_rms = est_l2 / np.sqrt(float(b.size))
            if verbose:
                print(
                    f"supercell cGW GMRES iter {total_it:4d}: "
                    f"estimated_residual_rms={est_rms:.3e}, "
                    f"restart={restart}"
                )

            ref_scale = max(float(np.linalg.norm(apply_A(basis[j]).ravel())), 1.0)
            if hnext <= breakdown_eps * ref_scale:
                happy_breakdown = True
                break
            basis.append(w / hnext)

        if used == 0 or y_last is None:
            break

        dx = np.zeros_like(x)
        for i in range(used):
            dx += y_last[i] * basis[i]
        x += dx

        r = b - apply_A(x)
        err = _maxabs(r)
        if verbose:
            tag = "happy-breakdown" if happy_breakdown else "restart"
            print(
                f"supercell cGW GMRES {tag}: total_it={total_it}, "
                f"residual_max={err:.3e}"
            )
        if err < float(tol):
            return x, True, total_it, err

        if happy_breakdown:
            # Arnoldi found an invariant subspace but the exact max-norm residual
            # is still above tolerance.  Restarting from that residual is safer
            # than falsely declaring convergence.
            if not np.all(np.isfinite(r)):
                break

    return x, False, total_it, float(err)


def _solve_vertex_linear(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    Kfield: np.ndarray,
    Gamma0: np.ndarray,
    grid: MatsubaraGrid,
    opts: SupercellVertexOptions,
) -> tuple[np.ndarray, bool, int, float]:
    """Legacy damped fixed-point iteration, retained for diagnostics."""
    Gamma = np.asarray(Gamma0, dtype=complex).copy()
    converged = False
    err = float("inf")
    it = 0
    for it in range(1, int(opts.max_iter) + 1):
        kernel, _ = _vertex_kernel_sum(G, W, Vq, Gamma, grid, opts)
        equation_residual = Kfield + kernel - Gamma
        err = _maxabs(equation_residual)
        if opts.verbose:
            print(
                f"supercell cGW linear vertex iter {it:4d}: "
                f"residual_max={err:.3e}, backend={opts.momentum_backend}"
            )
        if err < float(opts.tol):
            converged = True
            break
        Gamma += float(opts.mixing) * equation_residual
    return Gamma, converged, it, float(err)


def solve_vertex_q0(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    K: np.ndarray,
    grid: MatsubaraGrid,
    opts: SupercellVertexOptions = SupercellVertexOptions(),
    initial_gamma: np.ndarray | None = None,
) -> SupercellVertexResult:
    """Solve one q_sc=0 current vertex on an arbitrary matrix dimension.

    The default ``solver='gmres'`` solves the linear equation

        [I - L] Gamma = K

    directly with restarted matrix-free GMRES.  ``solver='linear'`` reproduces
    the old damped fixed-point strategy and is mainly useful as a diagnostic.
    """
    norb = int(G.shape[-1])
    if K.shape != (norb, norb):
        raise ValueError(f"K shape {K.shape} != {(norb, norb)}")
    if Vq.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected Vq shape")
    if W.shape != (grid.nb, grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected W shape")

    solver = _check_vertex_solver(opts.solver)
    Kfield = np.broadcast_to(K, G.shape).copy()
    Gamma0 = _initial_gamma_field(initial_gamma, Kfield)

    if solver == "gmres":
        def apply_A(field):
            kernel, _ = _vertex_kernel_sum(G, W, Vq, field, grid, opts)
            return np.asarray(field, dtype=complex) - kernel

        Gamma, converged, it, err = _gmres_matrix_free(
            apply_A,
            Kfield,
            Gamma0,
            tol=float(opts.tol),
            max_iter=int(opts.max_iter),
            restart=int(opts.gmres_restart),
            verbose=bool(opts.verbose),
        )
    else:
        Gamma, converged, it, err = _solve_vertex_linear(
            G, W, Vq, Kfield, Gamma0, grid, opts
        )

    # Re-evaluate the decomposed corrections once at the final Gamma.  Besides
    # populating diagnostics, this provides an independent exact equation
    # residual rather than relying on a Krylov residual estimate.
    kernel, parts = _vertex_kernel_sum(G, W, Vq, Gamma, grid, opts)
    gh, gf, gmt, gal1, gal2 = parts
    err = _maxabs(Kfield + kernel - Gamma)
    converged = bool(np.isfinite(err) and err < float(opts.tol))

    return SupercellVertexResult(
        Gamma=Gamma,
        Gamma_H=gh,
        Gamma_F=gf,
        Gamma_MT=gmt,
        Gamma_AL1=gal1,
        Gamma_AL2=gal2,
        converged=converged,
        iterations=int(it),
        final_error=float(err),
        solver=solver,
    )


def supercell_current_vertices() -> tuple[np.ndarray, list[str]]:
    """Return the six local current vertices [+,s=0..2; -,s=0..2].

    ``+`` retains the project convention PHYSICAL OPPOSITE circulation and
    ``-`` retains PHYSICAL SAME circulation.
    """
    _, _, kp, km = eta_vertices()
    vertices = []
    labels = []
    for label, k6 in (("opposite", kp), ("same", km)):
        for s in range(NSECTOR):
            mat = np.zeros((NSUP, NSUP), dtype=complex)
            sl = slice(NSUB * s, NSUB * (s + 1))
            mat[sl, sl] = k6
            vertices.append(mat)
            labels.append(f"{label}_s{s}")
    return np.stack(vertices, axis=0), labels


def sector_harmonic_matrix() -> np.ndarray:
    """Orthogonal real transform (s0,s1,s2) -> (q0,Qc,Qs)."""
    return np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, -1.0, -1.0],
            [0.0, 1.0, -1.0],
        ],
        dtype=float,
    ) / np.array([[np.sqrt(3.0)], [np.sqrt(6.0)], [np.sqrt(2.0)]])


def current_harmonic_transform() -> tuple[np.ndarray, list[str]]:
    """6x6 transform local currents -> q0/Qc/Qs in each physical channel."""
    u = sector_harmonic_matrix()
    T = np.zeros((2 * NSECTOR, 2 * NSECTOR), dtype=float)
    T[:NSECTOR, :NSECTOR] = u
    T[NSECTOR:, NSECTOR:] = u
    labels = [
        "opposite_q0", "opposite_Qc", "opposite_Qs",
        "same_q0", "same_Qc", "same_Qs",
    ]
    return T, labels


def susceptibility_matrix_q0(
    G: np.ndarray,
    left_vertices: np.ndarray,
    gammas: list[np.ndarray] | np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Return chi_ab=-int_k Tr[K_a G Gamma_b G]."""
    K = np.asarray(left_vertices, dtype=complex)
    nchan = int(K.shape[0])
    if len(gammas) != nchan:
        raise ValueError("number of right vertices must match left vertices")
    chi = np.zeros((nchan, nchan), dtype=complex)
    pref = -(grid.T / grid.nk)
    for b in range(nchan):
        Gamma = np.asarray(gammas[b], dtype=complex)
        chi[:, b] = pref * np.einsum(
            "iab,nxybc,nxycd,nxyda->i",
            K,
            G,
            Gamma,
            G,
            optimize=True,
        )
    return chi


def physical_symmetric_susceptibility(chi: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the real symmetric static response and discarded imaginary scale."""
    chi = np.asarray(chi, dtype=complex)
    herm = 0.5 * (chi + chi.conj().T)
    imag_max = float(np.max(np.abs(herm.imag)))
    return np.asarray(herm.real, dtype=float), imag_max


def curvature_from_susceptibility(
    chi: np.ndarray,
    rcond: float = 1e-12,
) -> dict:
    """Analyze R=chi^{-1} in the six local-current basis."""
    chi_sym, imag_max = physical_symmetric_susceptibility(chi)
    evals_chi, evecs = np.linalg.eigh(chi_sym)
    scale = max(float(np.max(np.abs(evals_chi))), 1.0)
    cutoff = float(rcond) * scale
    if np.any(np.abs(evals_chi) <= cutoff):
        raise np.linalg.LinAlgError(
            "current susceptibility is singular within requested rcond"
        )

    R = np.linalg.inv(chi_sym)
    R = 0.5 * (R + R.T)
    evals_r, evecs_r = np.linalg.eigh(R)
    soft = evecs_r[:, 0]
    soft = soft / np.linalg.norm(soft)

    T, harmonic_labels = current_harmonic_transform()
    chi_harm = T @ chi_sym @ T.T
    R_harm = T @ R @ T.T
    soft_harm = T @ soft

    uniform_idx = np.array([0, 3], dtype=int)
    chi_uniform = chi_harm[np.ix_(uniform_idx, uniform_idx)]
    R_uniform_relaxed = np.linalg.inv(chi_uniform)
    R_uniform_constrained = R_harm[np.ix_(uniform_idx, uniform_idx)]

    w_opposite = float(np.sum(np.abs(soft[:NSECTOR]) ** 2))
    w_same = float(np.sum(np.abs(soft[NSECTOR:]) ** 2))
    w_q0 = float(abs(soft_harm[0]) ** 2 + abs(soft_harm[3]) ** 2)
    w_Q = float(np.sum(np.abs(soft_harm[[1, 2, 4, 5]]) ** 2))

    return {
        "chi_symmetric": chi_sym,
        "chi_imag_max": imag_max,
        "chi_eigenvalues": evals_chi,
        "chi_eigenvectors": evecs,
        "R": R,
        "R_eigenvalues": evals_r,
        "R_eigenvectors": evecs_r,
        "soft_vector_local": soft,
        "harmonic_transform": T,
        "harmonic_labels": harmonic_labels,
        "chi_harmonic": chi_harm,
        "R_harmonic": R_harm,
        "soft_vector_harmonic": soft_harm,
        "chi_uniform": chi_uniform,
        "R_uniform_relaxed": R_uniform_relaxed,
        "R_uniform_constrained": R_uniform_constrained,
        "soft_weight_opposite": w_opposite,
        "soft_weight_same": w_same,
        "soft_weight_q0": w_q0,
        "soft_weight_Q": w_Q,
    }

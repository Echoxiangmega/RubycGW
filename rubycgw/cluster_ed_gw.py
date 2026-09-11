"""Self-consistent 6-site Ruby cluster ED + lattice GW self-energy embedding.

All density interactions of the microscopic Ruby model are primitive-cell local.
This makes a six-site primitive cell a complete interacting cluster: inter-cell
couplings are hopping only.  The approximation implemented here is

    Sigma_emb(k,iw) = Sigma_GW^lattice(k,iw)
                    - Sigma_GW^cluster(iw)
                    + Sigma_ED^cluster(iw),

where ``Sigma_GW^cluster`` is the same weak-coupling skeleton evaluated with the
projected cluster Green function, and ``Sigma_ED^cluster`` is obtained from a
finite-bath Anderson impurity solved by exact diagonalization.

The impurity Weiss field is updated from the lattice projection,

    G0_C^{-1} = G_C^{-1} + Sigma_ED_C,

and its hybridization is represented by a finite real bath.  The default six
bath orbitals give a 12-orbital impurity (4096 total Fock states).

This is a self-energy embedding diagnostic.  The polarization is still the
lattice GW polarization; an eventual Sigma+Pi / GW+EDMFT extension is separate.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from .grids import MatsubaraGrid
from .gw import GWOptions, GWResult, _check_backend
from .impurity_ed import FiniteBathImpurityED
from .model import RubyParameters, NSUB, ruby_hoppings, ruby_interaction_bonds
from .supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    compute_sigma_gw_matrix,
    hartree_self_energy_matrix,
)
from .supercell_gw_fast import _solve_mu_matrix_fast, solve_matrix_gw_fast
from .supercell_gw_split import (
    compute_sigma_gw_split_matrix,
    compute_static_fock_matrix,
    one_body_density_matrix_tail,
)


@dataclass
class BathParameters:
    energies: np.ndarray
    couplings: np.ndarray
    fit_error: float
    nfev: int


@dataclass(frozen=True)
class ClusterEDGWOptions:
    max_iter: int = 12
    tol: float = 2.0e-5
    mixing: float = 0.30
    impurity_mixing: float = 0.70
    nbath: int = 6
    bath_fit_nfreq: int = 12
    bath_fit_max_nfev: int = 300
    bath_energy_window: float = 4.0
    bath_coupling_bound: float = 4.0
    bath_fit_xtol: float = 1.0e-9
    discard_weight_tol: float = 1.0e-11
    verbose: bool = True


@dataclass
class ClusterEDGWResult:
    G: np.ndarray
    W: np.ndarray
    P: np.ndarray
    Sigma_H: np.ndarray
    Sigma_emb: np.ndarray
    Sigma_GW_lattice: np.ndarray
    Sigma_GW_cluster: np.ndarray
    Sigma_ED_cluster: np.ndarray
    G_cluster: np.ndarray
    G_impurity: np.ndarray
    mu: float
    density: np.ndarray
    bath: BathParameters
    converged: bool
    iterations: int
    final_error: float
    impurity_mismatch: float
    bath_fit_error: float
    background: GWResult


def build_intracell_h0(params: RubyParameters) -> np.ndarray:
    """Return only primitive-cell (R=0) hopping as a 6x6 Hermitian matrix.

    For a finite torus with Lx=1 or Ly=1 this is not necessarily the local block
    of that torus, because a nonzero primitive translation can wrap back onto the
    same cluster.  The embedding solver therefore uses the k-average of the actual
    finite-torus h0(k) as its impurity one-body block.
    """
    p0 = RubyParameters(ti=params.ti, t1=params.t1, t2=params.t2, V=0.0)
    h = np.zeros((NSUB, NSUB), dtype=complex)
    for i, j, R, amp in ruby_hoppings(p0):
        if np.any(np.asarray(R, dtype=int) != 0):
            continue
        h[int(i), int(j)] += complex(amp)
    return 0.5 * (h + h.conj().T)


def ruby_cluster_interactions(params: RubyParameters):
    """Return all six primitive-cell density bonds as ``(i,j,V)`` terms."""
    out = []
    for i, j, R, u in ruby_interaction_bonds(params):
        if np.any(np.asarray(R, dtype=int) != 0):
            raise RuntimeError("Ruby interaction unexpectedly crosses primitive cells")
        out.append((int(i), int(j), float(np.real(u))))
    return tuple(out)


def bath_hybridization(
    omega: np.ndarray,
    mu: float,
    energies: np.ndarray,
    couplings: np.ndarray,
) -> np.ndarray:
    """Return Delta_ab(iw)=sum_p t_bath[a,p] t_bath[b,p]^*/(iw+mu-eps_p).

    ``couplings[a,p]`` is the one-particle cluster-bath hybridization amplitude
    (often written as mathcal V_{ap}); it is unrelated to the physical density
    interaction strength ``params.V`` of the Ruby model.
    """
    w = np.asarray(omega, dtype=float).reshape(-1)
    eps = np.asarray(energies, dtype=float).reshape(-1)
    bath_hyb = np.asarray(couplings, dtype=complex)
    if bath_hyb.ndim != 2 or bath_hyb.shape[1] != len(eps):
        raise ValueError("bath coupling shape mismatch")
    den = 1.0 / (1j * w[:, None] + float(mu) - eps[None, :])
    return np.einsum(
        "ap,np,bp->nab", bath_hyb, den, bath_hyb.conj(), optimize=True
    )


def _initial_bath_guess(
    target_delta: np.ndarray,
    omega: np.ndarray,
    mu: float,
    nbath: int,
    energy_window: float,
) -> tuple[np.ndarray, np.ndarray]:
    norb = int(target_delta.shape[-1])
    eps = float(mu) + np.linspace(-float(energy_window), float(energy_window), nbath)
    # i*w*Delta(iw) -> t_bath t_bath^dag at high frequency.  Use the largest
    # available positive frequency to obtain a stable low-rank hybridization seed.
    pos = np.flatnonzero(np.asarray(omega) > 0.0)
    idx = int(pos[np.argmax(np.asarray(omega)[pos])]) if pos.size else len(omega) - 1
    moment = (1j * float(omega[idx])) * np.asarray(target_delta[idx], dtype=complex)
    moment = 0.5 * (moment + moment.conj().T)
    evals, evecs = np.linalg.eigh(moment.real)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 1.0e-8)
    evecs = evecs[:, order]
    bath_hyb = np.zeros((norb, nbath), dtype=float)
    rank = min(norb, nbath)
    bath_hyb[:, :rank] = evecs[:, :rank] * np.sqrt(evals[:rank])[None, :]
    if nbath > rank:
        bath_hyb[:, rank:] = 1.0e-3
    return eps, bath_hyb


def fit_finite_bath(
    target_delta: np.ndarray,
    omega: np.ndarray,
    mu: float,
    *,
    nbath: int = 6,
    nfit: int = 12,
    max_nfev: int = 300,
    energy_window: float = 4.0,
    coupling_bound: float = 4.0,
    xtol: float = 1.0e-9,
    initial: BathParameters | None = None,
) -> BathParameters:
    """Least-squares fit of a full-matrix hybridization to a finite real bath."""
    delta = np.asarray(target_delta, dtype=complex)
    w = np.asarray(omega, dtype=float).reshape(-1)
    if delta.ndim != 3 or delta.shape[0] != len(w) or delta.shape[1] != delta.shape[2]:
        raise ValueError("target_delta must have shape (nf,norb,norb)")
    norb = int(delta.shape[-1])
    nbath = int(nbath)
    if nbath < 1:
        raise ValueError("nbath must be positive")

    pos = np.flatnonzero(w > 0.0)
    if pos.size == 0:
        raise ValueError("bath fit requires positive Matsubara frequencies")
    pos = pos[np.argsort(w[pos])[: min(int(nfit), len(pos))]]
    wfit = w[pos]
    dfit = delta[pos]
    w0 = max(float(wfit[0]), 1e-12)
    weights = 1.0 / np.sqrt(wfit * wfit + w0 * w0)
    weights /= float(np.max(weights))

    if initial is not None:
        eps0 = np.asarray(initial.energies, dtype=float)
        bath_hyb0 = np.asarray(initial.couplings, dtype=float)
        if eps0.shape != (nbath,) or bath_hyb0.shape != (norb, nbath):
            eps0, bath_hyb0 = _initial_bath_guess(
                delta, w, mu, nbath, energy_window
            )
    else:
        eps0, bath_hyb0 = _initial_bath_guess(delta, w, mu, nbath, energy_window)

    lo_e = float(mu) - float(energy_window)
    hi_e = float(mu) + float(energy_window)
    eps0 = np.clip(eps0, lo_e + 1e-8, hi_e - 1e-8)
    bath_hyb0 = np.clip(
        bath_hyb0,
        -float(coupling_bound) + 1e-8,
        float(coupling_bound) - 1e-8,
    )
    x0 = np.concatenate([eps0, bath_hyb0.ravel()])
    lower = np.concatenate([
        np.full(nbath, lo_e),
        np.full(norb * nbath, -float(coupling_bound)),
    ])
    upper = np.concatenate([
        np.full(nbath, hi_e),
        np.full(norb * nbath, float(coupling_bound)),
    ])

    def unpack(x):
        return x[:nbath], x[nbath:].reshape(norb, nbath)

    def residual(x):
        eps, bath_hyb = unpack(x)
        model = bath_hybridization(wfit, mu, eps, bath_hyb)
        diff = (model - dfit) * weights[:, None, None]
        return np.concatenate([diff.real.ravel(), diff.imag.ravel()])

    opt = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
        xtol=float(xtol),
        ftol=float(xtol),
        gtol=float(xtol),
    )
    eps, bath_hyb = unpack(opt.x)
    order = np.argsort(eps)
    eps = np.asarray(eps[order], dtype=float)
    bath_hyb = np.asarray(bath_hyb[:, order], dtype=float)
    fit = bath_hybridization(wfit, mu, eps, bath_hyb)
    den = max(float(np.linalg.norm(dfit.ravel())), 1e-300)
    err = float(np.linalg.norm((fit - dfit).ravel()) / den)
    return BathParameters(eps, bath_hyb, err, int(opt.nfev))


def build_impurity_one_body(
    h_cluster: np.ndarray,
    bath: BathParameters,
) -> np.ndarray:
    h = np.asarray(h_cluster, dtype=complex)
    bath_hyb = np.asarray(bath.couplings, dtype=complex)
    eps = np.asarray(bath.energies, dtype=float)
    norb = int(h.shape[0])
    nbath = len(eps)
    if h.shape != (norb, norb) or bath_hyb.shape != (norb, nbath):
        raise ValueError("cluster/bath shape mismatch")
    out = np.zeros((norb + nbath, norb + nbath), dtype=complex)
    out[:norb, :norb] = h
    out[:norb, norb:] = bath_hyb
    out[norb:, :norb] = bath_hyb.conj().T
    out[norb:, norb:] = np.diag(eps)
    return 0.5 * (out + out.conj().T)


def cluster_gw_self_energy(
    G_cluster: np.ndarray,
    rho_cluster: np.ndarray,
    V_cluster: np.ndarray,
    grid: MatsubaraGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Total local GW self-energy used for SEET-like double counting."""
    Gc = np.asarray(G_cluster, dtype=complex)
    rho = np.asarray(rho_cluster, dtype=complex)
    Vc = np.asarray(V_cluster, dtype=complex)
    norb = int(Gc.shape[-1])
    if Gc.shape != (grid.nf, norb, norb) or rho.shape != (norb, norb):
        raise ValueError("cluster G/rho shape mismatch")
    cg = MatsubaraGrid(nk1=1, nk2=1, nw=grid.nw, nOmega=grid.nOmega, T=grid.T)
    G5 = Gc[:, None, None]
    Vq = Vc[None, None]
    P = compute_polarization_matrix(G5, cg, backend="direct")
    W = compute_screened_interaction_matrix(P, Vq)
    density = np.real(np.diag(rho))
    sigma_h = hartree_self_energy_matrix(density, Vc)
    sigma_f = compute_static_fock_matrix(rho[None, None], Vq, cg, backend="direct")[0, 0]
    Wc = W - Vq[None, ...]
    sigma_c = compute_sigma_gw_matrix(G5, Wc, cg, backend="direct")[:, 0, 0]
    total = sigma_h[None, ...] + sigma_f[None, ...] + sigma_c
    return total, P[:, 0, 0], W[:, 0, 0]


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(b).ravel())), 1e-300)
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)).ravel()) / den)


def _maxabs(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a)), initial=0.0))


def solve_cluster_ed_gw(
    h0: np.ndarray,
    Vq: np.ndarray,
    params: RubyParameters,
    grid: MatsubaraGrid,
    *,
    gw_opts: GWOptions = GWOptions(),
    embed_opts: ClusterEDGWOptions = ClusterEDGWOptions(),
    background: GWResult | None = None,
) -> ClusterEDGWResult:
    """Solve the coupled lattice-GW / 6-site finite-bath ED fixed point."""
    backend = _check_backend(gw_opts.momentum_backend)
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    if h0.shape != (grid.nk1, grid.nk2, NSUB, NSUB):
        raise ValueError("cluster ED+GW currently expects the primitive 6-site lattice basis")
    if Vq.shape != h0.shape:
        raise ValueError("Vq/h0 shape mismatch")
    if not (0.0 < float(embed_opts.mixing) <= 1.0):
        raise ValueError("embedding mixing must lie in (0,1]")
    if not (0.0 < float(embed_opts.impurity_mixing) <= 1.0):
        raise ValueError("impurity mixing must lie in (0,1]")

    if background is None:
        if embed_opts.verbose:
            print("[cluster-ED+GW] solving initial lattice GW background ...", flush=True)
        background = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    if not background.converged:
        raise RuntimeError(f"initial lattice GW background is not converged: {background.final_error:.3e}")

    # The impurity one-body block must be the local block of the *actual finite
    # torus*, which is exactly the k-average of h0(k).  This differs from the
    # strict R=0 primitive-cell block when Lx=1 or Ly=1 because a translated
    # hopping can wrap around the PBC torus and return to the same cluster.  If
    # that static wrap-around term were omitted here, Delta_target would retain
    # a nonzero constant at |omega|->infinity, which no finite bath can represent
    # because every bath hybridization decays as 1/(i omega).
    h_cluster_strict = build_intracell_h0(params)
    h_cluster = np.mean(h0, axis=(0, 1))
    h_cluster = 0.5 * (h_cluster + h_cluster.conj().T)
    alias_norm = _maxabs(h_cluster - h_cluster_strict)
    if embed_opts.verbose and alias_norm > 1.0e-12:
        print(
            f"[cluster-ED+GW] finite-torus local-block correction: "
            f"max|h_local-h_R0|={alias_norm:.3e}",
            flush=True,
        )

    interactions = ruby_cluster_interactions(params)
    V_cluster = np.asarray(Vq[0, 0], dtype=complex)

    sigma_h = np.asarray(background.Sigma_H, dtype=complex).copy()
    sigma_emb = np.asarray(background.Sigma_GW, dtype=complex).copy()
    mu = float(background.mu)
    G = np.asarray(background.G, dtype=complex).copy()
    bath = None

    rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    Gc = np.mean(G, axis=(1, 2))
    rho_c = np.mean(rho_k, axis=(0, 1))
    sigma_cgw, _, _ = cluster_gw_self_energy(Gc, rho_c, V_cluster, grid)
    sigma_imp = np.array(sigma_cgw, copy=True)

    converged = False
    err = float("inf")
    mismatch = float("inf")
    sigma_gw_lattice = np.asarray(background.Sigma_GW, dtype=complex)
    Gimp = np.asarray(Gc)
    W = np.asarray(background.W)
    P = np.asarray(background.P)
    density = np.asarray(background.density)
    it = 0

    for it in range(1, int(embed_opts.max_iter) + 1):
        t0 = perf_counter()
        if embed_opts.verbose:
            print(f"[cluster-ED+GW] outer {it:02d}: build lattice GW map", flush=True)

        rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
        rho_c = np.mean(rho_k, axis=(0, 1))
        density = np.real(np.diag(rho_c))
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
        P = compute_polarization_matrix(G, grid, backend=backend)
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_lattice = compute_sigma_gw_split_matrix(
            G, W, Vq, grid, h0, mu, sigma_h, backend=backend
        )

        Gc = np.mean(G, axis=(1, 2))
        sigma_cgw, _, _ = cluster_gw_self_energy(Gc, rho_c, V_cluster, grid)

        # Impurity Weiss field corresponding to the current projected lattice G
        # and the previous/mixed impurity self-energy.
        g0_inv = np.linalg.inv(Gc) + sigma_imp
        eye = np.eye(NSUB, dtype=complex)
        delta_target = (
            (1j * grid.omega[:, None, None] + float(mu)) * eye[None, :, :]
            - h_cluster[None, :, :]
            - g0_inv
        )

        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW] outer {it:02d}: fit {embed_opts.nbath} bath orbitals ...",
                flush=True,
            )
        bath = fit_finite_bath(
            delta_target,
            grid.omega,
            mu,
            nbath=int(embed_opts.nbath),
            nfit=int(embed_opts.bath_fit_nfreq),
            max_nfev=int(embed_opts.bath_fit_max_nfev),
            energy_window=float(embed_opts.bath_energy_window),
            coupling_bound=float(embed_opts.bath_coupling_bound),
            xtol=float(embed_opts.bath_fit_xtol),
            initial=bath,
        )
        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW] outer {it:02d}: bath relerr={bath.fit_error:.3e}, "
                f"nfev={bath.nfev}; diagonalize impurity ...",
                flush=True,
            )

        himp = build_impurity_one_body(h_cluster, bath)
        impurity = FiniteBathImpurityED(
            himp,
            interactions,
            correlated_orbitals=tuple(range(NSUB)),
        )
        impurity.diagonalize()
        Gimp, selection = impurity.green_iomega(
            1j * grid.omega,
            mu,
            grid.T,
            orbitals=tuple(range(NSUB)),
            discard_weight_tol=float(embed_opts.discard_weight_tol),
        )
        delta_fit = bath_hybridization(
            grid.omega, mu, bath.energies, bath.couplings
        )
        g0_fit_inv = (
            (1j * grid.omega[:, None, None] + float(mu)) * eye[None, :, :]
            - h_cluster[None, :, :]
            - delta_fit
        )
        sigma_ed_raw = g0_fit_inv - np.linalg.inv(Gimp)
        beta_imp = float(embed_opts.impurity_mixing)
        sigma_ed = (1.0 - beta_imp) * sigma_imp + beta_imp * sigma_ed_raw
        mismatch = _relative_error(Gimp, Gc)

        correction = sigma_ed - sigma_cgw
        sigma_emb_out = sigma_gw_lattice + correction[:, None, None, :, :]
        err = max(
            _maxabs(sigma_h_out - sigma_h),
            _maxabs(sigma_emb_out - sigma_emb),
        )

        elapsed = perf_counter() - t0
        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW] outer {it:02d}: residual={err:.3e}, "
                f"Gimp/Gc mismatch={mismatch:.3e}, bath={bath.fit_error:.3e}, "
                f"Ntot_imp={selection.average_particles:.6f}, mu={mu:+.9f}, dt={elapsed:.1f}s",
                flush=True,
            )
        sigma_imp = sigma_ed
        if err < float(embed_opts.tol):
            converged = True
            break

        a = float(embed_opts.mixing)
        sigma_h = sigma_h + a * (sigma_h_out - sigma_h)
        sigma_emb = sigma_emb + a * (sigma_emb_out - sigma_emb)
        if gw_opts.target_filling is None:
            from .supercell_gw import dyson_from_sigma_matrix
            G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_emb)
        else:
            mu, G, _, _ = _solve_mu_matrix_fast(
                h0,
                sigma_h,
                sigma_emb,
                grid,
                float(gw_opts.target_filling),
                mu,
                float(gw_opts.mu_tol),
                int(gw_opts.mu_max_iter),
            )

    # Put the accepted target through one final mixed Dyson step so the returned
    # Green function corresponds to the reported embedded self-energy.
    a = float(embed_opts.mixing)
    sigma_h = sigma_h + a * (sigma_h_out - sigma_h)
    sigma_emb = sigma_emb + a * (sigma_emb_out - sigma_emb)
    if gw_opts.target_filling is None:
        from .supercell_gw import dyson_from_sigma_matrix
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_emb)
    else:
        mu, G, _, _ = _solve_mu_matrix_fast(
            h0,
            sigma_h,
            sigma_emb,
            grid,
            float(gw_opts.target_filling),
            mu,
            float(gw_opts.mu_tol),
            int(gw_opts.mu_max_iter),
        )
    rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    density = np.real(np.diag(np.mean(rho_k, axis=(0, 1))))
    Gc = np.mean(G, axis=(1, 2))

    return ClusterEDGWResult(
        G=np.asarray(G),
        W=np.asarray(W),
        P=np.asarray(P),
        Sigma_H=np.asarray(sigma_h),
        Sigma_emb=np.asarray(sigma_emb),
        Sigma_GW_lattice=np.asarray(sigma_gw_lattice),
        Sigma_GW_cluster=np.asarray(sigma_cgw),
        Sigma_ED_cluster=np.asarray(sigma_imp),
        G_cluster=np.asarray(Gc),
        G_impurity=np.asarray(Gimp),
        mu=float(mu),
        density=np.asarray(density),
        bath=bath,
        converged=bool(converged),
        iterations=int(it),
        final_error=float(err),
        impurity_mismatch=float(mismatch),
        bath_fit_error=float(bath.fit_error if bath is not None else np.nan),
        background=background,
    )


__all__ = [
    "BathParameters",
    "ClusterEDGWOptions",
    "ClusterEDGWResult",
    "build_intracell_h0",
    "ruby_cluster_interactions",
    "bath_hybridization",
    "fit_finite_bath",
    "build_impurity_one_body",
    "cluster_gw_self_energy",
    "solve_cluster_ed_gw",
]

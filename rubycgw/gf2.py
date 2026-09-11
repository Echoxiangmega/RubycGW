"""Self-consistent second-order Green-function theory (GF2).

GF2 is used here as an alternative weak solver for self-energy embedding.  For
an instantaneous density-density interaction its self-energy is

    Sigma = Sigma_H + Sigma_F + Sigma_direct^(2) + Sigma_exchange^(2).

The direct second-order skeleton is the O(V^2) term in the GW expansion,
obtained from W^(2)-V = V P V with P=GG.  The exchange second-order skeleton is
the bare SOX functional already used elsewhere in this repository.  Both are
evaluated with the dressed Green function and iterated to self-consistency.

For compatibility with existing embedding infrastructure ``solve_matrix_gf2``
returns a :class:`GWResult`: its ``Sigma_GW`` field stores the non-Hartree GF2
self-energy and its ``W`` field stores V+VPV (not a fully screened W).  Callers
must use the accompanying weak-solver label rather than interpreting those
field names literally.
"""
from __future__ import annotations

import numpy as np

from .grids import MatsubaraGrid
from .gw import (
    GWOptions,
    GWResult,
    _check_backend,
    _check_mixing_method,
    _mixed_self_energies,
    _residual_error,
)
from .sox_covariant import SOXOptions, compute_sox_self_energy_periodic
from .supercell_gw import (
    compute_polarization_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
    screening_soft_modes_matrix,
)
from .supercell_gw_fast import _solve_mu_matrix_fast
from .supercell_gw_split import (
    compute_sigma_gw_split_components,
    one_body_density_matrix_tail,
)


def second_order_bare_interaction(P: np.ndarray, Vq: np.ndarray) -> np.ndarray:
    """Return V + V P V on the represented bosonic grid."""
    P = np.asarray(P, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    if P.ndim != 5 or Vq.ndim != 4:
        raise ValueError("expected P(m,q,a,b) and V(q,a,b)")
    if P.shape[1:] != Vq.shape:
        raise ValueError("P/Vq shape mismatch")
    Vb = Vq[None, :, :, :, :]
    return Vb + np.matmul(np.matmul(Vb, P), Vb)


def compute_gf2_nonhartree(
    G: np.ndarray,
    Vq: np.ndarray,
    h0: np.ndarray,
    mu: float,
    sigma_h: np.ndarray,
    grid: MatsubaraGrid,
    *,
    backend: str = "fft",
    sox_opts: SOXOptions = SOXOptions(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Sigma_GF2_nonH, P, W2, Sigma_F, Sigma_SOX)."""
    backend = _check_backend(backend)
    sox_opts.validate()
    G = np.asarray(G, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    h0 = np.asarray(h0, dtype=complex)
    sigma_h = np.asarray(sigma_h, dtype=complex)

    P = compute_polarization_matrix(G, grid, backend=backend)
    W2 = second_order_bare_interaction(P, Vq)
    sigma_direct, sigma_f, _, _ = compute_sigma_gw_split_components(
        G, W2, Vq, grid, h0, float(mu), sigma_h, backend=backend
    )
    h_ref = h0 + sigma_h[None, None, :, :] + sigma_f
    sigma_sox = compute_sox_self_energy_periodic(
        G, Vq, h_ref, float(mu), grid, opts=sox_opts
    )
    return sigma_direct + sigma_sox, P, W2, sigma_f, sigma_sox


def cluster_gf2_self_energy(
    G_cluster: np.ndarray,
    rho_cluster: np.ndarray,
    h_cluster: np.ndarray,
    V_cluster: np.ndarray,
    mu: float,
    grid: MatsubaraGrid,
    *,
    sox_opts: SOXOptions = SOXOptions(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GF2 cluster self-energy for SEET-like double counting.

    The returned self-energy includes Hartree, matching the convention of
    ``cluster_gw_self_energy`` used by the historical embedding code.
    """
    Gc = np.asarray(G_cluster, dtype=complex)
    rho = np.asarray(rho_cluster, dtype=complex)
    hc = np.asarray(h_cluster, dtype=complex)
    Vc = np.asarray(V_cluster, dtype=complex)
    norb = int(Gc.shape[-1])
    if Gc.shape != (grid.nf, norb, norb):
        raise ValueError("cluster G shape mismatch")
    if rho.shape != (norb, norb) or hc.shape != (norb, norb) or Vc.shape != (norb, norb):
        raise ValueError("cluster matrix shape mismatch")

    cg = MatsubaraGrid(nk1=1, nk2=1, nw=grid.nw, nOmega=grid.nOmega, T=grid.T)
    G5 = Gc[:, None, None, :, :]
    h5 = hc[None, None, :, :]
    V5 = Vc[None, None, :, :]
    density = np.real(np.diag(rho))
    sigma_h = hartree_self_energy_matrix(density, Vc)
    sigma_nonh, P, W2, _, _ = compute_gf2_nonhartree(
        G5,
        V5,
        h5,
        float(mu),
        sigma_h,
        cg,
        backend="direct",
        sox_opts=sox_opts,
    )
    total = sigma_h[None, :, :] + sigma_nonh[:, 0, 0]
    return total, P[:, 0, 0], W2[:, 0, 0]


def solve_matrix_gf2(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    *,
    sox_opts: SOXOptions = SOXOptions(n_quad=64),
    initial: GWResult | None = None,
) -> GWResult:
    """Self-consistent periodic GF2 with the same fixed-filling controls as GW."""
    backend = _check_backend(opts.momentum_backend)
    method = _check_mixing_method(opts.mixing_method)
    sox_opts.validate()
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(h0.shape[-1])
    expected_h = (grid.nk1, grid.nk2, norb, norb)
    expected_dyn = (grid.nf, grid.nk1, grid.nk2, norb, norb)
    if h0.shape != expected_h or Vq.shape != expected_h:
        raise ValueError("h0/Vq shape mismatch")

    compatible = (
        initial is not None
        and np.asarray(initial.Sigma_H).shape == (norb, norb)
        and np.asarray(initial.Sigma_GW).shape == expected_dyn
    )
    if compatible:
        sigma_h = np.asarray(initial.Sigma_H, dtype=complex).copy()
        sigma_corr = np.asarray(initial.Sigma_GW, dtype=complex).copy()
        mu = float(initial.mu)
    else:
        sigma_h = np.zeros((norb, norb), dtype=complex)
        sigma_corr = np.zeros(expected_dyn, dtype=complex)
        mu = float(opts.mu)

    if opts.target_filling is None:
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_corr)
    else:
        mu, G, _, _ = _solve_mu_matrix_fast(
            h0,
            sigma_h,
            sigma_corr,
            grid,
            float(opts.target_filling),
            mu,
            float(opts.mu_tol),
            int(opts.mu_max_iter),
        )

    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    converged = False
    err = float("inf")
    P = np.zeros((grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    W2 = np.broadcast_to(Vq[None], P.shape).copy()
    density = np.zeros(norb, dtype=float)
    it = 0

    for it in range(1, int(opts.max_iter) + 1):
        rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
        rho_c = np.mean(rho_k, axis=(0, 1))
        density = np.real(np.diag(rho_c))
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
        sigma_corr_out, P, W2, _, sigma_sox = compute_gf2_nonhartree(
            G,
            Vq,
            h0,
            mu,
            sigma_h,
            grid,
            backend=backend,
            sox_opts=sox_opts,
        )
        err = _residual_error(sigma_h_out - sigma_h, sigma_corr_out - sigma_corr)
        if opts.verbose:
            print(
                f"SC-GF2 iter {it:4d}: residual={err:.3e}, mu={mu:.10f}, "
                f"n={np.sum(density):.10f}, max|Sigma_SOX|={np.max(np.abs(sigma_sox)):.3e}, "
                f"method={method}, backend={backend}"
            )
        if np.isfinite(err) and err < float(opts.tol):
            sigma_h = np.asarray(sigma_h_out)
            sigma_corr = np.asarray(sigma_corr_out)
            converged = True
        else:
            sigma_h, sigma_corr = _mixed_self_energies(
                sigma_h,
                sigma_corr,
                sigma_h_out,
                sigma_corr_out,
                opts,
                it,
                history,
            )

        if opts.target_filling is None:
            G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_corr)
        else:
            mu, G, _, _ = _solve_mu_matrix_fast(
                h0,
                sigma_h,
                sigma_corr,
                grid,
                float(opts.target_filling),
                mu,
                float(opts.mu_tol),
                int(opts.mu_max_iter),
            )
        if converged:
            break

    # Rebuild the map once on the accepted state for consistent saved pieces.
    rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    rho_c = np.mean(rho_k, axis=(0, 1))
    density = np.real(np.diag(rho_c))
    sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
    sigma_corr_out, P, W2, _, _ = compute_gf2_nonhartree(
        G,
        Vq,
        h0,
        mu,
        sigma_h,
        grid,
        backend=backend,
        sox_opts=sox_opts,
    )
    err = _residual_error(sigma_h_out - sigma_h, sigma_corr_out - sigma_corr)
    converged = bool(np.isfinite(err) and err < float(opts.tol))

    (
        smin,
        mmin,
        omin,
        q1min,
        q2min,
        screening_mode,
        density_mode,
        density_mode_residual,
    ) = screening_soft_modes_matrix(P, Vq, grid)

    return GWResult(
        G=np.asarray(G),
        W=np.asarray(W2),
        P=np.asarray(P),
        Sigma_H=np.asarray(sigma_h),
        Sigma_GW=np.asarray(sigma_corr),
        mu=float(mu),
        density=np.asarray(density),
        converged=bool(converged),
        iterations=int(it),
        final_error=float(err),
        mixing_method=method,
        min_screening_singular_value=float(smin),
        min_screening_m=int(mmin),
        min_screening_Omega=float(omin),
        min_screening_q1=float(q1min),
        min_screening_q2=float(q2min),
        min_screening_mode=np.asarray(screening_mode),
        min_density_mode=np.asarray(density_mode),
        min_density_mode_residual=float(density_mode_residual),
    )


__all__ = [
    "second_order_bare_interaction",
    "compute_gf2_nonhartree",
    "cluster_gf2_self_energy",
    "solve_matrix_gf2",
]

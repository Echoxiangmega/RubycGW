"""Frequency-dependent screened second-order exchange for periodic Ruby lattices.

Write W(Q)=V(q)+Wp(Q).  Three production modes are supported:

  sosex  = Sigma_SOX + 1/2 [Sigma_(Wp,V) + Sigma_(V,Wp)]
  2sosex = Sigma_SOX +     [Sigma_(Wp,V) + Sigma_(V,Wp)]
  g3w2   = Sigma_SOX + Sigma_(Wp,V) + Sigma_(V,Wp) + Sigma_(Wp,Wp)

The last line is the full two-screened-line dynamic G3W2 / WW skeleton.  The
V,V piece is evaluated by the existing tail-completed bare-SOX implementation;
only the polarizable part Wp=W-V is explicitly truncated on the represented
bosonic Matsubara grid.

For the full finite torus, with fermionic n and bosonic m,

  Sigma_(Wp,V),ij(n)
    = T sum_m,kl Wp_il(m) G_ik(n+m) B_klj(m) V_kj,
  B_klj(m) = T sum_r G_kl(r+m) G_lj(r),

  Sigma_(V,Wp),ij(n)
    = T sum_m,kl V_il C_ikl(m) Wp_kj(m) G_lj(n+m),
  C_ikl(m) = T sum_r G_ik(r) G_kl(r+m),

and the genuinely double-dynamic contribution is

  Sigma_(Wp,Wp),ij(n)
    = T^2 sum_m,m',k,l Wp_il(m) Wp_kj(m')
        G_ik(n+m) G_kl(n+m+m') G_lj(n+m').

The pair sums B,C are completed analytically with a static H+F reference.
Shifted Green functions in the Wp,Wp term are reference-tail completed whenever
n+m, n+m', or n+m+m' leaves the represented fermionic box.  Spatial momentum
routing is exact because k/q matrices are unfolded to the finite real-space
torus before contraction.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grids import MatsubaraGrid
from .sox_covariant import SOXOptions, _check_shapes, _check_static_kfield
from .sox_fast import (
    _full_periodic_to_kfield_batch,
    _interaction_neighbor_tables,
    _kfield_to_full_periodic_batch,
    compute_sox_self_energy_periodic_fast,
)


@dataclass(frozen=True)
class DynamicSOSEXOptions:
    n_quad: int = 128
    interaction_tol: float = 1.0e-13
    tail_complete: bool = True
    tail_edge_points: int = 2
    mode: str = "sosex"          # sosex, 2sosex, or g3w2
    max_full_sites: int = 36

    def validate(self):
        if int(self.n_quad) < 16:
            raise ValueError("dynamic SOSEX n_quad must be at least 16")
        if float(self.interaction_tol) < 0.0:
            raise ValueError("interaction_tol must be non-negative")
        if int(self.tail_edge_points) < 1:
            raise ValueError("tail_edge_points must be positive")
        if str(self.mode).strip().lower() not in {"sosex", "2sosex", "g3w2"}:
            raise ValueError("mode must be 'sosex', '2sosex', or 'g3w2'")
        if int(self.max_full_sites) < 1:
            raise ValueError("max_full_sites must be positive")
        return self


@dataclass
class DynamicSOSEXParts:
    Sigma: np.ndarray
    Sigma_SOX: np.ndarray
    Sigma_WpV: np.ndarray
    Sigma_VWp: np.ndarray
    Sigma_WpWp: np.ndarray
    mixed_line_relative_difference: float
    mode: str


def _fermi(x, T):
    y = np.asarray(x, dtype=float) / float(T)
    out = np.empty_like(y)
    hi = y > 40.0
    lo = y < -40.0
    mid = ~(hi | lo)
    out[hi] = 0.0
    out[lo] = 1.0
    out[mid] = 1.0 / (np.exp(y[mid]) + 1.0)
    return out


def _reference_full_eigensystem(h_ref, mu, grid):
    hfull = _kfield_to_full_periodic_batch(
        np.asarray(h_ref, dtype=complex)[None, ...]
    )[0]
    hfull = 0.5 * (hfull + hfull.conj().T)
    evals, U = np.linalg.eigh(hfull)
    xi = np.asarray(evals - float(mu), dtype=float)
    return U, xi, _fermi(xi, grid.T)


def _reference_green_full(U, xi, fermion_indices, T):
    n = np.asarray(fermion_indices, dtype=int)
    omega = (2.0 * n + 1.0) * np.pi * float(T)
    d = 1.0 / (1j * omega[:, None] - xi[None, :])
    return np.einsum("ia,na,ja->nij", U, d, U.conj(), optimize=True)


def _reference_pair_kernel(xi, occ, m, T):
    """Exact T sum_n g_p(n+m) g_q(n) for the static reference."""
    Om = 2.0 * np.pi * float(T) * int(m)
    num = occ[:, None] - occ[None, :]
    den = xi[:, None] - xi[None, :] - 1j * Om
    out = np.empty_like(den, dtype=complex)
    small = np.abs(den) < 1.0e-12
    np.divide(num, den, out=out, where=~small)
    if np.any(small):
        xavg = 0.5 * (xi[:, None] + xi[None, :])
        f = _fermi(xavg, T)
        fp = -(f * (1.0 - f)) / float(T)
        out[small] = fp[small]
    return out


def _reference_pair_tensors(U, xi, occ, m, T):
    S = _reference_pair_kernel(xi, occ, m, T)
    nsite = int(U.shape[0])
    Uc = U.conj()

    B = np.empty((nsite, nsite, nsite), dtype=complex)
    for l in range(nsite):
        left = U * Uc[l][None, :]
        right = U[l][:, None] * Uc.T
        B[:, l, :] = left @ S @ right

    C = np.empty((nsite, nsite, nsite), dtype=complex)
    for k in range(nsite):
        left = U * Uc[k][None, :]
        right = U[k][:, None] * Uc.T
        C[:, k, :] = left @ S.T @ right
    return B, C


def _shifted_green_with_reference_tail(Gfull, Gref_base, U, xi, m, grid):
    """Return G(n+m), continuing missing frequencies with the H+F reference."""
    base_n = np.asarray(grid.n_values, dtype=int)
    target_n = base_n + int(m)
    Gref_shift = _reference_green_full(U, xi, target_n, grid.T)
    out = np.array(Gref_shift, copy=True)
    delta = np.asarray(Gfull) - np.asarray(Gref_base)
    valid = (target_n >= -int(grid.nw)) & (target_n < int(grid.nw))
    if np.any(valid):
        src = target_n[valid] + int(grid.nw)
        out[valid] += delta[src]
    return out, Gref_shift


def _tail_completed_pair_tensors(
    Gfull, Gref_base, Gshift, Gref_shift, Bref, Cref, T
):
    Bcorr = float(T) * (
        np.einsum("nkl,nlj->klj", Gshift, Gfull, optimize=True)
        - np.einsum("nkl,nlj->klj", Gref_shift, Gref_base, optimize=True)
    )
    Ccorr = float(T) * (
        np.einsum("nik,nkl->ikl", Gfull, Gshift, optimize=True)
        - np.einsum("nik,nkl->ikl", Gref_base, Gref_shift, optimize=True)
    )
    return Bref + Bcorr, Cref + Ccorr


def _mixed_sparse_contractions(
    Gshift, Wp, B, C, row_idx, row_w, col_idx, col_w
):
    """Two mixed line orderings for one bosonic transfer."""
    nf, nsite, _ = Gshift.shape
    A = np.zeros((nf, nsite, nsite), dtype=complex)  # (Wp,V)
    D = np.zeros_like(A)                             # (V,Wp)

    for j in range(nsite):
        for a in range(col_idx.shape[1]):
            k = int(col_idx[j, a])
            vkj = col_w[j, a]
            if abs(vkj) == 0.0:
                continue
            vec_i = Wp @ B[k, :, j]
            A[:, :, j] += Gshift[:, :, k] * vec_i[None, :] * vkj

    for i in range(nsite):
        for a in range(row_idx.shape[1]):
            l = int(row_idx[i, a])
            vil = row_w[i, a]
            if abs(vil) == 0.0:
                continue
            vec_j = C[i, :, l] @ Wp
            D[:, i, :] += Gshift[:, l, :] * vec_j[None, :] * vil
    return A, D


def _double_screened_contraction(Wpm, Wpp, Gm, Gmm, Gmp):
    """One (m,m') contribution to the Wp,Wp G3W2 term.

    Computes, for every external fermionic frequency n,

      sum_kl Wpm[i,l] Wpp[k,j]
             Gm[n,i,k] Gmm[n,k,l] Gmp[n,l,j].

    The explicit j loop reduces the contraction to batched matrix products and
    is much faster than a generic five-factor einsum for the 12-site benchmark.
    """
    Gm = np.asarray(Gm, dtype=complex)
    Gmm = np.asarray(Gmm, dtype=complex)
    Gmp = np.asarray(Gmp, dtype=complex)
    Wpm = np.asarray(Wpm, dtype=complex)
    Wpp = np.asarray(Wpp, dtype=complex)
    nf, nsite, _ = Gm.shape
    out = np.zeros((nf, nsite, nsite), dtype=complex)
    for j in range(nsite):
        # X[n,i,k] = G(n+m)[i,k] Wp(m')[k,j]
        X = Gm * Wpp[:, j][None, None, :]
        # Y[n,i,l] = sum_k X[n,i,k] G(n+m+m')[k,l]
        Y = np.matmul(X, Gmm)
        # Finish the l contraction with Wp(m)[i,l] G(n+m')[l,j].
        out[:, :, j] = np.sum(
            Y * Wpm[None, :, :] * Gmp[:, :, j][:, None, :], axis=2
        )
    return out


def compute_dynamic_sosex_self_energy_periodic_fast(
    G,
    Vq,
    W,
    h_ref,
    mu,
    grid: MatsubaraGrid,
    opts: DynamicSOSEXOptions = DynamicSOSEXOptions(),
    *,
    return_parts: bool = False,
):
    """Dynamic SOSEX / G3W2 with full W(iOmega) dependence.

    mode='sosex':  SOX + 1/2[(Wp,V)+(V,Wp)]
    mode='2sosex': SOX +     [(Wp,V)+(V,Wp)]
    mode='g3w2':   SOX + (Wp,V)+(V,Wp)+(Wp,Wp) = full W,W skeleton
    """
    opts.validate()
    norb = _check_shapes(G, grid, "G")
    Vq = _check_static_kfield(Vq, grid, norb, "Vq")
    href = _check_static_kfield(h_ref, grid, norb, "h_ref")
    W = np.asarray(W, dtype=complex)
    if W.shape != (grid.nb, grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected W shape")

    nsite = int(grid.nk) * int(norb)
    if nsite > int(opts.max_full_sites):
        raise ValueError(
            f"dynamic SOSEX full-torus kernel has {nsite} sites; "
            f"increase max_full_sites={opts.max_full_sites} deliberately"
        )

    sox_opts = SOXOptions(
        n_quad=int(opts.n_quad),
        interaction_tol=float(opts.interaction_tol),
        tail_complete=bool(opts.tail_complete),
        tail_edge_points=int(opts.tail_edge_points),
    )
    sigma_sox = compute_sox_self_energy_periodic_fast(
        G, Vq, href, float(mu), grid, opts=sox_opts
    )

    Gfull = _kfield_to_full_periodic_batch(np.asarray(G, dtype=complex))
    Vfull = _kfield_to_full_periodic_batch(Vq[None, ...])[0]
    Wp_full = _kfield_to_full_periodic_batch(W - Vq[None, ...])
    row_idx, row_w, col_idx, col_w = _interaction_neighbor_tables(
        Vfull, float(opts.interaction_tol)
    )

    U, xi, occ = _reference_full_eigensystem(href, float(mu), grid)
    Gref_base = _reference_green_full(U, xi, grid.n_values, grid.T)

    WpV_full = np.zeros_like(Gfull)
    VWp_full = np.zeros_like(Gfull)
    shift_cache: dict[int, np.ndarray] = {0: np.asarray(Gfull)}

    def shifted(s: int):
        s = int(s)
        if s not in shift_cache:
            shift_cache[s] = _shifted_green_with_reference_tail(
                Gfull, Gref_base, U, xi, s, grid
            )[0]
        return shift_cache[s]

    for im, m_raw in enumerate(grid.m_values):
        m = int(m_raw)
        Wpm = np.asarray(Wp_full[im], dtype=complex)
        if np.max(np.abs(Wpm)) <= float(opts.interaction_tol):
            continue
        Gshift, Gref_shift = _shifted_green_with_reference_tail(
            Gfull, Gref_base, U, xi, m, grid
        )
        shift_cache[m] = Gshift
        Bref, Cref = _reference_pair_tensors(U, xi, occ, m, grid.T)
        B, C = _tail_completed_pair_tensors(
            Gfull, Gref_base, Gshift, Gref_shift, Bref, Cref, grid.T
        )
        A, D = _mixed_sparse_contractions(
            Gshift, Wpm, B, C, row_idx, row_w, col_idx, col_w
        )
        WpV_full += float(grid.T) * A
        VWp_full += float(grid.T) * D

    sigma_WpV = _full_periodic_to_kfield_batch(
        WpV_full, grid.nk1, grid.nk2, norb
    )
    sigma_VWp = _full_periodic_to_kfield_batch(
        VWp_full, grid.nk1, grid.nk2, norb
    )

    mode = str(opts.mode).strip().lower()
    WpWp_full = np.zeros_like(Gfull)
    if mode == "g3w2":
        active = [
            (im, int(m)) for im, m in enumerate(grid.m_values)
            if np.max(np.abs(Wp_full[im])) > float(opts.interaction_tol)
        ]
        T2 = float(grid.T) ** 2
        for im, m in active:
            Wpm = np.asarray(Wp_full[im], dtype=complex)
            Gm = shifted(m)
            for ip, mp in active:
                Wpp = np.asarray(Wp_full[ip], dtype=complex)
                WpWp_full += T2 * _double_screened_contraction(
                    Wpm, Wpp, Gm, shifted(m + mp), shifted(mp)
                )

    sigma_WpWp = _full_periodic_to_kfield_batch(
        WpWp_full, grid.nk1, grid.nk2, norb
    )

    if mode == "sosex":
        sigma = sigma_sox + 0.5 * (sigma_WpV + sigma_VWp)
    elif mode == "2sosex":
        sigma = sigma_sox + sigma_WpV + sigma_VWp
    else:
        sigma = sigma_sox + sigma_WpV + sigma_VWp + sigma_WpWp

    # The two mixed orderings are related by orbital transpose.  Compare them
    # in that symmetry channel rather than using the misleading raw A-D norm.
    den = max(
        float(np.linalg.norm(sigma_WpV.ravel())),
        float(np.linalg.norm(sigma_VWp.ravel())),
        1.0e-300,
    )
    asym = float(
        np.linalg.norm(
            (sigma_WpV - np.swapaxes(sigma_VWp, -1, -2)).ravel()
        ) / den
    )

    if return_parts:
        return DynamicSOSEXParts(
            Sigma=np.asarray(sigma),
            Sigma_SOX=np.asarray(sigma_sox),
            Sigma_WpV=np.asarray(sigma_WpV),
            Sigma_VWp=np.asarray(sigma_VWp),
            Sigma_WpWp=np.asarray(sigma_WpWp),
            mixed_line_relative_difference=asym,
            mode=mode,
        )
    return np.asarray(sigma)


__all__ = [
    "DynamicSOSEXOptions",
    "DynamicSOSEXParts",
    "compute_dynamic_sosex_self_energy_periodic_fast",
]

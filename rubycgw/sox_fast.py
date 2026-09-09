"""Fast periodic bare-SOX self-energy evaluation.

This module is a drop-in accelerator for the self-consistent GW+SOX background.
It does not change the SOX skeleton,

    Sigma_SOX,ij(tau) = sum_kl v_il v_kj G_ik(tau) G_kl(-tau) G_lj(tau),

or the tail completion used by :mod:`rubycgw.sox_covariant`.  The speedup comes
from implementation only:

* cache the static-reference eigensystem and G_ref(iw) once per SOX call;
* reconstruct several tau points at once;
* vectorize k <-> full-periodic real-space transforms;
* exploit the sparse interaction neighbor lists in one batched contraction.

The original transparent kernels remain in ``sox_covariant.py`` and are kept as
reference implementations for regression tests and the covariant SOX vertex.
"""
from __future__ import annotations

import numpy as np

from .grids import MatsubaraGrid
from .sox_covariant import (
    SOXOptions,
    _check_shapes,
    _check_static_kfield,
    _quadrature,
    _reference_eigensystem,
    reference_green_iomega,
)


# Small chunks keep the temporary (tau, i, j, z, z) tensor bounded on larger
# meshes while still removing the Python loop over every (i,j) pair.
_DEFAULT_TAU_BATCH = 8


def _kfield_to_full_periodic_batch(field_k: np.ndarray) -> np.ndarray:
    """Vectorized k-field -> full torus transform for a leading batch axis.

    Parameters
    ----------
    field_k
        Shape ``(nbatch,nk1,nk2,norb,norb)``.
    """
    F = np.asarray(field_k, dtype=complex)
    if F.ndim != 5 or F.shape[-1] != F.shape[-2]:
        raise ValueError("field_k must have shape (batch,nk1,nk2,norb,norb)")
    nbatch, nk1, nk2, norb, _ = F.shape
    blocks = np.fft.ifftn(F, axes=(1, 2))
    ncell = int(nk1) * int(nk2)
    cells = np.arange(ncell, dtype=int)
    c1 = cells // int(nk2)
    c2 = cells % int(nk2)
    d1 = (c1[:, None] - c1[None, :]) % int(nk1)
    d2 = (c2[:, None] - c2[None, :]) % int(nk2)
    # (batch, rcell, scell, a, b) -> (batch, rcell*a, scell*b)
    picked = blocks[:, d1, d2]
    return picked.transpose(0, 1, 3, 2, 4).reshape(
        nbatch, ncell * norb, ncell * norb
    )


def _full_periodic_to_kfield_batch(
    matrix: np.ndarray,
    nk1: int,
    nk2: int,
    norb: int,
) -> np.ndarray:
    """Vectorized translation projection/full torus -> k-field."""
    M = np.asarray(matrix, dtype=complex)
    ncell = int(nk1) * int(nk2)
    n = ncell * int(norb)
    if M.ndim != 3 or M.shape[1:] != (n, n):
        raise ValueError("matrix must have shape (batch,ncell*norb,ncell*norb)")
    nbatch = int(M.shape[0])
    blocks_rs = M.reshape(nbatch, ncell, norb, ncell, norb).transpose(
        0, 1, 3, 2, 4
    )

    cells = np.arange(ncell, dtype=int)
    s1 = cells // int(nk2)
    s2 = cells % int(nk2)
    # Flattened displacement ordering is d1*nk2+d2, identical to cell ordering.
    d1 = s1
    d2 = s2
    r1 = (d1[:, None] + s1[None, :]) % int(nk1)
    r2 = (d2[:, None] + s2[None, :]) % int(nk2)
    ridx = r1 * int(nk2) + r2
    sidx = cells[None, :]
    # selected: (batch, displacement, source-cell, a, b)
    selected = blocks_rs[:, ridx, sidx]
    blocks = selected.mean(axis=2).reshape(nbatch, nk1, nk2, norb, norb)
    return np.fft.fftn(blocks, axes=(1, 2))


def _interaction_neighbor_tables(v: np.ndarray, tol: float):
    """Build padded row/column interaction-neighbor tables once per SOX call."""
    v = np.asarray(v, dtype=complex)
    if v.ndim != 2 or v.shape[0] != v.shape[1]:
        raise ValueError("v must be square")
    n = int(v.shape[0])
    rows = [np.flatnonzero(np.abs(v[i]) > float(tol)) for i in range(n)]
    cols = [np.flatnonzero(np.abs(v[:, j]) > float(tol)) for j in range(n)]
    zr = max(1, max((x.size for x in rows), default=0))
    zc = max(1, max((x.size for x in cols), default=0))
    row_idx = np.zeros((n, zr), dtype=int)
    row_w = np.zeros((n, zr), dtype=complex)
    col_idx = np.zeros((n, zc), dtype=int)
    col_w = np.zeros((n, zc), dtype=complex)
    for i, ls in enumerate(rows):
        m = int(ls.size)
        if m:
            row_idx[i, :m] = ls
            row_w[i, :m] = v[i, ls]
    for j, ks in enumerate(cols):
        m = int(ks.size)
        if m:
            col_idx[j, :m] = ks
            col_w[j, :m] = v[ks, j]
    return row_idx, row_w, col_idx, col_w


def _sox_self_energy_sparse_batch(
    Gp: np.ndarray,
    Gm: np.ndarray,
    row_idx: np.ndarray,
    row_w: np.ndarray,
    col_idx: np.ndarray,
    col_w: np.ndarray,
) -> np.ndarray:
    """Batched sparse SOX contraction for full real-space torus matrices.

    Implements exactly

        sum_kl v_il v_kj Gp_ik Gm_kl Gp_lj

    while replacing the Python ``for i``/``for j`` loops by padded neighbor
    gathers.  For the Ruby interaction each site has only two nonzero V bonds,
    so the last two contraction dimensions are typically just 2 x 2.
    """
    Gp = np.asarray(Gp, dtype=complex)
    Gm = np.asarray(Gm, dtype=complex)
    if Gp.ndim != 3 or Gp.shape != Gm.shape or Gp.shape[1] != Gp.shape[2]:
        raise ValueError("Gp and Gm must have shape (batch,n,n)")
    n = int(Gp.shape[1])
    if row_idx.shape[0] != n or col_idx.shape[0] != n:
        raise ValueError("interaction neighbor tables incompatible with G")

    ii = np.arange(n, dtype=int)[:, None, None]
    jj = np.arange(n, dtype=int)[None, :, None]

    # left[t,i,j,a]  = Gp[t,i,k(j,a)] v[k(j,a),j]
    left = Gp[:, ii, col_idx[None, :, :]] * col_w[None, None, :, :]
    # right[t,i,j,b] = v[i,l(i,b)] Gp[t,l(i,b),j]
    right = Gp[:, row_idx[:, None, :], jj] * row_w[None, :, None, :]
    # middle[t,i,j,a,b] = Gm[t,k(j,a),l(i,b)]
    middle = Gm[:, col_idx[None, :, :, None], row_idx[:, None, None, :]]
    return np.einsum("tija,tijab,tijb->tij", left, middle, right, optimize=True)


def _reference_tau_pair_batch(
    U: np.ndarray,
    xi: np.ndarray,
    occ: np.ndarray,
    tau: np.ndarray,
):
    """Analytic static-reference G(+/-tau) using a cached eigensystem."""
    t = np.asarray(tau, dtype=float)[:, None, None, None]
    gp = -np.exp(-t * xi[None, ...]) * (1.0 - occ)[None, ...]
    gm = np.exp(+t * xi[None, ...]) * occ[None, ...]
    Gp = np.einsum("xyia,txya,xyja->txyij", U, gp, U.conj(), optimize=True)
    Gm = np.einsum("xyia,txya,xyja->txyij", U, gm, U.conj(), optimize=True)
    return Gp, Gm


def compute_sox_self_energy_periodic_fast(
    G: np.ndarray,
    Vq: np.ndarray,
    h_ref: np.ndarray,
    mu: float,
    grid: MatsubaraGrid,
    opts: SOXOptions = SOXOptions(),
) -> np.ndarray:
    """Fast, numerically equivalent periodic bare-SOX self-energy."""
    norb = _check_shapes(G, grid, "G")
    Vq = _check_static_kfield(Vq, grid, norb, "Vq")
    href = _check_static_kfield(h_ref, grid, norb, "h_ref")
    G = np.asarray(G, dtype=complex)

    # V is fixed throughout this call.  Build the full torus interaction and
    # its sparse neighbor tables only once, rather than at every tau point.
    vfull = _kfield_to_full_periodic_batch(Vq[None, ...])[0]
    row_idx, row_w, col_idx, col_w = _interaction_neighbor_tables(
        vfull, float(opts.interaction_tol)
    )

    tau, weight = _quadrature(grid, opts)
    omega = np.asarray(grid.omega, dtype=float)
    out = np.zeros_like(G)

    # Everything below is invariant across quadrature points in one SOX call.
    if opts.tail_complete:
        _, U, xi, occ = _reference_eigensystem(href, mu, grid.T)
        Gref_iw = reference_green_iomega(href, mu, grid)
        delta = G - Gref_iw
    else:
        U = xi = occ = delta = None

    nt = int(tau.size)
    batch = min(_DEFAULT_TAU_BATCH, max(nt, 1))
    for start in range(0, nt, batch):
        stop = min(start + batch, nt)
        tb = np.asarray(tau[start:stop], dtype=float)
        wb = np.asarray(weight[start:stop], dtype=float)
        phase_p = np.exp(-1j * tb[:, None] * omega[None, :])
        phase_m = np.exp(+1j * tb[:, None] * omega[None, :])

        if opts.tail_complete:
            Gp_ref, Gm_ref = _reference_tau_pair_batch(U, xi, occ, tb)
            Gp = Gp_ref + float(grid.T) * np.einsum(
                "tn,nxyab->txyab", phase_p, delta, optimize=True
            )
            Gm = Gm_ref + float(grid.T) * np.einsum(
                "tn,nxyab->txyab", phase_m, delta, optimize=True
            )
        else:
            Gp = float(grid.T) * np.einsum(
                "tn,nxyab->txyab", phase_p, G, optimize=True
            )
            Gm = float(grid.T) * np.einsum(
                "tn,nxyab->txyab", phase_m, G, optimize=True
            )

        Gp_full = _kfield_to_full_periodic_batch(Gp)
        Gm_full = _kfield_to_full_periodic_batch(Gm)
        sig_full = _sox_self_energy_sparse_batch(
            Gp_full, Gm_full, row_idx, row_w, col_idx, col_w
        )
        sig_k = _full_periodic_to_kfield_batch(
            sig_full, grid.nk1, grid.nk2, norb
        )
        phase_out = np.exp(+1j * omega[:, None] * tb[None, :])
        out += np.einsum(
            "nt,t,txyab->nxyab", phase_out, wb, sig_k, optimize=True
        )

    return out


__all__ = [
    "compute_sox_self_energy_periodic_fast",
]

"""Statically screened SOSEX self-energy for periodic Ruby-lattice calculations.

This module generalizes the existing bare-SOX skeleton to two *static* density
interactions U and X,

    S[U,X]_ij(tau)
      = sum_kl U_il X_kj G_ik(tau) G_kl(-tau) G_lj(tau).

The production screened-SOSEX diagnostic uses the symmetrized one-screened-line
form

    Sigma_sSOSEX = 1/2 { S[V,W0] + S[W0,V] },

where W0(q) = W(q,Omega=0).  It has three useful properties for the present
project:

1. W0 -> V reproduces the existing bare SOX *exactly*;
2. expanding W0 = V + V P(0) V + ... resums bubble-screened crossed-exchange
   diagrams beyond O(V^2);
3. only one screened line is present in each term, so the cost remains modest
   even when W0 is dense in the finite-torus real-space representation.

For comparison, ``mode='twoW'`` evaluates S[W0,W0], i.e. the fully statically
screened G3W2/SOSEX skeleton.  This is substantially more expensive because
both interaction neighbor tables are generally dense.

This is a *static-screening* approximation.  It is not the fully dynamic G3W2
self-energy, which contains two independent internal time/frequency variables.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grids import MatsubaraGrid
from .sox_covariant import (
    _check_shapes,
    _check_static_kfield,
    _quadrature,
    _reference_eigensystem,
    reference_green_iomega,
)
from .sox_fast import (
    _DEFAULT_TAU_BATCH,
    _full_periodic_to_kfield_batch,
    _interaction_neighbor_tables,
    _kfield_to_full_periodic_batch,
    _reference_tau_pair_batch,
    _sox_self_energy_sparse_batch,
)


@dataclass(frozen=True)
class ScreenedSOSEXOptions:
    """Numerical controls for the static screened-SOSEX diagnostic."""

    n_quad: int = 128
    interaction_tol: float = 1.0e-13
    tail_complete: bool = True
    tail_edge_points: int = 2
    mode: str = "oneW-sym"

    def validate(self) -> "ScreenedSOSEXOptions":
        if int(self.n_quad) < 16:
            raise ValueError("sSOSEX n_quad must be at least 16")
        if float(self.interaction_tol) < 0.0:
            raise ValueError("interaction_tol must be non-negative")
        if int(self.tail_edge_points) < 1:
            raise ValueError("tail_edge_points must be positive")
        mode = str(self.mode).strip().lower()
        if mode not in {"onew-sym", "twow"}:
            raise ValueError("sSOSEX mode must be 'oneW-sym' or 'twoW'")
        return self


def static_W0(W: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    """Return W(q,Omega=0) from a bosonic Matsubara interaction array."""
    arr = np.asarray(W, dtype=complex)
    if arr.ndim != 5:
        raise ValueError("W must have shape (nb,nk1,nk2,norb,norb)")
    im = np.flatnonzero(np.asarray(grid.m_values, dtype=int) == 0)
    if im.size != 1:
        raise ValueError("bosonic grid must contain a unique m=0 sector")
    return np.asarray(arr[int(im[0])], dtype=complex)


def _static_two_interaction_sox_fast(
    G: np.ndarray,
    Uq: np.ndarray,
    Xq: np.ndarray,
    h_ref: np.ndarray,
    mu: float,
    grid: MatsubaraGrid,
    opts: ScreenedSOSEXOptions,
) -> np.ndarray:
    """Evaluate S[U,X] with the same tau/tail convention as bare SOX."""
    norb = _check_shapes(G, grid, "G")
    Uq = _check_static_kfield(Uq, grid, norb, "Uq")
    Xq = _check_static_kfield(Xq, grid, norb, "Xq")
    href = _check_static_kfield(h_ref, grid, norb, "h_ref")
    G = np.asarray(G, dtype=complex)

    ufull = _kfield_to_full_periodic_batch(Uq[None, ...])[0]
    xfull = _kfield_to_full_periodic_batch(Xq[None, ...])[0]
    urow_idx, urow_w, _, _ = _interaction_neighbor_tables(
        ufull, float(opts.interaction_tol)
    )
    _, _, xcol_idx, xcol_w = _interaction_neighbor_tables(
        xfull, float(opts.interaction_tol)
    )

    # Reuse the production SOX quadrature/tail machinery.  _quadrature only
    # requires these option attributes, so the screened options deliberately
    # keep the same names.
    tau, weight = _quadrature(grid, opts)
    omega = np.asarray(grid.omega, dtype=float)
    out = np.zeros_like(G)

    if opts.tail_complete:
        _, Ueig, xi, occ = _reference_eigensystem(href, mu, grid.T)
        Gref_iw = reference_green_iomega(href, mu, grid)
        delta = G - Gref_iw
    else:
        Ueig = xi = occ = delta = None

    nt = int(tau.size)
    batch = min(_DEFAULT_TAU_BATCH, max(nt, 1))
    for start in range(0, nt, batch):
        stop = min(start + batch, nt)
        tb = np.asarray(tau[start:stop], dtype=float)
        wb = np.asarray(weight[start:stop], dtype=float)
        phase_p = np.exp(-1j * tb[:, None] * omega[None, :])
        phase_m = np.exp(+1j * tb[:, None] * omega[None, :])

        if opts.tail_complete:
            Gp_ref, Gm_ref = _reference_tau_pair_batch(Ueig, xi, occ, tb)
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
            Gp_full,
            Gm_full,
            urow_idx,
            urow_w,
            xcol_idx,
            xcol_w,
        )
        sig_k = _full_periodic_to_kfield_batch(
            sig_full, grid.nk1, grid.nk2, norb
        )
        phase_out = np.exp(+1j * omega[:, None] * tb[None, :])
        out += np.einsum(
            "nt,t,txyab->nxyab", phase_out, wb, sig_k, optimize=True
        )

    return out


def compute_static_screened_sosex_self_energy_periodic_fast(
    G: np.ndarray,
    Vq: np.ndarray,
    W: np.ndarray,
    h_ref: np.ndarray,
    mu: float,
    grid: MatsubaraGrid,
    opts: ScreenedSOSEXOptions = ScreenedSOSEXOptions(),
) -> np.ndarray:
    """Return the statically screened SOSEX self-energy.

    ``oneW-sym`` (default):

        1/2 [ S[V,W0] + S[W0,V] ].

    ``twoW``:

        S[W0,W0].

    In either mode, setting W0=V gives the bare-SOX skeleton exactly.
    """
    opts.validate()
    Vq = np.asarray(Vq, dtype=complex)
    W0 = static_W0(W, grid)
    mode = str(opts.mode).strip().lower()

    if mode == "onew-sym":
        left = _static_two_interaction_sox_fast(
            G, Vq, W0, h_ref, mu, grid, opts
        )
        right = _static_two_interaction_sox_fast(
            G, W0, Vq, h_ref, mu, grid, opts
        )
        return 0.5 * (left + right)

    return _static_two_interaction_sox_fast(
        G, W0, W0, h_ref, mu, grid, opts
    )


__all__ = [
    "ScreenedSOSEXOptions",
    "static_W0",
    "compute_static_screened_sosex_self_energy_periodic_fast",
]

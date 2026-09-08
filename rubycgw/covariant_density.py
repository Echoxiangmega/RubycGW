"""Full dynamic covariant density-density susceptibility for post-GW.

The screened interaction couples to orbital densities, so the response required
by post-GW is the reducible two-index matrix

    chi_nn,ab(Q) = delta <n_a(Q)> / delta phi_b(Q),

not a projected pseudospin susceptibility.  In the RubycGW sign convention
X=G Gamma G=-dG/dh this becomes

    chi_nn,ab(Q) = -(T/Nk) sum_k,n X_b(k;Q)_{aa}.

This module computes that matrix for every represented bosonic Q.  The same
provider can be used on a GW background or, with ``include_sox=True``, on a
GW+SOX background.  In the latter case the finite-transfer derivative of the
SOX skeleton is included in the vertex equation, so the response matches the
one-particle functional being post-corrected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grids import MatsubaraGrid
from .response_tail import build_tail_reference
from .sox_covariant import SOXOptions
from .sox_transfer import compute_sox_vertex_transfer_periodic
from .supercell_cgw import SupercellVertexOptions
from .transfer_cgw import (
    negative_transfer,
    normalize_q_index,
    solve_vertex_transfer_tail,
    transfer_response_tail_completed,
)


@dataclass
class CovariantDensityResult:
    chi_raw: np.ndarray
    chi_completed: np.ndarray
    tail_correction: np.ndarray
    transfer_converged: np.ndarray
    transfer_max_error: np.ndarray
    solved_explicitly: np.ndarray
    include_sox: bool
    m_max: int | None


def _density_projectors(norb: int) -> np.ndarray:
    out = np.zeros((norb, norb, norb), dtype=complex)
    idx = np.arange(norb)
    out[idx, idx, idx] = 1.0
    return out


def _representative_transfer(
    q_index: tuple[int, int], m: int, grid: MatsubaraGrid
) -> bool:
    """Choose one member of Q/-Q; prefer positive Matsubara frequency."""
    p = normalize_q_index(q_index, grid)
    pn, mn = negative_transfer(p, int(m), grid)
    if int(m) > 0:
        return True
    if int(m) < 0:
        return False
    return p <= pn


def compute_covariant_density_susceptibility(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    h0: np.ndarray,
    mu: float,
    sigma_h: np.ndarray,
    sigma_f: np.ndarray,
    grid: MatsubaraGrid,
    *,
    vertex_opts: SupercellVertexOptions = SupercellVertexOptions(),
    include_sox: bool = False,
    sox_opts: SOXOptions = SOXOptions(),
    m_max: int | None = None,
    allow_unconverged: bool = False,
) -> CovariantDensityResult:
    """Compute chi_nn(Q) for all represented Q, optionally with SOX vertices.

    ``m_max`` is a diagnostic acceleration knob.  When supplied, only
    |m|<=m_max is post-corrected and the remaining entries are left NaN so a
    caller can explicitly fall back to the background W.  ``m_max=None`` is the
    full post-GW calculation.
    """
    G = np.asarray(G, dtype=complex)
    W = np.asarray(W, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    h0 = np.asarray(h0, dtype=complex)
    sigma_h = np.asarray(sigma_h, dtype=complex)
    sigma_f = np.asarray(sigma_f, dtype=complex)
    norb = int(G.shape[-1])
    expected_g = (grid.nf, grid.nk1, grid.nk2, norb, norb)
    if G.shape != expected_g:
        raise ValueError("unexpected G shape")
    if W.shape != (grid.nb, grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected W shape")
    if Vq.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected Vq shape")
    if h0.shape != Vq.shape:
        raise ValueError("h0 and Vq must use the same periodic matrix shape")
    if sigma_h.shape != (norb, norb):
        raise ValueError("sigma_h shape mismatch")
    if sigma_f.shape != h0.shape:
        raise ValueError("sigma_f shape mismatch")
    if m_max is not None and int(m_max) < 0:
        raise ValueError("m_max must be non-negative or None")

    reference_hf = build_tail_reference(h0, float(mu), sigma_h, grid)
    h_static = h0 + sigma_h[None, None, :, :] + sigma_f
    Kdens = _density_projectors(norb)
    shape = (grid.nb, grid.nk1, grid.nk2, norb, norb)
    raw = np.full(shape, np.nan + 0j)
    completed = np.full(shape, np.nan + 0j)
    tail = np.full(shape, np.nan + 0j)
    converged = np.zeros((grid.nb, grid.nk1, grid.nk2), dtype=bool)
    max_error = np.full((grid.nb, grid.nk1, grid.nk2), np.nan)
    explicit = np.zeros((grid.nb, grid.nk1, grid.nk2), dtype=bool)

    # Warm starts are kept separately for each density source.  Adjacent
    # transfers usually have similar vertices, while the transfer solver masks
    # frequencies that are invalid for the new m_ext.
    initial_gamma: list[np.ndarray | None] = [None] * norb

    for im, m_raw in enumerate(grid.m_values):
        m = int(m_raw)
        if m_max is not None and abs(m) > int(m_max):
            continue
        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                p = (iq1, iq2)
                if not _representative_transfer(p, m, grid):
                    continue

                gammas: list[np.ndarray] = []
                errs = []
                oks = []
                for b in range(norb):
                    Kb = Kdens[b]
                    if include_sox:
                        def extra(field, p=p, m=m):
                            return compute_sox_vertex_transfer_periodic(
                                G,
                                field,
                                Vq,
                                h_static,
                                float(mu),
                                p,
                                m,
                                grid,
                                opts=sox_opts,
                            )
                    else:
                        extra = None

                    vr = solve_vertex_transfer_tail(
                        G,
                        W,
                        Vq,
                        Kb,
                        p,
                        m,
                        grid,
                        reference_hf,
                        opts=vertex_opts,
                        initial_gamma=initial_gamma[b],
                        extra_kernel=extra,
                    )
                    initial_gamma[b] = vr.Gamma
                    gammas.append(vr.Gamma)
                    errs.append(float(vr.final_error))
                    oks.append(bool(vr.converged))
                    if not vr.converged and not allow_unconverged:
                        label = "cGW+SOX" if include_sox else "cGW"
                        raise RuntimeError(
                            f"{label} density vertex did not converge at "
                            f"q={p}, m={m:+d}, source={b}: "
                            f"{vr.final_error:.3e}"
                        )

                resp = transfer_response_tail_completed(
                    G,
                    Kdens,
                    gammas,
                    p,
                    m,
                    grid,
                    h_static,
                    float(mu),
                    edge_points=int(sox_opts.tail_edge_points),
                )
                raw[im, iq1, iq2] = resp["raw"]
                completed[im, iq1, iq2] = resp["completed"]
                tail[im, iq1, iq2] = resp["tail_correction"]
                converged[im, iq1, iq2] = bool(all(oks))
                max_error[im, iq1, iq2] = float(max(errs)) if errs else 0.0
                explicit[im, iq1, iq2] = True

                pn, mn = negative_transfer(p, m, grid)
                neg_matches = np.flatnonzero(np.asarray(grid.m_values) == int(mn))
                if neg_matches.size:
                    jm = int(neg_matches[0])
                    j1, j2 = pn
                    if (jm, j1, j2) != (im, iq1, iq2):
                        raw[jm, j1, j2] = resp["raw"].conj().T
                        completed[jm, j1, j2] = resp["completed"].conj().T
                        tail[jm, j1, j2] = resp["tail_correction"].conj().T
                        converged[jm, j1, j2] = bool(all(oks))
                        max_error[jm, j1, j2] = max_error[im, iq1, iq2]

    return CovariantDensityResult(
        chi_raw=raw,
        chi_completed=completed,
        tail_correction=tail,
        transfer_converged=converged,
        transfer_max_error=max_error,
        solved_explicitly=explicit,
        include_sox=bool(include_sox),
        m_max=None if m_max is None else int(m_max),
    )


__all__ = ["CovariantDensityResult", "compute_covariant_density_susceptibility"]

"""Warm-started covariant density response for iterative GW-Gamma_P feedback.

This module preserves the equations of :mod:`rubycgw.covariant_density` but
allows converged three-point density vertices to be reused between successive
outer feedback iterations.  The expensive transfer BSE/GMRES systems change
only smoothly when G and W are mixed, so this cache can substantially reduce
Krylov work after the first outer iteration without changing the fixed point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .covariant_density import CovariantDensityResult, _density_projectors, _representative_transfer
from .grids import MatsubaraGrid
from .response_tail import build_tail_reference
from .sox_covariant import SOXOptions
from .sox_transfer import compute_sox_vertex_transfer_periodic
from .supercell_cgw import SupercellVertexOptions
from .transfer_cgw import (
    negative_transfer,
    solve_vertex_transfer_tail,
    transfer_response_tail_completed,
)


@dataclass(frozen=True)
class DensityVertexStats:
    n_solves: int
    total_iterations: int
    max_iterations: int
    mean_iterations: float
    cache_hits: int


def compute_covariant_density_susceptibility_fast(
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
    initial_gamma_cache: dict[tuple[int, int, int, int], np.ndarray] | None = None,
) -> tuple[
    CovariantDensityResult,
    dict[tuple[int, int, int, int], np.ndarray],
    DensityVertexStats,
]:
    """Compute the same covariant density response with reusable Gamma seeds.

    Cache keys are ``(m, iq1, iq2, density_source)`` for the explicitly solved
    representative transfers.  If an exact-transfer seed is unavailable, the
    historical adjacent-transfer warm start for the same density source is
    retained as a fallback.
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

    gamma_cache = {} if initial_gamma_cache is None else dict(initial_gamma_cache)
    adjacent_seed: list[np.ndarray | None] = [None] * norb
    total_iterations = 0
    max_iterations = 0
    n_solves = 0
    cache_hits = 0

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
                errs: list[float] = []
                oks: list[bool] = []
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

                    key = (m, iq1, iq2, b)
                    seed = gamma_cache.get(key)
                    if seed is not None and np.asarray(seed).shape == G.shape:
                        cache_hits += 1
                    else:
                        seed = adjacent_seed[b]

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
                        initial_gamma=seed,
                        extra_kernel=extra,
                    )
                    gamma_cache[key] = vr.Gamma
                    adjacent_seed[b] = vr.Gamma
                    gammas.append(vr.Gamma)
                    errs.append(float(vr.final_error))
                    oks.append(bool(vr.converged))
                    nit = int(vr.iterations)
                    total_iterations += nit
                    max_iterations = max(max_iterations, nit)
                    n_solves += 1
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

    result = CovariantDensityResult(
        chi_raw=raw,
        chi_completed=completed,
        tail_correction=tail,
        transfer_converged=converged,
        transfer_max_error=max_error,
        solved_explicitly=explicit,
        include_sox=bool(include_sox),
        m_max=None if m_max is None else int(m_max),
    )
    stats = DensityVertexStats(
        n_solves=int(n_solves),
        total_iterations=int(total_iterations),
        max_iterations=int(max_iterations),
        mean_iterations=(float(total_iterations) / n_solves if n_solves else 0.0),
        cache_hits=int(cache_hits),
    )
    return result, gamma_cache, stats


__all__ = [
    "DensityVertexStats",
    "compute_covariant_density_susceptibility_fast",
]

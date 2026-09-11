"""Covariant n/B/J channel response and Gamma_P feedback for same-torus Fierz GW.

This module is restricted to ``nk1=nk2=1``.  It generalizes the density-only
Gamma_P feedback to the joint Fierz channel space used by weighted n/B/J GW.
The bosonic coupling ``g_ab`` and bilinear vertices ``K_a`` are kept exactly as
specified by :class:`~rubycgw.fierz_channel_gw.ChannelDefinition`.

For a covariant reducible channel response

    chi_ab(Q) = -T sum_n Tr[K_a G(n+Q) Gamma_b(n;Q) G(n)],

the screened interaction is updated through the exact channel-space identity

    W_Gamma(Q) = g - g chi(Q) g.

The outer fixed point remains a Gamma_P *diagnostic*: the covariant vertex is
fed back through screening, while the self-energy keeps the channel-GW form
Sigma_c = -K G K (W_Gamma-g).  It is not a full Hedin GWGamma closure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fierz_channel_gw import (
    ChannelDefinition,
    ChannelGWResult,
    _channel_sigma_from_boson,
    _pair_polarization,
    channel_static_self_energy,
    compute_channel_correlation_self_energy,
)
from .grids import MatsubaraGrid, frequency_shift_slices
from .gw import GWOptions
from .hedin_gamma_fast import (
    GammaPFeedbackOptions,
    _feedback_error,
    _mixed_feedback,
    _dyson_fixed_filling,
)
from .response_tail import build_tail_reference, reference_tail_remainder
from .supercell_cgw import SupercellVertexOptions, _gmres_matrix_free
from .supercell_gw_split import one_body_density_matrix_tail
from .transfer_cgw import (
    estimate_transfer_static_vertex,
    transfer_response_tail_completed,
    transfer_x_field,
)


@dataclass
class ChannelTransferVertexResult:
    Gamma: np.ndarray
    Gamma_static: np.ndarray
    Gamma_MT: np.ndarray
    Gamma_AL1: np.ndarray
    Gamma_AL2: np.ndarray
    m_ext: int
    converged: bool
    iterations: int
    final_error: float


@dataclass
class CovariantChannelResult:
    chi_raw: np.ndarray
    chi_completed: np.ndarray
    tail_correction: np.ndarray
    transfer_converged: np.ndarray
    transfer_max_error: np.ndarray
    solved_explicitly: np.ndarray
    m_max: int | None


@dataclass
class ChannelGammaPResult:
    G: np.ndarray
    W: np.ndarray
    P_gamma: np.ndarray
    chi_cov: np.ndarray
    Sigma_static: np.ndarray
    Sigma_c: np.ndarray
    mu: float
    rho: np.ndarray
    converged: bool
    iterations: int
    final_error: float
    response: CovariantChannelResult
    fallback_mask: np.ndarray
    screening_identity_error: float
    background: ChannelGWResult
    vertex_gamma_cache: dict[tuple[int, int], np.ndarray]


def _require_same_torus(grid: MatsubaraGrid):
    if int(grid.nk1) != 1 or int(grid.nk2) != 1:
        raise ValueError("Fierz channel Gamma_P feedback currently requires nk1=nk2=1")


def _mask_transfer(field: np.ndarray, m_ext: int, grid: MatsubaraGrid) -> np.ndarray:
    arr = np.asarray(field, dtype=complex)
    out = np.zeros_like(arr)
    _, dst = frequency_shift_slices(grid.nf, int(m_ext))
    if dst.stop != dst.start:
        out[dst] = arr[dst]
    return out


def _bare_vertex_field(K: np.ndarray, G: np.ndarray, m_ext: int, grid: MatsubaraGrid):
    out = np.zeros_like(G, dtype=complex)
    _, dst = frequency_shift_slices(grid.nf, int(m_ext))
    if dst.stop != dst.start:
        out[dst, 0, 0] = np.asarray(K, dtype=complex)
    return out


def _screened_channel_at_m(
    G: np.ndarray,
    definition: ChannelDefinition,
    m: int,
    grid: MatsubaraGrid,
) -> np.ndarray:
    G0 = np.asarray(G, dtype=complex)[:, 0, 0]
    src, dst = frequency_shift_slices(grid.nf, int(m))
    nch = len(definition.labels)
    if src.stop == src.start:
        Pm = np.zeros((nch, nch), dtype=complex)
    else:
        Pm = _pair_polarization(G0[src], G0[dst], definition.vertices, grid.T)
    g = np.asarray(definition.coupling, dtype=complex)
    eye = np.eye(nch, dtype=complex)
    return np.linalg.solve(eye - g @ Pm, g)


def _build_channel_w_lookup(
    G: np.ndarray,
    W: np.ndarray,
    definition: ChannelDefinition,
    m_ext: int,
    grid: MatsubaraGrid,
) -> dict[int, np.ndarray]:
    stored = {int(m): np.asarray(W[i], dtype=complex) for i, m in enumerate(grid.m_values)}
    needed = {int(mi) - int(m_ext) for mi in grid.m_values}
    out = dict(stored)
    for m in sorted(needed):
        if m not in out:
            out[m] = _screened_channel_at_m(G, definition, m, grid)
    return out


def _completed_drho_from_x(
    X: np.ndarray,
    Gamma: np.ndarray,
    h_static_ref: np.ndarray,
    mu: float,
    m_ext: int,
    grid: MatsubaraGrid,
    *,
    edge_points: int,
) -> np.ndarray:
    """Tail-completed ``T sum_n X_n`` for the folded torus."""
    raw = float(grid.T) * np.sum(np.asarray(X, dtype=complex)[:, 0, 0], axis=0)
    norb = int(X.shape[-1])
    ref = build_tail_reference(
        np.asarray(h_static_ref, dtype=complex),
        float(mu),
        np.zeros((norb, norb), dtype=complex),
        grid,
    )
    S = estimate_transfer_static_vertex(Gamma, int(m_ext), grid, edge_points=edge_points)
    corr = reference_tail_remainder(ref, S, grid, q_index=(0, 0), m_ext=int(m_ext))
    return raw + np.asarray(corr[0, 0], dtype=complex)


def _static_vertex_from_x(
    X: np.ndarray,
    Gamma: np.ndarray,
    definition: ChannelDefinition,
    h_static_ref: np.ndarray,
    mu: float,
    m_ext: int,
    grid: MatsubaraGrid,
    opts: SupercellVertexOptions,
    *,
    edge_points: int,
) -> np.ndarray:
    if not opts.include_hartree and not opts.include_fock:
        return np.zeros_like(Gamma)
    drho = _completed_drho_from_x(
        X, Gamma, h_static_ref, mu, m_ext, grid, edge_points=edge_points
    )
    K = np.asarray(definition.vertices, dtype=complex)
    g = np.asarray(definition.coupling, dtype=complex)
    dobs = np.einsum("aij,ji->a", K, drho, optimize=True)
    d_tad = np.einsum("aij,ab,b->ij", K, g, dobs, optimize=True)
    d_ex = -np.einsum("ba,aik,kl,blj->ij", g, K, drho, K, optimize=True)
    d_static = np.zeros_like(d_tad)
    if opts.include_hartree:
        d_static += d_tad
    if opts.include_fock:
        d_static += d_ex
    if int(m_ext) == 0:
        d_static = 0.5 * (d_static + d_static.conj().T)
    out = np.zeros_like(Gamma)
    _, dst = frequency_shift_slices(grid.nf, int(m_ext))
    if dst.stop != dst.start:
        out[dst, 0, 0] = d_static
    return out


def _dynamic_vertex_parts(
    G: np.ndarray,
    W: np.ndarray,
    definition: ChannelDefinition,
    X: np.ndarray,
    m_ext: int,
    grid: MatsubaraGrid,
    opts: SupercellVertexOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    G0 = np.asarray(G, dtype=complex)[:, 0, 0]
    X0 = np.asarray(X, dtype=complex)[:, 0, 0]
    g = np.asarray(definition.coupling, dtype=complex)
    Wc = np.asarray(W, dtype=complex) - g[None, ...]
    lookup = _build_channel_w_lookup(G, W, definition, m_ext, grid)
    K = np.asarray(definition.vertices, dtype=complex)

    gmt0 = np.zeros_like(G0)
    gal10 = np.zeros_like(G0)
    gal20 = np.zeros_like(G0)
    for im, mi_raw in enumerate(grid.m_values):
        mi = int(mi_raw)
        src2, dst2 = frequency_shift_slices(grid.nf, mi)
        if src2.stop == src2.start:
            continue

        if opts.include_mt:
            gmt0[dst2] += _channel_sigma_from_boson(
                X0[src2], Wc[im], K, grid.T
            )

        if not opts.include_al:
            continue

        mleft = mi - int(m_ext)
        src1, dst1 = frequency_shift_slices(grid.nf, mleft)
        if src1.stop == src1.start:
            L1 = np.zeros_like(g)
        else:
            L1 = _pair_polarization(X0[src1], G0[dst1], K, grid.T)
        L2 = _pair_polarization(G0[src2], X0[dst2], K, grid.T)
        Wleft = lookup[mleft]
        Wright = np.asarray(W[im], dtype=complex)
        M1 = Wleft @ L1 @ Wright
        M2 = Wleft @ L2 @ Wright
        gal10[dst2] += _channel_sigma_from_boson(G0[src2], M1, K, grid.T)
        gal20[dst2] += _channel_sigma_from_boson(G0[src2], M2, K, grid.T)

    return (
        _mask_transfer(gmt0[:, None, None], m_ext, grid),
        _mask_transfer(gal10[:, None, None], m_ext, grid),
        _mask_transfer(gal20[:, None, None], m_ext, grid),
    )


def solve_channel_vertex_transfer(
    background: ChannelGWResult,
    definition: ChannelDefinition,
    Kext: np.ndarray,
    h0: np.ndarray,
    grid: MatsubaraGrid,
    m_ext: int,
    *,
    opts: SupercellVertexOptions = SupercellVertexOptions(),
    initial_gamma: np.ndarray | None = None,
    tail_edge_points: int = 2,
) -> ChannelTransferVertexResult:
    """Solve the finite-Matsubara-transfer covariant channel vertex at q=0."""
    _require_same_torus(grid)
    G = np.asarray(background.G, dtype=complex)
    W = np.asarray(background.W, dtype=complex)
    h0 = np.asarray(h0, dtype=complex)
    Kext = np.asarray(Kext, dtype=complex)
    if h0.shape != (1, 1, G.shape[-1], G.shape[-1]):
        raise ValueError("h0 must have folded-torus shape (1,1,norb,norb)")
    if W.shape[0] != grid.nb:
        raise ValueError("W/grid bosonic-frequency shape mismatch")
    if int(tail_edge_points) < 1:
        raise ValueError("tail_edge_points must be positive")

    m_ext = int(m_ext)
    rhs = _bare_vertex_field(Kext, G, m_ext, grid)
    if initial_gamma is None or np.asarray(initial_gamma).shape != G.shape:
        gamma0 = np.array(rhs, copy=True)
    else:
        gamma0 = _mask_transfer(np.asarray(initial_gamma, dtype=complex), m_ext, grid)
    h_static_ref = h0 + np.asarray(background.Sigma_static, dtype=complex)[None, None]

    def parts(field):
        field = _mask_transfer(np.asarray(field, dtype=complex), m_ext, grid)
        X = transfer_x_field(G, field, (0, 0), m_ext, grid)
        gs = _static_vertex_from_x(
            X,
            field,
            definition,
            h_static_ref,
            background.mu,
            m_ext,
            grid,
            opts,
            edge_points=int(tail_edge_points),
        )
        gm, ga1, ga2 = _dynamic_vertex_parts(
            G, W, definition, X, m_ext, grid, opts
        )
        return gs, gm, ga1, ga2

    def kernel(field):
        gs, gm, ga1, ga2 = parts(field)
        return gs + gm + ga1 + ga2

    def apply_A(field):
        field = _mask_transfer(np.asarray(field, dtype=complex), m_ext, grid)
        return field - kernel(field)

    if str(opts.solver).strip().lower() != "gmres":
        raise ValueError("finite-transfer Fierz channel vertex currently requires solver='gmres'")
    Gamma, conv, nit, err = _gmres_matrix_free(
        apply_A,
        rhs,
        gamma0,
        float(opts.tol),
        int(opts.max_iter),
        int(opts.gmres_restart),
        bool(opts.verbose),
    )
    Gamma = _mask_transfer(Gamma, m_ext, grid)
    gs, gm, ga1, ga2 = parts(Gamma)
    residual = rhs + gs + gm + ga1 + ga2 - Gamma
    err = float(np.max(np.abs(residual)))
    conv = bool(np.isfinite(err) and err < float(opts.tol))
    return ChannelTransferVertexResult(
        Gamma=np.asarray(Gamma),
        Gamma_static=np.asarray(gs),
        Gamma_MT=np.asarray(gm),
        Gamma_AL1=np.asarray(ga1),
        Gamma_AL2=np.asarray(ga2),
        m_ext=m_ext,
        converged=conv,
        iterations=int(nit),
        final_error=err,
    )


def compute_covariant_channel_susceptibility(
    background: ChannelGWResult,
    definition: ChannelDefinition,
    h0: np.ndarray,
    grid: MatsubaraGrid,
    *,
    vertex_opts: SupercellVertexOptions = SupercellVertexOptions(),
    m_max: int | None = 0,
    allow_unconverged: bool = False,
    initial_gamma_cache: dict[tuple[int, int], np.ndarray] | None = None,
    tail_edge_points: int = 2,
) -> tuple[CovariantChannelResult, dict[tuple[int, int], np.ndarray]]:
    """Return the full reducible covariant response matrix in n/B/J space."""
    _require_same_torus(grid)
    K = np.asarray(definition.vertices, dtype=complex)
    nch = len(definition.labels)
    raw = np.full((grid.nb, nch, nch), np.nan + 0j, dtype=complex)
    completed = np.full_like(raw, np.nan + 0j)
    tail = np.full_like(raw, np.nan + 0j)
    converged = np.zeros(grid.nb, dtype=bool)
    max_error = np.full(grid.nb, np.nan)
    explicit = np.zeros(grid.nb, dtype=bool)
    cache = {} if initial_gamma_cache is None else dict(initial_gamma_cache)
    adjacent: list[np.ndarray | None] = [None] * nch
    h_static_ref = np.asarray(h0, dtype=complex) + np.asarray(background.Sigma_static)[None, None]

    m_to_i = {int(m): i for i, m in enumerate(grid.m_values)}
    for im, m_raw in enumerate(grid.m_values):
        m = int(m_raw)
        if m < 0:
            continue
        if m_max is not None and abs(m) > int(m_max):
            continue
        gammas = []
        errs = []
        oks = []
        for b in range(nch):
            key = (m, b)
            seed = cache.get(key)
            if seed is None or np.asarray(seed).shape != background.G.shape:
                seed = adjacent[b]
            vr = solve_channel_vertex_transfer(
                background,
                definition,
                K[b],
                h0,
                grid,
                m,
                opts=vertex_opts,
                initial_gamma=seed,
                tail_edge_points=tail_edge_points,
            )
            cache[key] = vr.Gamma
            adjacent[b] = vr.Gamma
            gammas.append(vr.Gamma)
            errs.append(float(vr.final_error))
            oks.append(bool(vr.converged))
            if not vr.converged and not allow_unconverged:
                raise RuntimeError(
                    f"Fierz channel vertex failed at m={m:+d}, source={b} "
                    f"({definition.labels[b]}): residual={vr.final_error:.3e}"
                )

        resp = transfer_response_tail_completed(
            background.G,
            K,
            gammas,
            (0, 0),
            m,
            grid,
            h_static_ref,
            background.mu,
            edge_points=int(tail_edge_points),
        )
        raw[im] = resp["raw"]
        completed[im] = resp["completed"]
        tail[im] = resp["tail_correction"]
        converged[im] = bool(all(oks))
        max_error[im] = float(max(errs)) if errs else 0.0
        explicit[im] = True
        if -m in m_to_i and m != 0:
            jm = m_to_i[-m]
            raw[jm] = raw[im].conj().T
            completed[jm] = completed[im].conj().T
            tail[jm] = tail[im].conj().T
            converged[jm] = converged[im]
            max_error[jm] = max_error[im]

    return (
        CovariantChannelResult(
            chi_raw=raw,
            chi_completed=completed,
            tail_correction=tail,
            transfer_converged=converged,
            transfer_max_error=max_error,
            solved_explicitly=explicit,
            m_max=None if m_max is None else int(m_max),
        ),
        cache,
    )


def _right_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(a.T, b.T).T


def channel_chi_to_irreducible_p(
    definition: ChannelDefinition,
    chi_cov: np.ndarray,
) -> np.ndarray:
    """Convert reducible channel response to ``P_Gamma`` using repo signs."""
    g = np.asarray(definition.coupling, dtype=complex)
    chi = np.asarray(chi_cov, dtype=complex)
    nch = g.shape[0]
    eye = np.eye(nch, dtype=complex)
    out = np.full_like(chi, np.nan + 0j)
    for im, c in enumerate(chi):
        if not np.all(np.isfinite(c)):
            continue
        out[im] = _right_solve(eye - g @ c, -c)
    return out


def build_channel_gamma_screening(
    definition: ChannelDefinition,
    chi_cov: np.ndarray,
    *,
    background_W: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build ``W=g-g chi g`` and the equivalent Hedin ``P_Gamma`` form."""
    g = np.asarray(definition.coupling, dtype=complex)
    chi = np.asarray(chi_cov, dtype=complex)
    if chi.ndim != 3 or chi.shape[1:] != g.shape:
        raise ValueError("chi_cov/channel coupling shape mismatch")
    bg = None if background_W is None else np.asarray(background_W, dtype=complex)
    if bg is not None and bg.shape != chi.shape:
        raise ValueError("background_W shape mismatch")
    Wdirect = np.empty_like(chi)
    fallback = np.zeros(chi.shape[0], dtype=bool)
    for im, c in enumerate(chi):
        if not np.all(np.isfinite(c)):
            if bg is None:
                raise ValueError("chi_cov contains unsolved transfers without background_W")
            Wdirect[im] = bg[im]
            fallback[im] = True
        else:
            Wdirect[im] = g - g @ c @ g

    Pgamma = channel_chi_to_irreducible_p(definition, chi)
    eye = np.eye(g.shape[0], dtype=complex)
    Whedin = np.full_like(chi, np.nan + 0j)
    finite = np.all(np.isfinite(Pgamma), axis=(-2, -1))
    for im in np.flatnonzero(finite):
        Whedin[im] = np.linalg.solve(eye - g @ Pgamma[im], g)
    if np.any(finite):
        identity_error = float(np.max(np.abs(Wdirect[finite] - Whedin[finite])))
    else:
        identity_error = float("nan")
    return Wdirect, Pgamma, fallback, identity_error


def solve_channel_gamma_feedback(
    h0: np.ndarray,
    definition: ChannelDefinition,
    grid: MatsubaraGrid,
    *,
    gw_opts: GWOptions,
    vertex_opts: SupercellVertexOptions,
    feedback_opts: GammaPFeedbackOptions,
    background: ChannelGWResult,
    tail_edge_points: int = 2,
) -> ChannelGammaPResult:
    """Iterate full n/B/J covariant screening back into the channel-GW self-energy."""
    _require_same_torus(grid)
    h0 = np.asarray(h0, dtype=complex)
    if gw_opts.target_filling is None:
        raise ValueError("Fierz Gamma_P feedback currently requires fixed filling")
    if not (0.0 < float(feedback_opts.mixing) <= 1.0):
        raise ValueError("feedback mixing must lie in (0,1]")

    sigma_static = np.asarray(background.Sigma_static, dtype=complex).copy()
    sigma_c = np.asarray(background.Sigma_c, dtype=complex).copy()
    W = np.asarray(background.W, dtype=complex).copy()
    mu = float(background.mu)
    gamma_cache: dict[tuple[int, int], np.ndarray] = {}
    mu, G, _ = _dyson_fixed_filling(h0, grid, sigma_static, sigma_c, gw_opts, mu)

    history = []
    response = None
    Pgamma = None
    fallback = None
    identity_error = float("nan")
    converged = False
    err = float("inf")
    rho0 = np.asarray(background.rho)
    it = 0

    for it in range(1, int(feedback_opts.max_iter) + 1):
        rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_static)
        rho0 = np.asarray(rho[0, 0], dtype=complex)
        static_out, _, _ = channel_static_self_energy(rho0, definition)

        current = ChannelGWResult(
            G=np.asarray(G),
            W=np.asarray(W),
            P=np.asarray(background.P),
            Sigma_static=np.asarray(sigma_static),
            Sigma_tadpole=np.zeros_like(sigma_static),
            Sigma_exchange=np.zeros_like(sigma_static),
            Sigma_c=np.asarray(sigma_c),
            mu=float(mu),
            rho=np.asarray(rho0),
            density=np.real(np.diag(rho0)),
            converged=True,
            iterations=it,
            final_error=float(err),
            mixing_method="fierz-gamma-feedback",
            mode=definition.mode,
            min_screening_singular_value=np.nan,
        )
        response, gamma_cache = compute_covariant_channel_susceptibility(
            current,
            definition,
            h0,
            grid,
            vertex_opts=vertex_opts,
            m_max=feedback_opts.m_max,
            allow_unconverged=feedback_opts.allow_unconverged_vertex,
            initial_gamma_cache=gamma_cache,
            tail_edge_points=tail_edge_points,
        )
        W_out, Pgamma, fallback, identity_error = build_channel_gamma_screening(
            definition,
            response.chi_completed,
            background_W=W if feedback_opts.m_max is not None else None,
        )
        sigma_c_out = compute_channel_correlation_self_energy(
            G, W_out, definition, grid
        )
        err = _feedback_error(
            sigma_static, static_out, sigma_c, sigma_c_out, W, W_out
        )
        if feedback_opts.verbose:
            vmax = float(np.nanmax(response.transfer_max_error))
            print(
                f"Fierz-GammaP iter {it:3d}: residual={err:.3e}, mu={mu:+.10f}, "
                f"vertex={vmax:.3e}, W-id={identity_error:.3e}"
            )
        if np.isfinite(err) and err < float(feedback_opts.tol):
            converged = True
            break
        if it >= int(feedback_opts.max_iter):
            break

        sigma_static, sigma_c, W, _, _ = _mixed_feedback(
            sigma_static,
            sigma_c,
            W,
            static_out,
            sigma_c_out,
            W_out,
            feedback_opts,
            it,
            history,
        )
        mu, G, _ = _dyson_fixed_filling(
            h0, grid, sigma_static, sigma_c, gw_opts, mu
        )

    if response is None or Pgamma is None or fallback is None:
        raise RuntimeError("Fierz Gamma_P feedback performed no iterations")

    mu, G, _ = _dyson_fixed_filling(h0, grid, sigma_static, sigma_c, gw_opts, mu)
    rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_static)
    rho0 = np.asarray(rho[0, 0], dtype=complex)
    return ChannelGammaPResult(
        G=np.asarray(G),
        W=np.asarray(W),
        P_gamma=np.asarray(Pgamma),
        chi_cov=np.asarray(response.chi_completed),
        Sigma_static=np.asarray(sigma_static),
        Sigma_c=np.asarray(sigma_c),
        mu=float(mu),
        rho=np.asarray(rho0),
        converged=bool(converged),
        iterations=int(it),
        final_error=float(err),
        response=response,
        fallback_mask=np.asarray(fallback),
        screening_identity_error=float(identity_error),
        background=background,
        vertex_gamma_cache=gamma_cache,
    )


__all__ = [
    "ChannelTransferVertexResult",
    "CovariantChannelResult",
    "ChannelGammaPResult",
    "solve_channel_vertex_transfer",
    "compute_covariant_channel_susceptibility",
    "channel_chi_to_irreducible_p",
    "build_channel_gamma_screening",
    "solve_channel_gamma_feedback",
]

"""Fierz-channel GW/cGW on an exactly matched finite Ruby torus.

This module is intentionally restricted to ``nk1=nk2=1``. The full finite
torus is represented as one large orbital cell (12 orbitals for the production
2x1 benchmark), while the bosonic channel index resolves either site density
or bond bilinears.

For one interacting spinless bond (i,j), coherent-state Grassmann bilinears obey

    V n_i n_j = -(V/2) B_ij^2 = -(V/2) J_ij^2,

with B_ij = c_i^dag c_j + c_j^dag c_i and
J_ij = i(c_i^dag c_j - c_j^dag c_i).
The symmetric Fierz choice uses -(V/4)(B_ij^2+J_ij^2).

All representations reproduce the same Hartree-Fock self-energy exactly. They
differ beyond first order because GW resums different bosonic channels.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grids import MatsubaraGrid, frequency_shift_slices
from .gw import (
    GWOptions,
    _check_mixing_method,
    _mixed_self_energies,
    _residual_error,
)
from .supercell_cgw import SupercellVertexOptions, _gmres_matrix_free
from .supercell_gw import dyson_from_sigma_matrix
from .supercell_gw_fast import (
    _build_tail_cache,
    _effective_mu_tol,
    _solve_mu_matrix_fast,
    _strict_refine_fixed_filling,
)
from .supercell_gw_split import one_body_density_matrix_tail


@dataclass(frozen=True)
class ChannelDefinition:
    mode: str
    vertices: np.ndarray
    coupling: np.ndarray
    labels: tuple[str, ...]
    interaction_pairs: tuple[tuple[int, int], ...]


@dataclass
class ChannelGWResult:
    G: np.ndarray
    W: np.ndarray
    P: np.ndarray
    Sigma_static: np.ndarray
    Sigma_tadpole: np.ndarray
    Sigma_exchange: np.ndarray
    Sigma_c: np.ndarray
    mu: float
    rho: np.ndarray
    density: np.ndarray
    converged: bool
    iterations: int
    final_error: float
    mixing_method: str
    mode: str
    min_screening_singular_value: float

    @property
    def Sigma_total(self):
        return self.Sigma_static[None, None, None, :, :] + self.Sigma_c


@dataclass
class ChannelVertexResult:
    Gamma: np.ndarray
    Gamma_static: np.ndarray
    Gamma_MT: np.ndarray
    Gamma_AL: np.ndarray
    converged: bool
    iterations: int
    final_error: float
    solver: str


def _require_same_torus(grid: MatsubaraGrid):
    if int(grid.nk1) != 1 or int(grid.nk2) != 1:
        raise ValueError(
            "Fierz-channel diagnostic currently requires nk1=nk2=1; "
            "fold the finite torus into the orbital cell"
        )


def build_channel_definition(
    interaction_pairs,
    nsite: int,
    V: float,
    mode: str,
) -> ChannelDefinition:
    """Build density, B, J, or symmetric B+J Fierz channels."""
    mode_key = str(mode).strip().lower()
    aliases = {
        "density": "density",
        "n": "density",
        "b": "b",
        "bond": "b",
        "j": "j",
        "current": "j",
        "bj": "bj",
        "b+j": "bj",
        "symmetric": "bj",
    }
    if mode_key not in aliases:
        raise ValueError("mode must be one of density, B, J, BJ")
    mode_key = aliases[mode_key]
    pairs = tuple((int(i), int(j)) for i, j in interaction_pairs)
    nsite = int(nsite)
    V = float(V)

    if mode_key == "density":
        K = np.zeros((nsite, nsite, nsite), dtype=complex)
        idx = np.arange(nsite)
        K[idx, idx, idx] = 1.0
        g = np.zeros((nsite, nsite), dtype=complex)
        for i, j in pairs:
            g[i, j] += V
            g[j, i] += V
        labels = tuple(f"n{i}" for i in range(nsite))
        return ChannelDefinition("density", K, g, labels, pairs)

    vertices = []
    labels = []
    weights = []
    for i, j in pairs:
        if mode_key in {"b", "bj"}:
            Kb = np.zeros((nsite, nsite), dtype=complex)
            Kb[i, j] = 1.0
            Kb[j, i] = 1.0
            vertices.append(Kb)
            labels.append(f"B({i},{j})")
            weights.append(-V if mode_key == "b" else -0.5 * V)
        if mode_key in {"j", "bj"}:
            Kj = np.zeros((nsite, nsite), dtype=complex)
            Kj[i, j] = 1j
            Kj[j, i] = -1j
            vertices.append(Kj)
            labels.append(f"J({i},{j})")
            weights.append(-V if mode_key == "j" else -0.5 * V)

    K = np.asarray(vertices, dtype=complex)
    g = np.diag(np.asarray(weights, dtype=complex))
    return ChannelDefinition(mode_key, K, g, tuple(labels), pairs)


def channel_static_self_energy(
    rho: np.ndarray,
    definition: ChannelDefinition,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact first-order tadpole+exchange for the chosen bilinear representation."""
    rho = np.asarray(rho, dtype=complex)
    K = np.asarray(definition.vertices, dtype=complex)
    g = np.asarray(definition.coupling, dtype=complex)
    obs = np.einsum("aij,ji->a", K, rho, optimize=True)
    tad = np.einsum("aij,ab,b->ij", K, g, obs, optimize=True)
    exchange = -np.einsum(
        "ba,aik,kl,blj->ij", g, K, rho, K, optimize=True
    )
    tad = 0.5 * (tad + tad.conj().T)
    exchange = 0.5 * (exchange + exchange.conj().T)
    return tad + exchange, tad, exchange


def _pair_polarization(
    Afield: np.ndarray,
    Bfield: np.ndarray,
    vertices: np.ndarray,
    T: float,
) -> np.ndarray:
    """Return T sum_n Tr[K_a A_n K_b B_n]."""
    K = np.asarray(vertices, dtype=complex)
    A = np.einsum("aij,njk->naik", K, Afield, optimize=True)
    B = np.einsum("bij,njk->nbik", K, Bfield, optimize=True)
    return float(T) * np.einsum("naij,nbji->ab", A, B, optimize=True)


def compute_channel_polarization(
    G: np.ndarray,
    definition: ChannelDefinition,
    grid: MatsubaraGrid,
) -> np.ndarray:
    _require_same_torus(grid)
    G0 = np.asarray(G, dtype=complex)[:, 0, 0]
    out = np.zeros(
        (grid.nb, len(definition.labels), len(definition.labels)), dtype=complex
    )
    for im, m_raw in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m_raw))
        if src.stop == src.start:
            continue
        out[im] = _pair_polarization(
            G0[src], G0[dst], definition.vertices, grid.T
        )
    return out


def compute_channel_screened_interaction(
    P: np.ndarray,
    definition: ChannelDefinition,
) -> np.ndarray:
    g = np.asarray(definition.coupling, dtype=complex)
    eye = np.eye(g.shape[0], dtype=complex)
    out = np.empty_like(P, dtype=complex)
    for im in range(P.shape[0]):
        out[im] = np.linalg.solve(eye - g @ P[im], g)
    return out


def _channel_sigma_from_boson(
    Gfield: np.ndarray,
    boson: np.ndarray,
    vertices: np.ndarray,
    T: float,
) -> np.ndarray:
    """Return -T sum_ab K_a G K_b boson_ba for a batch of G."""
    K = np.asarray(vertices, dtype=complex)
    R = np.einsum("ba,bij->aij", boson, K, optimize=True)
    KG = np.einsum("aij,njk->naik", K, Gfield, optimize=True)
    return -float(T) * np.einsum("naij,ajk->nik", KG, R, optimize=True)


def compute_channel_correlation_self_energy(
    G: np.ndarray,
    W: np.ndarray,
    definition: ChannelDefinition,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Dynamic GW correlation part with only W-g in the finite bosonic box."""
    _require_same_torus(grid)
    G0 = np.asarray(G, dtype=complex)[:, 0, 0]
    g = np.asarray(definition.coupling, dtype=complex)
    sigma0 = np.zeros_like(G0)
    for im, m_raw in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m_raw))
        if src.stop == src.start:
            continue
        Wc = np.asarray(W[im], dtype=complex) - g
        sigma0[dst] += _channel_sigma_from_boson(
            G0[src], Wc, definition.vertices, grid.T
        )
    return sigma0[:, None, None]


def _screening_smin(P, definition):
    g = np.asarray(definition.coupling, dtype=complex)
    eye = np.eye(g.shape[0], dtype=complex)
    return float(
        min(
            np.linalg.svd(eye - g @ Pm, compute_uv=False)[-1]
            for Pm in np.asarray(P)
        )
    )


def solve_channel_gw_same_torus(
    h0: np.ndarray,
    definition: ChannelDefinition,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
) -> ChannelGWResult:
    """Self-consistent GW in one chosen Fierz channel on the folded finite torus."""
    _require_same_torus(grid)
    method = _check_mixing_method(opts.mixing_method)
    h0 = np.asarray(h0, dtype=complex)
    norb = int(h0.shape[-1])
    if h0.shape != (1, 1, norb, norb):
        raise ValueError("h0 must have shape (1,1,norb,norb)")
    if definition.vertices.shape[1:] != (norb, norb):
        raise ValueError("channel vertex/orbital shape mismatch")

    sigma_static = np.zeros((norb, norb), dtype=complex)
    sigma_c = np.zeros((grid.nf, 1, 1, norb, norb), dtype=complex)
    mu = float(opts.mu)
    if opts.target_filling is None:
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_static, sigma_c)
        tail_cache = _build_tail_cache(h0, sigma_static)
    else:
        mu, G, tail_cache, _ = _solve_mu_matrix_fast(
            h0, sigma_static, sigma_c, grid, float(opts.target_filling),
            mu, opts.mu_tol, opts.mu_max_iter,
        )

    history = []
    converged = False
    err = float("inf")
    P = None
    W = None

    for it in range(1, int(opts.max_iter) + 1):
        rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_static)
        rho0 = np.asarray(rho[0, 0], dtype=complex)
        static_out, tad_out, exchange_out = channel_static_self_energy(
            rho0, definition
        )
        P = compute_channel_polarization(G, definition, grid)
        W = compute_channel_screened_interaction(P, definition)
        sigma_c_out = compute_channel_correlation_self_energy(
            G, W, definition, grid
        )
        err = _residual_error(
            static_out - sigma_static, sigma_c_out - sigma_c
        )
        if opts.verbose:
            print(
                f"SC-{definition.mode}-GW iter {it:4d}: residual={err:.3e}, "
                f"mu={mu:+.10f}, smin={_screening_smin(P, definition):.3e}"
            )
        if err < float(opts.tol):
            converged = True
            break

        static_next, sigma_c_next = _mixed_self_energies(
            sigma_static, sigma_c, static_out, sigma_c_out,
            opts, it, history,
        )
        mu_tol_next = _effective_mu_tol(opts.mu_tol, err)
        if opts.target_filling is None:
            Gnext = dyson_from_sigma_matrix(
                h0, grid, mu, static_next, sigma_c_next
            )
            cache_next = _build_tail_cache(h0, static_next)
        else:
            mu, Gnext, cache_next, _ = _solve_mu_matrix_fast(
                h0, static_next, sigma_c_next, grid,
                float(opts.target_filling), mu, mu_tol_next, opts.mu_max_iter,
            )
        sigma_static = static_next
        sigma_c = sigma_c_next
        G = Gnext
        tail_cache = cache_next

    mu, G, tail_cache, _ = _strict_refine_fixed_filling(
        h0, sigma_static, sigma_c, grid, opts.target_filling, mu,
        opts.mu_tol, opts.mu_max_iter,
    )
    rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_static)
    rho0 = np.asarray(rho[0, 0], dtype=complex)
    static_out, tad_out, exchange_out = channel_static_self_energy(
        rho0, definition
    )
    P = compute_channel_polarization(G, definition, grid)
    W = compute_channel_screened_interaction(P, definition)
    sigma_c_out = compute_channel_correlation_self_energy(
        G, W, definition, grid
    )
    err = _residual_error(
        static_out - sigma_static, sigma_c_out - sigma_c
    )
    converged = bool(np.isfinite(err) and err < float(opts.tol))
    density = np.real(np.diag(rho0))

    return ChannelGWResult(
        G=np.asarray(G),
        W=np.asarray(W),
        P=np.asarray(P),
        Sigma_static=np.asarray(static_out),
        Sigma_tadpole=np.asarray(tad_out),
        Sigma_exchange=np.asarray(exchange_out),
        Sigma_c=np.asarray(sigma_c_out),
        mu=float(mu),
        rho=np.asarray(rho0),
        density=np.asarray(density),
        converged=converged,
        iterations=int(it),
        final_error=float(err),
        mixing_method=method,
        mode=definition.mode,
        min_screening_singular_value=_screening_smin(P, definition),
    )


def channel_vertex_kernel_parts(
    G: np.ndarray,
    W: np.ndarray,
    definition: ChannelDefinition,
    Gamma: np.ndarray,
    grid: MatsubaraGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return static, MT(W-g), and AL=dW pieces of the covariant q=0 kernel."""
    _require_same_torus(grid)
    G0 = np.asarray(G, dtype=complex)[:, 0, 0]
    Gamma0 = np.asarray(Gamma, dtype=complex)[:, 0, 0]
    X0 = np.einsum("nij,njk,nkl->nil", G0, Gamma0, G0, optimize=True)
    K = np.asarray(definition.vertices, dtype=complex)
    g = np.asarray(definition.coupling, dtype=complex)

    drho = float(grid.T) * np.sum(X0, axis=0)
    dobs = np.einsum("aij,ji->a", K, drho, optimize=True)
    d_tad = np.einsum("aij,ab,b->ij", K, g, dobs, optimize=True)
    d_ex = -np.einsum(
        "ba,aik,kl,blj->ij", g, K, drho, K, optimize=True
    )
    d_static = d_tad + d_ex
    # The background map explicitly Hermitian-symmetrizes its static
    # tadpole/exchange matrices. Differentiate that numerical map itself, rather
    # than its unsymmetrized algebraic precursor, so the analytic cGW kernel is
    # exactly the tangent of the fixed-point equations used above.
    d_static = 0.5 * (d_static + d_static.conj().T)
    gamma_static = np.broadcast_to(
        d_static[None, None, None], np.asarray(Gamma).shape
    ).copy()

    gamma_mt0 = np.zeros_like(G0)
    gamma_al0 = np.zeros_like(G0)
    for im, m_raw in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m_raw))
        if src.stop == src.start:
            continue
        Wm = np.asarray(W[im], dtype=complex)
        Wc = Wm - g
        gamma_mt0[dst] += _channel_sigma_from_boson(
            X0[src], Wc, K, grid.T
        )

        dP = _pair_polarization(X0[src], G0[dst], K, grid.T)
        dP += _pair_polarization(G0[src], X0[dst], K, grid.T)
        dW = Wm @ dP @ Wm
        gamma_al0[dst] += _channel_sigma_from_boson(
            G0[src], dW, K, grid.T
        )

    return (
        gamma_static,
        gamma_mt0[:, None, None],
        gamma_al0[:, None, None],
    )


def solve_channel_vertex_q0(
    background: ChannelGWResult,
    definition: ChannelDefinition,
    Kext: np.ndarray,
    grid: MatsubaraGrid,
    opts: SupercellVertexOptions = SupercellVertexOptions(),
) -> ChannelVertexResult:
    """Analytic covariant q=0 vertex of the chosen self-consistent channel-GW map."""
    _require_same_torus(grid)
    Kext = np.asarray(Kext, dtype=complex)
    norb = int(background.G.shape[-1])
    if Kext.shape != (norb, norb):
        raise ValueError("external vertex shape mismatch")
    Kfield = np.broadcast_to(
        Kext[None, None, None], background.G.shape
    ).copy()

    def kernel(gamma):
        gs, gm, ga = channel_vertex_kernel_parts(
            background.G, background.W, definition, gamma, grid
        )
        return gs + gm + ga

    def apply_A(gamma):
        return gamma - kernel(gamma)

    solver = str(opts.solver).strip().lower()
    if solver != "gmres":
        raise ValueError("channel cGW currently supports solver='gmres' only")
    Gamma, conv, nit, err = _gmres_matrix_free(
        apply_A,
        Kfield,
        Kfield,
        float(opts.tol),
        int(opts.max_iter),
        int(opts.gmres_restart),
        bool(opts.verbose),
    )
    gs, gm, ga = channel_vertex_kernel_parts(
        background.G, background.W, definition, Gamma, grid
    )
    return ChannelVertexResult(
        Gamma=np.asarray(Gamma),
        Gamma_static=np.asarray(gs),
        Gamma_MT=np.asarray(gm),
        Gamma_AL=np.asarray(ga),
        converged=bool(conv),
        iterations=int(nit),
        final_error=float(err),
        solver=solver,
    )


def susceptibility_from_vertex_q0(
    G: np.ndarray,
    Kleft: np.ndarray,
    Gamma_right: np.ndarray,
    grid: MatsubaraGrid,
) -> complex:
    """chi=-T Tr[K_left G Gamma_right G] on the folded finite torus."""
    _require_same_torus(grid)
    G0 = np.asarray(G, dtype=complex)[:, 0, 0]
    Gam0 = np.asarray(Gamma_right, dtype=complex)[:, 0, 0]
    return complex(
        -float(grid.T)
        * np.einsum(
            "ab,nbc,ncd,nda->",
            np.asarray(Kleft, dtype=complex),
            G0,
            Gam0,
            G0,
            optimize=True,
        )
    )


__all__ = [
    "ChannelDefinition",
    "ChannelGWResult",
    "ChannelVertexResult",
    "build_channel_definition",
    "channel_static_self_energy",
    "compute_channel_polarization",
    "compute_channel_screened_interaction",
    "compute_channel_correlation_self_energy",
    "solve_channel_gw_same_torus",
    "channel_vertex_kernel_parts",
    "solve_channel_vertex_q0",
    "susceptibility_from_vertex_q0",
]

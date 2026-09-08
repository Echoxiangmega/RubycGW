"""Covariant second-order exchange (SOX) self-energy and vertex.

This module promotes the strict free-background O(V^2) diagnostic in
:mod:`rubycgw.sox_diagnostic` to an interacting periodic Green-function
background.  The implementation is deliberately kept separate from the GW
solver so that the same SOX functional can be used in three ways:

1. strict weak-coupling diagnostics on the free background;
2. fixed-GW-background response diagnostics;
3. a self-consistent GW+SOX solver together with its covariant derivative.

For a spinless density-density interaction written in the full real-space site
basis, the second-order exchange skeleton is

    Sigma_SOX,ij(tau)
      = sum_kl v_il v_kj G_ik(tau) G_kl(-tau) G_lj(tau).

Its directional derivative in the positive cGW tangent convention
X = G Gamma G is the sum of the three terms obtained by replacing, in turn,
one of the three internal Green functions by X.

The periodic implementation evaluates the skeleton on the finite real-space
torus corresponding to the supplied k mesh.  G(k,tau) and V(q) are transformed
to translation-covariant real-space matrices, the real-space contraction is
performed there, and the result is transformed back to k.  This avoids a
second, separately maintained set of momentum-routing formulae.

A static reference tail is used when reconstructing G(tau).  The X(tau)
reconstruction differentiates the same reference analytically.  This is
important because a finite fermionic Matsubara box can otherwise shift the SOX
response by a few percent even though X itself is absolutely summable.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss

from .grids import MatsubaraGrid


@dataclass(frozen=True)
class SOXOptions:
    """Numerical controls for periodic SOX evaluation."""

    n_quad: int = 128
    interaction_tol: float = 1.0e-13
    tail_complete: bool = True
    tail_edge_points: int = 2

    def validate(self) -> "SOXOptions":
        if int(self.n_quad) < 16:
            raise ValueError("SOX n_quad must be at least 16")
        if float(self.interaction_tol) < 0.0:
            raise ValueError("interaction_tol must be non-negative")
        if int(self.tail_edge_points) < 1:
            raise ValueError("tail_edge_points must be positive")
        return self


def _fermi(xi: np.ndarray, T: float) -> np.ndarray:
    x = np.clip(np.asarray(xi, dtype=float) / float(T), -700.0, 700.0)
    return 1.0 / (np.exp(x) + 1.0)


def _divided_difference(values, derivatives, energies, tol=1.0e-11):
    val = np.asarray(values, dtype=complex)
    der = np.asarray(derivatives, dtype=complex)
    e = np.asarray(energies, dtype=float)
    de = e[:, None] - e[None, :]
    numer = val[:, None] - val[None, :]
    out = np.empty_like(de, dtype=complex)
    mask = np.abs(de) > float(tol)
    out[mask] = numer[mask] / de[mask]
    dlim = 0.5 * (der[:, None] + der[None, :])
    out[~mask] = dlim[~mask]
    return out


def _check_shapes(field: np.ndarray, grid: MatsubaraGrid, name: str) -> int:
    arr = np.asarray(field)
    if arr.ndim != 5:
        raise ValueError(f"{name} must have shape (nf,nk1,nk2,norb,norb)")
    norb = int(arr.shape[-1])
    expected = (grid.nf, grid.nk1, grid.nk2, norb, norb)
    if arr.shape != expected:
        raise ValueError(f"{name} shape {arr.shape} != expected {expected}")
    return norb


def _check_static_kfield(field: np.ndarray, grid: MatsubaraGrid, norb: int, name: str):
    arr = np.asarray(field, dtype=complex)
    if arr.shape == (norb, norb):
        return np.broadcast_to(arr, (grid.nk1, grid.nk2, norb, norb)).copy()
    expected = (grid.nk1, grid.nk2, norb, norb)
    if arr.shape != expected:
        raise ValueError(f"{name} shape {arr.shape} != expected {expected}")
    return np.array(arr, copy=True)


def estimate_static_vertex(Gamma: np.ndarray, edge_points: int = 2) -> np.ndarray:
    """Linear high-frequency estimate of the static part of a q=0 vertex.

    The average of the outer positive/negative fermionic frequencies is a
    linear map of ``Gamma``.  Therefore using it inside the SOX kernel preserves
    the linearity required by matrix-free GMRES.
    """
    arr = np.asarray(Gamma, dtype=complex)
    if arr.ndim != 5:
        raise ValueError("Gamma must have shape (nf,nk1,nk2,norb,norb)")
    p = min(max(int(edge_points), 1), max(arr.shape[0] // 2, 1))
    idx = list(range(p)) + list(range(arr.shape[0] - p, arr.shape[0]))
    return np.mean(arr[idx], axis=0)


def _reference_eigensystem(h_ref: np.ndarray, mu: float, T: float):
    h = np.asarray(h_ref, dtype=complex)
    h = 0.5 * (h + np.swapaxes(h.conj(), -1, -2))
    evals, evecs = np.linalg.eigh(h)
    xi = evals - float(mu)
    occ = _fermi(xi, T)
    return evals, evecs, xi, occ


def reference_green_iomega(
    h_ref: np.ndarray,
    mu: float,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Static-reference Green function on the represented Matsubara box."""
    h = np.asarray(h_ref, dtype=complex)
    norb = int(h.shape[-1])
    expected = (grid.nk1, grid.nk2, norb, norb)
    if h.shape != expected:
        raise ValueError(f"h_ref shape {h.shape} != expected {expected}")
    eye = np.eye(norb, dtype=complex)
    return np.linalg.inv(
        (1j * np.asarray(grid.omega)[:, None, None, None, None] + float(mu))
        * eye[None, None, None, :, :]
        - h[None, :, :, :, :]
    )


def reference_tau_and_direction(
    h_ref: np.ndarray,
    mu: float,
    source_static: np.ndarray,
    tau: float,
    T: float,
):
    """Analytic G_ref(+-tau) and D G_ref[source_static] for every k."""
    h = np.asarray(h_ref, dtype=complex)
    norb = int(h.shape[-1])
    S = np.asarray(source_static, dtype=complex)
    if S.shape == (norb, norb):
        S = np.broadcast_to(S, h.shape)
    if S.shape != h.shape:
        raise ValueError("source_static must be orbital or k-resolved static field")
    beta = 1.0 / float(T)
    if not 0.0 < float(tau) < beta:
        raise ValueError("tau must lie in (0,beta)")

    evals, U, xi, f = _reference_eigensystem(h, mu, T)
    t = float(tau)
    gp = -np.exp(-xi * t) * (1.0 - f)
    gm = np.exp(xi * t) * f
    dgp = np.exp(-xi * t) * (1.0 - f) * (t - beta * f)
    dgm = np.exp(xi * t) * f * (t - beta * (1.0 - f))

    nk1, nk2 = h.shape[:2]
    Gp = np.empty_like(h)
    Gm = np.empty_like(h)
    Xp = np.empty_like(h)
    Xm = np.empty_like(h)
    for i in range(nk1):
        for j in range(nk2):
            Ui = U[i, j]
            Se = Ui.conj().T @ S[i, j] @ Ui
            Xpe = _divided_difference(gp[i, j], dgp[i, j], evals[i, j]) * Se
            Xme = _divided_difference(gm[i, j], dgm[i, j], evals[i, j]) * Se
            Gp[i, j] = (Ui * gp[i, j][None, :]) @ Ui.conj().T
            Gm[i, j] = (Ui * gm[i, j][None, :]) @ Ui.conj().T
            Xp[i, j] = Ui @ Xpe @ Ui.conj().T
            Xm[i, j] = Ui @ Xme @ Ui.conj().T
    return Gp, Gm, Xp, Xm


def _reference_tau_only(h_ref: np.ndarray, mu: float, tau: float, T: float):
    h = np.asarray(h_ref, dtype=complex)
    _, U, xi, f = _reference_eigensystem(h, mu, T)
    t = float(tau)
    gp = -np.exp(-xi * t) * (1.0 - f)
    gm = np.exp(xi * t) * f
    Gp = np.einsum("xyia,xya,xyja->xyij", U, gp, U.conj(), optimize=True)
    Gm = np.einsum("xyia,xya,xyja->xyij", U, gm, U.conj(), optimize=True)
    return Gp, Gm


def reconstruct_tau_pair(
    G: np.ndarray,
    h_ref: np.ndarray,
    mu: float,
    grid: MatsubaraGrid,
    tau: float,
):
    """Tail-completed interacting G(+tau), G(-tau) on the periodic k mesh."""
    norb = _check_shapes(G, grid, "G")
    href = _check_static_kfield(h_ref, grid, norb, "h_ref")
    G = np.asarray(G, dtype=complex)
    Gref_iw = reference_green_iomega(href, mu, grid)
    Gp_ref, Gm_ref = _reference_tau_only(href, mu, tau, grid.T)
    delta = G - Gref_iw
    phase_p = np.exp(-1j * np.asarray(grid.omega) * float(tau))
    phase_m = np.exp(+1j * np.asarray(grid.omega) * float(tau))
    Gp = Gp_ref + float(grid.T) * np.einsum(
        "n,nxyab->xyab", phase_p, delta, optimize=True
    )
    Gm = Gm_ref + float(grid.T) * np.einsum(
        "n,nxyab->xyab", phase_m, delta, optimize=True
    )
    return Gp, Gm


def reconstruct_tau_direction(
    G: np.ndarray,
    Gamma: np.ndarray,
    h_ref: np.ndarray,
    mu: float,
    grid: MatsubaraGrid,
    tau: float,
    *,
    gamma_static: np.ndarray | None = None,
    edge_points: int = 2,
):
    """Tail-completed G(+-tau) and X(+-tau), X=G Gamma G, for q=0."""
    norb = _check_shapes(G, grid, "G")
    _check_shapes(Gamma, grid, "Gamma")
    href = _check_static_kfield(h_ref, grid, norb, "h_ref")
    G = np.asarray(G, dtype=complex)
    Gamma = np.asarray(Gamma, dtype=complex)
    if gamma_static is None:
        S = estimate_static_vertex(Gamma, edge_points=edge_points)
    else:
        S = _check_static_kfield(gamma_static, grid, norb, "gamma_static")

    X = np.einsum("nxyab,nxybc,nxycd->nxyad", G, Gamma, G, optimize=True)
    Gref = reference_green_iomega(href, mu, grid)
    Sfield = S[None, :, :, :, :]
    Xref = np.einsum(
        "nxyab,nxybc,nxycd->nxyad", Gref, Sfield, Gref, optimize=True
    )
    Gp_ref, Gm_ref, Xp_ref, Xm_ref = reference_tau_and_direction(
        href, mu, S, tau, grid.T
    )
    dG = G - Gref
    dX = X - Xref
    phase_p = np.exp(-1j * np.asarray(grid.omega) * float(tau))
    phase_m = np.exp(+1j * np.asarray(grid.omega) * float(tau))
    Gp = Gp_ref + float(grid.T) * np.einsum(
        "n,nxyab->xyab", phase_p, dG, optimize=True
    )
    Gm = Gm_ref + float(grid.T) * np.einsum(
        "n,nxyab->xyab", phase_m, dG, optimize=True
    )
    Xp = Xp_ref + float(grid.T) * np.einsum(
        "n,nxyab->xyab", phase_p, dX, optimize=True
    )
    Xm = Xm_ref + float(grid.T) * np.einsum(
        "n,nxyab->xyab", phase_m, dX, optimize=True
    )
    return Gp, Gm, Xp, Xm


def kfield_to_full_periodic(field_k: np.ndarray) -> np.ndarray:
    """Convert a translation-invariant k field to the full finite-torus matrix."""
    F = np.asarray(field_k, dtype=complex)
    if F.ndim != 4 or F.shape[-1] != F.shape[-2]:
        raise ValueError("field_k must have shape (nk1,nk2,norb,norb)")
    nk1, nk2, norb, _ = F.shape
    blocks = np.fft.ifftn(F, axes=(0, 1))
    ncell = nk1 * nk2
    out = np.empty((ncell * norb, ncell * norb), dtype=complex)

    def cell_index(r1, r2):
        return int(r1) * nk2 + int(r2)

    for r1 in range(nk1):
        for r2 in range(nk2):
            ir = cell_index(r1, r2)
            sli = slice(ir * norb, (ir + 1) * norb)
            for s1 in range(nk1):
                for s2 in range(nk2):
                    js = cell_index(s1, s2)
                    slj = slice(js * norb, (js + 1) * norb)
                    d1 = (r1 - s1) % nk1
                    d2 = (r2 - s2) % nk2
                    out[sli, slj] = blocks[d1, d2]
    return out


def full_periodic_to_kfield(
    matrix: np.ndarray,
    nk1: int,
    nk2: int,
    norb: int,
) -> np.ndarray:
    """Project a full finite-torus matrix onto its translation-invariant k field."""
    M = np.asarray(matrix, dtype=complex)
    ncell = int(nk1) * int(nk2)
    expected = (ncell * int(norb), ncell * int(norb))
    if M.shape != expected:
        raise ValueError(f"matrix shape {M.shape} != expected {expected}")
    blocks = np.zeros((nk1, nk2, norb, norb), dtype=complex)

    def cell_index(r1, r2):
        return int(r1) * nk2 + int(r2)

    for s1 in range(nk1):
        for s2 in range(nk2):
            js = cell_index(s1, s2)
            slj = slice(js * norb, (js + 1) * norb)
            for d1 in range(nk1):
                for d2 in range(nk2):
                    r1 = (s1 + d1) % nk1
                    r2 = (s2 + d2) % nk2
                    ir = cell_index(r1, r2)
                    sli = slice(ir * norb, (ir + 1) * norb)
                    blocks[d1, d2] += M[sli, slj]
    blocks /= float(ncell)
    return np.fft.fftn(blocks, axes=(0, 1))


def sox_self_energy_full(
    Gp: np.ndarray,
    Gm: np.ndarray,
    v: np.ndarray,
    *,
    interaction_tol: float = 1.0e-13,
) -> np.ndarray:
    """SOX self-energy on one finite real-space torus at one tau."""
    Gp = np.asarray(Gp, dtype=complex)
    Gm = np.asarray(Gm, dtype=complex)
    v = np.asarray(v, dtype=complex)
    if Gp.shape != Gm.shape or Gp.shape != v.shape:
        raise ValueError("Gp, Gm and v must have the same square shape")
    n = int(v.shape[0])
    out = np.zeros_like(v)
    rows = [np.flatnonzero(np.abs(v[i]) > interaction_tol) for i in range(n)]
    cols = [np.flatnonzero(np.abs(v[:, j]) > interaction_tol) for j in range(n)]
    for i in range(n):
        ls = rows[i]
        if ls.size == 0:
            continue
        for j in range(n):
            ks = cols[j]
            if ks.size == 0:
                continue
            left = Gp[i, ks] * v[ks, j]
            right = v[i, ls] * Gp[ls, j]
            out[i, j] = left @ Gm[np.ix_(ks, ls)] @ right
    return out


def sox_vertex_full(
    Gp: np.ndarray,
    Gm: np.ndarray,
    Xp: np.ndarray,
    Xm: np.ndarray,
    v: np.ndarray,
    *,
    interaction_tol: float = 1.0e-13,
) -> np.ndarray:
    """D Sigma_SOX[G][X] on one finite real-space torus at one tau."""
    Gp = np.asarray(Gp, dtype=complex)
    Gm = np.asarray(Gm, dtype=complex)
    Xp = np.asarray(Xp, dtype=complex)
    Xm = np.asarray(Xm, dtype=complex)
    v = np.asarray(v, dtype=complex)
    if not (Gp.shape == Gm.shape == Xp.shape == Xm.shape == v.shape):
        raise ValueError("all SOX real-space matrices must have the same shape")
    n = int(v.shape[0])
    out = np.zeros_like(v)
    rows = [np.flatnonzero(np.abs(v[i]) > interaction_tol) for i in range(n)]
    cols = [np.flatnonzero(np.abs(v[:, j]) > interaction_tol) for j in range(n)]
    for i in range(n):
        ls = rows[i]
        if ls.size == 0:
            continue
        for j in range(n):
            ks = cols[j]
            if ks.size == 0:
                continue
            gleft = Gp[i, ks] * v[ks, j]
            xleft = Xp[i, ks] * v[ks, j]
            gright = v[i, ls] * Gp[ls, j]
            xright = v[i, ls] * Xp[ls, j]
            sub_gm = Gm[np.ix_(ks, ls)]
            sub_xm = Xm[np.ix_(ks, ls)]
            out[i, j] = (
                xleft @ sub_gm @ gright
                + gleft @ sub_xm @ gright
                + gleft @ sub_gm @ xright
            )
    return out


def _quadrature(grid: MatsubaraGrid, opts: SOXOptions):
    opts.validate()
    x, w = leggauss(int(opts.n_quad))
    beta = 1.0 / float(grid.T)
    tau = 0.5 * beta * (x + 1.0)
    weight = 0.5 * beta * w
    return tau, weight


def compute_sox_self_energy_periodic(
    G: np.ndarray,
    Vq: np.ndarray,
    h_ref: np.ndarray,
    mu: float,
    grid: MatsubaraGrid,
    opts: SOXOptions = SOXOptions(),
) -> np.ndarray:
    """Evaluate the periodic bare-SOX skeleton on an interacting background."""
    norb = _check_shapes(G, grid, "G")
    Vq = _check_static_kfield(Vq, grid, norb, "Vq")
    href = _check_static_kfield(h_ref, grid, norb, "h_ref")
    vfull = kfield_to_full_periodic(Vq)
    tau, weight = _quadrature(grid, opts)
    out = np.zeros_like(np.asarray(G, dtype=complex))
    omega = np.asarray(grid.omega, dtype=float)
    for t, wt in zip(tau, weight):
        if opts.tail_complete:
            Gp, Gm = reconstruct_tau_pair(G, href, mu, grid, float(t))
        else:
            phase_p = np.exp(-1j * omega * float(t))
            phase_m = np.exp(+1j * omega * float(t))
            Gp = float(grid.T) * np.einsum(
                "n,nxyab->xyab", phase_p, G, optimize=True
            )
            Gm = float(grid.T) * np.einsum(
                "n,nxyab->xyab", phase_m, G, optimize=True
            )
        sig_full = sox_self_energy_full(
            kfield_to_full_periodic(Gp),
            kfield_to_full_periodic(Gm),
            vfull,
            interaction_tol=float(opts.interaction_tol),
        )
        sig_k = full_periodic_to_kfield(sig_full, grid.nk1, grid.nk2, norb)
        phase = np.exp(1j * omega * float(t))
        out += float(wt) * phase[:, None, None, None, None] * sig_k[None, ...]
    return out


def compute_sox_vertex_periodic(
    G: np.ndarray,
    Gamma: np.ndarray,
    Vq: np.ndarray,
    h_ref: np.ndarray,
    mu: float,
    grid: MatsubaraGrid,
    opts: SOXOptions = SOXOptions(),
    *,
    gamma_static: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate D Sigma_SOX[G][G Gamma G] for a static q=0 vertex."""
    norb = _check_shapes(G, grid, "G")
    _check_shapes(Gamma, grid, "Gamma")
    Vq = _check_static_kfield(Vq, grid, norb, "Vq")
    href = _check_static_kfield(h_ref, grid, norb, "h_ref")
    vfull = kfield_to_full_periodic(Vq)
    tau, weight = _quadrature(grid, opts)
    out = np.zeros_like(np.asarray(Gamma, dtype=complex))
    omega = np.asarray(grid.omega, dtype=float)
    Xiw = np.einsum(
        "nxyab,nxybc,nxycd->nxyad", G, Gamma, G, optimize=True
    )
    for t, wt in zip(tau, weight):
        if opts.tail_complete:
            Gp, Gm, Xp, Xm = reconstruct_tau_direction(
                G,
                Gamma,
                href,
                mu,
                grid,
                float(t),
                gamma_static=gamma_static,
                edge_points=int(opts.tail_edge_points),
            )
        else:
            phase_p = np.exp(-1j * omega * float(t))
            phase_m = np.exp(+1j * omega * float(t))
            Gp = float(grid.T) * np.einsum(
                "n,nxyab->xyab", phase_p, G, optimize=True
            )
            Gm = float(grid.T) * np.einsum(
                "n,nxyab->xyab", phase_m, G, optimize=True
            )
            Xp = float(grid.T) * np.einsum(
                "n,nxyab->xyab", phase_p, Xiw, optimize=True
            )
            Xm = float(grid.T) * np.einsum(
                "n,nxyab->xyab", phase_m, Xiw, optimize=True
            )
        vert_full = sox_vertex_full(
            kfield_to_full_periodic(Gp),
            kfield_to_full_periodic(Gm),
            kfield_to_full_periodic(Xp),
            kfield_to_full_periodic(Xm),
            vfull,
            interaction_tol=float(opts.interaction_tol),
        )
        vert_k = full_periodic_to_kfield(vert_full, grid.nk1, grid.nk2, norb)
        phase = np.exp(1j * omega * float(t))
        out += float(wt) * phase[:, None, None, None, None] * vert_k[None, ...]
    return out


__all__ = [
    "SOXOptions",
    "estimate_static_vertex",
    "reference_green_iomega",
    "reference_tau_and_direction",
    "reconstruct_tau_pair",
    "reconstruct_tau_direction",
    "kfield_to_full_periodic",
    "full_periodic_to_kfield",
    "sox_self_energy_full",
    "sox_vertex_full",
    "compute_sox_self_energy_periodic",
    "compute_sox_vertex_periodic",
]

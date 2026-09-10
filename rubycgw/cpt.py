"""Cluster perturbation theory (CPT) for the spinless Ruby lattice.

The reference system is one *isolated* six-site primitive cell.  All density
interactions of the production Ruby model lie inside this cluster, so the
cluster problem treats V nonperturbatively by full finite-temperature ED.  The
remaining inter-cluster hopping is restored at the one-particle level through

    G_CPT(k,iw) = [G_cl(iw)^(-1) - T_inter(k)]^(-1),

with T_inter(k)=h0(k)-h_cl.  Standard CPT has no lattice many-body
self-consistency.  For fixed filling we only solve the scalar physical chemical
potential; at each trial mu the isolated-cluster Green function is evaluated
with the same mu.

The module deliberately keeps the construction transparent because its primary
use is as a strong-coupling benchmark against the exact 2x1 torus.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grids import MatsubaraGrid
from .model import NSUB, RubyParameters, build_h0, ruby_hoppings
from .small_cluster_exact import ExactSmallRubyThermal
from .supercell_gw import density_from_G_matrix


@dataclass(frozen=True)
class CPTResult:
    mu: float
    G: np.ndarray
    G_cluster: np.ndarray
    Sigma_cluster: np.ndarray
    Sigma_static_tail: np.ndarray
    density: np.ndarray
    filling: float
    cluster_filling: float
    h_cluster: np.ndarray
    h0_lattice: np.ndarray
    intercluster_hopping: np.ndarray
    source_per_cell: float
    mu_iterations: int
    mu_residual: float


def isolated_primitive_h0(params: RubyParameters) -> np.ndarray:
    """Return the 6x6 one-body Hamiltonian of an isolated primitive cell.

    Only hopping terms with primitive-cell displacement R=(0,0) are retained.
    This is *not* the 1x1 periodic torus, where nonzero-R hoppings wrap back into
    the same cell.  The distinction is essential for CPT.
    """
    p0 = RubyParameters(ti=params.ti, t1=params.t1, t2=params.t2, V=0.0)
    h = np.zeros((NSUB, NSUB), dtype=complex)
    for i, j, R, amp in ruby_hoppings(p0):
        if np.all(np.asarray(R, dtype=int) == 0):
            h[int(i), int(j)] += complex(amp)
    return 0.5 * (h + h.conj().T)


def intercluster_hopping(
    kpts: np.ndarray,
    params: RubyParameters,
    h_cluster: np.ndarray | None = None,
) -> np.ndarray:
    """Return T_inter(k)=h0(k)-h_cluster on the supplied primitive k points."""
    hcl = isolated_primitive_h0(params) if h_cluster is None else np.asarray(h_cluster, dtype=complex)
    hk = np.asarray(build_h0(kpts, params), dtype=complex)
    return hk - hcl


def _prepare_cluster(
    params: RubyParameters,
    V: float,
    h_cluster: np.ndarray,
) -> ExactSmallRubyThermal:
    """Build a 6-site grand-canonical exact solver with a custom open cluster h0."""
    p0 = RubyParameters(ti=params.ti, t1=params.t1, t2=params.t2, V=0.0)
    exact = ExactSmallRubyThermal(1, 1, p0)
    exact.h0 = np.asarray(h_cluster, dtype=complex).copy()
    exact.h0 = 0.5 * (exact.h0 + exact.h0.conj().T)
    exact.diagonalize(float(V))
    return exact


def _cluster_green_sigma(
    exact: ExactSmallRubyThermal,
    h_cluster: np.ndarray,
    mu: float,
    T: float,
    omega: np.ndarray,
    *,
    discard_weight_tol: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return exact isolated-cluster G, self-energy, and cluster filling."""
    Gcl, _ = exact.green_iomega(
        1j * np.asarray(omega, dtype=float),
        float(mu),
        float(T),
        discard_weight_tol=float(discard_weight_tol),
    )
    Gcl = np.asarray(Gcl, dtype=complex)
    eye = np.eye(NSUB, dtype=complex)
    G0inv = (
        (1j * np.asarray(omega, dtype=float)[:, None, None] + float(mu))
        * eye[None, :, :]
        - np.asarray(h_cluster, dtype=complex)[None, :, :]
    )
    Sigma = G0inv - np.linalg.inv(Gcl)
    _, ncluster, _ = exact._normalized_probabilities(float(mu), float(T))
    return Gcl, Sigma, float(ncluster)


def _static_tail_from_sigma(Sigma: np.ndarray, omega: np.ndarray, edge_points: int = 4) -> np.ndarray:
    """Estimate the Hermitian Sigma(iw->infinity) used only for tail completion."""
    arr = np.asarray(Sigma, dtype=complex)
    om = np.asarray(omega, dtype=float)
    nkeep = min(max(2 * int(edge_points), 2), len(om))
    idx = np.argsort(np.abs(om))[-nkeep:]
    s = np.mean(arr[idx], axis=0)
    return 0.5 * (s + s.conj().T)


def cpt_green_from_cluster(
    G_cluster: np.ndarray,
    inter_hopping: np.ndarray,
) -> np.ndarray:
    """Embed an exact cluster Green function into the primitive lattice."""
    Gcl = np.asarray(G_cluster, dtype=complex)
    tinter = np.asarray(inter_hopping, dtype=complex)
    if Gcl.ndim != 3 or Gcl.shape[-2:] != (NSUB, NSUB):
        raise ValueError("G_cluster must have shape (nf,6,6)")
    if tinter.ndim != 4 or tinter.shape[-2:] != (NSUB, NSUB):
        raise ValueError("inter_hopping must have shape (nk1,nk2,6,6)")
    inv_cl = np.linalg.inv(Gcl)
    inv_lattice = inv_cl[:, None, None, :, :] - tinter[None, :, :, :, :]
    return np.linalg.inv(inv_lattice)


def solve_primitive_cpt_fixed_filling(
    params: RubyParameters,
    grid: MatsubaraGrid,
    *,
    V: float,
    filling: float,
    source_vertex: np.ndarray | None = None,
    source_per_cell: float = 0.0,
    mu0: float = 0.0,
    mu_tol: float = 1e-10,
    mu_max_iter: int = 100,
    discard_weight_tol: float = 0.0,
    tail_edge_points: int = 4,
) -> CPTResult:
    """Solve standard primitive-cell CPT at a fixed average filling per cell.

    ``source_per_cell`` multiplies the primitive 6x6 one-body source vertex in
    both the reference cluster and the reconstructed lattice Hamiltonian.  For
    comparison with an Ncell exact torus whose normalized source is
    K_N=(1/sqrt(Ncell))*sum_R K_R, use source_per_cell=h_ref/sqrt(Ncell).
    """
    target = float(filling)
    if not 0.0 <= target <= NSUB:
        raise ValueError("filling must lie between 0 and 6 particles per primitive cell")
    if grid.T <= 0.0:
        raise ValueError("CPT finite-temperature benchmark requires T>0")

    K = np.zeros((NSUB, NSUB), dtype=complex)
    if source_vertex is not None:
        K = np.asarray(source_vertex, dtype=complex)
        if K.shape != (NSUB, NSUB):
            raise ValueError("source_vertex must have shape (6,6)")
    K = 0.5 * (K + K.conj().T)

    p0 = RubyParameters(ti=params.ti, t1=params.t1, t2=params.t2, V=0.0)
    hcl0 = isolated_primitive_h0(p0)
    hcl = hcl0 - float(source_per_cell) * K
    hcl = 0.5 * (hcl + hcl.conj().T)

    h0_lattice = np.asarray(build_h0(grid.kmesh(), p0), dtype=complex)
    h0_lattice = h0_lattice - float(source_per_cell) * K[None, None, :, :]
    h0_lattice = 0.5 * (h0_lattice + np.swapaxes(h0_lattice.conj(), -1, -2))

    # The source is cell-local and therefore cancels in h0(k)-h_cluster.
    tinter = h0_lattice - hcl[None, None, :, :]
    exact = _prepare_cluster(p0, float(V), hcl)

    def evaluate(mu: float):
        Gcl, Sigma, ncl = _cluster_green_sigma(
            exact,
            hcl,
            float(mu),
            float(grid.T),
            grid.omega,
            discard_weight_tol=float(discard_weight_tol),
        )
        G = cpt_green_from_cluster(Gcl, tinter)
        sigma_static = _static_tail_from_sigma(
            Sigma, grid.omega, edge_points=int(tail_edge_points)
        )
        density = density_from_G_matrix(
            G,
            grid,
            h0=h0_lattice,
            mu=float(mu),
            sigma_h=sigma_static,
        )
        return float(np.sum(density) - target), Gcl, Sigma, sigma_static, G, density, ncl

    width = 2.0
    lo, hi = float(mu0) - width, float(mu0) + width
    flo, *_ = evaluate(lo)
    fhi, *_ = evaluate(hi)
    for _ in range(30):
        if flo <= 0.0 <= fhi:
            break
        width *= 2.0
        lo, hi = float(mu0) - width, float(mu0) + width
        flo, *_ = evaluate(lo)
        fhi, *_ = evaluate(hi)
    else:
        raise RuntimeError(
            "Could not bracket CPT chemical potential; "
            f"target={target:g}, f(lo)={flo:.6e}, f(hi)={fhi:.6e}"
        )

    mid = 0.5 * (lo + hi)
    payload = None
    fmid = np.inf
    nit = 0
    for nit in range(1, int(mu_max_iter) + 1):
        mid = 0.5 * (lo + hi)
        payload = evaluate(mid)
        fmid = float(payload[0])
        if abs(fmid) < float(mu_tol):
            break
        if fmid > 0.0:
            hi = mid
        else:
            lo = mid
    if payload is None:
        raise RuntimeError("CPT chemical-potential solve produced no iterate")

    _, Gcl, Sigma, sigma_static, G, density, ncl = payload
    return CPTResult(
        mu=float(mid),
        G=np.asarray(G, dtype=complex),
        G_cluster=np.asarray(Gcl, dtype=complex),
        Sigma_cluster=np.asarray(Sigma, dtype=complex),
        Sigma_static_tail=np.asarray(sigma_static, dtype=complex),
        density=np.asarray(density, dtype=float),
        filling=float(np.sum(density)),
        cluster_filling=float(ncl),
        h_cluster=np.asarray(hcl, dtype=complex),
        h0_lattice=np.asarray(h0_lattice, dtype=complex),
        intercluster_hopping=np.asarray(tinter, dtype=complex),
        source_per_cell=float(source_per_cell),
        mu_iterations=int(nit),
        mu_residual=float(abs(fmid)),
    )


__all__ = [
    "CPTResult",
    "isolated_primitive_h0",
    "intercluster_hopping",
    "cpt_green_from_cluster",
    "solve_primitive_cpt_fixed_filling",
]

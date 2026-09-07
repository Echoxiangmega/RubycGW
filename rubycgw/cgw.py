"""Modern q=0 covariant-GW response for the six-site primitive Ruby cell.

The primitive solver now shares the mature arbitrary-matrix response kernel
with the supercell implementation.  Consequently it differentiates the same
SC-GW map used by :mod:`rubycgw.primitive_gw`:

    Sigma_GW = Sigma_F + Sigma_c[W-V],

and solves

    Gamma = K + Gamma_H + Gamma_F + Gamma_MT,c + Gamma_AL1 + Gamma_AL2

as ``(I-L) Gamma = K`` with restarted matrix-free GMRES by default.  A legacy
linear fixed-point solver remains available through ``VertexOptions.solver``.

The public ``solve_vertex_q0`` accepts either the full ``Vq`` array or the old
6x6 ``Vq0`` argument.  The latter is broadcast over q for backward
compatibility; this is exact for the present Ruby model because V contains only
intra-triangle, intra-cell density bonds and is q independent.
"""

from __future__ import annotations

import numpy as np

from .grids import MatsubaraGrid
from .supercell_cgw import (
    SupercellVertexOptions,
    SupercellVertexResult,
    physical_symmetric_susceptibility,
    solve_vertex_q0 as _solve_vertex_q0_matrix,
    susceptibility_matrix_q0,
    vertex_corrections_q0 as _vertex_corrections_q0_matrix,
)


VertexOptions = SupercellVertexOptions
VertexResult = SupercellVertexResult


def _full_vq(Vq_or_vq0: np.ndarray, G: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    """Normalize old ``Vq0`` or modern full ``Vq`` input to full q shape."""
    arr = np.asarray(Vq_or_vq0, dtype=complex)
    norb = int(G.shape[-1])
    if arr.shape == (grid.nk1, grid.nk2, norb, norb):
        return arr
    if arr.shape == (norb, norb):
        return np.broadcast_to(
            arr[None, None, :, :],
            (grid.nk1, grid.nk2, norb, norb),
        ).copy()
    raise ValueError(
        f"Vq shape {arr.shape} is neither full "
        f"{(grid.nk1, grid.nk2, norb, norb)} nor Vq0 {(norb, norb)}"
    )


def solve_vertex_q0(
    G: np.ndarray,
    W: np.ndarray,
    Vq_or_vq0: np.ndarray,
    K: np.ndarray,
    grid: MatsubaraGrid,
    opts: VertexOptions = VertexOptions(),
    initial_gamma: np.ndarray | None = None,
) -> VertexResult:
    """Solve one primitive-cell q=0 covariant vertex.

    Use a full ``Vq`` whenever possible so the static-Fock derivative and
    ``W-V`` MT term exactly match the background SC-GW map.
    """
    Vq = _full_vq(Vq_or_vq0, G, grid)
    return _solve_vertex_q0_matrix(
        G, W, Vq, K, grid, opts=opts, initial_gamma=initial_gamma
    )


def vertex_corrections_q0(
    G: np.ndarray,
    W: np.ndarray,
    Vq_or_vq0: np.ndarray,
    Gamma: np.ndarray,
    grid: MatsubaraGrid,
    opts: VertexOptions = VertexOptions(),
):
    """Return ``(H,F,MTc,AL1,AL2)`` for one trial primitive vertex."""
    Vq = _full_vq(Vq_or_vq0, G, grid)
    return _vertex_corrections_q0_matrix(G, W, Vq, Gamma, grid, opts)


def gamma_h_q0(
    G: np.ndarray,
    Gamma: np.ndarray,
    Vq0: np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Backward-compatible Hartree-only derivative helper."""
    Wzero = np.zeros(
        (grid.nb, grid.nk1, grid.nk2, G.shape[-1], G.shape[-1]), dtype=complex
    )
    opts = VertexOptions(
        include_hartree=True,
        include_fock=False,
        include_mt=False,
        include_al=False,
        verbose=False,
    )
    return vertex_corrections_q0(G, Wzero, Vq0, Gamma, grid, opts)[0]


def gamma_f_q0(
    G: np.ndarray,
    Gamma: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    backend: str = "fft",
) -> np.ndarray:
    """Static bare-V Fock derivative of the primitive response."""
    Wzero = np.zeros(
        (grid.nb, grid.nk1, grid.nk2, G.shape[-1], G.shape[-1]), dtype=complex
    )
    opts = VertexOptions(
        include_hartree=False,
        include_fock=True,
        include_mt=False,
        include_al=False,
        verbose=False,
        momentum_backend=backend,
    )
    return vertex_corrections_q0(G, Wzero, Vq, Gamma, grid, opts)[1]


def gamma_mt_q0(
    G: np.ndarray,
    W: np.ndarray,
    Gamma: np.ndarray,
    grid: MatsubaraGrid,
    backend: str = "fft",
) -> np.ndarray:
    """Legacy MT helper using full W, retained for FFT/direct regression tests.

    Production cGW should use :func:`solve_vertex_q0`, where the MT correction
    uses the physically correct decaying ``W-V`` part.
    """
    Vzero = np.zeros((G.shape[-1], G.shape[-1]), dtype=complex)
    opts = VertexOptions(
        include_hartree=False,
        include_fock=False,
        include_mt=True,
        include_al=False,
        verbose=False,
        momentum_backend=backend,
    )
    return vertex_corrections_q0(G, W, Vzero, Gamma, grid, opts)[2]


def gamma_al_q0(
    G: np.ndarray,
    W: np.ndarray,
    Gamma: np.ndarray,
    grid: MatsubaraGrid,
    backend: str = "fft",
):
    """Return AL1 and AL2; AL uses full W and is independent of bare-V split."""
    Vzero = np.zeros((G.shape[-1], G.shape[-1]), dtype=complex)
    opts = VertexOptions(
        include_hartree=False,
        include_fock=False,
        include_mt=False,
        include_al=True,
        verbose=False,
        momentum_backend=backend,
    )
    parts = vertex_corrections_q0(G, W, Vzero, Gamma, grid, opts)
    return parts[3], parts[4]


__all__ = [
    "VertexOptions",
    "VertexResult",
    "solve_vertex_q0",
    "vertex_corrections_q0",
    "susceptibility_matrix_q0",
    "physical_symmetric_susceptibility",
    "gamma_h_q0",
    "gamma_f_q0",
    "gamma_mt_q0",
    "gamma_al_q0",
]

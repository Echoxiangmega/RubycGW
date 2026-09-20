"""Fold a discrete primitive-cell momentum torus into one real-space supercell.

For an Lx x Ly primitive momentum mesh, the exact discrete Fourier transform
produces a single 6*Lx*Ly orbital torus Hamiltonian.  Its spectrum is the union
of the primitive Bloch matrices on the original mesh.  This representation is
used to realize static finite-q ordered states by allowing inequivalent local
self-energies on each primitive cell of the torus.
"""
from __future__ import annotations

import numpy as np


def torus_cells(Lx: int, Ly: int) -> list[tuple[int, int]]:
    return [(r1, r2) for r1 in range(int(Lx)) for r2 in range(int(Ly))]


def _validate_k_field(field: np.ndarray, Lx: int, Ly: int):
    a = np.asarray(field, dtype=complex)
    if a.shape[-4:-2] != (int(Lx), int(Ly)):
        raise ValueError(
            f"momentum field has mesh {a.shape[-4:-2]}, expected {(Lx, Ly)}"
        )
    if a.shape[-1] != a.shape[-2]:
        raise ValueError("last two field axes must be square orbital matrices")
    return a


def fold_static_k_field(field: np.ndarray) -> np.ndarray:
    """Fold field[k1,k2,a,b] to one block-circulant real-space supercell."""
    a = np.asarray(field, dtype=complex)
    if a.ndim != 4 or a.shape[-1] != a.shape[-2]:
        raise ValueError("static field must have shape (Lx,Ly,norb,norb)")
    Lx, Ly, norb = a.shape[0], a.shape[1], a.shape[-1]
    cells = torus_cells(Lx, Ly)
    ncell = len(cells)
    out = np.zeros((ncell * norb, ncell * norb), dtype=complex)
    for ic, R in enumerate(cells):
        for jc, Rp in enumerate(cells):
            block = np.zeros((norb, norb), dtype=complex)
            d1 = R[0] - Rp[0]
            d2 = R[1] - Rp[1]
            for k1 in range(Lx):
                for k2 in range(Ly):
                    phase = np.exp(
                        2j * np.pi * (
                            (k1 / float(Lx)) * d1
                            + (k2 / float(Ly)) * d2
                        )
                    )
                    block += phase * a[k1, k2]
            block /= float(ncell)
            i0 = ic * norb
            j0 = jc * norb
            out[i0:i0+norb, j0:j0+norb] = block
    return 0.5 * (out + out.conj().T)


def fold_dynamic_k_field(field: np.ndarray) -> np.ndarray:
    """Fold field[n,k1,k2,a,b] to field[n,1,1,I,J]."""
    a = np.asarray(field, dtype=complex)
    if a.ndim != 5 or a.shape[-1] != a.shape[-2]:
        raise ValueError("dynamic field must have shape (nf,Lx,Ly,norb,norb)")
    nf, Lx, Ly, norb = a.shape[0], a.shape[1], a.shape[2], a.shape[-1]
    cells = torus_cells(Lx, Ly)
    ncell = len(cells)
    out = np.zeros((nf, 1, 1, ncell*norb, ncell*norb), dtype=complex)
    for ic, R in enumerate(cells):
        for jc, Rp in enumerate(cells):
            block = np.zeros((nf, norb, norb), dtype=complex)
            d1 = R[0] - Rp[0]
            d2 = R[1] - Rp[1]
            for k1 in range(Lx):
                for k2 in range(Ly):
                    phase = np.exp(
                        2j * np.pi * (
                            (k1 / float(Lx)) * d1
                            + (k2 / float(Ly)) * d2
                        )
                    )
                    block += phase * a[:, k1, k2]
            block /= float(ncell)
            i0 = ic * norb
            j0 = jc * norb
            out[:, 0, 0, i0:i0+norb, j0:j0+norb] = block
    return out


def fold_static_field_as_grid(field: np.ndarray) -> np.ndarray:
    """Fold and add the one-point reduced-BZ axes expected by matrix GW."""
    H = fold_static_k_field(field)
    return H[None, None, :, :]


def commensurate_source_matrix(
    form_factor: np.ndarray,
    q_index: tuple[int, int],
    Lx: int,
    Ly: int,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Return the Hermitian real-space source for one q/-q pair.

    The local block on cell R is

        1/2 [ M_q exp(i q.R) + M_q^dagger exp(-i q.R) ].

    At q=0 this reduces to the Hermitian part of M_q.  A global identity
    component is removed because fixed-filling calculations absorb it into mu.
    """
    M = np.asarray(form_factor, dtype=complex)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("form_factor must be square")
    norb = M.shape[0]
    iq1 = int(q_index[0]) % int(Lx)
    iq2 = int(q_index[1]) % int(Ly)
    cells = torus_cells(Lx, Ly)
    out = np.zeros((len(cells)*norb, len(cells)*norb), dtype=complex)
    for ic, (r1, r2) in enumerate(cells):
        phase = np.exp(
            2j * np.pi * (
                iq1 * r1 / float(Lx)
                + iq2 * r2 / float(Ly)
            )
        )
        block = 0.5 * (phase * M + np.conj(phase) * M.conj().T)
        i0 = ic * norb
        out[i0:i0+norb, i0:i0+norb] = block
    out = 0.5 * (out + out.conj().T)
    out -= np.trace(out) * np.eye(out.shape[0], dtype=complex) / float(out.shape[0])
    if normalize:
        scale = float(np.max(np.abs(out), initial=0.0))
        if not np.isfinite(scale) or scale <= 1e-14:
            raise ValueError("commensurate source has zero norm")
        out /= scale
    return out


def cell_block(matrix: np.ndarray, cell: int, norb: int = 6) -> np.ndarray:
    a = np.asarray(matrix)
    i0 = int(cell) * int(norb)
    return a[..., i0:i0+norb, i0:i0+norb]


def set_cell_block(
    matrix: np.ndarray,
    cell: int,
    block: np.ndarray,
    norb: int = 6,
) -> None:
    i0 = int(cell) * int(norb)
    matrix[..., i0:i0+norb, i0:i0+norb] = block


__all__ = [
    "torus_cells",
    "fold_static_k_field",
    "fold_dynamic_k_field",
    "fold_static_field_as_grid",
    "commensurate_source_matrix",
    "cell_block",
    "set_cell_block",
]

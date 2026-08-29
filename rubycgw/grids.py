"""Momentum and Matsubara grids for the reference cGW implementation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MatsubaraGrid:
    nk1: int = 4
    nk2: int = 4
    nw: int = 16          # fermion n = -nw,...,nw-1
    nOmega: int = 8       # boson m = -nOmega,...,+nOmega
    T: float = 0.05

    @property
    def n_values(self):
        return np.arange(-self.nw, self.nw, dtype=int)

    @property
    def m_values(self):
        return np.arange(-self.nOmega, self.nOmega + 1, dtype=int)

    @property
    def omega(self):
        n = self.n_values
        return (2 * n + 1) * np.pi * self.T

    @property
    def Omega(self):
        return 2 * self.m_values * np.pi * self.T

    @property
    def nk(self):
        return self.nk1 * self.nk2

    @property
    def nf(self):
        return 2 * self.nw

    @property
    def nb(self):
        return 2 * self.nOmega + 1

    def kmesh(self):
        k1 = np.arange(self.nk1, dtype=float) / self.nk1
        k2 = np.arange(self.nk2, dtype=float) / self.nk2
        a, b = np.meshgrid(k1, k2, indexing="ij")
        return np.stack([a, b], axis=-1)

    def qmesh(self):
        return self.kmesh()


def frequency_shift_slices(nf: int, m: int) -> tuple[slice, slice]:
    """Return ``(src, dst)`` slices for a fermionic shift by bosonic index m.

    For a field ``F[n]`` the shifted field satisfies

        F_shifted[dst] = F[src]

    and values outside ``dst`` are zero.  The returned slices allow the hot
    convolution loops to work only on the valid Matsubara window instead of
    allocating a full zero-padded array for every internal Q.
    """
    m = int(m)
    if abs(m) >= nf:
        return slice(0, 0), slice(0, 0)
    if m >= 0:
        return slice(m, nf), slice(0, nf - m)
    s = -m
    return slice(0, nf - s), slice(s, nf)


def roll_spatial(field: np.ndarray, dq1: int, dq2: int) -> np.ndarray:
    """Return the periodic spatial momentum shift ``F(k+q)``.

    ``field`` may contain only the valid Matsubara slice.  Only the two spatial
    momentum axes are rolled.
    """
    return np.roll(field, shift=(-int(dq1), -int(dq2)), axis=(1, 2))


def shift_fermion_field(field: np.ndarray, dq1: int, dq2: int, m: int) -> np.ndarray:
    """Return F(k+Q) on the full base fermion grid.

    This compatibility helper keeps the original public behavior.  Performance
    critical GW/cGW loops use :func:`frequency_shift_slices` and
    :func:`roll_spatial` directly so they do not repeatedly allocate the
    zero-padded Matsubara box.
    """
    out = np.zeros_like(field)
    src, dst = frequency_shift_slices(field.shape[0], int(m))
    if src.stop == src.start:
        return out
    out[dst] = roll_spatial(field[src], dq1, dq2)
    return out

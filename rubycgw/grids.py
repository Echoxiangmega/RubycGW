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


def shift_fermion_field(field: np.ndarray, dq1: int, dq2: int, m: int) -> np.ndarray:
    """Return F(k+Q) on the base fermion grid.

    ``field`` has shape (Nf, Nk1, Nk2, ..., ...). Spatial momentum is periodic;
    the finite Matsubara box is not. Values shifted outside the stored fermion
    frequency box are set to zero. This is a controlled prototype truncation;
    production calculations must verify convergence with ``nw``.
    """
    out = np.zeros_like(field)
    nf = field.shape[0]

    if abs(m) >= nf:
        return out
    if m >= 0:
        src = slice(m, nf)
        dst = slice(0, nf - m)
    else:
        s = -m
        src = slice(0, nf - s)
        dst = slice(s, nf)
    out[dst] = field[src]
    out = np.roll(out, shift=(-dq1, -dq2), axis=(1, 2))
    return out

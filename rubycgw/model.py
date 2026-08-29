"""Ruby-lattice single-particle model, density interaction, and eta vertices.

The conventions intentionally match the earlier
``ruby_selection_rule_check_physical_labels.py`` implementation:

* six sites per unit cell, indexed 0,...,5;
* reduced reciprocal coordinates k=(k1,k2);
* a cell offset R contributes exp(2*pi*i*k.R);
* eta_plus = (eta_A + eta_B)/sqrt(2) is PHYSICAL OPPOSITE circulation;
* eta_minus = (eta_A - eta_B)/sqrt(2) is PHYSICAL SAME circulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

NSUB = 6


@dataclass(frozen=True)
class RubyParameters:
    ti: float = 0.4
    t1: float = 0.2
    t2: float = 0.2
    V: float = 0.2


def _base_bonds(params: RubyParameters):
    """Undirected NN bonds in exactly the previous Ruby convention."""
    return [
        (0, 1, (0, 0), params.ti),
        (0, 2, (0, 0), params.ti),
        (2, 1, (0, 0), params.ti),
        (3, 4, (0, 0), params.ti),
        (3, 5, (0, 0), params.ti),
        (4, 5, (0, 0), params.ti),
        (1, 4, (0, 0), params.t1),
        (5, 0, (0, -1), params.t1),
        (2, 3, (-1, 0), params.t1),
        (3, 1, (0, -1), params.t2),
        (2, 5, (0, 0), params.t2),
        (0, 4, (-1, 0), params.t2),
    ]


def ruby_hoppings(params: RubyParameters):
    out = []
    for i, j, R, amp in _base_bonds(params):
        Rv = np.asarray(R, dtype=int)
        out.append((i, j, Rv, complex(amp)))
        out.append((j, i, -Rv, complex(np.conj(amp))))
    return out


def build_h0(kpts: np.ndarray, params: RubyParameters) -> np.ndarray:
    """Return h0(k), shape (..., 6, 6), for reduced k coordinates."""
    kpts = np.asarray(kpts, dtype=float)
    flat = kpts.reshape(-1, 2)
    h0 = np.zeros((flat.shape[0], NSUB, NSUB), dtype=complex)
    for ik, k in enumerate(flat):
        for i, j, R, amp in ruby_hoppings(params):
            h0[ik, i, j] += amp * np.exp(2j * np.pi * np.dot(k, R))
    h0 = 0.5 * (h0 + np.swapaxes(h0.conj(), -1, -2))
    return h0.reshape(kpts.shape[:-1] + (NSUB, NSUB))


def build_interaction(qpts: np.ndarray, params: RubyParameters) -> np.ndarray:
    """Build the 6x6 Fourier-space NN density interaction V_ab(q).

    The same 12 undirected nearest-neighbour bonds as the hopping model are
    assigned the same density-density coupling ``params.V``. Both directed
    orientations are included; the Hamiltonian convention is

        H_V = (1/2N) sum_q n_a(q) V_ab(q) n_b(-q).
    """
    qpts = np.asarray(qpts, dtype=float)
    flat = qpts.reshape(-1, 2)
    vq = np.zeros((flat.shape[0], NSUB, NSUB), dtype=complex)
    for iq, q in enumerate(flat):
        for i, j, R, _ in _base_bonds(params):
            Rv = np.asarray(R, dtype=int)
            phase = np.exp(2j * np.pi * np.dot(q, Rv))
            vq[iq, i, j] += params.V * phase
            vq[iq, j, i] += params.V * np.conj(phase)
    vq = 0.5 * (vq + np.swapaxes(vq.conj(), -1, -2))
    return vq.reshape(qpts.shape[:-1] + (NSUB, NSUB))


def _add_eta_bond(mat: np.ndarray, i: int, j: int, coeff: float = 1.0):
    mat[i, j] += 1j * coeff
    mat[j, i] -= 1j * coeff


def eta_vertices():
    """Return K_A, K_B, K_plus, K_minus.

    Oriented algebraic loops:
      A: 0 -> 1 -> 2 -> 0
      B: 3 -> 4 -> 5 -> 3

    Because these two arrow loops have opposite geometric handedness in the
    chosen real-space embedding:
      K_plus  -> PHYSICAL OPPOSITE circulation,
      K_minus -> PHYSICAL SAME circulation.
    """
    ka = np.zeros((NSUB, NSUB), dtype=complex)
    kb = np.zeros((NSUB, NSUB), dtype=complex)
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        _add_eta_bond(ka, i, j)
    for i, j in [(3, 4), (4, 5), (5, 3)]:
        _add_eta_bond(kb, i, j)
    kp = (ka + kb) / np.sqrt(2.0)
    km = (ka - kb) / np.sqrt(2.0)
    return ka, kb, kp, km

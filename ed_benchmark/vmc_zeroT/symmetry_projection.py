"""Finite-torus symmetry projection for Hex18 ED/VMC comparisons.

This module second-quantizes verified *ordinary* site permutations from
native_ed.geometry. It intentionally starts with translations + ordinary C3v.
Gauge-compensated C6 operations at t1=-t2 are a separate extension because
their single-particle action includes site-dependent phases.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FockSymmetry:
    name: str
    site_map: np.ndarray
    target_index: np.ndarray
    fermion_phase: np.ndarray

    def apply(self, psi: np.ndarray) -> np.ndarray:
        x = np.asarray(psi, complex)
        if x.ndim != 1 or len(x) != len(self.target_index):
            raise ValueError("State-vector dimension mismatch")
        out = np.empty_like(x, dtype=np.complex128)
        out[self.target_index] = self.fermion_phase * x
        return out


def _permutation_parity(values):
    """Return +/-1 parity needed to sort a list of distinct integers."""
    inv = 0
    for i in range(len(values)):
        vi = values[i]
        for j in range(i + 1, len(values)):
            inv += int(vi > values[j])
    return -1.0 if inv % 2 else 1.0


def second_quantize_site_map(states, ns: int, site_map, *, name="sym") -> FockSymmetry:
    """Build the fixed-N Fock-space representation of c_i^dag -> c_p(i)^dag."""
    states = np.asarray(states, dtype=np.uint64)
    p = np.asarray(site_map, dtype=np.int64)
    if p.shape != (ns,) or sorted(map(int, p)) != list(range(ns)):
        raise ValueError(f"{name}: site_map is not a permutation")

    target = np.empty(len(states), dtype=np.int64)
    phase = np.empty(len(states), dtype=np.complex128)
    for a, s0 in enumerate(states):
        s = int(s0)
        occ = [i for i in range(ns) if s & (1 << (ns - 1 - i))]
        mapped = [int(p[i]) for i in occ]
        sign = _permutation_parity(mapped)
        st = 0
        for j in mapped:
            st |= 1 << (ns - 1 - j)
        b = int(np.searchsorted(states, np.uint64(st)))
        if b >= len(states) or int(states[b]) != st:
            raise RuntimeError(f"{name}: mapped Fock state not found")
        target[a] = b
        phase[a] = sign
    if len(np.unique(target)) != len(target):
        raise RuntimeError(f"{name}: Fock action is not bijective")
    return FockSymmetry(name, p, target, phase)


def compose_site_maps(p_after, p_before):
    """Map for U_after U_before: i -> p_after[p_before[i]]."""
    a = np.asarray(p_after, dtype=np.int64)
    b = np.asarray(p_before, dtype=np.int64)
    return a[b]


def ordinary_space_group_site_maps(g):
    """Unique translations x ordinary C3v site maps on the ring3 torus."""
    ident = np.arange(g.Ns, dtype=np.int64)
    point = g.point_maps()
    c3 = point.get("C3")
    if c3 is None:
        raise RuntimeError("ring3 ordinary C3 symmetry is unavailable")

    c3 = np.asarray(c3, dtype=np.int64)
    point_ops = [("E", ident), ("C3", c3), ("C3^2", compose_site_maps(c3, c3))]
    for angle, p in sorted(point.get("mirrors", {}).items(), key=lambda x: float(x[0])):
        point_ops.append((f"M{angle}", np.asarray(p, dtype=np.int64)))

    translations = []
    for R in g.reps:
        translations.append((f"T{tuple(map(int, R))}", np.asarray(g.translate(R), dtype=np.int64)))

    out = {}
    for tn, tp in translations:
        for pn, pp in point_ops:
            # Apply translation first, then point operation. The full set is
            # independent of this naming convention after de-duplication.
            q = compose_site_maps(pp, tp)
            key = tuple(map(int, q))
            out.setdefault(key, (f"{pn}*{tn}", q))
    return list(out.values())


def build_ordinary_space_group(g, states):
    return [
        second_quantize_site_map(states, g.Ns, p, name=name)
        for name, p in ordinary_space_group_site_maps(g)
    ]


def symmetry_character(op: FockSymmetry, psi: np.ndarray):
    x = np.asarray(psi, complex)
    x = x / np.linalg.norm(x)
    ux = op.apply(x)
    lam = np.vdot(x, ux)
    residual = np.linalg.norm(ux - lam * x)
    return complex(lam), float(residual)


def characterize_unique_state(group, psi: np.ndarray, tol=1e-8):
    """Characters of a unique eigenstate under every group operation."""
    chars = []
    for op in group:
        lam, residual = symmetry_character(op, psi)
        if residual > tol or abs(abs(lam) - 1.0) > 10 * tol:
            raise RuntimeError(
                f"State is not a one-dimensional symmetry eigenstate under {op.name}: "
                f"lambda={lam}, residual={residual:.3e}"
            )
        chars.append(lam / abs(lam))
    return np.asarray(chars, dtype=np.complex128)


def project_to_characters(group, characters, psi: np.ndarray, tol=1e-12):
    """P psi = |G|^-1 sum_g chi(g)^* U_g psi for a 1D target irrep."""
    x = np.asarray(psi, complex)
    n0 = np.linalg.norm(x)
    if n0 < tol:
        raise ValueError("Cannot project a zero-norm state")
    x = x / n0
    chars = np.asarray(characters, complex)
    if len(chars) != len(group):
        raise ValueError("Character/group length mismatch")
    raw = np.zeros_like(x, dtype=np.complex128)
    for chi, op in zip(chars, group):
        raw += np.conj(chi) * op.apply(x)
    raw /= len(group)
    weight = float(np.vdot(raw, raw).real)
    if weight < tol:
        return None, weight
    return raw / np.sqrt(weight), weight


def projected_sector_residual(group, characters, psi):
    """Maximum ||U_g psi-chi_g psi|| over the projected state."""
    x = np.asarray(psi, complex)
    x = x / np.linalg.norm(x)
    return float(max(
        np.linalg.norm(op.apply(x) - chi * x)
        for chi, op in zip(characters, group)
    ))


def commutator_error(group, H, n_tests=2, seed=2601):
    """Random-vector check of [H,U_g]=0 for every generated operation."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    worst_name = None
    n = H.shape[0]
    for op in group:
        for _ in range(n_tests):
            x = rng.normal(size=n) + 1j * rng.normal(size=n)
            x /= np.linalg.norm(x)
            a = H @ op.apply(x)
            b = op.apply(H @ x)
            err = float(np.linalg.norm(a - b))
            if err > worst:
                worst, worst_name = err, op.name
    return worst, worst_name

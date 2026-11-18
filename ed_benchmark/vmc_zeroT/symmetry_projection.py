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


def translation_c3_site_maps(g):
    """Unique ordinary primitive translations times C3 rotations (abelian on ring3)."""
    ident = np.arange(g.Ns, dtype=np.int64)
    c3 = g.point_maps().get("C3")
    if c3 is None:
        raise RuntimeError("ordinary C3 unavailable")
    c3 = np.asarray(c3, dtype=np.int64)
    point = [ident, c3, compose_site_maps(c3, c3)]
    out = {}
    for R in g.reps:
        tr = np.asarray(g.translate(R), dtype=np.int64)
        for p in point:
            q = compose_site_maps(p, tr)
            out.setdefault(tuple(map(int, q)), (f"C3xT{tuple(map(int,R))}", q))
    return list(out.values())


def build_translation_c3_group(g, states):
    return [
        second_quantize_site_map(states, g.Ns, p, name=name)
        for name, p in translation_c3_site_maps(g)
    ]


def _cluster_unit_circle_eigenvalues(vals, tol=1e-7):
    """Group nearly equal unit-circle eigenvalues into index lists."""
    groups = []
    used = np.zeros(len(vals), dtype=bool)
    for i, z in enumerate(vals):
        if used[i]:
            continue
        idx = [j for j, w in enumerate(vals) if (not used[j]) and abs(z-w) < tol]
        for j in idx:
            used[j] = True
        groups.append(idx)
    return groups


def decompose_ground_manifold_abelian(group, psi_states, tol=1e-7):
    """Decompose a complete degenerate ground manifold into 1D abelian sectors.

    Returns one record per distinct character sector. A sector may have
    multiplicity >1 if the ground manifold contains an accidental degeneracy
    within the same symmetry irrep.
    """
    V = np.asarray(psi_states, complex)
    if V.ndim == 1:
        V = V[:, None]
    V, _ = np.linalg.qr(V)
    d = V.shape[1]
    # Coefficient-space blocks spanning simultaneous eigenspaces.
    blocks = [np.eye(d, dtype=complex)]

    for op in group:
        UV = np.column_stack([op.apply(V[:, j]) for j in range(d)])
        M = V.conj().T @ UV
        # Ground manifold must be invariant under every exact symmetry.
        leak = np.linalg.norm(UV - V @ M)
        if leak > 100*tol:
            raise RuntimeError(
                f"Ground manifold incomplete under {op.name}: leakage={leak:.3e}"
            )

        new_blocks = []
        for B in blocks:
            m = B.conj().T @ M @ B
            # Exact symmetry matrices are unitary and mutually commuting.
            # Diagonalize the restricted normal matrix directly; do not
            # Hermitianize it because its eigenvalues are generally phases.
            vals, U = np.linalg.eig(m)
            # Normalize phases and orthonormalize eigenvectors within clusters.
            vals = np.asarray([z/abs(z) if abs(z) else 1+0j for z in vals])
            for idx in _cluster_unit_circle_eigenvalues(vals, tol=tol):
                C = B @ U[:, idx]
                C, _ = np.linalg.qr(C)
                new_blocks.append(C)
        blocks = new_blocks

    sectors = []
    for B in blocks:
        W = V @ B
        W, _ = np.linalg.qr(W)
        chars = []
        residual = 0.0
        for op in group:
            UW = np.column_stack([op.apply(W[:, j]) for j in range(W.shape[1])])
            m = W.conj().T @ UW
            chi = np.trace(m) / W.shape[1]
            if abs(chi) < 1e-12:
                raise RuntimeError(f"Failed to identify character for {op.name}")
            chi /= abs(chi)
            residual = max(residual, float(np.linalg.norm(UW - chi*W)))
            chars.append(chi)
        if residual > 100*tol:
            raise RuntimeError(
                f"Ground-sector block is not a 1D character sector: residual={residual:.3e}"
            )
        sectors.append({
            "characters": np.asarray(chars, np.complex128),
            "basis": W,
            "dimension": int(W.shape[1]),
            "residual": float(residual),
        })

    # Deduplicate blocks with numerically identical characters.
    merged = []
    for sec in sectors:
        found = None
        for m in merged:
            if np.max(np.abs(m["characters"]-sec["characters"])) < 10*tol:
                found = m
                break
        if found is None:
            merged.append(sec)
        else:
            W = np.column_stack([found["basis"], sec["basis"]])
            W, _ = np.linalg.qr(W)
            found["basis"] = W
            found["dimension"] = int(W.shape[1])
            found["residual"] = max(found["residual"], sec["residual"])
    return merged

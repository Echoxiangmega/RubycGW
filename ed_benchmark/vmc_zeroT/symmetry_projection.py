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


def _uint64_parity(values):
    """Vectorized popcount parity for uint64 arrays."""
    x=np.asarray(values,dtype=np.uint64).copy()
    for shift in (32,16,8,4):
        x ^= x >> np.uint64(shift)
    return ((np.uint64(0x6996) >> (x & np.uint64(15))) & np.uint64(1)).astype(np.int8)


def second_quantize_site_map(states, ns: int, site_map, *, name="sym") -> FockSymmetry:
    """Build the fixed-N Fock representation of c_i^dag -> c_p(i)^dag.

    The previous implementation looped over every Fock state in Python.  This
    version performs the same permutation and fermionic inversion parity with
    vectorized uint64 operations, which is important for the 24-site 2x2
    benchmark (735471 states at Nf=8).
    """
    states=np.asarray(states,dtype=np.uint64)
    p=np.asarray(site_map,dtype=np.int64)
    if p.shape!=(ns,) or sorted(map(int,p))!=list(range(ns)):
        raise ValueError(f"{name}: site_map is not a permutation")
    if ns>62:
        raise ValueError("uint64 Fock mapping supports at most 62 sites")

    mapped=np.zeros(len(states),dtype=np.uint64)
    prefix=np.zeros(len(states),dtype=np.uint64)
    parity=np.zeros(len(states),dtype=np.int8)

    # Process occupied source orbitals in canonical source order.  Whenever
    # source i is occupied, count previously mapped occupied orbitals p(j)
    # with j<i and p(j)>p(i); those are exactly the inversions produced by
    # sorting the mapped creation operators.
    for i in range(ns):
        srcbit=np.uint64(1 << (ns-1-i))
        occ=(states & srcbit)!=0
        pi=int(p[i])
        if pi+1<ns:
            greater_mask=np.uint64((1 << (ns-1-pi)) - 1)
            odd=_uint64_parity(prefix & greater_mask).astype(bool)
            parity ^= (occ & odd).astype(np.int8)
        dstbit=np.uint64(1 << (ns-1-pi))
        mapped[occ] |= dstbit
        prefix[occ] |= dstbit

    target=np.searchsorted(states,mapped)
    if np.any(target>=len(states)) or np.any(states[target]!=mapped):
        raise RuntimeError(f"{name}: mapped Fock state not found")
    if len(np.unique(target))!=len(target):
        raise RuntimeError(f"{name}: Fock action is not bijective")
    phase=(1-2*parity).astype(np.int8)
    return FockSymmetry(name,p,target.astype(np.int64),phase)


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



def translation_site_maps(g):
    """Unique ordinary finite-torus translations only."""
    out = {}
    for R in g.reps:
        p = np.asarray(g.translate(R), dtype=np.int64)
        out.setdefault(tuple(map(int, p)), (f"T{tuple(map(int,R))}", p))
    return list(out.values())


def build_translation_group(g, states):
    return [
        second_quantize_site_map(states, g.Ns, p, name=name)
        for name, p in translation_site_maps(g)
    ]


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


def c3_site_maps(g):
    """Ordinary C3 subgroup only: E, C3, C3^2."""
    ident = np.arange(g.Ns, dtype=np.int64)
    c3 = g.point_maps().get("C3")
    if c3 is None:
        raise RuntimeError("ordinary C3 unavailable")
    c3 = np.asarray(c3, dtype=np.int64)
    return [
        ("E", ident),
        ("C3", c3),
        ("C3^2", compose_site_maps(c3, c3)),
    ]


def build_c3_group(g, states):
    return [
        second_quantize_site_map(states, g.Ns, p, name=name)
        for name, p in c3_site_maps(g)
    ]


def c3v_site_maps(g):
    """Ordinary C3v point group only: E, C3, C3^2 and three mirrors.

    No translations are included.  This is the residual point group used by
    the ring3-supercell ansatz, where supercell translations are exact by
    construction.
    """
    ident = np.arange(g.Ns, dtype=np.int64)
    point = g.point_maps()
    c3 = point.get("C3")
    if c3 is None:
        raise RuntimeError("ordinary C3 unavailable")
    mirrors = point.get("mirrors", {})
    if len(mirrors) != 3:
        raise RuntimeError(
            f"ordinary C3v requires exactly three mirrors; found {len(mirrors)}"
        )
    c3 = np.asarray(c3, dtype=np.int64)
    out = [
        ("E", ident),
        ("C3", c3),
        ("C3^2", compose_site_maps(c3, c3)),
    ]
    out.extend(
        (f"M{angle}", np.asarray(p, dtype=np.int64))
        for angle, p in sorted(mirrors.items(), key=lambda x: float(x[0]))
    )
    # Guard against an accidental duplicate map on a very small torus.
    keys = [tuple(map(int, p)) for _, p in out]
    if len(set(keys)) != 6:
        raise RuntimeError("C3v point maps are not six distinct operations")
    return out


def build_c3v_group(g, states):
    return [
        second_quantize_site_map(states, g.Ns, p, name=name)
        for name, p in c3v_site_maps(g)
    ]


def internal_translation_site_maps(g):
    """Residual primitive translations inside one 18-site ring3 supercell.

    The ring3 superlattice has index three in the primitive translation group.
    Once ring3-supercell translations are built into the ansatz, only these
    three cosets remain to be restored explicitly. Their number is independent
    of the total L1 x L2 supercell size.
    """
    reps=[(0,0),(1,0),(1,1)]
    out=[]
    seen=set()
    for R in reps:
        p=np.asarray(g.translate(R),dtype=np.int64)
        key=tuple(map(int,p))
        if key in seen:
            raise RuntimeError(
                "Internal ring3 translation cosets collapsed on this torus"
            )
        seen.add(key)
        out.append((f"Tint{R}",p))
    if len(out)!=3:
        raise RuntimeError("Expected exactly three internal translations")
    return out


def build_internal_translation_group(g, states):
    return [
        second_quantize_site_map(states,g.Ns,p,name=name)
        for name,p in internal_translation_site_maps(g)
    ]


def internal_c3v_space_group_site_maps(g):
    """Residual (T_primitive/T_ring3) semidirect C3v point group.

    Ordering is internal-translation outer, C3v inner. There are exactly
    3*6=18 site maps for a compatible ring3-supercell torus.
    """
    trans=internal_translation_site_maps(g)
    point=c3v_site_maps(g)
    out={}
    for tn,tp in trans:
        for pn,pp in point:
            q=compose_site_maps(pp,tp)
            out.setdefault(tuple(map(int,q)),(f"{pn}*{tn}",q))
    vals=list(out.values())
    if len(vals)!=18:
        raise RuntimeError(
            f"Residual internal-translation x C3v group has {len(vals)} "
            "distinct maps; expected 18"
        )
    return vals


def build_internal_c3v_space_group(g, states):
    return [
        second_quantize_site_map(states,g.Ns,p,name=name)
        for name,p in internal_c3v_space_group_site_maps(g)
    ]


def internal_gamma_c3v_characters(irrep: str) -> np.ndarray:
    """One-dimensional Gamma-sector characters of residual Tint semidirect C3v.

    At internal Gamma all three translation characters equal +1, so the full
    18-character array is simply the C3v A1/A2 character row repeated for each
    translation coset. This matches internal_c3v_space_group_site_maps ordering.
    """
    ch=c3v_irrep_characters(irrep)
    if str(irrep).upper() not in ("A1","A2"):
        raise ValueError("Internal-Gamma residual projector supports A1/A2")
    return np.tile(ch,3).astype(np.complex128)


def c3v_irrep_characters(irrep: str) -> np.ndarray:
    """Characters for the one-dimensional ordinary C3v irreps A1/A2.

    Ordering matches c3v_site_maps: E, C3, C3^2, M1, M2, M3.
    """
    ir = str(irrep).upper()
    if ir == "A1":
        return np.ones(6, dtype=np.complex128)
    if ir == "A2":
        return np.asarray([1, 1, 1, -1, -1, -1], np.complex128)
    if ir == "E":
        # Character projector for the two-dimensional E irrep.  The missing
        # overall irrep-dimension factor is irrelevant after normalization.
        return np.asarray([2, -1, -1, 0, 0, 0], np.complex128)
    raise ValueError("Unknown C3v irrep; choose A1, A2 or E")


def c3v_from_characters(characters, tol=1e-6):
    """Identify A1/A2 from six one-dimensional C3v characters."""
    ch = np.asarray(characters, complex)
    if ch.shape != (6,):
        return {"irrep": None, "label": "unknown"}
    a1 = c3v_irrep_characters("A1")
    a2 = c3v_irrep_characters("A2")
    e1 = float(np.max(np.abs(ch-a1)))
    e2 = float(np.max(np.abs(ch-a2)))
    if min(e1, e2) > tol:
        return {"irrep": None, "label": "unknown", "error": min(e1, e2)}
    ir = "A1" if e1 <= e2 else "A2"
    return {"irrep": ir, "label": ir, "error": min(e1, e2)}


def _find_group_operation(group, site_map):
    target = np.asarray(site_map, dtype=np.int64)
    for i, op in enumerate(group):
        if np.array_equal(np.asarray(op.site_map, dtype=np.int64), target):
            return i
    return None


def momentum_from_characters(g, group, characters, tol=1e-6):
    """Identify the finite-torus momentum from pure-translation characters.

    Reciprocal coordinates q are in primitive reciprocal-lattice units and
    obey U_T(R)|psi> = exp[-2*pi*i*q.R]|psi> in the convention used here.
    Returns a dictionary. If the supplied group has no complete primitive
    translation subgroup (for example C3-only), name is 'unfixed'.
    """
    chars = np.asarray(characters, complex)
    observations = []
    # Use all finite-torus translation representatives, including identity.
    for R in g.reps:
        p = np.asarray(g.translate(R), dtype=np.int64)
        idx = _find_group_operation(group, p)
        if idx is None:
            continue
        observations.append((tuple(map(int, R)), chars[idx]))

    if len(observations) < len(g.reps):
        return {
            "name": "unfixed",
            "q": None,
            "error": None,
            "translation_characters": [
                {
                    "R": list(R),
                    "real": float(z.real),
                    "imag": float(z.imag),
                    "phase_over_pi": float(np.angle(z)/np.pi),
                }
                for R, z in observations
            ],
        }

    candidates = []
    for q in g.permitted_momenta():
        qv = np.asarray(q, float)
        err = 0.0
        for R, z in observations:
            pred = np.exp(-2j*np.pi*np.dot(qv, np.asarray(R, float)))
            err = max(err, float(abs(pred-z)))
        candidates.append((err, tuple(float(x) for x in q)))
    candidates.sort(key=lambda x: x[0])
    err, q = candidates[0]
    if err > tol:
        return {
            "name": "unknown",
            "q": list(q),
            "error": float(err),
            "translation_characters": [
                {
                    "R": list(R),
                    "real": float(z.real),
                    "imag": float(z.imag),
                    "phase_over_pi": float(np.angle(z)/np.pi),
                }
                for R, z in observations
            ],
        }

    qarr = np.mod(np.asarray(q, float), 1.0)
    if np.linalg.norm(qarr - np.rint(qarr)) < tol:
        name = "Gamma"
    elif g.name == "ring3":
        # The two nonzero momenta are the K/K' pair. Assign K to the first
        # canonical permitted momentum and K' to its time-reversed partner.
        nonzero = [
            tuple(float(x) for x in qq)
            for qq in g.permitted_momenta()
            if np.linalg.norm(np.mod(np.asarray(qq),1.0)-np.rint(np.mod(np.asarray(qq),1.0))) >= tol
        ]
        nonzero = sorted(nonzero)
        name = "K" if tuple(q) == nonzero[0] else "K'"
    else:
        name = "q=(" + ",".join(f"{x:.6g}" for x in q) + ")"

    return {
        "name": name,
        "q": list(q),
        "error": float(err),
        "translation_characters": [
            {
                "R": list(R),
                "real": float(z.real),
                "imag": float(z.imag),
                "phase_over_pi": float(np.angle(z)/np.pi),
            }
            for R, z in observations
        ],
    }


def c3_from_characters(g, group, characters, tol=1e-6):
    """Return C3 eigenvalue label m=0,1,2 with chi(C3)=exp(-2*pi*i*m/3)."""
    c3 = g.point_maps().get("C3")
    if c3 is None:
        return {"m": None, "label": "unavailable", "character": None}
    idx = _find_group_operation(group, c3)
    if idx is None:
        return {"m": None, "label": "unfixed", "character": None}
    z = complex(np.asarray(characters, complex)[idx])
    candidates = [(abs(z-np.exp(-2j*np.pi*m/3)), m) for m in range(3)]
    err, m = min(candidates)
    return {
        "m": int(m) if err < tol else None,
        "label": f"C3:m{m}" if err < tol else "C3:unknown",
        "error": float(err),
        "character": {
            "real": float(z.real),
            "imag": float(z.imag),
            "phase_over_pi": float(np.angle(z)/np.pi),
        },
    }


def sector_quantum_numbers(g, group, characters):
    return {
        "momentum": momentum_from_characters(g, group, characters),
        "c3": c3_from_characters(g, group, characters),
    }

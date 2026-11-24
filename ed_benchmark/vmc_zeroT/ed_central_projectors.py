"""Central projectors for a complete ED ground manifold.

The returned arrays are projector coefficients d*chi (or sums thereof),
NOT one-dimensional eigenvalues.  With the existing VAP kernel the
overall division by |G| is immaterial; for exact diagnostics it matters.
"""
from __future__ import annotations

import numpy as np


def _group_classes(group):
    """Conjugacy classes from the actual finite-torus site permutations."""
    maps = [np.asarray(op.site_map, dtype=int) for op in group]
    lookup = {tuple(p): i for i, p in enumerate(maps)}
    if len(lookup) != len(maps):
        raise ValueError("Duplicate group maps")
    identity = lookup[tuple(np.arange(len(maps[0])))]
    mult = np.asarray([[lookup[tuple(a[b])] for b in maps] for a in maps])
    inv = [int(np.flatnonzero(mult[i] == identity)[0])
           for i in range(len(maps))]
    todo = set(range(len(maps)))
    classes = []
    while todo:
        i = min(todo)
        cls = sorted({int(mult[mult[j, i], inv[j]]) for j in range(len(maps))})
        classes.append(cls)
        todo.difference_update(cls)
    return classes, identity


def ed_central_sectors(group, ground, tol=1e-7):
    """Resolve distinct isotypic components, including repeated irreps.

    Restrict the symmetry representation to the complete ED ground space.
    Commuting Hermitian parts of class sums distinguish its irreps, without
    assuming a particular character table or a unique ground state.
    """
    V, _ = np.linalg.qr(np.asarray(ground, complex))
    d0 = V.shape[1]
    if d0 == 0:
        raise ValueError("Empty ED ground manifold")
    rep = []
    for op in group:
        UV = np.column_stack([op.apply(V[:, i]) for i in range(d0)])
        mat = V.conj().T @ UV
        err = np.linalg.norm(UV - V @ mat)
        if err > tol * max(1., np.sqrt(d0)):
            raise RuntimeError(f"Incomplete ED symmetry manifold under {op.name}: {err:.3e}")
        rep.append(mat)
    rep = np.asarray(rep)
    classes, identity = _group_classes(group)
    blocks = [np.eye(d0, dtype=complex)]
    for cls in classes:
        A = np.sum(rep[cls], axis=0)
        for B in ((A + A.conj().T)/2, (A - A.conj().T)/(2j)):
            new = []
            for W in blocks:
                C = W.conj().T @ B @ W
                vals, vecs = np.linalg.eigh((C+C.conj().T)/2)
                cuts = np.flatnonzero(np.diff(vals) > tol * max(1., len(cls))) + 1
                for ids in np.split(np.arange(len(vals)), cuts):
                    new.append(W @ vecs[:, ids])
            blocks = new

    sectors = []
    for W in blocks:
        trace = np.asarray([np.trace(W.conj().T @ R @ W) for R in rep])
        mult_float = np.sqrt(np.mean(abs(trace)**2))
        mult = int(round(float(mult_float)))
        dim = W.shape[1] // max(mult, 1)
        if mult < 1 or abs(mult_float-mult) > 20*tol or dim*mult != W.shape[1]:
            raise RuntimeError("ED class sums did not resolve a valid isotypic component")
        chi = trace / mult
        if abs(chi[identity]-dim) > 20*tol:
            raise RuntimeError("Invalid ED irrep dimension")
        weights = dim * chi
        # Verify that the *normalized* central projector acts as the
        # orthogonal isotypic projector on the complete ED manifold.
        restricted = np.einsum("g,gij->ij", weights.conj(), rep) / len(group)
        if np.linalg.norm(restricted-W@W.conj().T) > 50*tol*max(1., np.sqrt(d0)):
            raise RuntimeError("Central projector failed the ED subspace test")
        weights.real[np.abs(weights.real) < 10*tol] = 0.
        weights.imag[np.abs(weights.imag) < 10*tol] = 0.
        sectors.append(dict(characters=weights, ground_dimension=W.shape[1],
                            irrep_dimension=dim, multiplicity=mult))
    sectors.sort(key=lambda s: (s["irrep_dimension"], tuple(
        np.round(np.column_stack((s["characters"].real,s["characters"].imag)).ravel(), 6))))
    return sectors


def pair_time_reversal_sectors(sectors, tol=1e-6):
    """A real spinless Hamiltonian must contain conjugate partner sectors."""
    out = []
    used = set()
    for i, sec in enumerate(sectors):
        if i in used:
            continue
        ch = sec["characters"]
        if np.max(abs(ch.imag)) < tol:
            out.append(dict(sec, characters=ch.real.astype(complex)))
            used.add(i)
            continue
        matches = [j for j, other in enumerate(sectors)
                   if j != i and j not in used
                   and np.max(abs(other["characters"]-ch.conj())) < tol]
        if len(matches) != 1:
            raise RuntimeError("Missing or ambiguous time-reversal partner in ED manifold")
        j = matches[0]
        other = sectors[j]
        if sec["ground_dimension"] != other["ground_dimension"]:
            raise RuntimeError("Time-reversal partners have unequal multiplicities")
        out.append(dict(characters=(ch+other["characters"]).real.astype(complex),
                        ground_dimension=sec["ground_dimension"]+other["ground_dimension"],
                        irrep_dimension=sec["irrep_dimension"],
                        multiplicity=sec["multiplicity"], tr_paired=True))
        used.update((i,j))
    return out


def central_projector_diagnostics(group, weights, psi):
    """Return ||P psi||^2 and ||P psi-psi|| for a normalized state."""
    x = np.asarray(psi, complex)
    x = x / np.linalg.norm(x)
    projected = sum((np.conj(w)*op.apply(x)
                     for w, op in zip(weights, group)), np.zeros_like(x)) / len(group)
    return float(np.vdot(projected, projected).real), float(np.linalg.norm(projected-x))

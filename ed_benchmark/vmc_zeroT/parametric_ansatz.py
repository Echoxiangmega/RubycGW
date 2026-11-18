"""Symmetry-tied Varma-style one-body parameterization for Ruby VMC.

The auxiliary one-body Hamiltonian is not represented by a free Ns x Nf
orbital matrix. Instead,

    h_var = h0 + sum_a lambda_a D_a
                 + sum_b [x_b K_b^R + y_b K_b^I],

where D_a and K_b are independent under primitive translations and ordinary
C3 rotations of the finite torus. Mirrors and time reversal are NOT imposed,
so complex loop-current states remain allowed. The parameter count is
independent of the number of occupied orbitals and is suitable for later
large-cluster reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from symmetry_projection import compose_site_maps


@dataclass
class SymmetryTiedBasis:
    h0: np.ndarray
    onsite_matrices: np.ndarray
    bond_real_matrices: np.ndarray
    bond_imag_matrices: np.ndarray
    site_orbits: list
    bond_orbits: list
    subgroup_maps: list

    @property
    def n_onsite(self):
        return int(self.onsite_matrices.shape[0])

    @property
    def n_bond(self):
        return int(self.bond_real_matrices.shape[0])

    @property
    def n_one_body_parameters(self):
        return self.n_onsite + 2 * self.n_bond


def translation_c3_site_maps(g):
    """Unique primitive translations times ordinary C3 rotations."""
    ident = np.arange(g.Ns, dtype=np.int64)
    c3 = g.point_maps().get("C3")
    if c3 is None:
        raise RuntimeError("ordinary C3 unavailable")
    c3 = np.asarray(c3, dtype=np.int64)
    point = [
        ident,
        c3,
        compose_site_maps(c3, c3),
    ]
    out = {}
    for R in g.reps:
        tr = np.asarray(g.translate(R), dtype=np.int64)
        for p in point:
            q = compose_site_maps(p, tr)
            out.setdefault(tuple(map(int, q)), q)
    return list(out.values())


def _site_orbits(ns, maps):
    unseen = set(range(ns))
    out = []
    while unseen:
        seed = min(unseen)
        orb = sorted({int(p[seed]) for p in maps})
        # group closure should make one application enough, but close robustly.
        changed = True
        cur = set(orb)
        while changed:
            old = len(cur)
            cur.update(int(p[i]) for p in maps for i in list(cur))
            changed = len(cur) != old
        orb = sorted(cur)
        out.append(orb)
        unseen.difference_update(orb)
    return out


def _bond_lookup(g):
    lookup = {}
    for ib, e in enumerate(g.edges):
        if (e.i, e.j) in lookup or (e.j, e.i) in lookup:
            raise RuntimeError("duplicate finite-torus edge")
        lookup[(e.i, e.j)] = (ib, +1)
        lookup[(e.j, e.i)] = (ib, -1)
    return lookup


def _bond_orbits(g, maps):
    """Return [(edge_index, orientation_sign), ...] per symmetry orbit."""
    lookup = _bond_lookup(g)
    assigned = set()
    out = []
    for seed in range(len(g.edges)):
        if seed in assigned:
            continue
        rel = {seed: +1}
        queue = [seed]
        while queue:
            ib = queue.pop()
            s0 = rel[ib]
            e = g.edges[ib]
            a, b = (e.i, e.j) if s0 > 0 else (e.j, e.i)
            for p in maps:
                key = (int(p[a]), int(p[b]))
                if key not in lookup:
                    raise RuntimeError(f"mapped edge missing: {key}")
                jb, orient = lookup[key]
                sj = int(orient)
                if jb in rel:
                    if rel[jb] != sj:
                        raise RuntimeError(
                            "orientation inconsistency inside translation+C3 bond orbit; "
                            "imaginary hopping would be symmetry-forced to zero"
                        )
                else:
                    rel[jb] = sj
                    queue.append(jb)
        orb = sorted((int(i), int(s)) for i, s in rel.items())
        out.append(orb)
        assigned.update(i for i, _ in orb)
    if len(assigned) != len(g.edges):
        raise RuntimeError("incomplete bond orbit partition")
    return out


def build_symmetry_tied_basis(g):
    maps = translation_c3_site_maps(g)
    sites = _site_orbits(g.Ns, maps)
    bonds = _bond_orbits(g, maps)

    h0 = np.zeros((g.Ns, g.Ns), dtype=np.complex128)
    for e in g.edges:
        t = float(g.t[e.kind])
        h0[e.i, e.j] += t
        h0[e.j, e.i] += t

    # Drop the last site orbit as the arbitrary uniform onsite-energy gauge.
    onsite = []
    if len(sites) > 1:
        ref = sites[-1]
        for orb in sites[:-1]:
            m = np.zeros_like(h0)
            # Difference of orbit averages keeps a clean relative-potential gauge.
            for i in orb:
                m[i, i] += 1.0 / len(orb)
            for i in ref:
                m[i, i] -= 1.0 / len(ref)
            onsite.append(m.real)
    onsite = np.asarray(onsite, dtype=np.float64)
    if onsite.size == 0:
        onsite = np.zeros((0, g.Ns, g.Ns), dtype=np.float64)

    bre = []
    bim = []
    for orb in bonds:
        mr = np.zeros_like(h0)
        mi = np.zeros_like(h0)
        for ib, s in orb:
            e = g.edges[ib]
            mr[e.i, e.j] += 1.0
            mr[e.j, e.i] += 1.0
            mi[e.i, e.j] += 1j * s
            mi[e.j, e.i] -= 1j * s
        bre.append(mr.real)
        bim.append(mi)
    return SymmetryTiedBasis(
        h0=h0,
        onsite_matrices=np.asarray(onsite, dtype=np.float64),
        bond_real_matrices=np.asarray(bre, dtype=np.float64),
        bond_imag_matrices=np.asarray(bim, dtype=np.complex128),
        site_orbits=sites,
        bond_orbits=bonds,
        subgroup_maps=maps,
    )


def seed_tied_parameters(basis: SymmetryTiedBasis, h_seed: np.ndarray):
    """Least-squares-project a seed one-body Hamiltonian onto tied corrections."""
    delta = np.asarray(h_seed, complex) - basis.h0
    mats = []
    labels = []
    for i, m in enumerate(basis.onsite_matrices):
        mats.append(np.asarray(m, complex)); labels.append(("onsite", i))
    for i, m in enumerate(basis.bond_real_matrices):
        mats.append(np.asarray(m, complex)); labels.append(("re", i))
    for i, m in enumerate(basis.bond_imag_matrices):
        mats.append(np.asarray(m, complex)); labels.append(("im", i))

    if mats:
        A = np.column_stack([m.reshape(-1) for m in mats])
        # Real coefficients, complex matrix target: stack real and imag equations.
        Ar = np.vstack([A.real, A.imag])
        br = np.concatenate([delta.reshape(-1).real, delta.reshape(-1).imag])
        coeff, *_ = np.linalg.lstsq(Ar, br, rcond=None)
    else:
        coeff = np.zeros(0)

    no = basis.n_onsite
    nb = basis.n_bond
    return {
        "onsite": np.asarray(coeff[:no], np.float64),
        "bond_re": np.asarray(coeff[no:no+nb], np.float64),
        "bond_im": np.asarray(coeff[no+nb:no+2*nb], np.float64),
    }


def reconstruct_one_body(basis: SymmetryTiedBasis, params: dict):
    h = np.array(basis.h0, copy=True)
    if basis.n_onsite:
        h += np.einsum("a,aij->ij", params["onsite"], basis.onsite_matrices)
    if basis.n_bond:
        h += np.einsum("a,aij->ij", params["bond_re"], basis.bond_real_matrices)
        h += np.einsum("a,aij->ij", params["bond_im"], basis.bond_imag_matrices)
    return h


def build_parametric_netket_model(
    basis: SymmetryTiedBasis,
    nf: int,
    init_params: dict,
    shell_matrices: np.ndarray,
    use_jastrow: bool,
    *,
    projection_site_maps=None,
    projection_characters=None,
):
    """Build unprojected or VAP symmetry-tied Slater(-Jastrow) log wavefunction."""
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    import netket as nk

    h0 = np.asarray(basis.h0, np.complex128)
    onsite_m = np.asarray(basis.onsite_matrices, np.float64)
    bre_m = np.asarray(basis.bond_real_matrices, np.float64)
    bim_m = np.asarray(basis.bond_imag_matrices, np.complex128)
    shells_np = np.asarray(shell_matrices, np.float64)
    ns = h0.shape[0]
    maps_np = None if projection_site_maps is None else np.asarray(projection_site_maps, np.int32)
    chars_np = None if projection_characters is None else np.asarray(projection_characters, np.complex128)
    projected = maps_np is not None
    theta0_np = (
        np.zeros(tan_np.shape[0], dtype=np.float64)
        if initial_theta is None
        else np.asarray(initial_theta, np.float64)
    )
    if theta0_np.shape != (tan_np.shape[0],):
        raise ValueError("initial_theta shape mismatch")
    if projected and chars_np.shape != (maps_np.shape[0],):
        raise ValueError("projection character/map mismatch")

    p0_o = np.asarray(init_params["onsite"], np.float64)
    p0_r = np.asarray(init_params["bond_re"], np.float64)
    p0_i = np.asarray(init_params["bond_im"], np.float64)

    class ParametricSlaterJastrow(nn.Module):
        @nn.compact
        def __call__(self, n):
            onsite = self.param("onsite", lambda key: jnp.asarray(p0_o))
            bre = self.param("bond_re", lambda key: jnp.asarray(p0_r))
            bim = self.param("bond_im", lambda key: jnp.asarray(p0_i))
            if use_jastrow:
                v = self.param(
                    "v", nn.initializers.zeros_init(), (shells_np.shape[0],), jnp.float64
                )
            else:
                v = None

            h = jnp.asarray(h0)
            if onsite_m.shape[0]:
                h = h + jnp.einsum("a,aij->ij", onsite, jnp.asarray(onsite_m))
            if bre_m.shape[0]:
                h = h + jnp.einsum("a,aij->ij", bre, jnp.asarray(bre_m))
                h = h + jnp.einsum("a,aij->ij", bim, jnp.asarray(bim_m))
            _, u = jnp.linalg.eigh(h)
            Q = u[:, :nf]
            shells = jnp.asarray(shells_np)
            filling = nf / ns

            def base_logpsi(x):
                occ = jnp.nonzero(x, size=nf)[0]
                out = nk.jax.logdet_cmplx(Q[occ])
                if use_jastrow and shells.shape[0] > 0:
                    dn = x - filling
                    feat = 0.5 * jnp.einsum("i,sij,j->s", dn, shells, dn)
                    out = out + jnp.dot(v, feat)
                return out

            if not projected:
                return jnp.vectorize(base_logpsi, signature="(n)->()")(n)

            maps = jnp.asarray(maps_np)
            chars = jnp.asarray(chars_np)

            def fock_sign(p, source):
                occupied = jnp.nonzero(source, size=nf)[0]
                mapped = p[occupied]
                cmp = mapped[:, None] > mapped[None, :]
                inv = jnp.sum(jnp.triu(cmp, k=1), dtype=jnp.int32)
                return jnp.where((inv & 1) == 0, 1.0, -1.0)

            def projected_logpsi(x):
                sources = jax.vmap(lambda p: x[p])(maps)
                logs = jax.vmap(base_logpsi)(sources)
                signs = jax.vmap(fock_sign)(maps, sources)
                coeff = jnp.conj(chars) * signs
                shift = jnp.max(jnp.real(logs))
                z = jnp.sum(coeff * jnp.exp(logs-shift))
                tiny = jnp.asarray(1e-300+0j, dtype=z.dtype)
                z = jnp.where(jnp.abs(z) > 1e-300, z, tiny)
                return shift + jnp.log(z)

            return jnp.vectorize(projected_logpsi, signature="(n)->()")(n)

    return ParametricSlaterJastrow()


def build_orbital_rotation_data(
    basis: SymmetryTiedBasis,
    Q0: np.ndarray,
    tol: float = 1e-10,
):
    """Build a stable reduced tangent basis for the occupied Slater subspace.

    This replaces autodifferentiating through eigh(h_var). Let G_a be the
    translation+C3 independent Hermitian one-body generators. Around a fixed
    occupied subspace Q0, only occupied-virtual mixing changes the determinant:

        dQ_a = (1-P0) G_a Q0,   P0=Q0 Q0^dag.

    Real-linear dependent tangent directions are removed by SVD. The resulting
    real parameters are symmetry-tied linear combinations of the original
    onsite/real-bond/imag-bond generators, but the differentiable path contains
    no eigendecomposition.
    """
    Q0 = np.asarray(Q0, np.complex128)
    ns, nf = Q0.shape
    Q0, _ = np.linalg.qr(Q0, mode="reduced")
    P0 = Q0 @ Q0.conj().T
    Iminus = np.eye(ns, dtype=complex) - P0

    generators = []
    labels = []
    for i, m in enumerate(basis.onsite_matrices):
        generators.append(np.asarray(m, complex)); labels.append(f"onsite[{i}]")
    for i, m in enumerate(basis.bond_real_matrices):
        generators.append(np.asarray(m, complex)); labels.append(f"bond_re[{i}]")
    for i, m in enumerate(basis.bond_imag_matrices):
        generators.append(np.asarray(m, complex)); labels.append(f"bond_im[{i}]")

    tangents = []
    active_labels = []
    for lab, G in zip(labels, generators):
        dQ = Iminus @ G @ Q0
        nrm = float(np.linalg.norm(dQ))
        if nrm > tol:
            tangents.append(dQ)
            active_labels.append(lab)

    if not tangents:
        return {
            "Q0": Q0,
            "tangent_basis": np.zeros((0, ns, nf), np.complex128),
            "singular_values": np.zeros(0),
            "generator_labels": labels,
            "active_generator_labels": [],
            "generator_count": len(generators),
            "tangent_rank": 0,
        }

    # Real coefficients multiply complex tangent directions, so determine rank
    # in the real vector space [Re(dQ), Im(dQ)].
    A = np.column_stack([
        np.concatenate([d.real.ravel(), d.imag.ravel()]) for d in tangents
    ])
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    cutoff = tol * max(1.0, float(s[0]))
    rank = int(np.count_nonzero(s > cutoff))
    basis_vecs = []
    nflat = ns * nf
    for j in range(rank):
        u = U[:, j]
        d = u[:nflat].reshape(ns, nf) + 1j*u[nflat:].reshape(ns, nf)
        # Re-project away tiny numerical occupied-space components.
        d = Iminus @ d
        d /= np.linalg.norm(d)
        basis_vecs.append(d)

    return {
        "Q0": Q0,
        "tangent_basis": np.asarray(basis_vecs, np.complex128),
        "singular_values": np.asarray(s[:rank], float),
        "generator_labels": labels,
        "active_generator_labels": active_labels,
        "generator_count": len(generators),
        "tangent_rank": rank,
        "generator_to_tangent_vh": np.asarray(Vh[:rank, :], float),
    }


def rotate_orbitals_numpy(Q0, tangent_basis, theta):
    """Reference NumPy map used for seed/projector diagnostics."""
    Q0 = np.asarray(Q0, complex)
    tangent_basis = np.asarray(tangent_basis, complex)
    theta = np.asarray(theta, float)
    M = np.array(Q0, copy=True)
    if len(theta):
        M += np.einsum("a,aij->ij", theta, tangent_basis)
    Q, R = np.linalg.qr(M, mode="reduced")
    # Fix QR column phases continuously enough for deterministic exact checks.
    d = np.diag(R)
    phase = np.where(np.abs(d) > 1e-14, d/np.abs(d), 1.0+0j)
    Q = Q * phase.conj()[None, :]
    return Q


def build_stable_parametric_netket_model(
    Q0: np.ndarray,
    tangent_basis: np.ndarray,
    shell_matrices: np.ndarray,
    use_jastrow: bool,
    *,
    initial_theta=None,
    projection_site_maps=None,
    projection_characters=None,
):
    """Reduced Slater(-Jastrow) model without differentiating through eigh.

    Q(theta) = qr[ Q0 + sum_a theta_a dQ_a ], with real theta_a and fixed
    translation+C3 symmetry-tied tangent directions dQ_a.
    """
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    import netket as nk

    Q0_np = np.asarray(Q0, np.complex128)
    tan_np = np.asarray(tangent_basis, np.complex128)
    shells_np = np.asarray(shell_matrices, np.float64)
    ns, nf = Q0_np.shape
    maps_np = None if projection_site_maps is None else np.asarray(projection_site_maps, np.int32)
    chars_np = None if projection_characters is None else np.asarray(projection_characters, np.complex128)
    projected = maps_np is not None
    if projected and chars_np.shape != (maps_np.shape[0],):
        raise ValueError("projection character/map mismatch")

    class StableReducedSlaterJastrow(nn.Module):
        @nn.compact
        def __call__(self, n):
            theta = self.param("theta", lambda key: jnp.asarray(theta0_np))
            if use_jastrow:
                v = self.param(
                    "v", nn.initializers.zeros_init(), (shells_np.shape[0],), jnp.float64
                )
            else:
                v = None

            M = jnp.asarray(Q0_np)
            if tan_np.shape[0]:
                M = M + jnp.einsum("a,aij->ij", theta, jnp.asarray(tan_np))
            Q, R = jnp.linalg.qr(M, mode="reduced")
            d = jnp.diag(R)
            phase = jnp.where(jnp.abs(d) > 1e-14, d/jnp.abs(d), 1.0+0j)
            Q = Q * jnp.conj(phase)[None, :]

            shells = jnp.asarray(shells_np)
            filling = nf / ns

            def base_logpsi(x):
                occ = jnp.nonzero(x, size=nf)[0]
                out = nk.jax.logdet_cmplx(Q[occ])
                if use_jastrow and shells.shape[0] > 0:
                    dn = x - filling
                    feat = 0.5*jnp.einsum("i,sij,j->s", dn, shells, dn)
                    out = out + jnp.dot(v, feat)
                return out

            if not projected:
                return jnp.vectorize(base_logpsi, signature="(n)->()")(n)

            maps = jnp.asarray(maps_np)
            chars = jnp.asarray(chars_np)

            def fock_sign(p, source):
                occupied = jnp.nonzero(source, size=nf)[0]
                mapped = p[occupied]
                cmp = mapped[:, None] > mapped[None, :]
                inv = jnp.sum(jnp.triu(cmp, k=1), dtype=jnp.int32)
                return jnp.where((inv & 1) == 0, 1.0, -1.0)

            def projected_logpsi(x):
                sources = jax.vmap(lambda p: x[p])(maps)
                logs = jax.vmap(base_logpsi)(sources)
                signs = jax.vmap(fock_sign)(maps, sources)
                coeff = jnp.conj(chars) * signs
                shift = jnp.max(jnp.real(logs))
                z = jnp.sum(coeff*jnp.exp(logs-shift))
                tiny = jnp.asarray(1e-300+0j, dtype=z.dtype)
                z = jnp.where(jnp.abs(z) > 1e-300, z, tiny)
                return shift + jnp.log(z)

            return jnp.vectorize(projected_logpsi, signature="(n)->()")(n)

    return StableReducedSlaterJastrow()

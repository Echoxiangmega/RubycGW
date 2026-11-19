"""Translation-built-in VMC ansatz using the 18-site ring3 variational supercell.

The total torus contains L1 x L2 copies of the 18-site ring3 cell. The
one-body orbitals are block diagonal in ring3-superlattice momentum. For each
allowed k, a trainable complex 18 x n_occ(k) matrix is QR-orthonormalized and
Bloch-expanded over all supercells.

For L1=L2=1 this is exactly the same full 18 x Nf orbital freedom as the
original unrestricted Hex18 Slater ansatz. For larger systems the one-body
parameter count scales as 18*Nf = O(N), instead of Ns*Nf = O(N^2), while
ring3-supercell translation is exact by construction.
"""
from __future__ import annotations

import numpy as np

from benchmark_observables import ruby_star_formfactor
from vmc_ansatz import one_body_seed_hamiltonian


def ring3_periodic_star_formfactor(g, l1: int, l2: int) -> np.ndarray:
    """Tile the existing Hex18 SD/ISD star form factor over ring3 supercells."""
    import geometry as native_geometry

    ref = native_geometry.RubyPBC(
        "ring3", t=g.t["t"], t1=g.t["t1"], t2=g.t["t2"]
    )
    fref = np.asarray(ruby_star_formfactor(ref), float)
    _, internal = native_geometry.ring3_supercell_layout(g, l1, l2)

    if len(g.edges) != 12 * g.num_cells:
        raise RuntimeError("Unexpected Ruby edge ordering")
    out = np.empty(len(g.edges), float)
    for ie in range(len(g.edges)):
        cell = ie // 12
        local_edge = ie % 12
        internal_cell = int(internal[6 * cell] // 6)
        out[ie] = fref[12 * internal_cell + local_edge]
    return out


def bloch_layout(g, l1: int, l2: int):
    """Return site supercell coordinates, internal indices and k points."""
    import geometry as native_geometry

    sc, internal = native_geometry.ring3_supercell_layout(g, l1, l2)
    kpoints = np.asarray(
        native_geometry.ring3_supercell_momenta(l1, l2), dtype=float
    )
    return np.asarray(sc, np.int32), np.asarray(internal, np.int32), kpoints


def bloch_basis_matrix(
    sc_coords: np.ndarray,
    internal_sites: np.ndarray,
    k: np.ndarray,
    n_supercells: int,
) -> np.ndarray:
    """Ns x 18 orthonormal Bloch embedding for one superlattice momentum."""
    ns = len(internal_sites)
    F = np.zeros((ns, 18), np.complex128)
    phase = np.exp(
        2j * np.pi * (np.asarray(sc_coords, float) @ np.asarray(k, float))
    ) / np.sqrt(float(n_supercells))
    F[np.arange(ns), internal_sites] = phase
    err = np.linalg.norm(F.conj().T @ F - np.eye(18))
    if err > 1e-10:
        raise RuntimeError(f"Nonorthonormal supercell Bloch embedding: {err}")
    return F


def seed_bloch_blocks(
    g,
    nf: int,
    l1: int,
    l2: int,
    seed: str,
    seed_amplitude: float,
):
    """Construct translation-compatible initial occupied orbital blocks.

    The ring3-periodic seed Hamiltonian is transformed to each superlattice
    momentum block. The globally lowest nf one-body levels determine the
    fixed occupation count n_occ(k) used by the variational family.
    """
    sc, internal, kpoints = bloch_layout(g, l1, l2)
    nsc = int(l1) * int(l2)
    star = ring3_periodic_star_formfactor(g, l1, l2)
    h = one_body_seed_hamiltonian(g, seed, seed_amplitude, star)

    evecs = []
    embeddings = []
    levels = []
    for ik, k in enumerate(kpoints):
        F = bloch_basis_matrix(sc, internal, k, nsc)
        hk = F.conj().T @ h @ F
        hk = 0.5 * (hk + hk.conj().T)
        w, U = np.linalg.eigh(hk)
        evecs.append(U)
        embeddings.append(F)
        for ib, en in enumerate(w):
            levels.append((float(en), ik, ib))

    levels.sort(key=lambda x: (x[0], x[1], x[2]))
    if not 0 <= int(nf) <= 18 * nsc:
        raise ValueError("nf outside supercell Hilbert-space range")
    selected = levels[: int(nf)]
    bands = [[] for _ in range(len(kpoints))]
    for _, ik, ib in selected:
        bands[ik].append(int(ib))

    blocks = []
    counts = []
    for ik in range(len(kpoints)):
        ids = sorted(bands[ik])
        counts.append(len(ids))
        blocks.append(
            np.asarray(evecs[ik][:, ids], np.complex128)
            if ids else np.empty((18, 0), np.complex128)
        )

    qparts = []
    for ik, block in enumerate(blocks):
        if block.shape[1]:
            qparts.append(embeddings[ik] @ block)
    Q = (
        np.concatenate(qparts, axis=1)
        if qparts
        else np.empty((g.Ns, 0), np.complex128)
    )
    if Q.shape != (g.Ns, nf):
        raise RuntimeError(f"Seed block shape mismatch: {Q.shape} != {(g.Ns,nf)}")
    if nf and np.linalg.norm(Q.conj().T @ Q - np.eye(nf)) > 1e-9:
        raise RuntimeError("Seed Bloch occupied orbitals are not orthonormal")

    shell_gap = None
    if 0 < nf < len(levels):
        shell_gap = float(levels[nf][0] - levels[nf - 1][0])
    total_k = np.mod(
        np.sum(np.asarray(counts,float)[:,None] * kpoints, axis=0),
        1.0,
    )
    total_k[np.isclose(total_k,1.0,atol=1e-10)] = 0.0

    return {
        "blocks": blocks,
        "occupations_per_k": np.asarray(counts, np.int32),
        "kpoints": kpoints,
        "sc_coords": sc,
        "internal_sites": internal,
        "star_formfactor": star,
        "one_body_levels": np.asarray([x[0] for x in levels], float),
        "shell_gap": shell_gap,
        "total_supercell_momentum": total_k,
        "Q0_full": Q,
    }


def assemble_bloch_orbitals_numpy(
    blocks,
    sc_coords: np.ndarray,
    internal_sites: np.ndarray,
    kpoints: np.ndarray,
) -> np.ndarray:
    """Assemble the full real-space occupied-orbital matrix from k blocks."""
    nsc = len(np.unique(np.asarray(sc_coords), axis=0))
    parts = []
    for block, k in zip(blocks, kpoints):
        block = np.asarray(block, np.complex128)
        if block.shape[1] == 0:
            continue
        phase = np.exp(
            2j * np.pi * (np.asarray(sc_coords, float) @ np.asarray(k, float))
        ) / np.sqrt(float(nsc))
        parts.append(phase[:, None] * block[np.asarray(internal_sites), :])
    if not parts:
        return np.empty((len(internal_sites), 0), np.complex128)
    return np.concatenate(parts, axis=1)


def build_supercell_netket_model(
    initial_blocks,
    sc_coords: np.ndarray,
    internal_sites: np.ndarray,
    kpoints: np.ndarray,
    shell_matrices: np.ndarray,
    *,
    use_jastrow: bool = True,
    backflow_matrices: np.ndarray | None = None,
    initial_v: np.ndarray | None = None,
    initial_eta: np.ndarray | None = None,
    point_site_maps: np.ndarray | None = None,
    point_characters: np.ndarray | None = None,
):
    """Build a translation-exact ring3-supercell Slater-Jastrow-backflow model.

    If point_site_maps/characters are supplied, only that fixed-size point
    group is explicitly projected. Supercell translations are never summed
    over: they are exact in the Bloch orbital construction.
    """
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    import netket as nk

    blocks0 = [np.asarray(x, np.complex128) for x in initial_blocks]
    sc_np = np.asarray(sc_coords, np.int32)
    internal_np = np.asarray(internal_sites, np.int32)
    k_np = np.asarray(kpoints, float)
    shells_np = np.asarray(shell_matrices, np.float64)
    bf_np = (
        None if backflow_matrices is None
        else np.asarray(backflow_matrices, np.float64)
    )
    maps_np = (
        None if point_site_maps is None
        else np.asarray(point_site_maps, np.int32)
    )
    chars_np = (
        None if point_characters is None
        else np.asarray(point_characters, np.complex128)
    )
    v0_np = None if initial_v is None else np.asarray(initial_v, np.float64)
    eta0_np = None if initial_eta is None else np.asarray(initial_eta, np.complex128)

    ns = len(internal_np)
    nf = int(sum(x.shape[1] for x in blocks0))
    nsc = len(np.unique(sc_np, axis=0))
    if k_np.shape[0] != len(blocks0):
        raise ValueError("kpoint/block count mismatch")
    if maps_np is not None:
        if maps_np.ndim != 2 or maps_np.shape[1] != ns:
            raise ValueError("point_site_maps must be (|G|,Ns)")
        if chars_np is None or chars_np.shape != (maps_np.shape[0],):
            raise ValueError("point characters do not match point maps")

    class SupercellBlochSlaterJastrow(nn.Module):
        @nn.compact
        def __call__(self, n):
            params = []
            for ik, b0 in enumerate(blocks0):
                if b0.shape[1] == 0:
                    params.append(None)
                    continue

                def init_block(key, arr=b0):
                    del key
                    return jnp.asarray(arr)

                params.append(self.param(f"M_{ik}", init_block))

            if use_jastrow:
                def init_v(key):
                    del key
                    if v0_np is None:
                        return jnp.zeros((shells_np.shape[0],), dtype=jnp.float64)
                    return jnp.asarray(v0_np, dtype=jnp.float64)
                v = self.param("v", init_v)
            else:
                v = None

            if bf_np is not None:
                def init_eta(key):
                    del key
                    if eta0_np is None:
                        return jnp.zeros((bf_np.shape[0],), dtype=jnp.complex128)
                    return jnp.asarray(eta0_np, dtype=jnp.complex128)
                eta = self.param("eta", init_eta)
            else:
                eta = None

            sc = jnp.asarray(sc_np)
            internal = jnp.asarray(internal_np)
            ks = jnp.asarray(k_np)
            shells = jnp.asarray(shells_np)
            backflow = None if bf_np is None else jnp.asarray(bf_np)
            maps = None if maps_np is None else jnp.asarray(maps_np)
            chars = None if chars_np is None else jnp.asarray(chars_np)
            filling = nf / ns

            qparts = []
            for ik, M in enumerate(params):
                if M is None:
                    continue
                U, _ = jnp.linalg.qr(M, mode="reduced")
                phase = jnp.exp(
                    2j * jnp.pi * jnp.sum(sc * ks[ik][None, :], axis=1)
                ) / jnp.sqrt(jnp.asarray(float(nsc)))
                qparts.append(phase[:, None] * U[internal, :])
            Q = jnp.concatenate(qparts, axis=1)

            def base_logpsi(x):
                occupied = jnp.nonzero(x, size=nf)[0]
                dn = x - filling
                if backflow is not None:
                    dQ = jnp.einsum("k,kij,j,ja->ia", eta, backflow, dn, Q)
                    Qx = Q + dQ
                else:
                    Qx = Q
                A = Qx[occupied]
                out = nk.jax.logdet_cmplx(A)
                if use_jastrow and shells.shape[0] > 0:
                    features = 0.5 * jnp.einsum("i,sij,j->s", dn, shells, dn)
                    out = out + jnp.dot(v, features)
                return out

            def fock_sign(p, source):
                occupied = jnp.nonzero(source, size=nf)[0]
                mapped = p[occupied]
                cmp = mapped[:, None] > mapped[None, :]
                inv = jnp.sum(jnp.triu(cmp, k=1), dtype=jnp.int32)
                return jnp.where((inv & 1) == 0, 1.0, -1.0)

            def one(x):
                if maps is None:
                    return base_logpsi(x)
                sources = jax.vmap(lambda p: x[p])(maps)
                logs = jax.vmap(base_logpsi)(sources)
                signs = jax.vmap(fock_sign)(maps, sources)
                shift = jnp.max(jnp.real(logs))
                amps = jnp.exp(logs - shift)
                z = jnp.sum(jnp.conj(chars) * signs * amps)
                tiny = jnp.asarray(1e-300 + 0j, dtype=z.dtype)
                return shift + jnp.log(jnp.where(jnp.abs(z) > 1e-300, z, tiny))

            return jnp.vectorize(one, signature="(n)->()")(n)

    return SupercellBlochSlaterJastrow()


def extract_supercell_parameters(parameters, nkpoints: int) -> dict:
    """Convert a NetKet/Flax parameter tree to NumPy warm-start arrays."""
    blocks = []
    for ik in range(int(nkpoints)):
        key = f"M_{ik}"
        if key in parameters:
            blocks.append(np.asarray(parameters[key], np.complex128))
        else:
            blocks.append(np.empty((18, 0), np.complex128))
    return {
        "blocks": blocks,
        "v": (
            np.asarray(parameters["v"], np.float64)
            if "v" in parameters else np.empty(0, float)
        ),
        "eta": (
            np.asarray(parameters["eta"], np.complex128)
            if "eta" in parameters else np.empty(0, complex)
        ),
    }

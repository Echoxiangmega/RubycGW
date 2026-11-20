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
    *,
    random_seed: int | None = None,
):
    """Construct translation-compatible initial occupied orbital blocks.

    Standard seeds use the existing ring3-periodic one-body seed Hamiltonian.
    For seed="random", the normal-seed spectrum is used only to fix the
    occupation counts n_occ(k); each occupied k-block is then replaced by an
    independent random complex orthonormal subspace.  This keeps the same
    momentum-sector occupation pattern as the normal variational family while
    removing hand-designed CO/BOW/LC bias from the initial orbitals.
    """
    sc, internal, kpoints = bloch_layout(g, l1, l2)
    nsc = int(l1) * int(l2)
    star = ring3_periodic_star_formfactor(g, l1, l2)
    seed_key=str(seed).lower()
    random_mode=(seed_key=="random")
    reference_seed="normal" if random_mode else seed
    h = one_body_seed_hamiltonian(g, reference_seed, seed_amplitude, star)

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
    rng=(
        np.random.default_rng(0 if random_seed is None else int(random_seed))
        if random_mode else None
    )
    for ik in range(len(kpoints)):
        ids = sorted(bands[ik])
        nocc=len(ids)
        counts.append(nocc)
        if nocc==0:
            block=np.empty((18,0),np.complex128)
        elif random_mode:
            z=(
                rng.normal(size=(18,nocc))
                +1j*rng.normal(size=(18,nocc))
            )
            block=np.linalg.qr(z,mode="reduced")[0].astype(np.complex128)
        else:
            block=np.asarray(evecs[ik][:,ids],np.complex128)
        blocks.append(block)

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
        "seed_mode":("random_complex_orthonormal" if random_mode else "named"),
        "random_seed":(
            int(0 if random_seed is None else random_seed)
            if random_mode else None
        ),
        "occupation_reference_seed":("normal" if random_mode else str(seed)),
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


def _sparse_jastrow_channels(shell_matrices: np.ndarray):
    """Compress symmetric Jastrow matrices to upper-triangle pair lists.

    For symmetric S, 0.5*dn^T S dn equals the sum over i<j of
    S_ij*dn_i*dn_j plus half of diagonal terms.  Current Ruby channels have
    zero diagonal, but the conversion keeps diagonal support for completeness.
    """
    mats=np.asarray(shell_matrices,np.float64)
    if mats.ndim!=3 or mats.shape[1]!=mats.shape[2]:
        raise ValueError("shell_matrices must have shape (n_channels,Ns,Ns)")
    if np.max(np.abs(mats-mats.transpose(0,2,1)))>1e-12:
        raise ValueError("Jastrow channel matrices must be symmetric")
    ch=[]; ii=[]; jj=[]; vv=[]
    for s,m in enumerate(mats):
        rows,cols=np.nonzero(np.triu(np.abs(m)>0,k=0))
        for i,j in zip(rows,cols):
            val=float(m[i,j])*(0.5 if i==j else 1.0)
            if val!=0.0:
                ch.append(s);ii.append(i);jj.append(j);vv.append(val)
    return (
        np.asarray(ch,np.int32),
        np.asarray(ii,np.int32),
        np.asarray(jj,np.int32),
        np.asarray(vv,np.float64),
    )


def _sparse_backflow_channels(backflow_matrices: np.ndarray):
    """Compress directed nonzero backflow adjacency entries.

    The dense formula sum_j B[k,i,j] dn[j] Q[j,:] is evaluated as a sparse
    scatter-add over only nonzero directed graph entries.
    """
    mats=np.asarray(backflow_matrices,np.float64)
    if mats.ndim!=3 or mats.shape[1]!=mats.shape[2]:
        raise ValueError("backflow_matrices must have shape (n_channels,Ns,Ns)")
    ch=[]; dst=[]; src=[]; vv=[]
    for k,m in enumerate(mats):
        rows,cols=np.nonzero(np.abs(m)>0)
        for i,j in zip(rows,cols):
            ch.append(k);dst.append(i);src.append(j);vv.append(float(m[i,j]))
    return (
        np.asarray(ch,np.int32),
        np.asarray(dst,np.int32),
        np.asarray(src,np.int32),
        np.asarray(vv,np.float64),
    )


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
    time_reversal_even: bool = False,
    initial_tr_phase: float = 0.0,
    sparse_correlators: bool = False,
    packed_orbitals: bool = False,
    orthonormalize_orbitals: bool = True,
):
    """Build a translation-exact ring3-supercell Slater-Jastrow-backflow model.

    If point_site_maps/characters are supplied, only that fixed-size point
    group is explicitly projected. Supercell translations are never summed
    over: they are exact in the Bloch orbital construction.

    If orthonormalize_orbitals=False, the trainable occupied blocks are used
    directly instead of QR-orthonormalising them on every model evaluation.
    This is an exact gauge choice for unprojected Slater-Jastrow-linear-
    backflow states: right-multiplying an occupied block by any invertible
    matrix changes the many-body amplitude only by a configuration-independent
    determinant factor.
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

    if sparse_correlators:
        j_ch_np,j_i_np,j_j_np,j_v_np=_sparse_jastrow_channels(shells_np)
        if bf_np is not None:
            bf_ch_np,bf_dst_np,bf_src_np,bf_v_np=_sparse_backflow_channels(bf_np)
        else:
            bf_ch_np=bf_dst_np=bf_src_np=bf_v_np=None
    else:
        j_ch_np=j_i_np=j_j_np=j_v_np=None
        bf_ch_np=bf_dst_np=bf_src_np=bf_v_np=None

    ns = len(internal_np)
    nf = int(sum(x.shape[1] for x in blocks0))
    nocc_np=np.asarray([x.shape[1] for x in blocks0],np.int32)
    offsets_np=np.concatenate(([0],np.cumsum(nocc_np))).astype(np.int32)
    packed0=(
        np.concatenate([x for x in blocks0 if x.shape[1]],axis=1)
        if nf else np.empty((18,0),np.complex128)
    )
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
            if packed_orbitals:
                def init_packed(key):
                    del key
                    return jnp.asarray(packed0)
                Mpacked=self.param("M_packed",init_packed)
                for ik,b0 in enumerate(blocks0):
                    nki=int(nocc_np[ik])
                    if nki==0:
                        params.append(None)
                    else:
                        lo=int(offsets_np[ik]);hi=int(offsets_np[ik+1])
                        params.append(Mpacked[:,lo:hi])
            else:
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

            if time_reversal_even:
                tr_phase = self.param(
                    "tr_phase",
                    lambda key: jnp.asarray(initial_tr_phase, dtype=jnp.float64),
                )
            else:
                tr_phase = None

            sc = jnp.asarray(sc_np)
            internal = jnp.asarray(internal_np)
            ks = jnp.asarray(k_np)
            shells = None if sparse_correlators else jnp.asarray(shells_np)
            backflow = (
                None if (bf_np is None or sparse_correlators)
                else jnp.asarray(bf_np)
            )
            if sparse_correlators:
                j_ch=jnp.asarray(j_ch_np); j_i=jnp.asarray(j_i_np)
                j_j=jnp.asarray(j_j_np); j_v=jnp.asarray(j_v_np)
                if bf_np is not None:
                    bf_ch=jnp.asarray(bf_ch_np)
                    bf_dst=jnp.asarray(bf_dst_np)
                    bf_src=jnp.asarray(bf_src_np)
                    bf_v=jnp.asarray(bf_v_np)
                else:
                    bf_ch=bf_dst=bf_src=bf_v=None
            else:
                j_ch=j_i=j_j=j_v=None
                bf_ch=bf_dst=bf_src=bf_v=None
            maps = None if maps_np is None else jnp.asarray(maps_np)
            chars = None if chars_np is None else jnp.asarray(chars_np)
            filling = nf / ns

            qparts = []
            for ik, M in enumerate(params):
                if M is None:
                    continue
                if orthonormalize_orbitals:
                    U, _ = jnp.linalg.qr(M, mode="reduced")
                else:
                    U = M
                phase = jnp.exp(
                    2j * jnp.pi * jnp.sum(sc * ks[ik][None, :], axis=1)
                ) / jnp.sqrt(jnp.asarray(float(nsc)))
                qparts.append(phase[:, None] * U[internal, :])
            Q = jnp.concatenate(qparts, axis=1)

            def base_logpsi(x):
                occupied = jnp.nonzero(x, size=nf)[0]
                dn = x - filling
                if bf_np is not None:
                    if sparse_correlators:
                        contrib=(
                            eta[bf_ch,None]
                            * bf_v[:,None]
                            * dn[bf_src,None]
                            * Q[bf_src,:]
                        )
                        dQ=jnp.zeros_like(Q).at[bf_dst,:].add(contrib)
                    else:
                        dQ = jnp.einsum(
                            "k,kij,j,ja->ia", eta, backflow, dn, Q
                        )
                    Qx = Q + dQ
                else:
                    Qx = Q
                A = Qx[occupied]
                out = nk.jax.logdet_cmplx(A)
                if use_jastrow and shells_np.shape[0] > 0:
                    if sparse_correlators:
                        vals=j_v*dn[j_i]*dn[j_j]
                        features=jnp.zeros(
                            (shells_np.shape[0],),dtype=dn.dtype
                        ).at[j_ch].add(vals)
                    else:
                        features = 0.5 * jnp.einsum(
                            "i,sij,j->s", dn, shells, dn
                        )
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
                sources = x[maps]
                logs = jax.vmap(base_logpsi)(sources)
                signs = jax.vmap(fock_sign)(maps, sources)
                shift = jnp.max(jnp.real(logs))
                amps = jnp.exp(logs - shift)
                z = jnp.sum(jnp.conj(chars) * signs * amps)
                if time_reversal_even:
                    # Spinless T=K in the occupation basis.  The trainable
                    # phase spans all TR-even real rays associated with the
                    # C3-projected state, without restoring translations or
                    # mirrors.
                    phase = jnp.exp(1j * tr_phase)
                    z = phase * z + jnp.conj(phase * z)
                tiny = jnp.asarray(1e-300 + 0j, dtype=z.dtype)
                return shift + jnp.log(jnp.where(jnp.abs(z) > 1e-300, z, tiny))

            n_arr=jnp.asarray(n)
            leading=n_arr.shape[:-1]
            flat=n_arr.reshape((-1,ns))
            vals=jax.vmap(one)(flat)
            return vals.reshape(leading)

    return SupercellBlochSlaterJastrow()


def make_fast_raw_local_energy_kernel(
    *,
    sc_coords,
    internal_sites,
    kpoints,
    occupations_per_k,
    shell_matrices,
    backflow_matrices,
    edge_i,
    edge_j,
    edge_t,
    edge_has_interaction,
    interaction_V,
    return_diagnostics=True,
    orthonormalize_orbitals=True,
):
    """Build a JAX Woodbury local-energy kernel for the unprojected Raw ansatz.

    This kernel is specialised to the translation-built-in
    Slater-Jastrow-linear-backflow state used by the large-supercell workflow.
    It is intentionally separate from the generic model path so it can first
    be validated against full log-wavefunction recomputation before becoming
    a production NetKet local-estimator backend.

    For one fermion hop s->d, linear backflow gives

        Q_BF(x') - Q_BF(x)
          = -W[:,s] Q[s,:] + W[:,d] Q[d,:],

    which has rank <= 2.  Keeping the root determinant row ordering and
    replacing the source row by the destination adds one more rank-1 update.
    Therefore every hopping determinant ratio is obtained from a 3x3
    determinant after one solve of the root Nf x Nf Slater matrix.

    Set return_diagnostics=False for the production scalar local-estimator
    kernel so XLA can eliminate ratio/source/destination outputs used only by
    the validation path.

    The fermionic sign of c_d^dag c_s is exactly the row-permutation sign
    between the replacement ordering and canonical sorted occupation order,
    so those two signs cancel in the kinetic local energy.  The returned
    per-edge ratio is therefore the replacement-order wavefunction ratio that
    multiplies the bare hopping t_ij directly.
    """
    import jax
    import jax.numpy as jnp

    sc_np=np.asarray(sc_coords,np.int32)
    internal_np=np.asarray(internal_sites,np.int32)
    k_np=np.asarray(kpoints,float)
    counts_np=np.asarray(occupations_per_k,np.int32)
    shells_np=np.asarray(shell_matrices,np.float64)
    bf_np=np.asarray(backflow_matrices,np.float64)
    ei_np=np.asarray(edge_i,np.int32)
    ej_np=np.asarray(edge_j,np.int32)
    et_np=np.asarray(edge_t,np.float64)
    ev_np=np.asarray(edge_has_interaction,np.float64)

    ns=int(len(internal_np))
    nf=int(np.sum(counts_np))
    nsc=int(len(np.unique(sc_np,axis=0)))
    offsets=np.concatenate(([0],np.cumsum(counts_np))).astype(np.int32)

    sc=jnp.asarray(sc_np)
    internal=jnp.asarray(internal_np)
    ks=jnp.asarray(k_np)
    shells=jnp.asarray(shells_np)
    bf=jnp.asarray(bf_np)
    edge_i_j=jnp.asarray(ei_np)
    edge_j_j=jnp.asarray(ej_np)
    edge_t_j=jnp.asarray(et_np)
    edge_v_j=jnp.asarray(ev_np)

    bf_ch_np,bf_dst_np,bf_src_np,bf_val_np=_sparse_backflow_channels(bf_np)
    j_ch_np,j_i_np,j_j_np,j_val_np=_sparse_jastrow_channels(shells_np)
    bf_ch=jnp.asarray(bf_ch_np)
    bf_dst=jnp.asarray(bf_dst_np)
    bf_src=jnp.asarray(bf_src_np)
    bf_val=jnp.asarray(bf_val_np)
    j_ch=jnp.asarray(j_ch_np)
    j_i=jnp.asarray(j_i_np)
    j_j=jnp.asarray(j_j_np)
    j_val=jnp.asarray(j_val_np)

    # Only these Jastrow matrix entries are required by a nearest-neighbour
    # hop.  Precompute channel features instead of materialising C(v) as a
    # dense Ns x Ns matrix for every parameter update.
    edge_shell=jnp.asarray(
        shells_np[:,ei_np,ej_np].T,
        dtype=jnp.float64,
    )
    diag_shell=jnp.asarray(
        np.diagonal(shells_np,axis1=1,axis2=2).T,
        dtype=jnp.float64,
    )

    filling=float(nf/ns)
    interaction_V=float(interaction_V)

    def assemble_Q(parameters):
        packed=parameters["M_packed"]
        qparts=[]
        for ik in range(len(counts_np)):
            nki=int(counts_np[ik])
            if nki==0:
                continue
            lo=int(offsets[ik]); hi=int(offsets[ik+1])
            M=packed[:,lo:hi]
            if orthonormalize_orbitals:
                U,_=jnp.linalg.qr(M,mode="reduced")
            else:
                U=M
            phase=jnp.exp(
                2j*jnp.pi*jnp.sum(sc*ks[ik][None,:],axis=1)
            )/jnp.sqrt(jnp.asarray(float(nsc)))
            qparts.append(phase[:,None]*U[internal,:])
        return jnp.concatenate(qparts,axis=1)

    def kernel(parameters,configurations):
        Q=assemble_Q(parameters)
        eta=parameters["eta"]
        v=parameters["v"]

        # W is kept dense only for the O(Nf*Nedge) Woodbury element lookups
        # below.  The expensive configuration-dependent Q_BF construction is
        # sparse and scales with the number of physical directed bonds.
        W=jnp.einsum("k,kij->ij",eta,bf)
        eye3=jnp.eye(3,dtype=Q.dtype)
        edge_C=edge_shell@v
        diag_C=diag_shell@v
        j_coeff=v[j_ch]*j_val

        def one(x):
            occupied=jnp.nonzero(x,size=nf)[0]
            dn=x-filling

            bf_contrib=(
                eta[bf_ch,None]
                *bf_val[:,None]
                *dn[bf_src,None]
                *Q[bf_src,:]
            )
            dQ=jnp.zeros_like(Q).at[bf_dst,:].add(bf_contrib)
            Qx=Q+dQ
            A=Qx[occupied]

            xi=x[edge_i_j]
            xj=x[edge_j_j]
            forward=(xi>0.5)&(xj<0.5)
            reverse=(xj>0.5)&(xi<0.5)
            active=forward|reverse
            src=jnp.where(forward,edge_i_j,edge_j_j)
            dst=jnp.where(forward,edge_j_j,edge_i_j)

            # Rank-2 backflow update restricted to the old occupied rows.
            u1=-W[occupied[:,None],src[None,:]]
            u2= W[occupied[:,None],dst[None,:]]

            # Rank-1 row replacement.  For inactive edges this can be zero;
            # their final contribution is masked out.
            u3=(occupied[:,None]==src[None,:]).astype(Q.dtype)
            Ue=jnp.stack((u1,u2,u3),axis=-1)  # (Nf,Nedge,3)

            # One LU factorisation of A with all 3*Nedge RHS at once.
            X=jnp.linalg.solve(A,Ue.reshape((nf,-1)))
            X=X.reshape((nf,len(ei_np),3)).transpose((1,0,2))

            qsrc=Q[src]
            qdst=Q[dst]
            d_src=(
                -W[src,src,None]*qsrc
                +W[src,dst,None]*qdst
            )
            d_dst=(
                -W[dst,src,None]*qsrc
                +W[dst,dst,None]*qdst
            )
            replacement=(
                (Qx[dst]+d_dst)
                -(Qx[src]+d_src)
            )
            Ve=jnp.stack((qsrc,qdst,replacement),axis=-1)  # (E,Nf,3)
            small=eye3+jnp.einsum("eni,enj->eij",Ve,X)
            det_ratio=jnp.linalg.det(small)

            # Exact Jastrow ratio for delta n = -e_src + e_dst.
            # For J=1/2 dn^T C dn, h=C dn.  The sparse upper-triangle
            # representation contributes to both endpoints; diagonal entries
            # are stored with the built-in 1/2 factor and therefore the two
            # scatter-adds correctly reconstruct C_ii dn_i.
            h=jnp.zeros((ns,),dtype=dn.dtype)
            h=h.at[j_i].add(j_coeff*dn[j_j])
            h=h.at[j_j].add(j_coeff*dn[j_i])
            delta_J=(
                -h[src]+h[dst]
                +0.5*(
                    diag_C[src]+diag_C[dst]
                    -2.0*edge_C
                )
            )
            replacement_ratio=det_ratio*jnp.exp(delta_J)

            kinetic=jnp.sum(
                jnp.where(
                    active,
                    edge_t_j*replacement_ratio,
                    jnp.asarray(0.0+0.0j,dtype=Q.dtype),
                )
            )
            diagonal=interaction_V*jnp.sum(
                edge_v_j*xi*xj
            )
            energy=diagonal+kinetic
            if return_diagnostics:
                return energy, replacement_ratio, active, src, dst
            return energy

        return jax.vmap(one)(configurations)

    return jax.jit(kernel)


def extract_supercell_parameters(
    parameters, nkpoints: int, occupations_per_k=None
) -> dict:
    """Convert a NetKet/Flax parameter tree to NumPy warm-start arrays."""
    blocks = []
    if "M_packed" in parameters:
        if occupations_per_k is None:
            raise ValueError(
                "occupations_per_k is required to unpack M_packed"
            )
        packed=np.asarray(parameters["M_packed"],np.complex128)
        counts=np.asarray(occupations_per_k,int)
        off=np.concatenate(([0],np.cumsum(counts)))
        for ik in range(int(nkpoints)):
            blocks.append(packed[:,off[ik]:off[ik+1]].copy())
    else:
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
        "tr_phase": (
            float(np.asarray(parameters["tr_phase"]))
            if "tr_phase" in parameters else None
        ),
    }

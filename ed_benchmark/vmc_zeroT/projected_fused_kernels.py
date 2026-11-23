"""Fused symmetry-projected Woodbury energy and LC local-current estimators.

Builds common Bloch orbitals Q, complex backflow W and real Jastrow C exactly
once per parameter batch, then visits each spatial image with a rank-three
Woodbury update.  The per-image hopping updates are mapped BACK to the same
physical edge.  Thus the local current estimators work even when a point-group
operation flips the current channel, and for the two-dimensional C3v E irrep.

For spinless T=K, the hopping energy numerator is TR even; current numerator
is TR odd.  Both rules are applied explicitly.  Does not change the ansatz.

The static image loop uses jax.lax.scan, so large intermediate arrays need
not be materialized for all symmetry images at once.
"""
from __future__ import annotations

import numpy as np


def make_fused_projected_kernel(
    *, seed_data, shell_matrices, backflow_matrices, edge_i, edge_j,
    edge_t, edge_has_interaction, interaction_V, site_maps, characters,
    w_same=None, w_opposite=None, time_reversal_even=False,
    orthonormalize_orbitals=True, return_currents=True,
):
    """Return jit(kernel(parameters, configurations) -> energy or (E,Js,Jo)).

    The returned current amplitudes are the *complex local estimators*:
        Jloc(x) = <x|J|psi> / <x|psi>.
    For Hermitian J, E[abs(Jloc)**2] equals <J**2> exactly.
    Requires nonzero projection amplitude at each root configuration.
    """
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)

    sc_np = np.asarray(seed_data['sc_coords'], dtype=np.int32)
    internal_np = np.asarray(seed_data['internal_sites'], dtype=np.int32)
    kpts_np = np.asarray(seed_data['kpoints'], dtype=np.float64)
    counts = np.asarray(seed_data['occupations_per_k'], dtype=np.int32)
    shells_np = np.asarray(shell_matrices, dtype=np.float64)
    backflow_np = np.asarray(backflow_matrices, dtype=np.float64)
    maps_np = np.asarray(site_maps, dtype=np.int32)
    chars_np = np.asarray(characters, dtype=np.complex128)
    ei_np = np.asarray(edge_i, dtype=np.int32)
    ej_np = np.asarray(edge_j, dtype=np.int32)
    ts_np = np.asarray(edge_t, dtype=np.float64)
    iv_np = np.asarray(edge_has_interaction, dtype=np.float64)
    ns = len(internal_np)
    nf = int(np.sum(counts))
    ne = len(ei_np)
    nsc = len(np.unique(sc_np, axis=0))
    assert maps_np.ndim == 2 and maps_np.shape[1] == ns
    assert chars_np.shape == (len(maps_np),)
    assert shells_np.shape[1:] == backflow_np.shape[1:] == (ns, ns)
    assert len(ts_np) == len(iv_np) == ne
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    inv_maps_np = np.argsort(maps_np, axis=1).astype(np.int32)

    # Compact directed backflow gather, independent of the current configuration.
    ch, dst, src = np.nonzero(np.transpose(backflow_np, (1, 2, 0)))
    # numpy nonzero above gives (destination,source,channel)
    ch_np = np.asarray(src, dtype=np.int32)
    dst_np = np.asarray(ch, dtype=np.int32)
    src_np = np.asarray(dst, dtype=np.int32)
    bf_val_np = backflow_np[ch_np, dst_np, src_np]

    ks = jnp.asarray(kpts_np)
    sc = jnp.asarray(sc_np)
    internal = jnp.asarray(internal_np)
    maps = jnp.asarray(maps_np)
    inverses = jnp.asarray(inv_maps_np)
    chars = jnp.asarray(chars_np)
    ei = jnp.asarray(ei_np)
    ej = jnp.asarray(ej_np)
    te = jnp.asarray(ts_np)
    ve = jnp.asarray(iv_np)
    shells = jnp.asarray(shells_np)
    BF = jnp.asarray(backflow_np)
    bf_ch = jnp.asarray(ch_np)
    bf_dst = jnp.asarray(dst_np)
    bf_src = jnp.asarray(src_np)
    bf_val = jnp.asarray(bf_val_np)
    ws = jnp.asarray(np.zeros(ne) if w_same is None else w_same)
    wo = jnp.asarray(np.zeros(ne) if w_opposite is None else w_opposite)
    filling = float(nf / ns)
    V = float(interaction_V)
    eye3 = jnp.eye(3, dtype=jnp.complex128)

    def build_shared(pars):
        packed = pars['M_packed']
        parts = []
        for ik, nki in enumerate(counts):
            if nki == 0:
                continue
            block = packed[:, int(offsets[ik]):int(offsets[ik + 1])]
            if orthonormalize_orbitals:
                block, _ = jnp.linalg.qr(block, mode='reduced')
            phase = jnp.exp(2j * jnp.pi * jnp.sum(sc * ks[ik][None, :], axis=1))
            parts.append((phase[:, None] / jnp.sqrt(float(nsc))) * block[internal, :])
        Q = jnp.concatenate(parts, axis=1)
        eta = pars['eta']
        W = jnp.einsum('k,kij->ij', eta, BF)
        C = jnp.einsum('k,kij->ij', pars['v'], shells)
        weights = eta[bf_ch] * bf_val
        return Q, W, C, weights

    def one_root(y, original, p, invp, Q, W, C, weights):
        occ = jnp.nonzero(y, size=nf)[0]
        dn = y - filling
        contrib = (weights[:, None] * dn[bf_src, None] * Q[bf_src, :])
        Qx = Q + jnp.zeros_like(Q).at[bf_dst, :].add(contrib)
        A = Qx[occ, :]
        det_phase, log_det = jnp.linalg.slogdet(A)
        J = 0.5 * (dn @ (C @ dn))
        mapped = p[occ]
        inversions = jnp.triu(mapped[:, None] > mapped[None, :], k=1)
        fock = jnp.where((jnp.sum(inversions) & 1) == 0, 1.0, -1.0)
        # Same physical directed hopping, represented in this symmetry image.
        forward = (original[ei] > 0.5) & (original[ej] < 0.5)
        reverse = (original[ej] > 0.5) & (original[ei] < 0.5)
        active = forward | reverse
        src_phys = jnp.where(forward, ei, ej)
        dst_phys = jnp.where(forward, ej, ei)
        src = invp[src_phys]
        dst = invp[dst_phys]

        u1 = -W[occ[:, None], src[None, :]]
        u2 = W[occ[:, None], dst[None, :]]
        u3 = (occ[:, None] == src[None, :]).astype(Q.dtype)
        U = jnp.stack((u1, u2, u3), axis=-1)
        X = jnp.linalg.solve(A, U.reshape((nf, -1)))
        X = X.reshape((nf, ne, 3)).transpose((1, 0, 2))
        qsrc = Q[src, :]
        qdst = Q[dst, :]
        dsrc = -W[src, src, None] * qsrc + W[src, dst, None] * qdst
        ddst = -W[dst, src, None] * qsrc + W[dst, dst, None] * qdst
        repl = Qx[dst, :] + ddst - Qx[src, :] - dsrc
        Ve = jnp.stack((qsrc, qdst, repl), axis=-1)
        small = eye3 + jnp.einsum('eni,enj->eij', Ve, X)
        r_det = jnp.linalg.det(small)
        h = C @ dn
        delta_J = (-h[src] + h[dst] +
                   0.5 * (C[src, src] + C[dst, dst] - C[src, dst] - C[dst, src]))
        ratio = jnp.where(active, r_det * jnp.exp(delta_J), 0.0 + 0j)
        kinetic = jnp.sum(te * ratio)
        direction = jnp.where(forward, 1.0, -1.0)
        same = 1j * jnp.sum(ws * direction * ratio)
        opposite = 1j * jnp.sum(wo * direction * ratio)
        return (log_det + J, det_phase * fock, kinetic, same, opposite)

    def one_image(x, p, invp, Q, W, C, weights):
        y = x[:, p]
        return jax.vmap(one_root, in_axes=(0, 0, None, None,
                                             None, None, None, None))(
            y, x, p, invp, Q, W, C, weights)

    @jax.jit
    def kernel(parameters, configurations):
        Q, W, C, weights = build_shared(parameters)
        x = jnp.asarray(configurations)
        x = x.reshape((-1, ns))
        def image_step(_, item):
            p, invp = item
            return None, one_image(x, p, invp, Q, W, C, weights)
        _, out = jax.lax.scan(image_step, None, (maps, inverses))
        logabs, detphase, kin, js, jo = out  # each (G, batch)
        shift = jnp.max(logabs, axis=0, keepdims=True)
        weights_img = jnp.conj(chars)[:, None] * detphase * jnp.exp(logabs - shift)
        denominator = jnp.sum(weights_img, axis=0)
        en_num = jnp.sum(weights_img * kin, axis=0)
        js_num = jnp.sum(weights_img * js, axis=0)
        jo_num = jnp.sum(weights_img * jo, axis=0)
        if time_reversal_even:
            phase = jnp.exp(1j * parameters['tr_phase'])
            denominator = phase * denominator + jnp.conj(phase * denominator)
            en_num = phase * en_num + jnp.conj(phase * en_num)
            # The current operator is odd under spinless T=complex conjugation.
            js_num = phase * js_num - jnp.conj(phase * js_num)
            jo_num = phase * jo_num - jnp.conj(phase * jo_num)
        safe = jnp.where(jnp.abs(denominator) > 1e-300, denominator, 1e-300 + 0j)
        diag = V * jnp.sum(ve[None, :] * x[:, ei] * x[:, ej], axis=-1)
        en = diag + en_num / safe
        if return_currents:
            return en, js_num / safe, jo_num / safe
        return en

    return kernel

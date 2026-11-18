"""C3-constrained orbital-matrix Slater ansatz for Hex18.

The occupied subspace is constrained to be invariant under ordinary C3 but is
NOT constrained by primitive translations. This permits translation-breaking
supercell patterns while avoiding eigendecomposition gradients and fixed
tangent approximations.

For the 18-site ring3 torus, each one-particle C3 eigenspace has dimension 6.
The variational orbitals are arbitrary complex subspaces inside those C3
eigenspaces. The occupation counts (n0,n1,n2) are discrete and are chosen from
the C3-symmetrized seed, optionally constrained to a requested many-body C3
character.
"""
from __future__ import annotations

from itertools import product
import numpy as np


def one_particle_c3_matrix(g):
    p = g.point_maps().get("C3")
    if p is None:
        raise RuntimeError("ordinary C3 unavailable")
    p = np.asarray(p, dtype=int)
    R = np.zeros((g.Ns, g.Ns), dtype=np.complex128)
    # U c_i^dag U^-1 = c_p(i)^dag => orbital coefficient q transforms as
    # (R q)_{p(i)} = q_i.
    for i, j in enumerate(p):
        R[int(j), int(i)] = 1.0
    return R


def c3_single_particle_bases(g, tol=1e-10):
    """Return orthonormal bases B_m with eigenvalue exp(-2*pi*i*m/3)."""
    R = one_particle_c3_matrix(g)
    bases = []
    lambdas = []
    I = np.eye(g.Ns, dtype=complex)
    for m in range(3):
        lam = np.exp(-2j*np.pi*m/3)
        P = (I + np.conj(lam)*R + np.conj(lam)**2*(R@R))/3.0
        # Hermitize projector against roundoff and extract its range.
        P = 0.5*(P+P.conj().T)
        w, U = np.linalg.eigh(P)
        B = U[:, w > 0.5]
        if np.linalg.norm(R@B-lam*B) > 100*tol:
            raise RuntimeError(f"C3 single-particle basis m={m} failed")
        bases.append(np.asarray(B, np.complex128))
        lambdas.append(complex(lam))
    if sum(B.shape[1] for B in bases) != g.Ns:
        raise RuntimeError("incomplete C3 single-particle decomposition")
    return bases, np.asarray(lambdas, complex), R


def manybody_c3_m_from_character(character):
    z = complex(character)
    candidates = [(abs(z-np.exp(-2j*np.pi*m/3)), m) for m in range(3)]
    return int(min(candidates)[1])


def manybody_c3_character_from_counts(counts):
    phase_m = sum(m*int(n) for m, n in enumerate(counts)) % 3
    return np.exp(-2j*np.pi*phase_m/3), int(phase_m)


def _all_counts(dims, nf, target_m=None):
    out = []
    for n0 in range(min(dims[0], nf)+1):
        for n1 in range(min(dims[1], nf-n0)+1):
            n2 = nf-n0-n1
            if n2 < 0 or n2 > dims[2]:
                continue
            counts = (n0,n1,n2)
            _, m = manybody_c3_character_from_counts(counts)
            if target_m is None or m == int(target_m):
                out.append(counts)
    return out


def c3_seed_subspace(g, Qseed, nf, target_manybody_m=None):
    """Closest C3-invariant occupied subspace to a seed projector.

    We C3-average the seed one-body projector and choose the rank-nf invariant
    subspace maximizing Tr(P_seedbar P_trial), optionally at fixed many-body
    C3 character.
    """
    Qseed = np.asarray(Qseed, complex)
    Qseed, _ = np.linalg.qr(Qseed, mode="reduced")
    bases, lambdas, R = c3_single_particle_bases(g)

    P0 = Qseed@Qseed.conj().T
    Pbar = (P0 + R@P0@R.conj().T + (R@R)@P0@(R.conj().T@R.conj().T))/3.0
    Pbar = 0.5*(Pbar+Pbar.conj().T)

    spectra = []
    eigvecs = []
    for B in bases:
        M = 0.5*(B.conj().T@Pbar@B + (B.conj().T@Pbar@B).conj().T)
        w, U = np.linalg.eigh(M)
        order = np.argsort(w)[::-1]
        spectra.append(np.asarray(w[order], float))
        eigvecs.append(np.asarray(U[:, order], complex))

    dims = tuple(B.shape[1] for B in bases)
    candidates = _all_counts(dims, nf, target_m=target_manybody_m)
    if not candidates:
        raise RuntimeError(
            f"No C3 occupation count for nf={nf}, target m={target_manybody_m}"
        )

    best = None
    for counts in candidates:
        score = sum(
            float(np.sum(spectra[m][:counts[m]])) for m in range(3)
        )
        if best is None or score > best[0]:
            best = (score, counts)
    score, counts = best

    A0 = []
    Qblocks = []
    for m, (B,U,n) in enumerate(zip(bases,eigvecs,counts)):
        A = np.asarray(U[:, :n], complex)
        A0.append(A)
        if n:
            Qblocks.append(B@A)
    Q0 = np.concatenate(Qblocks, axis=1) if Qblocks else np.zeros((g.Ns,0),complex)
    if Q0.shape != (g.Ns, nf):
        raise RuntimeError("C3 seed occupied-subspace shape mismatch")

    char, mb_m = manybody_c3_character_from_counts(counts)
    return {
        "bases": bases,
        "single_particle_characters": lambdas,
        "A0": A0,
        "Q0": Q0,
        "counts": tuple(int(x) for x in counts),
        "manybody_character": complex(char),
        "manybody_m": int(mb_m),
        "seed_overlap_score": float(score),
        "physical_real_dof": int(sum(
            2*n*(B.shape[1]-n) for B,n in zip(bases,counts)
        )),
    }


def build_c3_constrained_netket_model(
    seed_data,
    shell_matrices,
    use_jastrow=True,
    *,
    projection_site_maps=None,
    projection_characters=None,
):
    """C3-constrained Slater(-Jastrow), optionally followed by a full projector."""
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    import netket as nk

    bases_np = [np.asarray(B,np.complex128) for B in seed_data["bases"]]
    A0_np = [np.asarray(A,np.complex128) for A in seed_data["A0"]]
    shells_np = np.asarray(shell_matrices,np.float64)
    ns = bases_np[0].shape[0]
    nf = sum(A.shape[1] for A in A0_np)
    maps_np = None if projection_site_maps is None else np.asarray(projection_site_maps,np.int32)
    chars_np = None if projection_characters is None else np.asarray(projection_characters,np.complex128)
    projected = maps_np is not None
    if projected and chars_np.shape != (maps_np.shape[0],):
        raise ValueError("projection character/map mismatch")

    class C3ConstrainedSlaterJastrow(nn.Module):
        @nn.compact
        def __call__(self,n):
            blocks = []
            for m,(B,A0) in enumerate(zip(bases_np,A0_np)):
                if A0.shape[1] == 0:
                    continue
                A = self.param(f"A{m}", lambda key, x=A0: jnp.asarray(x))
                QA, RA = jnp.linalg.qr(A, mode="reduced")
                d = jnp.diag(RA)
                phase = jnp.where(jnp.abs(d)>1e-14,d/jnp.abs(d),1.0+0j)
                QA = QA*jnp.conj(phase)[None,:]
                blocks.append(jnp.asarray(B)@QA)
            Q = jnp.concatenate(blocks,axis=1)

            if use_jastrow:
                v = self.param(
                    "v",nn.initializers.zeros_init(),(shells_np.shape[0],),jnp.float64
                )
            else:
                v = None
            shells = jnp.asarray(shells_np)
            filling = nf/ns

            def base_logpsi(x):
                occ = jnp.nonzero(x,size=nf)[0]
                out = nk.jax.logdet_cmplx(Q[occ])
                if use_jastrow and shells.shape[0]>0:
                    dn=x-filling
                    feat=0.5*jnp.einsum("i,sij,j->s",dn,shells,dn)
                    out=out+jnp.dot(v,feat)
                return out

            if not projected:
                return jnp.vectorize(base_logpsi,signature="(n)->()")(n)

            maps=jnp.asarray(maps_np)
            chars=jnp.asarray(chars_np)

            def fock_sign(p,source):
                occupied=jnp.nonzero(source,size=nf)[0]
                mapped=p[occupied]
                cmp=mapped[:,None]>mapped[None,:]
                inv=jnp.sum(jnp.triu(cmp,k=1),dtype=jnp.int32)
                return jnp.where((inv&1)==0,1.0,-1.0)

            def projected_logpsi(x):
                sources=jax.vmap(lambda p:x[p])(maps)
                logs=jax.vmap(base_logpsi)(sources)
                signs=jax.vmap(fock_sign)(maps,sources)
                coeff=jnp.conj(chars)*signs
                shift=jnp.max(jnp.real(logs))
                z=jnp.sum(coeff*jnp.exp(logs-shift))
                tiny=jnp.asarray(1e-300+0j,dtype=z.dtype)
                z=jnp.where(jnp.abs(z)>1e-300,z,tiny)
                return shift+jnp.log(z)

            return jnp.vectorize(projected_logpsi,signature="(n)->()")(n)

    return C3ConstrainedSlaterJastrow()

"""Grand-canonical low-T Green G(k,iw) with *all* Lanczos Krylov operations
confined to many-body translation-momentum sectors.

ED initial states are low-lying per-(N,K) eigenstates. The c_k and c_k^dag
operators transfer between (N,K) and (N+/-1,K-/+k), and the corresponding
N+/-1 resolvents are computed within these reduced blocks, not in the full
fixed-particle Hilbert space. As in the legacy thermal Lanczos method, thermal
state truncation and finite Krylov depth require convergence checks.

This function returns the selected primitive 6x6 Green matrix only. It does
not reconstruct the entire supercell Green, which would negate the K selection
advantage.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import eigsh

from .momentum_ed import MomentumTorusED
from .thermal_lanczos_green import (
    ThermalLanczosOptions,
    _solve_mu_truncated,
)
from .model import RubyParameters
from .three_torus_benchmark import TorusGeometry


def _low_spectrum(H, k, tol, maxiter):
    D=H.shape[0]
    if not D:
        return np.empty(0),np.empty((0,0),dtype=complex)
    k=max(1,min(int(k),D))
    if D<=max(10,k+2):
        vals,vecs=np.linalg.eigh(H.toarray())
        return np.asarray(vals[:k].real),np.asarray(vecs[:,:k],complex)
    k=min(k,D-2)
    vals,vecs=eigsh(H,k=k,which="SA",tol=tol,maxiter=maxiter)
    order=np.argsort(vals)
    return np.asarray(vals[order].real),np.asarray(vecs[:,order],complex)


def _manybody_transition_vectors(
    source: MomentumTorusED, target: MomentumTorusED,
    src_kidx: int,dst_kidx: int,psi: np.ndarray,k,
    *,creation:bool,
) -> np.ndarray:
    """Return columns c^dag_k,b|psi> or c_k,b|psi> in target K sector."""
    k=np.asarray(k,dtype=float).reshape(2)
    geom=source.geom
    srcblock=source.blocks[src_kidx]
    dstblock=target.blocks[dst_kidx]
    x=np.asarray(srcblock.projector@np.asarray(psi,dtype=complex)).reshape(-1)
    source_bits=np.asarray(source.basis,dtype=np.uint64)
    dest=np.zeros((target.full_dimension,6),dtype=complex)
    norm=1/np.sqrt(float(geom.ncell))
    dest_solver=target.solver
    for c,R in enumerate(geom.cells):
        phase=np.exp(2j*np.pi*np.dot(k,R)) if creation else np.exp(-2j*np.pi*np.dot(k,R))
        for a in range(6):
            site=6*c+a
            occ=(source_bits>>np.uint64(site))&np.uint64(1)
            indices=np.flatnonzero(occ==0 if creation else occ!=0)
            if not len(indices):
                continue
            mask=np.uint64(1<<site)
            states=source_bits[indices]
            mapped=(states|mask) if creation else (states&~mask)
            if hasattr(dest_solver,"_dense_lookup") and dest_solver._dense_lookup is not None:
                outidx=dest_solver._dense_lookup[mapped.astype(np.int64)]
            elif hasattr(dest_solver,"index"):
                outidx=np.fromiter(
                    (dest_solver.index[int(s)] for s in mapped),
                    count=len(mapped),dtype=np.int64
                )
            else:
                outidx=dest_solver._state_indices(mapped)
            parity=np.bitwise_count(states & np.uint64((1<<site)-1)) & 1
            sign=1.0-2.0*np.asarray(parity,dtype=float)
            dest[np.asarray(outidx,dtype=np.int64),a]+=(
                norm*phase*sign*x[indices]
            )
    return np.asarray(dstblock.projector.getH()@dest,dtype=complex)


def _cross_resolvent(
    H, right:np.ndarray, left:np.ndarray,z:np.ndarray,
    *,max_steps:int,tol:float,
) -> tuple[np.ndarray,int]:
    """Lanczos v_i^dag (z-H)^-1 right for six left vectors."""
    v=np.asarray(right,dtype=complex).reshape(-1)
    L=np.asarray(left,dtype=complex)
    z=np.asarray(z,dtype=complex).reshape(-1)
    out=np.zeros((len(z),L.shape[1]),dtype=complex)
    vn=float(np.linalg.norm(v))
    if vn<=1e-15:
        return out,0
    q=v/vn
    old=np.zeros_like(q)
    bprev=0.
    al=[]
    be=[]
    projs=[]
    for j in range(max(int(max_steps),1)):
        projs.append(L.conj().T@q)
        hq=H@q
        alpha=float(np.vdot(q,hq).real)
        w=hq-alpha*q
        if j:
            w-=bprev*old
        b=float(np.linalg.norm(w))
        al.append(alpha)
        if b<float(tol) or j+1>=int(max_steps):
            break
        be.append(b)
        old,q=q,w/b
        bprev=b
    tri=np.diag(np.asarray(al,float))
    if len(al)>1:
        off=np.asarray(be[:len(al)-1])
        tri+=np.diag(off,1)+np.diag(off,-1)
    energies,U=np.linalg.eigh(tri)
    p=np.asarray(projs,dtype=complex).T@U
    numer=p*(U[0][None,:]*vn)
    for j,w in enumerate(z):
        out[j]=np.sum(numer/(w-energies)[None,:],axis=1)
    return out,len(al)


@dataclass
class MomentumGreenResult:
    G: np.ndarray
    mu: float
    average_particles: float
    kept_thermal_weight: float
    sector_numbers: np.ndarray
    sector_probabilities: np.ndarray
    retained_states_per_sector: np.ndarray
    sector_dimensions_by_N: dict[int,list[int]]
    krylov_steps_mean: float
    krylov_steps_max: int
    n_retained_states: int
    source_momentum: tuple[float,float]


def momentum_thermal_lanczos_green(
    geom:TorusGeometry,params:RubyParameters,*,V:float,T:float,
    target_particles:float,omega:np.ndarray,K,
    opts:ThermalLanczosOptions=ThermalLanczosOptions(),
) -> MomentumGreenResult:
    if T<=0:
        raise ValueError("T must be positive")
    kv=geom.validate_momentum(K)
    Nsite=geom.nsites
    Ncenter=int(round(float(target_particles)))
    # Vacuum and full states are legitimate thermal sectors. Their
    # annihilation/addition transitions respectively vanish identically;
    # never instantiate an invalid N=-1 or N=Nsite+1 sector.
    Nlo=max(0,Ncenter-int(opts.sector_padding))
    Nhi=min(Nsite,Ncenter+int(opts.sector_padding))
    Ns=list(range(Nlo,Nhi+1))
    spaces:dict[int,MomentumTorusED]={}
    def space(N:int):
        N=int(N)
        if N not in spaces:
            spaces[N]=MomentumTorusED(geom,params,N)
        return spaces[N]

    # Exact spatial momentum sector labeling, low-T eigenstate truncation.
    levels={}
    spectra={}
    retained_by_sector={}
    for N in Ns:
        s=space(N)
        arr=[]
        for ik,b in enumerate(s.blocks):
            if b.dimension==0:
                continue
            ev,vec=_low_spectrum(
                s.hamiltonian(V,ik),int(opts.thermal_eigs),
                opts.eig_tol,opts.eig_maxiter
            )
            for j,e in enumerate(ev):
                arr.append((float(e),ik,np.asarray(vec[:,j],dtype=complex)))
        arr.sort(key=lambda x:x[0])
        if not arr:
            raise RuntimeError(f"empty low-energy states for N={N}")
        levels[N]=arr
        spectra[N]=np.asarray([x[0] for x in arr],dtype=float)
        if opts.verbose:
            print(f"[momentum G] N={N}: dims={[x.dimension for x in s.blocks]}, low-state count={len(arr)}",flush=True)
    mu, probs, Navg=_solve_mu_truncated(
        spectra,target_particles,T
    )
    prob_N=np.asarray([float(np.sum(probs[N])) for N in Ns])
    entries=[]
    for N in Ns:
        for j,prob in enumerate(probs[N]):
            entries.append((float(prob),N,j))
    entries.sort(key=lambda x:x[0],reverse=True)
    kept=[]
    cumulative=0.
    for p,N,j in entries:
        kept.append((p,N,j))
        cumulative+=p
        if cumulative>=1.-opts.thermal_discard_weight_tol:
            break

    G=np.zeros((len(omega),6,6),dtype=complex)
    steps=[]
    retained_per_N={N:0 for N in Ns}
    zadd=1j*np.asarray(omega,dtype=float)
    for p,N,j in kept:
        if p<=0:
            continue
        retained_per_N[N]+=1
        E,ik,psi=levels[N][j]
        src=space(N)
        kold=np.asarray(src.k_values[ik],dtype=float)
        if opts.verbose:
            print(f"[momentum G] source (N={N}, K={src.k_values[ik]},E={E:+.6g}) p={p:.3e}",flush=True)

        if N<Nsite:
            ka=src.k_index(kold+kv)
            plus=space(N+1)
            Ha=plus.hamiltonian(V,ka)
            va=_manybody_transition_vectors(
                src,plus,ik,ka,psi,kv,creation=True
            )
            for b in range(6):
                a,st=_cross_resolvent(
                    Ha,va[:,b],va,zadd+E+mu,
                    max_steps=opts.krylov_dim,tol=opts.krylov_tol
                )
                G[:,:,b]+=p*a
                steps.append(st)

        if N>0:
            kr=src.k_index(kold-kv)
            minus=space(N-1)
            Hr=minus.hamiltonian(V,kr)
            vr=_manybody_transition_vectors(
                src,minus,ik,kr,psi,kv,creation=False
            )
            for b in range(6):
                rem,st=_cross_resolvent(
                    Hr,vr[:,b],vr,E-mu-zadd,
                    max_steps=opts.krylov_dim,tol=opts.krylov_tol
                )
                G[:,b,:]-=p*rem
                steps.append(st)

    st=np.asarray(steps,int)
    return MomentumGreenResult(
        G=G,mu=float(mu),average_particles=float(Navg),
        kept_thermal_weight=float(cumulative),
        sector_numbers=np.asarray(Ns,int),
        sector_probabilities=prob_N,
        retained_states_per_sector=np.asarray(
            [retained_per_N[N] for N in Ns],dtype=int
        ),
        sector_dimensions_by_N={
            N:[b.dimension for b in spaces[N].blocks] for N in spaces
        },
        krylov_steps_mean=float(np.mean(st)) if len(st) else 0.,
        krylov_steps_max=int(np.max(st)) if len(st) else 0,
        n_retained_states=len(kept),
        source_momentum=tuple(map(float,kv))
    )

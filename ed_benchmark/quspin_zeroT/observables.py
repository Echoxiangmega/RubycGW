"""Fixed-N Fock-space observables; independent of QuSpin's symmetry sectors.

Site 0 is the most significant bit in QuSpin's general fermion basis.
Every fermionic operator is applied right-to-left with the usual sign from
occupied sites of LOWER site index. An optional self-check against QuSpin's
one-body matrix is performed by the runner when available.
"""
import numpy as np
from itertools import combinations
from scipy.linalg import eigh_tridiagonal


def apply_ops(ket, ket_states, bra_states, Ns, ops):
    """Apply ordered [(creation? (+/-), site),...] to a many-body vector.
    Operators are written from left to right, acted on from right to left.
    ket_states / bra_states use the QuSpin integer bit convention.
    """
    out=np.zeros(len(bra_states),dtype=np.complex128)
    index={int(s):i for i,s in enumerate(bra_states)}
    active=np.flatnonzero(np.abs(ket)>1.e-16)
    for a in active:
        state=int(ket_states[a]); sign=1; new=state
        for what,site in reversed(ops):
            bit=1 << (Ns-1-site)
            filled=(new&bit)!=0
            if (what=='+' and filled) or (what=='-' and not filled):
                sign=0;break
            # parity from occupied creation operators earlier in canonical order
            lower_sites=new >> (Ns-site)
            if lower_sites.bit_count()%2:sign=-sign
            if what=='+':new|=bit
            else:new^=bit
        if sign:
            j=index.get(new)
            if j is not None:out[j]+=sign*ket[a]
    return out


def op_sum(ket, states, Ns, terms):
    """terms = [(complex coefficient, [(+/- ,site),...]),...]."""
    out=np.zeros_like(ket,dtype=np.complex128)
    for coeff,ops in terms:
        out+=coeff*apply_ops(ket,states,states,Ns,ops)
    return out


def current_terms(g,loop):
    """Oriented particle-current sum / loop length (hbar=1)."""
    terms=[]
    for i,j in zip(loop,loop[1:]+loop[:1]):
        e,sign=g._directed_edge(i,j)
        if e is None:raise ValueError(f'Non-bond in loop {i}-{j}')
        t=g.t[e.kind]/len(loop)
        terms.extend([(1j*t,[('+',i),('-',j)]),(-1j*t,[('+',j),('-',i)])])
    return terms


def build_observables(g, psi, states, pairs=True):
    """Return equal-time data: rho, density covariance, loop-current gram,
    and optionally spinless nearest-neighbour pair correlation eigenvalues.
    No finite-size spontaneous TR-breaking order is assumed.
    """
    Ns=g.Ns
    assert len(states)==len(psi)
    n=np.array([[(int(s) >> (Ns-1-j))&1 for j in range(Ns)] for s in states],dtype=np.float64)
    prob=np.abs(psi)**2
    density=prob@n
    nn=n.T@(prob[:,None]*n)
    conn=nn-np.outer(density,density)
    rho=np.empty((Ns,Ns),dtype=np.complex128)
    for i in range(Ns):
        for j in range(Ns):
            if i==j:rho[i,j]=density[i]
            else:rho[i,j]=np.vdot(psi,apply_ops(psi,states,states,Ns,[('+',i),('-',j)]))
    assert np.allclose(rho,rho.conj().T,atol=1e-8)
    loops=g.cycles()
    vec=[];lc=[];loop_flux=[]
    for loop in loops:
        terms=current_terms(g,loop)
        x=op_sum(psi,states,Ns,terms)
        vec.append(x)
        lc.append(np.vdot(psi,x))
        ps=np.prod([rho[i,j] for i,j in zip(loop,loop[1:]+loop[:1])])
        loop_flux.append(np.angle(ps) if abs(ps)>1.e-12 else np.nan)
    C=np.array([[np.vdot(v,w) for w in vec] for v in vec],dtype=np.complex128)
    E=np.linalg.eigvalsh(C) if len(vec) else np.array([])
    result=dict(density=density,nn=nn,density_connected=conn,rho=rho,
                loops=np.array([list(l)+[-1]*(6-len(l)) for l in loops],dtype=int),
                loop_length=np.array([len(l) for l in loops]),
                current_mean=np.array(lc),loop_rho_flux=np.array(loop_flux),
                current_corr=C,current_eigs=E,
                charge_eigs=np.linalg.eigvalsh(conn))
    if pairs:
        Nf=int(states[0]).bit_count()
        if Nf<2: raise ValueError('Pair correlation requires Nf>=2')
        if Ns>18:
            raise ValueError('Pair correlation for 24-site cluster is opt-in disabled by default')
        # include all distinct nearest-neighbour bonds; spinless onsite pairs vanish
        edge_pairs=sorted({tuple(sorted((e.i,e.j))) for e in g.edges})
        # N-2 basis ordered numerically descending, consistent with QuSpin
        lower_states=np.array([sum(1 << (Ns-1-i) for i in occ)
                   for occ in combinations(range(Ns), Nf-2)],dtype=np.uint64)
        lower_states=np.sort(lower_states)[::-1]
        delta=np.column_stack([apply_ops(psi,states,lower_states,Ns,[('-',i),('-',j)])
                               for i,j in edge_pairs])
        pairC=delta.conj().T@delta
        result.update(pair_bonds=np.array(edge_pairs,dtype=int),pair_corr=pairC,
                      pair_eigs=np.linalg.eigvalsh(pairC))
    return result


def max_current_in_manifold(g, psi_states, states, delta_energies, window=1.e-8):
    """Maximize <J> by diagonalizing J within a *specified low-energy* manifold.
    Array psi_states has column vectors in the SAME fixed-N basis.
    The result cannot be called a spontaneous order unless a controlled
    near-degenerate manifold, finite-size scaling or h->0 analysis supports it.
    """
    mask=np.asarray(delta_energies)<=window+1.e-12
    vs=np.asarray(psi_states)[:,mask]
    if vs.ndim!=2 or not len(vs.T):raise ValueError('Empty manifold')
    # Numerical orthogonalization needed when states come from different blocks
    vs,_=np.linalg.qr(vs)
    loops=g.cycles()
    result=[]
    for loop in loops:
        Jvs=np.column_stack([op_sum(v,states,g.Ns,current_terms(g,loop)) for v in vs])
        m=vs.conj().T@Jvs
        m=.5*(m+m.conj().T)
        values,U=np.linalg.eigh(m)
        arg=int(np.argmax(np.abs(values)))
        result.append((len(loop), list(loop), float(values[arg]), float(np.max(np.abs(values))),
                       U[:,arg].tolist()))
    return result


def lanczos_spectral(H, start, e0, m=128, tol=1.e-11):
    """Non-reorthogonalized Lanczos spectral approximation (small systems).
    Returns poles E_j-e0, normalized positive weights for a given |start>.
    For cross Green functions use a block/overlap Lanczos extension later.
    """
    norm=np.linalg.norm(start)
    if norm<1.e-14:return np.array([]),np.array([])
    v=start/norm; prev=np.zeros_like(v); beta=0.
    aa=[];bb=[]
    for k in range(m):
        w=H.dot(v)-beta*prev
        alpha=float(np.vdot(v,w).real)
        w-=alpha*v
        if k: w-=0.0*prev
        # three-term recurrence, no full reorthogonalization: converged poles
        # may duplicate for very long runs; limit m and validate moments.
        newbeta=float(np.linalg.norm(w))
        aa.append(alpha)
        if newbeta<tol:break
        if k<m-1:bb.append(newbeta)
        prev,v=v,w/newbeta
        beta=newbeta
    e,U=eigh_tridiagonal(np.array(aa),np.array(bb))
    return e-e0, (norm**2)*(np.abs(U[0,:])**2)
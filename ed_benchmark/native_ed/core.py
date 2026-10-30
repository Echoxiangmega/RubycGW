"""NumPy/SciPy-only fixed-N, spinless fermion ED on Ruby PBC clusters.

Basis ordering: ascending integers, site i occupies bit (Ns-1-i).  Each
undirected hopping appears once and adds its Hermitian conjugate.  This file
contains no QuSpin imports and is independent of third-party binary ED code.
"""
from itertools import combinations
from math import comb
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh, LinearOperator


def make_states(ns, nf):
    if not 0 <= nf <= ns:
        raise ValueError('Nf must be between 0 and Ns')
    if ns > 62:
        raise ValueError('uint64 implementation supports at most 62 sites')
    states = np.fromiter((sum(1 << (ns-1-i) for i in occ)
                          for occ in combinations(range(ns), nf)),
                         dtype=np.uint64, count=comb(ns,nf))
    states.sort()
    return states


def fermionic_parity(values):
    """Vectorized parity of popcount(uint64), without NumPy-2-only APIs."""
    x = np.asarray(values, dtype=np.uint64).copy()
    for shift in (32,16,8,4):
        x ^= (x >> np.uint64(shift))
    return ((np.uint64(0x6996) >> (x & np.uint64(15))) & np.uint64(1)).astype(np.int8)


def edge_triplets(g, states, i_edge):
    e = g.edges[i_edge]
    ns=g.Ns
    bi=np.uint64(1 << (ns-1-e.i))
    bj=np.uint64(1 << (ns-1-e.j))
    between = sum(1 << (ns-1-k) for k in range(min(e.i,e.j)+1,max(e.i,e.j)))
    mask = ((states & bi)!=0) ^ ((states & bj)!=0)
    col=np.flatnonzero(mask)
    target=states[col] ^ (bi|bj)
    row=np.searchsorted(states, target)
    if np.any(row >= len(states)) or np.any(states[row]!=target):
        raise AssertionError('Invalid particle-number-conserving hopping')
    sign = 1-2*fermionic_parity(states[col] & np.uint64(between))
    return row.astype(np.int64),col.astype(np.int64),sign.astype(np.int8)


def interaction_diagonal(g,states,V):
    d=np.zeros(len(states),dtype=np.float64)
    if V==0:return d
    for e in g.edges:
        if e.kind!='t':continue
        a=np.uint64(1<<(g.Ns-1-e.i));b=np.uint64(1<<(g.Ns-1-e.j))
        d+=float(V)*(((states&a)!=0)&((states&b)!=0))
    return d


class FixedHamiltonian:
    """Create an explicit CSR operator, or a lower-memory matrix-free one.

    The sparse mode is fastest for 18-site benchmarks; matrixfree avoids
    allocating all nonzero hopping transitions for larger fixed-N sectors.
    """
    def __init__(self,g,nf,V,states=None,method='sparse',site_gauge=None):
        self.g,self.nf,self.V=g,nf,float(V)
        self.states=make_states(g.Ns,nf) if states is None else np.asarray(states,dtype=np.uint64)
        self.diagonal=interaction_diagonal(g,self.states,V)
        self.n=len(self.states)
        self.method=method
        self.site_gauge=site_gauge
        if method=='sparse':
            rows=[np.arange(self.n,dtype=np.int64)]
            cols=[np.arange(self.n,dtype=np.int64)]
            vals=[self.diagonal.astype(complex)]
            for ie,e in enumerate(g.edges):
                row,col,sign=edge_triplets(g,self.states,ie)
                amp=complex(g.t[e.kind])
                if site_gauge is not None:
                    amp*=site_gauge[e.i]*np.conj(site_gauge[e.j])
                # column with occupation at i: c^dag_i c_j is NOT allowed;
                # instead use the Hamiltonian matrix direction based on bit occupancy.
                col_i=((self.states[col] & np.uint64(1 << (g.Ns-1-e.i)))!=0)
                amp_for_col=np.where(col_i,np.conj(amp),amp)
                rows.append(row);cols.append(col)
                vals.append(sign*amp_for_col)
            self.operator=coo_matrix((np.concatenate(vals),
                (np.concatenate(rows),np.concatenate(cols))),
                shape=(self.n,self.n)).tocsr()
            self.operator.sum_duplicates()
            self.operator.eliminate_zeros()
        elif method=='matrixfree':
            def matvec(x):
                x=np.asarray(x)
                out=np.asarray(self.diagonal*x,dtype=np.complex128)
                # Compute transitions by vectorized bit shifts; O(Ns*D)
                for ie,e in enumerate(g.edges):
                    row,col,sign=edge_triplets(g,self.states,ie)
                    amp=complex(g.t[e.kind])
                    if site_gauge is not None:
                        amp*=site_gauge[e.i]*np.conj(site_gauge[e.j])
                    col_i=((self.states[col] & np.uint64(1<<(g.Ns-1-e.i)))!=0)
                    weights=np.where(col_i,np.conj(amp),amp)*sign
                    out[row]+=weights*x[col]
                return out
            self.operator=LinearOperator((self.n,self.n),matvec=matvec,dtype=np.complex128)
        else:raise ValueError('method must be sparse or matrixfree')


def lowest(operator,k=3):
    n=operator.shape[0]
    if n==0:return np.array([]),np.empty((0,0),complex)
    if n<=max(k+1,3):
        dense=operator.toarray() if hasattr(operator,'toarray') else np.column_stack(
            [operator.matvec(np.eye(n)[:,i]) for i in range(n)])
        w,u=np.linalg.eigh(dense)
        return w[:k],u[:,:k]
    w,u=eigsh(operator,k=min(k,n-2),which='SA',tol=1e-10,maxiter=max(3000,8*n))
    idx=np.argsort(w)
    return w[idx],u[:,idx]


def full_hermiticity_error(H,n_tests=2):
    rng=np.random.default_rng(1301)
    err=0.
    for _ in range(n_tests):
        a=rng.normal(size=H.shape[0])+1j*rng.normal(size=H.shape[0])
        b=rng.normal(size=H.shape[0])+1j*rng.normal(size=H.shape[0])
        z=np.vdot(a,H@b)-np.vdot(H@a,b)
        err=max(err,abs(z)/(np.linalg.norm(a)*np.linalg.norm(b)))
    return err
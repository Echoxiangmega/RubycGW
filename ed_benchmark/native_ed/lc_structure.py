"""Exact triangle-current correlation matrices and momentum decomposition.

Native Ruby ED, spinless. CCW triangular loop particle currents with the
existing +t hopping convention and a current averaged over the 3 loop edges.
All finite-cluster observables use the normalized *uniform mixture* over the
complete ground-state multiplet, not an arbitrary vector in that subspace.

C[p,q] = Tr(P0 I_p I_q)/d - Tr(P0 I_p)/d Tr(P0 I_q)/d.
I_{R,A/B}(q) = sum_R exp(-i q.R) I_{R,A/B}/sqrt(Nc).
S_ab(q) = <delta I_a(q)^dag delta I_b(q)>.
With this convention at Gamma, (1,+1)/sqrt(2) and (1,-1)/sqrt(2)
are the same/opposite physical triangle currents in scan_susceptibility.py.
"""
from __future__ import annotations

from itertools import product
import numpy as np


def triangle_index(g):
    """Return physical-CCW triangles sorted by cell, A then B."""
    loops=[tuple(int(x) for x in p) for p in g.cycles() if len(p)==3]
    expected=2*g.num_cells
    if len(loops)!=expected:
        raise AssertionError(f'Expected {expected} distinct triangles, got {len(loops)}')
    ordered=sorted(loops,key=lambda p:(min(p)//6,0 if min(p)%6<3 else 1))
    cells=np.array([min(p)//6 for p in ordered],dtype=int)
    sub=np.array([0 if min(p)%6<3 else 1 for p in ordered],dtype=int)
    for k,p in enumerate(ordered):
        if len({i//6 for i in p})!=1:
            raise AssertionError('A triangle must be contained in one primitive cell')
        if all(i%6<3 for i in p) != (sub[k]==0):
            raise AssertionError('Mixed A/B triangle loop')
    for c in range(g.num_cells):
        assert np.array_equal(sub[cells==c],np.array([0,1]))
    return ordered,cells,sub


def one_triangle_action(g,states,psi,loop):
    """Apply one averaged CCW triangle-current operator without Python Fock loops.

    edge_triplets encodes the fermion sign. Every undirected edge appears once
    in the canonical hopping table; a reversed loop edge changes current sign.
    """
    from core import edge_triplets
    states=np.asarray(states,dtype=np.uint64)
    psi=np.asarray(psi,dtype=np.complex128)
    y=np.zeros(len(states),dtype=np.complex128)
    edge_lookup={frozenset((e.i,e.j)):i for i,e in enumerate(g.edges)}
    for i,j in zip(loop,loop[1:]+loop[:1]):
        eidx=edge_lookup.get(frozenset((i,j)))
        if eidx is None:raise ValueError(f'Non-bond triangle edge {i}-{j}')
        e=g.edges[eidx]
        direction=1 if (e.i,e.j)==(i,j) else -1
        row,col,fermion_sign=edge_triplets(g,states,eidx)
        occupied_at_j=(states[col] & np.uint64(1<<(g.Ns-1-e.j))) != 0
        # On the representative edge, i->j particle current is
        # +i t c^dag_i c_j - i t c^dag_j c_i, with hbar=1.
        factors=1j*direction*g.t[e.kind]/len(loop)*fermion_sign
        y[row]+=factors*np.where(occupied_at_j,1.,-1.)*psi[col]
    return y


def lc_correlations(g,states,psis,check=True):
    """Ground-mixture C, ground-zero-mode C0, and finite-excitation Cfinite.

    Returns full connected (2Nc)x(2Nc) matrices, not just gamma patterns.
    A density matrix P0/d makes results independent of numerical eigenvector
    rotations inside exactly degenerate ground multiplets.
    """
    states=np.asarray(states,dtype=np.uint64)
    psis=np.asarray(psis,dtype=np.complex128)
    if psis.ndim!=2 or psis.shape[0]!=len(states):raise ValueError('Invalid ground vectors')
    d=psis.shape[1]
    if d<1:raise ValueError('Empty ground space')
    if not np.allclose(psis.conj().T@psis,np.eye(d),atol=1e-8):
        raise ValueError('Ground multiplet is not orthonormal')
    loops,cells,sub=triangle_index(g)
    nt=len(loops)
    full=np.zeros((nt,nt),complex)
    M=np.zeros((nt,d,d),complex)
    for ia in range(d):
        active=np.column_stack([one_triangle_action(g,states,psis[:,ia],p) for p in loops])
        full+=active.conj().T@active/d
        M[:,:,ia]=(psis.conj().T@active).T  # p,b,a = <b|I_p|a>
    M=(M+M.conj().transpose(0,2,1))/2
    mu=np.trace(M,axis1=1,axis2=2).real/d
    full=np.asarray(full-np.outer(mu,mu),complex)
    full=(full+full.conj().T)/2
    zero=np.einsum('pab,qba->pq',M,M,optimize=True)/d-np.outer(mu,mu)
    zero=(zero+zero.conj().T)/2
    finite=(full-zero)
    finite=(finite+finite.conj().T)/2
    if check:
        for matrix,name in ((full,'connected'),(zero,'ground zero-mode'),(finite,'finite-energy')):
            err=np.max(np.abs(matrix-matrix.conj().T))
            if err>1e-8:raise AssertionError(f'{name} not Hermitian, err={err}')
            if np.min(np.linalg.eigvalsh(matrix))<-5e-8:
                raise AssertionError(f'{name} is not positive semidefinite')
        if not np.allclose(full,zero+finite,atol=1e-8):raise AssertionError('Sum of spectral contributions failed')
    return {'loops':loops,'cells':cells,'sublattice':sub,'mean':mu,'C':full,'C_zero':zero,
            'C_finite':finite,'projected_current_matrices':M,'ground_dim':d}


def momentum_structure(g,result):
    """Fourier project the A/B triangle-current matrix at all PBC momenta.

    q in primitive reciprocal coordinates, q.R=2pi*(q1*R1+q2*R2).
    All matrices have current^2 units and order (A,B). Full S is PSD;
    S_nonlocal excludes diagonal p=q current variances and is NOT PSD.
    """
    nc=g.num_cells
    cell=np.asarray(result['cells'],int)
    sub=np.asarray(result['sublattice'],int)
    qs=np.asarray(g.permitted_momenta(),float)
    matrices={'S':result['C'],'S_zero':result['C_zero'],
              'S_finite':result['C_finite'],
              'S_nonlocal':result['C']-np.diag(np.diag(result['C']))}
    out={k:[] for k in matrices}
    eigmax=[];eigvec=[]
    for q in qs:
        F=np.zeros((2*nc,2),complex)
        for ip,(ci,ab) in enumerate(zip(cell,sub)):
            phi=2*np.pi*np.dot(q,np.asarray(g.reps[int(ci)],float))
            F[ip,ab]=np.exp(1j*phi)/np.sqrt(nc)
        for key,mat in matrices.items():
            z=F.conj().T@mat@F
            out[key].append((z+z.conj().T)/2)
        w,u=np.linalg.eigh(out['S'][-1]);eigmax.append(w[-1]);eigvec.append(u[:,-1])
    data={k:np.array(v) for k,v in out.items()}
    data.update(q=qs,eigenvalue_max=np.array(eigmax),eigenvector_max=np.array(eigvec))
    idx=np.argmin(np.linalg.norm(qs,axis=1))
    if np.linalg.norm(qs[idx])>1e-10:raise AssertionError('Gamma not in momentum grid')
    plus=np.array([1,1],complex)/np.sqrt(2)
    minus=np.array([1,-1],complex)/np.sqrt(2)
    for key in ('S','S_zero','S_finite','S_nonlocal'):
        data[key+'_same_gamma']=float(np.vdot(plus,data[key][idx]@plus).real)
        data[key+'_opposite_gamma']=float(np.vdot(minus,data[key][idx]@minus).real)
    if not np.allclose(np.sum(data['S'],axis=0).trace(),np.trace(result['C']),atol=1e-7):
        raise AssertionError('Fourier sum rule for connected current matrix failed')
    return data


def ground_jmax(g,result):
    """Exactly the same normalized same/opposite projected LC sources as ED scan."""
    M=result['projected_current_matrices']
    sub=result['sublattice']
    N=len(sub)
    out={}
    for key,sign in [('same',np.ones(N)),('opposite',np.where(sub==0,1.,-1.))]:
        O=np.einsum('p,pab->ab',sign/np.sqrt(N),M)
        O=(O+O.conj().T)/2
        w,u=np.linalg.eigh(O)
        out[key]={'matrix':O,'eigenvalues':w,'Jmax_abs':float(np.max(np.abs(w))),
                  'Jmax_per_triangle':float(np.max(np.abs(w))/np.sqrt(N)),
                  'zero_weight':float(np.vdot(O-np.trace(O)/len(w)*np.eye(len(w)),
                                                O-np.trace(O)/len(w)*np.eye(len(w))).real/len(w))}
    return out


def pairwise_nonlocal(g,result):
    """Minimum-image distances and opposite-staggered connected correlations."""
    loops=result['loops'];C=result['C'];sub=result['sublattice']
    centers=np.array([g.positions[np.asarray(p)].mean(axis=0) for p in loops]);rows=[]
    translations=[g.B@np.asarray([u,v],float) for u,v in product(range(-2,3),repeat=2)]
    for i in range(len(loops)):
        for j in range(i+1,len(loops)):
            d=min(np.linalg.norm(centers[j]+shift-centers[i]) for shift in translations)
            signs=(1 if sub[i]==0 else -1)*(1 if sub[j]==0 else -1)
            rows.append({'p':i,'q':j,'sublattice_p':'A' if sub[i]==0 else 'B',
                         'sublattice_q':'A' if sub[j]==0 else 'B',
                         'distance':float(d),'corr_real':float(np.real(C[i,j])),
                         'corr_imag':float(np.imag(C[i,j])),
                         'staggered_corr':float(signs*np.real(C[i,j])),
                         'zero_mode_corr':float(signs*np.real(result['C_zero'][i,j])),
                         'finite_corr':float(signs*np.real(result['C_finite'][i,j]))})
    return rows


def summarize_shells(rows,round_decimals=7):
    groups={}
    for row in rows:
        key=round(row['distance'],round_decimals)
        groups.setdefault(key,[]).append(row)
    return [{'distance':key,'pairs':len(v),
             'staggered_corr_mean':float(np.mean([z['staggered_corr'] for z in v])),
             'staggered_zero_mean':float(np.mean([z['zero_mode_corr'] for z in v])),
             'staggered_finite_mean':float(np.mean([z['finite_corr'] for z in v]))}
            for key,v in sorted(groups.items())]
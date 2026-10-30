"""Exact finite-group momentum/point-group projectors for spinless Fock states.

Projector columns are fully normalized, include the fermionic permutation
parity, and are checked to form an orthonormal basis.  Only one-dimensional
irreducible representations of the *little group* are imposed. Other irreps
are represented by separate rotation-eigenvalue blocks without mirrors.
"""
from itertools import product
from math import comb
import numpy as np
from scipy.sparse import csc_matrix, identity
from core import make_states
from geometry import twisted_c6_gauge


def order(p):
    x=np.arange(len(p),dtype=int)
    for k in range(1,25):
        x=p[x]
        if np.array_equal(x,np.arange(len(p))):return k
    raise ValueError('Permutation order >24')


def generators_trans(g):
    out=[]
    identity=np.arange(g.Ns)
    seen={tuple(identity)}
    for p in g.translations():
        if tuple(p) in seen:continue
        # Generate closure and accept translation only if independent.
        group={tuple(identity)}
        for old in out:
            nxt=set()
            for pp in group:
                for j in range(old[2]):
                    q=np.array(pp)
                    for _ in range(j):q=old[1][q]
                    nxt.add(tuple(q))
            group=nxt
        if tuple(p) in group:continue
        gen=('T'+str(len(out)+1),p,order(p))
        out.append(gen)
        seen.add(tuple(p))
    assert np.prod([x[2] for x in out])==g.num_cells,(g.name,'bad generators')
    return out


def keeps_momentum(p,trans,qs):
    if p is None:return False
    ident=np.arange(len(p),dtype=int)
    group={tuple(ident):tuple([0]*len(trans))}
    for exps in product(*[range(t[2]) for t in trans]):
        q=ident.copy()
        for (_,t,_),n in zip(trans,exps):
            for _ in range(n):q=t[q]
        group[tuple(q)]=exps
    inverse=np.argsort(p)
    for i,(_,t,ord_i) in enumerate(trans):
        conj=p[t[inverse]]
        exps=group.get(tuple(conj))
        if exps is None:return False
        v=sum(q*n/o for q,n,(_,_,o) in zip(qs,exps,trans))
        v-=qs[i]/ord_i
        if abs(v-round(v))>1e-8:return False
    return True


def sector_labels(g,kind='point',use_signed=True,nf=0):
    if kind not in ('none','translation','point'):
        raise ValueError('kind must be none, translation, or point')
    if kind=='none':
        yield {'K':[],'rotation':None,'mirror':None,'signed_rotation':False,'generators':[]}
        return
    trans=generators_trans(g)
    pm=g.point_maps()
    signed=None
    if kind=='point' and use_signed and pm['C6'] is None:
        signed=g.gauge_point_maps()
    allones=np.ones(g.Ns,dtype=np.int8)
    for qs in product(*[range(x[2]) for x in trans]):
        generic=[(p,np.exp(-2j*np.pi*q/o),allones) for (_,p,o),q in zip(trans,qs)]
        if kind=='translation':
            yield {'K':list(qs),'rotation':None,'mirror':None,'signed_rotation':False,'generators':generic}
            continue
        rot_name=next((k for k in ('C6','C3','C2') if keeps_momentum(pm[k],trans,qs)),None)
        rot=pm[rot_name] if rot_name else None
        signed_rotation=False
        if signed is not None and signed['C6'] is not None and keeps_momentum(signed['C6'][0],trans,qs):
            rot_name='C6';rot=signed['C6'][0];signed_rotation=True
        # The regular translations and the *gauge-signed* rotation commute
        # with H in the ORIGINAL gauge; do not change gauge and then wrongly
        # keep treating translations as plain permutations.
        if rot is None and signed is not None:
            pair=next(((k,signed[k]) for k in ('C6','C2')
                       if signed[k] is not None and keeps_momentum(signed[k][0],trans,qs)),None)
            if pair is not None:
                rot_name,pairmap=pair
                rot=pairmap[0]
                signed_rotation=True
        mirrors=[(lab,p) for lab,p in pm['mirrors'].items() if keeps_momentum(p,trans,qs)]
        if rot is None:
            if mirrors:
                for mq in (0,1):
                    yield {'K':list(qs),'rotation':None,'mirror':mq,'signed_rotation':False,
                           'generators':generic+[(mirrors[0][1],(-1.)**mq,allones)]}
            else:
                yield {'K':list(qs),'rotation':None,'mirror':None,'signed_rotation':False,'generators':generic}
            continue
        n=order(rot)
        onsite=signed[rot_name][1] if signed_rotation else allones
        # (G R_n)^n=(-1)^Nf for a signed rotation exchanging A/B.
        twist_phase=(nf%2)/2 if signed_rotation else 0
        for rq in range(n):
            gg=generic+[(rot,np.exp(-2j*np.pi*(rq+twist_phase)/n),onsite)]
            if signed_rotation:
                # Gauge-dressed mirrors can be added with a larger irrep
                # implementation; do not impose ordinary mirrors here.
                yield {'K':list(qs),'rotation':[rot_name,rq], 'mirror':None,
                       'signed_rotation':True,'generators':gg}
                continue
            eligible=[]
            for label,p in mirrors:
                commutes=np.array_equal(rot[p],p[rot])
                if commutes or rq==0 or (n%2==0 and rq==n//2):eligible.append(p)
            if eligible:
                for mq in (0,1):
                    yield {'K':list(qs),'rotation':[rot_name,rq], 'mirror':mq,
                           'signed_rotation':False,'generators':gg+[(eligible[0],(-1.)**mq,allones)]}
            else:
                yield {'K':list(qs),'rotation':[rot_name,rq], 'mirror':None,
                       'signed_rotation':False,'generators':gg}


def group_closure(generators,ns):
    """Closure of signed-site maps (permutation, on-site phase, character).

    A spatial+gauge group element maps c_i^dag to phase_i c_{p(i)}^dag.
    The same map may appear by different generator paths: its irrep
    character must agree, or the requested block is not well defined.
    """
    eye=tuple(range(ns));ones=tuple([1]*ns)
    identity=(eye,ones)
    group={identity:1+0j}
    pending=[identity]
    while pending:
        now=pending.pop()
        p0=np.asarray(now[0]);z0=np.asarray(now[1])
        for p,char,z in generators:
            pp=np.asarray(p,dtype=int); zz=np.asarray(z,dtype=int)
            new=(tuple(p0[pp]),tuple((zz*z0[pp]).tolist()))
            nc=group[now]*complex(char)
            if new not in group:
                group[new]=nc;pending.append(new)
                if len(group)>250:raise RuntimeError('Symmetry group unexpectedly large')
            elif abs(group[new]-nc)>1e-8:
                raise ValueError('Inconsistent one-dimensional symmetry character')
    return [(np.array(p,dtype=np.int32),np.asarray(z,dtype=int),ch) for (p,z),ch in group.items()]


def map_fock(state,perm,ns,onsite=None):
    occ=[i for i in range(ns) if (int(state)>>(ns-1-i))&1]
    new=[int(perm[i]) for i in occ]
    inversions=sum(new[i]>new[j] for i in range(len(new)) for j in range(i+1,len(new)))
    target=sum(1<<(ns-1-i) for i in new)
    phase=(-1 if inversions%2 else 1)
    if onsite is not None:
        for i in occ:phase*=int(onsite[i])
    return target,phase


def projected_basis(states,ns,generators):
    group=group_closure(generators,ns)
    indexes={int(s):i for i,s in enumerate(states)}
    visited=np.zeros(len(states),dtype=bool)
    rr,cc,dd=[],[],[]
    nc=0
    for ix,state in enumerate(states):
        if visited[ix]:continue
        orbit={};coefs={}
        for p,z,char in group:
            target,sgn=map_fock(int(state),p,ns,z)
            j=indexes[target]
            orbit[j]=True
            coefs[j]=coefs.get(j,0j)+np.conj(char)*sgn
        for j in orbit:visited[j]=True
        norm=np.sqrt(sum(abs(z)**2 for z in coefs.values()))
        if norm<1e-9:continue
        for j,z in coefs.items():
            if abs(z)>1e-12:
                rr.append(j);cc.append(nc);dd.append(z/norm)
        nc+=1
    Q=csc_matrix((np.asarray(dd,complex),(np.asarray(rr,dtype=int),np.asarray(cc,dtype=int))),
                shape=(len(states),nc))
    if nc:
        diff=Q.conj().T@Q-identity(nc,format='csc',dtype=complex)
        if diff.nnz and np.max(np.abs(diff.data))>1e-8:
            raise AssertionError('Nonorthonormal projected basis')
    return Q


def gauge_back(psi,states,ns,phases):
    if phases is None:return np.asarray(psi)
    w=np.ones(len(states),complex)
    for i,z in enumerate(phases):
        if abs(z-1)<1e-12:continue
        w[((states >> np.uint64(ns-1-i))&np.uint64(1))!=0]*=np.conj(z)
    return psi*w


def symmetry_projectors(g,nf,kind='point',max_basis=900_000,use_signed=True):
    states=make_states(g.Ns,nf)
    if len(states)>max_basis:
        raise MemoryError(f'Fixed-N dimension {len(states)} exceeds projector cap {max_basis}')
    count=0
    for label in sector_labels(g,kind,use_signed,nf):
        Q=projected_basis(states,g.Ns,label['generators'])
        count+=Q.shape[1]
        yield label,Q,states,None
    if count!=len(states):
        raise AssertionError(f'Incomplete/duplicated decomposition: {count} != {len(states)}')
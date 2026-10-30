#!/usr/bin/env python3
"""Symmetry-resolved zero-T QuSpin ED for Ruby spinless t-V lattice.

Run from this directory: python run_zeroT.py --cluster ring3 --nf 6 --V 1.0
QuSpin is needed for spectral calculations; geometry.py/tests work without it.
"""
import argparse
from itertools import product
import json
import csv
from pathlib import Path
import sys
import warnings
import numpy as np
from scipy.sparse.linalg import eigsh
from geometry import RubyPBC,twisted_c6_gauge
from observables import (apply_ops,build_observables,op_sum,current_terms,
                         lanczos_spectral,max_current_in_manifold)


def get_quspin():
    try:
        from quspin.basis import spinless_fermion_basis_general
        from quspin.operators import hamiltonian
    except ImportError as exc:
        raise SystemExit('QuSpin is not installed. Install it in a supported Python environment: pip install quspin') from exc
    return spinless_fermion_basis_general,hamiltonian


def hamiltonian_terms(g,V,site_gauge=None):
    fwd=[];rev=[];inter=[]
    for e in g.edges:
        t=g.t[e.kind]
        if site_gauge is not None:t=t*site_gauge[e.i]*np.conj(site_gauge[e.j])
        fwd.append([t,e.i,e.j]);rev.append([np.conj(t),e.j,e.i])
        if e.kind=='t':inter.append([V,e.i,e.j])
    return [['+-',fwd],['+-',rev],['nn',inter]]


def create_hamiltonian(g,V,basis,site_gauge=None):
    _,qh=get_quspin()
    return qh(hamiltonian_terms(g,V,site_gauge),[],basis=basis,dtype=np.complex128,
              check_symm=False,check_pcon=False,check_herm=False)


def perm_order(p):
    if p is None:return 1
    seq=np.arange(len(p),dtype=int)
    for o in range(1,13):
        seq=p[seq]
        if np.all(seq==np.arange(len(p))):return o
    raise ValueError('Permutation order exceeds 12')


def translation_generators(g):
    p1,p2=g.translations()
    result=[]
    if perm_order(p1)>1:result.append(('k1block',p1,perm_order(p1)))
    if perm_order(p2)>1 and not any(np.array_equal(p2,r[1]) for r in result):
        result.append(('k2block',p2,perm_order(p2)))
    assert int(np.prod([r[2] for r in result]))==g.num_cells
    return result


def _permutation_inverse(p):
    return np.argsort(p)


def _perm_product(a,b):
    """Composition a o b for maps acting on site indices."""
    return a[b]


def _commutes(a,b):
    return np.array_equal(a[b],b[a])


def point_preserves_momentum(point, trans, qs):
    """Does p T_i p^-1 act with the same translation character as T_i?

    This is the actual finite-torus little-group criterion, independent of
    k-vector sketches; it also handles non-rectangular area-3 supercells.
    """
    if point is None:return False
    if not trans:return True
    ident=np.arange(len(point),dtype=np.int32)
    inverse=_permutation_inverse(point)
    gens=[(p,o) for _,p,o in trans]
    table={}
    for exponents in product(*[range(o) for _,o in gens]):
        p=ident.copy()
        for (t,o),n in zip(gens,exponents):
            for _ in range(n):p=_perm_product(t,p)
        table[tuple(int(x) for x in p)]=exponents
    assert len(table)==int(np.prod([o for _,o in gens]))
    for i,(ti,oi) in enumerate(gens):
        conjugate=_perm_product(point,_perm_product(ti,inverse))
        exponents=table.get(tuple(int(x) for x in conjugate))
        if exponents is None:return False
        phase=sum(float(q)*n/o for q,n,(_,o) in zip(qs,exponents,gens))
        if abs((phase-qs[i]/oi)-round(phase-qs[i]/oi))>1.e-8:return False
    return True


def sectors(g,points=True,twisted=None):
    """Exhaustive commuting QuSpin blocks: T, then little-group C_n/mirror.

    Non-commuting point operators are never projected simultaneously. A
    gauge-twisted C6 (t1=-t2) is saved as a validated diagnostic only:
    spinless_fermion_basis_general negative indices denote particle-hole,
    not the required c_i -> -c_j gauge action.
    """
    trans=translation_generators(g)
    pm=g.point_maps()
    if twisted is not None:
        p,z=twisted
        # In the new gauge C6 and its powers are ORDINARY site
        # permutations. Mirrors do not generally become plain maps.
        pm['C6']=p
        pm['C3']=p[p]
        pm['C2']=p[p[p]]
        pm['mirrors']={}
    for qs in product(*[range(o) for _,_,o in trans]):
        blk={name:(t,int(q)) for (name,t,o),q in zip(trans,qs)}
        if not points:
            yield tuple(qs),None,None,blk
            continue
        usable={name:p for name,p in ((n,pm[n]) for n in ('C6','C3','C2'))
                if point_preserves_momentum(p,trans,qs)}
        rot_name=next((n for n in ('C6','C3','C2') if n in usable),None)
        rot=usable.get(rot_name)
        order=perm_order(rot)
        eligible_mirrors=[(a,p) for a,p in pm['mirrors'].items()
                          if point_preserves_momentum(p,trans,qs)]
        if rot is None or order<=1:
            if eligible_mirrors:
                mirror=eligible_mirrors[0][1]
                for mq in (0,1):
                    yield tuple(qs),None,mq,dict(blk,mirrorblock=(mirror,mq))
            else:
                yield tuple(qs),None,None,blk
            continue
        for rq in range(order):
            b=dict(blk,rotblock=(rot,rq))
            compatible=[]
            for axis,p in eligible_mirrors:
                # For dihedral symmetry: mirrors map q -> -q. Only
                # commute at q=0 or pi. Check actual permutation too.
                if _commutes(p,rot):compatible.append(p)
                elif rq==0 or (order%2==0 and rq==order//2):
                    compatible.append(p)
            if compatible:
                for mq in (0,1):
                    yield tuple(qs),rq,mq,dict(b,mirrorblock=(compatible[0],mq))
            else:
                yield tuple(qs),rq,None,b


def lowest_eigenpairs(H,k):
    n=int(H.Ns)
    if n==0:return np.empty(0),np.empty((0,0),dtype=complex)
    if n<=k+1:
        e,u=np.linalg.eigh(H.toarray())
        return e[:k],u[:,:k]
    e,u=H.eigsh(k=k,which='SA',tol=1.e-10,maxiter=max(5000,20*n))
    ix=np.argsort(e)
    return e[ix],u[:,ix]


def sector_scan(g,V,nf,n_eigs=3,points=True,twisted=None):
    basis_type,_=get_quspin()
    records=[]; candidates=[]; dimension=0
    for kval,rval,pval,blocks in sectors(g,points,twisted=twisted):
        b=basis_type(g.Ns,Nf=nf,**blocks)
        dimension+=b.Ns
        if b.Ns==0:continue
        H=create_hamiltonian(g,V,b,site_gauge=None if twisted is None else twisted[1])
        e,u=lowest_eigenpairs(H,min(n_eigs,b.Ns))
        records.append({'momentum_sector':kval,'rotation_q':rval,'mirror_q':pval,
                        'dimension':int(b.Ns),'eigvals':e.tolist()})
        for i in range(len(e)):
            candidates.append((float(e[i]),records[-1],b,u[:,i].copy()))
    from math import comb
    assert dimension==comb(g.Ns,nf),(dimension,comb(g.Ns,nf),'Symmetry sectors incomplete or overlapping')
    candidates.sort(key=lambda x:x[0])
    return records,candidates


def full_state(basis, vec):
    # For a particle-number conserving basis, pcon=True uses the fixed-N
    # full Fock space ordering (the same as basis_general(N,Nf=Nf).states).
    P=basis.get_proj(np.complex128,pcon=True)
    return np.asarray(P@vec).ravel()


def undo_one_body_gauge(psi, states, site_gauge, Ns):
    """H'=D H D^dag => psi_original=D^dag psi_gauge in Fock basis.

    QuSpin fixed-N bit order uses orbital 0 as the most-significant bit.
    """
    if site_gauge is None:return np.asarray(psi)
    phase=np.ones(len(states),dtype=np.complex128)
    for i,z in enumerate(site_gauge):
        if abs(z-1.)<1.e-14:continue
        occ=np.array([(int(b)>>(Ns-1-i))&1 for b in states],dtype=bool)
        phase[occ]*=np.conj(z)
    return np.asarray(psi)*phase


def check_fermion_op(g,V,nf,fullbasis,psi):
    """Cross-check signs in independent bitstring implementation."""
    H=create_hamiltonian(g,V,fullbasis)
    # Deterministic random vector tests prevent using only an eigenvector.
    rng=np.random.default_rng(120310)
    r=rng.normal(size=fullbasis.Ns)+1j*rng.normal(size=fullbasis.Ns)
    r/=np.linalg.norm(r)
    manual=np.zeros_like(r)
    states=fullbasis.states
    for e in g.edges:
        t=g.t[e.kind]
        manual+=t*apply_ops(r,states,states,g.Ns,[('+',e.i),('-',e.j)])
        manual+=t*apply_ops(r,states,states,g.Ns,[('+',e.j),('-',e.i)])
        if e.kind=='t':
            occ_i=np.array([(int(s)>>(g.Ns-1-e.i))&1 for s in states])
            occ_j=np.array([(int(s)>>(g.Ns-1-e.j))&1 for s in states])
            manual+=V*occ_i*occ_j*r
    rel=np.linalg.norm(manual-H.dot(r))
    if rel>1.e-8:raise AssertionError(f'Fermion sign convention fails: {rel}')
    return rel


def symmetry_diagnostics(g):
    plain=g.point_maps();gauge=g.gauge_point_maps()
    return {
        'ordinary':{k:(None if p is None else p.tolist()) for k,p in plain.items()
                    if k!='mirrors'},
        'ordinary_mirrors':{a:p.tolist() for a,p in plain['mirrors'].items()},
        'gauge':{k:(None if x is None else {'permutation':x[0].tolist(),
                         'source_gauge_signs':x[1].tolist()})
                 for k,x in gauge.items() if k!='mirrors'},
        'gauge_mirrors':{a:{'permutation':x[0].tolist(),
                            'source_gauge_signs':x[1].tolist()}
                         for a,x in gauge['mirrors'].items()},
        'note':'Gauge-signed operations verified but not given as QuSpin permutation blocks.'
    }


def build_full_observables(g,V,nf,e0,states,psi,pair,chi_m=0):
    out=build_observables(g,psi,states,pairs=pair)
    if chi_m:
        bt,_=get_quspin()
        bf=bt(g.Ns,Nf=nf)
        H=create_hamiltonian(g,V,bf)
        # Physical equal-orientation (CCW) current operators; all plaquettes
        # are oriented CCW by geometry.cycles(). Previous Gamma +/- labels
        # use opposite geometric bond-arrow senses between A/B triangles.
        triangles=[lp for lp in g.cycles() if len(lp)==3]
        for name,signf in [ ('lc_same',lambda tri:1.),
                           ('lc_opposite',lambda tri:1. if tri[0]%6<3 else -1.) ]:
            terms=[]
            for tri in triangles:
                terms.extend([(z*signf(tri)/len(triangles),ops) for z,ops in current_terms(g,tri)])
            Opsi=op_sum(psi,states,g.Ns,terms)
            mean=np.vdot(psi,Opsi)
            f=Opsi-mean*psi
            d,w=lanczos_spectral(H,f,e0,m=chi_m)
            mask=d>1.e-9
            out[f'{name}_poles']=d
            out[f'{name}_weights']=w
            out[f'{name}_chi0']=float(np.sum(2*w[mask]/d[mask]))
            out[f'{name}_chi_unresolved']=bool(np.any((d<=1.e-9)&(w>1.e-10)))
        # Largest connected charge covariance eigenmode, non-uniform
        evals,Vc=np.linalg.eigh(out['density_connected'])
        fpattern=Vc[:,-1].real
        ov=(np.array([[(int(s)>>(g.Ns-1-i))&1 for i in range(g.Ns)] for s in states])@fpattern)
        cv=ov*psi
        f=cv-np.vdot(psi,cv)*psi
        d,w=lanczos_spectral(H,f,e0,m=chi_m)
        good=d>1.e-9
        out.update(charge_mode=fpattern,charge_poles=d,charge_weights=w,
                   charge_chi0=float(np.sum(2*w[good]/d[good])),
                   charge_chi_unresolved=bool(np.any((d<=1.e-9)&(w>1.e-10))))
    return out


def run(args):
    g=RubyPBC(args.cluster,args.t,args.t1,args.t2)
    nf=args.nf
    if not (0<=nf<=g.Ns):raise ValueError('nf must be 0..Ns')
    folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
    twisted=None
    if args.point and args.twisted_c6 and g.point_maps()['C6'] is None:
        twisted=twisted_c6_gauge(g)
    spec=g.metadata()
    spec.update(V=args.V,Nf=nf,diagonalizer='QuSpin eigsh zero-T',
                lower_spin='spinless',energy_unit='t',temperature=0.,
                time_reversal='real hopping, h=0',
                current_convention='I_ij = i*t_ij*(c^dag_i c_j-c^dag_j c_i), hbar=1',
                loop_rho_flux='arg product <c^dag_i c_j>, not magnetic flux',
                green_definition='retarded/Matsubara to be evaluated in adjacent-N sectors',
                pair_definition='spinless bond pair; no onsite s-wave')
    spec['point_symmetry_maps']=symmetry_diagnostics(g)
    spec['spatial_blocks_include_gauge_signed_maps']=(twisted is not None)
    spec['twisted_c6_gauge_active']=(twisted is not None)
    if twisted is not None:
        spec['twisted_c6_site_phases']=[[float(z.real),float(z.imag)] for z in twisted[1]]
        spec['twisted_c6_map']=twisted[0].tolist()
        spec['twisted_c6_physical_eigenvalue']='exp(i*pi*Nf/6)*exp(-2*pi*i*rotation_q/6)' 
        spec['twisted_c6_note']='Diagonalize gauge-transformed H; rotate eigenvectors back before observables.' 
    (folder/'geometry.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    report,found=sector_scan(g,args.V,nf,args.low,args.point,twisted=twisted)
    with (folder/'sectors.json').open('w') as f:json.dump(report,f,indent=2)
    if not found:raise RuntimeError('No eigenstates found')
    e0=found[0][0]
    summary={'E0':e0,'Nf':nf,'Ns':g.Ns,'cluster':args.cluster,'V':args.V,
             'ground_blocks':{k:(v if not isinstance(v,tuple) else list(v))
                              for k,v in found[0][1].items()},
             'low_energies':[x[0] for x in found[:min(30,len(found))]],
             'degenerate_within_tolerance':sum(abs(x[0]-e0)<=args.degeneracy_tol for x in found),
             'point_symmetry_enabled':args.point}
    if args.observables:
        bt,_=get_quspin()
        fullbasis=bt(g.Ns,Nf=nf)
        if fullbasis.Ns>args.max_full:
            raise MemoryError(f'Full-N basis {fullbasis.Ns} exceeds --max-full {args.max_full}; rerun --no-observables')
        psi=full_state(found[0][2],found[0][3])
        psi=undo_one_body_gauge(psi,fullbasis.states,
                                None if twisted is None else twisted[1],g.Ns)
        psi/=np.linalg.norm(psi)
        assert len(psi)==fullbasis.Ns
        if args.check_signs:summary['fermion_sign_error']=check_fermion_op(g,args.V,nf,fullbasis,psi)
        data=build_full_observables(g,args.V,nf,e0,fullbasis.states,psi,
                                    pair=args.pair,chi_m=args.chi_lanczos)
        np.savez_compressed(folder/'equal_time.npz',**data)
        if args.save_psi:
            np.savez_compressed(folder/'ground_state.npz',psi=psi,
                                basis_states=fullbasis.states,e0=e0,Nf=nf)
        # Symmetry-breaking current: compare *only* quasi-degenerate low states.
        manifold=[x for x in found if x[0]-e0<=args.degeneracy_tol+1.e-12]
        if len(manifold)>1 and args.max_current:
            vectors=np.column_stack([undo_one_body_gauge(full_state(x[2],x[3]),
                       fullbasis.states,None if twisted is None else twisted[1],g.Ns)
                       for x in manifold])
            res=max_current_in_manifold(g,vectors,fullbasis.states,
                    [x[0]-e0 for x in manifold],window=args.degeneracy_tol)
            summary['max_current_manifold']=[{'loop_len':r[0], 'loop':r[1],
                    'J_at_max_abs':r[2],'max_abs_J':r[3],
                    'linear_combination_coefficients':[[z.real,z.imag] for z in r[4]]}
                    for r in res]
            summary['max_current_basis_energies']=[x[0] for x in manifold]
        else:
            summary['max_current_manifold']='not computed: no resolved multiplet or disabled'
        summary['charge_corr_max']=float(np.max(data['charge_eigs']))
        summary['current_corr_max']=float(np.max(data['current_eigs']))
        if 'pair_eigs' in data:summary['pair_corr_max']=float(np.max(data['pair_eigs']))
    (folder/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('ground_blocks','max_current_manifold')},indent=2,default=str))
    print('Saved:',folder)
    return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cluster',choices=['2x2','3x1','ring3'],default='ring3')
    p.add_argument('--nf',type=int,default=6)
    p.add_argument('--V',type=float,default=1.)
    p.add_argument('--t',type=float,default=1.)
    p.add_argument('--t1',type=float,default=.5)
    p.add_argument('--t2',type=float,default=.5)
    p.add_argument('--out',default='results/ring3_N6_V1')
    p.add_argument('--low',type=int,default=3,help='Low states per symmetry block')
    p.add_argument('--no-twisted-c6',dest='twisted_c6',action='store_false',
                   help='Disable gauge-transformed ordinary-C6 blocks at t1=-t2')
    p.set_defaults(twisted_c6=True)
    p.add_argument('--no-point',dest='point',action='store_false',help='Translation only')
    p.add_argument('--no-observables',dest='observables',action='store_false')
    p.add_argument('--pair',action='store_true',help='Spinless nn pair correlation (18 sites)')
    p.add_argument('--chi-lanczos',type=int,default=0,help='>0: current and charge chi(0) Krylov steps')
    p.add_argument('--degeneracy-tol',type=float,default=1.e-7)
    p.add_argument('--max-current',action='store_true')
    p.add_argument('--max-full',type=int,default=1000000)
    p.add_argument('--check-signs',action='store_true')
    p.add_argument('--save-psi',action='store_true')
    return p


if __name__=='__main__':
    run(parser().parse_args())
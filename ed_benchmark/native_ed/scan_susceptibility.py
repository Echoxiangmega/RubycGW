#!/usr/bin/env python3
"""Fixed-filling zero-temperature ED: V scan and reproducible susceptibility curves.

Native NumPy/SciPy only. Exact symmetries are used for ground-state search;
response resolvents act on the full fixed-N sector. Dynamic Lanczos poles and
weights, degeneracy diagnostics and raw channel definitions are saved.

No phase labels are inferred from comparing differently normalized responses.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from core import FixedHamiltonian, interaction_diagonal, make_states, lowest
from geometry import RubyPBC
from symmetry import symmetry_projectors
from observables import op_sum, current_terms, apply_ops


def safe_json(x):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (complex, np.complexfloating)):
        return {'real': float(x.real), 'imag': float(x.imag)}
    raise TypeError(f'Not serializable: {type(x)!r}')


def site_occupations(states, ns):
    return np.column_stack([((states >> np.uint64(ns-1-i)) & np.uint64(1)).astype(float)
                            for i in range(ns)])


def compact_label(lab):
    return {k:v for k,v in lab.items() if k!='generators'}


class VScanSolver:
    """Cache geometry, full H0 and all symmetry projectors once per filling."""
    def __init__(self,g,nf,symmetry='point',low=5,max_basis=70_000):
        self.g,self.nf,self.low=g,nf,low
        self.H0=FixedHamiltonian(g,nf,0.).operator
        self.states=make_states(g.Ns,nf)
        if len(self.states)>max_basis:
            raise MemoryError(f'D={len(self.states)} exceeds --max-basis={max_basis}. '
                              '18-site cases are the recommended starting point.')
        self.U=interaction_diagonal(g,self.states,1.)
        self.occ=site_occupations(self.states,g.Ns)
        self.blocks=[]
        dimensional_sum=0
        for lab,Q,states,_ in symmetry_projectors(g,nf,symmetry,max_basis=max_basis):
            if not np.array_equal(states,self.states):raise AssertionError('Basis mismatch')
            d=Q.shape[1];dimensional_sum+=d
            if d==0:continue
            h0=(Q.conj().T@(self.H0@Q)).tocsr()
            # Density interactions commute with every retained spatial symmetry.
            # The group-orbit columns of Q each have fixed interaction energy.
            # Orbit-projected interaction is diagonal: all members of one
            # symmetry orbit have the same density interaction energy.
            u=np.asarray(Q.multiply(self.U[:,None]).conj().multiply(Q).sum(axis=0)).ravel().real
            if not np.all(np.isfinite(u)):raise ValueError('Invalid interaction projection')
            self.blocks.append((lab,Q,h0,u))
        if dimensional_sum!=len(self.states):
            raise AssertionError(f'Incomplete symmetry blocks: {dimensional_sum}/{len(self.states)}')
        if not self.blocks:raise RuntimeError('No nonempty symmetry sectors')
        self.dim=len(self.states)

    def solve(self,V,deg_tol):
        winners=[];records=[]
        for lab,Q,h0,u in self.blocks:
            H=h0+diags(V*u,format='csr')
            energies,vectors=lowest(H,min(self.low,H.shape[0]))
            records.append({'labels':compact_label(lab),'dim':H.shape[0],
                            'energies':energies.tolist()})
            for i,e in enumerate(energies):winners.append((float(e),lab,Q,vectors[:,i]))
        winners.sort(key=lambda x:x[0])
        E0=winners[0][0]
        near=[x for x in winners if x[0]-E0<=deg_tol]
        psis=np.column_stack([np.asarray(Q@u).ravel() for _,lab,Q,u in near])
        # Q for different irreps is exactly orthogonal; perform QR in case
        # iterative eigenvectors from a degenerate block have numerical drift.
        psi_orth, R=np.linalg.qr(psis)
        if np.min(np.abs(np.diag(R)))<1.e-7:
            raise AssertionError('Numerically dependent vectors in ground manifold')
        Hfull=self.H0+diags(V*self.U,format='csr')
        residual=max(np.linalg.norm(Hfull@psi_orth[:,i]-E0*psi_orth[:,i])
                     for i in range(psi_orth.shape[1]))
        if residual>2e-6:raise AssertionError(f'Bad ED eigenstate residual: {residual}')
        # The eigensolver returned only `low` states in every block. If the
        # last returned state is still degenerate with E0, manifold is partial.
        incomplete=any(len(rec['energies'])==self.low and
                       rec['energies'][-1]-E0<=deg_tol for rec in records)
        return Hfull,E0,psi_orth,records,incomplete,residual


def leading_charge_mode(occ, psis, g):
    """Choose the leading charge covariance eigenmode among allowed q.

    F_{R,a}(q)=exp(+2 pi i q.R) u_a /sqrt(Nc). For real response
    use normalized Re/Im combinations, recording which component exists.
    """
    probs=np.mean(abs(psis)**2,axis=1)
    density=probs@occ
    conn=occ.T@(probs[:,None]*occ)-np.outer(density,density)
    nc=g.num_cells
    maximum=None
    for k in g.permitted_momenta():
        F=np.zeros((g.Ns,6),complex)
        for c,R in enumerate(g.reps):
            phase=np.exp(2j*np.pi*np.dot(k,R))/np.sqrt(nc)
            for a in range(6):F[6*c+a,a]=phase
        Cq=F.conj().T@conn@F
        w,u=np.linalg.eigh((Cq+Cq.conj().T)*.5)
        if maximum is None or w[-1]>maximum['structure_factor']:
            maximum={'q':list(k),'structure_factor':float(w[-1]),
                     'orbital_vector':u[:,-1], 'site_vector':F@u[:,-1]}
    return maximum


def charge_real_sources(mode):
    """Real orthonormal components of a complex charge mode, if independent."""
    vec=np.asarray(mode,dtype=complex)
    real=np.real(vec);imag=np.imag(vec)
    basis=[]
    for v in (real,imag):
        for prior in basis:v=v-np.dot(v,prior)*prior
        n=np.linalg.norm(v)
        if n>1e-10:basis.append(v/n)
    # A complex-k mode normally has two quadratures; each has unit norm.
    return basis


def pseudospin_e_charge_modes(g):
    """HF-compatible real E-type triangle CO operators in physical A/B frames.

    A=(0,1,2): tau_x=2n0-n1-n2, tau_y=sqrt(3)(n2-n1)
    B=(3,4,5): tau_x=2n3-n4-n5, tau_y=sqrt(3)(n4-n5)
    The B-y sign follows the opposite geometric chirality of triangle B.
    Physical density operators work at all fillings; their projected Pauli
    interpretation applies only inside the local one-fermion E doublet.
    The q=Gamma A +/- B combinations are Euclidean-unit normalized.
    """
    nc=g.num_cells
    xa=np.tile([2.,-1.,-1.,0.,0.,0.],nc)
    ya=np.tile([0.,-np.sqrt(3.),np.sqrt(3.),0.,0.,0.],nc)
    xb=np.tile([0.,0.,0.,2.,-1.,-1.],nc)
    yb=np.tile([0.,0.,0.,0.,np.sqrt(3.),-np.sqrt(3.)],nc)
    out={}
    for parity,sign in (('even',1.),('odd',-1.)):
        for axis,aa,bb in (('x',xa,xb),('y',ya,yb)):
            weights=aa+sign*bb
            out[f'CO_E_{parity}_{axis}']=weights/np.linalg.norm(weights)
    return out


def channel_specs(g,occ,ground,include_extra=True):
    ns=g.Ns;triangle=[p for p in g.cycles() if len(p)==3]
    nc=g.num_cells
    ab=np.tile([1.,1.,1.,-1.,-1.,-1.],nc)/np.sqrt(ns)
    lead=leading_charge_mode(occ,ground,g)
    channels=[{'name':'CO_AB','type':'charge','weights':ab,
               'definition':'A-B density, q=Gamma, sum w_i^2=1'},
              {'name':'CO_leading','type':'charge','weight_vectors':charge_real_sources(lead['site_vector']),
               'q':lead['q'], 'structure_factor':lead['structure_factor'],
               'orbital_vector':[[float(z.real),float(z.imag)] for z in lead['orbital_vector']],
               'definition':'leading equal-time charge eigenmode, real quadratures, sum w_i^2=1'}]
    for name,weights in pseudospin_e_charge_modes(g).items():
        parity='even' if '_even_' in name else 'odd'
        axis=name.rsplit('_',1)[1]
        channels.append({
            'name':name,'type':'charge','weights':weights,'q':[0.,0.],
            'definition':f'HF physical E-type tau_{axis}, '
                         f'A {"+" if parity=="even" else "-"} B, '
                         'q=Gamma, ||site weights||_2=1'
        })
    for key,signs in [('LC_same',[1.]*len(triangle)),
                      ('LC_opposite',[1. if p[0]%6<3 else -1. for p in triangle])]:
        channels.append({'name':key,'type':'current','loops':triangle,
                         'loop_coefficients':(np.asarray(signs)/np.sqrt(len(triangle))).tolist(),
                         'definition':'CCW triangle loop currents, average bond current, coefficients norm 1'})
    if include_extra:
        for k,nm in [(4,'LC_square'),(6,'LC_hexagon')]:
            polygons=[p for p in g.cycles() if len(p)==k]
            if polygons:
                channels.append({'name':nm,'type':'current','loops':polygons,
                                 'loop_coefficients':(np.ones(len(polygons))/np.sqrt(len(polygons))).tolist(),
                                 'definition':f'CCW {k}-site plaquette-current sum, coefficients norm 1'})
    return channels,lead


def source_action(ch,psi,states,ns,occ,g):
    if ch['type']=='charge':
        vectors=ch['weight_vectors'] if 'weight_vectors' in ch else [ch['weights']]
        return [np.asarray(occ@w,dtype=complex)*psi for w in vectors]
    terms=[]
    for loop,v in zip(ch['loops'],ch['loop_coefficients']):
        terms.extend((w*v,ops) for w,ops in current_terms(g,loop))
    return [op_sum(psi,states,ns,terms)]


def spectral_chi(H,source,e0,ground,depth=80,precision=1e-8):
    """Fully reorthogonalized, ground-manifold-deflated Hermitian Lanczos.

    Reports regular Lehmann χ=2Σ |<n|O|gs>|²/(En-E0), n outside G.
    A nonzero P_G O|gs> flags ambiguous/divergent h→0 response at T=0.
    """
    overlap=ground.conj().T@source
    zero_weight=float(np.vdot(overlap,overlap).real)
    f=source-ground@overlap
    norm=float(np.linalg.norm(f))
    if norm<1e-13:
        return {'regular':0.,'zero_weight':zero_weight,'poles':np.empty(0),
                'weights':np.empty(0),'convergence':0.,'unresolved':False}
    prev=np.zeros_like(f);q=f/norm;beta=0.;Q=[];aa=[];bb=[]
    # Reorthogonalization is essential near exact degeneracy.
    for j in range(min(depth,H.shape[0]-ground.shape[1])):
        v=H@q-beta*prev
        a=float(np.vdot(q,v).real)
        v-=a*q
        Q.append(q.copy())
        # Against the complete ground manifold and all previous Krylov vectors.
        v-=ground@(ground.conj().T@v)
        for old in Q:v-=old*np.vdot(old,v)
        # Two-pass reorth for stable low poles.
        for old in Q:v-=old*np.vdot(old,v)
        b=float(np.linalg.norm(v))
        aa.append(a)
        if b<1e-12 or j==depth-1:break
        bb.append(b)
        prev,q=q,v/b
        beta=b
    from scipy.linalg import eigh_tridiagonal
    poles,U=eigh_tridiagonal(np.asarray(aa),np.asarray(bb))
    poles=poles-e0
    weights=norm**2*abs(U[0,:])**2
    positive=poles>precision
    unresolved=bool(np.any((poles<=precision)&(weights>max(1e-10,norm**2*1e-8))))
    regular=float(np.sum(2*weights[positive]/poles[positive]))
    if len(poles)>=6:
        cutoff=max(2,len(poles)//2)
        from scipy.linalg import eigh_tridiagonal
        half_poles,half_U=eigh_tridiagonal(np.asarray(aa[:cutoff]),np.asarray(bb[:cutoff-1]))
        half_poles-=e0
        half_weights=norm**2*abs(half_U[0,:])**2
        good=half_poles>precision
        chi_half=float(np.sum(2*half_weights[good]/half_poles[good]))
        convergence=abs(regular-chi_half)/(1+abs(regular))
    else:convergence=np.nan
    return {'regular':regular,'zero_weight':zero_weight,'poles':poles,
            'weights':weights,'convergence':float(convergence),'unresolved':unresolved}


def one_channel(ch,H,e0,psis,states,occ,g,depth,precision,ground_incomplete=False):
    total=0.;z0=0.;conv=0.;danger=False;detail=[]
    actions=[source_action(ch,psi,states,g.Ns,occ,g) for psi in psis.T]
    if not actions:return None
    # Subtract the ensemble-average expectation, including the exact
    # degenerate manifold. Unlike raw P O P, this does NOT flag the
    # ordinary diagonal expectation value in a unique finite-size ground state.
    means=[sum(np.vdot(psis[:,j],actions[j][k]) for j in range(psis.shape[1]))
           /psis.shape[1] for k in range(len(actions[0]))]
    for j,psi in enumerate(psis.T):
        for k,source in enumerate(actions[j]):
            res=spectral_chi(H,source-means[k]*psi,e0,psis,depth,precision)
            total+=res['regular']/len(psis.T)
            z0+=res['zero_weight']/len(psis.T)
            conv=max(conv,res['convergence'])
            danger=danger or res['unresolved']
            detail.append(res)
    tol=1e-7
    status='incomplete_ground_manifold' if ground_incomplete else ('zero_mode' if z0>tol else ('unresolved_pole' if danger else 'ok'))
    # We intentionally leave the total static susceptibility undefined for a
    # zero-mode rather than dropping its divergent/ambiguous contribution.
    chi=total if status=='ok' else np.nan
    return {'chi_static':chi,'chi_regular':total,'zero_mode_weight':z0,
            'status':status,'max_krylov_change':conv,'spectra':detail}



def pairing_response(g,nf,V,e0,psis,states,depth,pole_tol,mu_override=None,
                     max_adjacent_basis=90_000):
    """Zero-T static response to a spinless bond-pair source Δ+Δ†.

    Δ=sum_b v_b c_{i_b} c_{j_b} with ||v||_2=1. The mode v is the
    leading equal-time pair Gram eigenvector at *this* V; thus SC_pair is
    a data-adaptive diagnostic, not one fixed p-/f-wave representation.
    Nf±2 resolvents use H-μN and require a stable grand-canonical sector.
    """
    if not 2<=nf<=g.Ns-2:
        return {'chi_static':np.nan,'chi_regular':np.nan,'status':'Nf_plusminus2_outside_range',
                'zero_mode_weight':np.nan,'max_krylov_change':np.nan},[],{}
    from itertools import combinations
    Ns=g.Ns
    edge_pairs=sorted({tuple(sorted((e.i,e.j))) for e in g.edges})
    st_minus=make_states(Ns,nf-2);st_plus=make_states(Ns,nf+2)
    if max(len(st_minus),len(st_plus))>max_adjacent_basis:
        return {'chi_static':np.nan,'chi_regular':np.nan,'status':'adjacent_basis_too_large',
                'zero_mode_weight':np.nan,'max_krylov_change':np.nan},[],{}
    # Pair Gram: Δ_b|ψ_a> in the (Nf-2) sector.
    A=[];gram=np.zeros((len(edge_pairs),len(edge_pairs)),complex)
    for psi in psis.T:
        mat=np.column_stack([apply_ops(psi,states,st_minus,Ns,[('-',i),('-',j)])
                             for i,j in edge_pairs])
        A.append(mat)
        gram+=mat.conj().T@mat/psis.shape[1]
    eigen,vec=np.linalg.eigh((gram+gram.conj().T)*.5)
    mode=vec[:,-1]
    # Fix arbitrary eigenvector phase for reproducible stored form factors.
    j=int(np.argmax(abs(mode)))
    mode=mode*np.exp(-1j*np.angle(mode[j]))
    HM=FixedHamiltonian(g,nf-2,V).operator
    HP=FixedHamiltonian(g,nf+2,V).operator
    # Evaluate μ from the finite-size single-particle addition/removal gap.
    if mu_override is None:
        Eplus=float(lowest(FixedHamiltonian(g,nf+1,V).operator,1)[0][0])
        Eminus=float(lowest(FixedHamiltonian(g,nf-1,V).operator,1)[0][0])
        mu_add=Eplus-e0;mu_remove=e0-Eminus
        mu=.5*(mu_add+mu_remove)
    else:
        mu=float(mu_override)
        mu_add=mu_remove=np.nan
    Em=float(lowest(HM,1)[0][0]);Ep=float(lowest(HP,1)[0][0])
    gap_m=Em-e0+2*mu;gap_p=Ep-e0-2*mu
    result={'mode_weights':[[float(z.real),float(z.imag)] for z in mode],
            'mode_bonds':[list(x) for x in edge_pairs],
            'pair_gram_leading_eigenvalue':float(eigen[-1]),
            'mu':float(mu),'mu_remove':float(mu_remove),'mu_add':float(mu_add),
            'gap_Nminus2':float(gap_m),'gap_Nplus2':float(gap_p)}
    if min(gap_m,gap_p)<=pole_tol:
        return {'chi_static':np.nan,'chi_regular':np.nan,
                'status':'pair_source_unstable_grand_canonical',
                'zero_mode_weight':np.nan,'max_krylov_change':np.nan},[],result
    one=[];regular=0.;conv=0.;unresolved=False
    empty_minus=np.empty((len(st_minus),0),complex)
    empty_plus=np.empty((len(st_plus),0),complex)
    for psi,mat in zip(psis.T,A):
        msrc=mat@mode
        psrc=np.zeros(len(st_plus),complex)
        for z,(i,j) in zip(mode,edge_pairs):
            psrc+=np.conj(z)*apply_ops(psi,states,st_plus,Ns,[('+',j),('+',i)])
        x=spectral_chi(HM,msrc,e0-2*mu,empty_minus,depth,pole_tol)
        y=spectral_chi(HP,psrc,e0+2*mu,empty_plus,depth,pole_tol)
        regular+=(x['regular']+y['regular'])/psis.shape[1]
        conv=max(conv,x['convergence'],y['convergence'])
        unresolved=unresolved or x['unresolved'] or y['unresolved']
        one.extend([x,y])
    status='unresolved_pole' if unresolved else 'ok'
    return {'chi_static':regular if status=='ok' else np.nan,
            'chi_regular':regular,'status':status,
            'zero_mode_weight':0.,'max_krylov_change':conv},one,result


def unique_dir(root,g,nf,V):
    return root/f'V_{V:+.9f}'


def metadata_match(path,scan):
    if not path.exists():return False
    try:
        d=json.loads(path.read_text(encoding='utf-8'))
        return all(d.get(k)==v for k,v in scan.items())
    except (OSError,ValueError):return False


def run(args):
    g=RubyPBC(args.cluster,t=args.t,t1=args.t1,t2=args.t2)
    if (args.n is None)==(args.nf is None):
        raise ValueError('Specify exactly one of --n (per primitive cell) or --nf (integer)')
    raw=args.n*g.num_cells if args.n is not None else args.nf
    nf=int(round(raw))
    if abs(nf-raw)>1e-8:
        raise ValueError(f'n={args.n:g} is not representable on {args.cluster}: '
                         f'Nf=n*{g.num_cells}={raw:g} must be an integer')
    if not 0<=nf<=g.Ns:raise ValueError('Particle number outside 0..Ns')
    if args.V is not None:Vs=list(args.V)
    else:
        if args.Vstep<=0 or args.Vmax<args.Vmin:raise ValueError('Invalid V grid')
        Vs=[float(round(x,12)) for x in np.arange(args.Vmin,args.Vmax+.5*args.Vstep,args.Vstep)]
    if not Vs:raise ValueError('Empty V grid')
    if any(not math.isfinite(x) for x in Vs):raise ValueError('Nonfinite V')
    Vs=sorted(set(Vs))
    ident={'channel_schema':'hf_pseudospin_E_v1','cluster':args.cluster,'t':args.t,'t1':args.t1,'t2':args.t2,
           'nf':nf,'symmetry':args.symmetry,'depth':args.depth,'low':args.low,
           'extra_loops':not args.no_extra_loops,'degeneracy_tol':args.degeneracy_tol,
           'pair':args.pair,'mu':args.mu,'max_adjacent_basis':args.max_adjacent_basis,
           'save_state':args.save_state,'pole_tol':args.pole_tol}
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    meta=root/'scan_metadata.json'
    if meta.exists() and not metadata_match(meta,ident):
        raise RuntimeError('Output folder belongs to different (t1,t2,n,symmetry,depth,...) scan. '
                           'Use a new --output directory.')
    meta.write_text(json.dumps(ident,indent=2),encoding='utf-8')
    print(f'Cluster {g.name}, Ns={g.Ns}, Nf={nf}, n={nf/g.num_cells:.8f}, '
          f't1={args.t1}, t2={args.t2}, symmetry={args.symmetry}',flush=True)
    solver=VScanSolver(g,nf,args.symmetry,args.low,args.max_basis)
    # An extended V grid must preserve previously completed V points.
    # Only accept files belonging to exactly the same physical scan identity.
    rows_by_v={}
    for existing in root.glob('V_*/responses.json'):
        saved=json.loads(existing.read_text(encoding='utf-8'))
        if saved.get('parameters')==ident:
            rows_by_v[float(saved['V'])]=saved['row']
    failed=[]
    for V in Vs:
        tic=time.monotonic()
        folder=unique_dir(root,g,nf,V)
        csvpath=folder/'responses.json'
        try:
            if csvpath.exists() and not args.recompute:
                data=json.loads(csvpath.read_text(encoding='utf-8'))
                if data.get('parameters')!=ident or abs(data.get('V',float('inf'))-V)>1e-12:
                    raise RuntimeError(f'Mismatched cached result {csvpath}')
                rows_by_v[V]=data['row']
                print(f'Reuse V={V:g}',flush=True)
                continue
            folder.mkdir(parents=True,exist_ok=True)
            H,e0,psis,records,incomplete,residual=solver.solve(V,args.degeneracy_tol)
            channels,lead=channel_specs(g,solver.occ,psis,include_extra=not args.no_extra_loops)
            row={'V':V,'n':nf/g.num_cells,'Nf':nf,'E0':e0,
                 'ground_dim':psis.shape[1],'ground_incomplete':incomplete,
                 'residual':residual}
            spectra={}
            definitions={}
            for ch in channels:
                name=ch['name']
                resp=one_channel(ch,H,e0,psis,solver.states,solver.occ,g,args.depth,
                                 args.pole_tol,incomplete)
                row[f'{name}_chi']=resp['chi_static']
                row[f'{name}_chi_regular']=resp['chi_regular']
                row[f'{name}_status']=resp['status']
                row[f'{name}_zero_weight']=resp['zero_mode_weight']
                row[f'{name}_krylov_difference']=resp['max_krylov_change']
                spectra[name]=resp['spectra']
                definitions[name]={k:v for k,v in ch.items() if k not in ('loops','weights','weight_vectors','orbital_vector')}
            # Fixed E-type CO susceptibility traces, averaged over the
            # two real components. Keep separate x/y poles and weights.
            for parity in ('even','odd'):
                label=f'CO_E_{parity}'
                names=[f'{label}_x',f'{label}_y']
                statuses=[row[f'{name}_status'] for name in names]
                status=next((s for s in
                    ('incomplete_ground_manifold','zero_mode','unresolved_pole')
                    if s in statuses),'ok')
                row[f'{label}_status']=status
                row[f'{label}_chi']=(
                    sum(row[f'{name}_chi'] for name in names)/2
                    if status=='ok' else np.nan)
                row[f'{label}_chi_regular']=sum(
                    row[f'{name}_chi_regular'] for name in names)/2
                row[f'{label}_zero_weight']=sum(
                    row[f'{name}_zero_weight'] for name in names)/2
                row[f'{label}_krylov_difference']=max(
                    row[f'{name}_krylov_difference'] for name in names)
                definitions[label]={'definition':
                    'Average of xx and yy static susceptibilities in fixed '
                    f'physical E-type {parity} q=Gamma sector. '
                    'x and y spectra are archived separately; this is not '
                    'a full off-diagonal dynamical response matrix.'}
            if args.pair:
                pres,pspec,pmode=pairing_response(g,nf,V,e0,psis,solver.states,
                    args.depth,args.pole_tol,mu_override=args.mu,
                    max_adjacent_basis=args.max_adjacent_basis)
                row['SC_pair_chi']=pres['chi_static']
                row['SC_pair_chi_regular']=pres['chi_regular']
                row['SC_pair_status']=pres['status']
                row['SC_pair_zero_weight']=pres['zero_mode_weight']
                row['SC_pair_krylov_difference']=pres['max_krylov_change']
                spectra['SC_pair']=pspec
                definitions['SC_pair']={'definition':'leading equal-time bond-pair eigenmode at each V; '
                  'source Δ+Δ†, ||v||=1; μ from Nf±1 midgap',**pmode}
            row['CO_leading_q1']=lead['q'][0];row['CO_leading_q2']=lead['q'][1]
            row['CO_leading_equal_time_lambda']=lead['structure_factor']
            row['seconds']=time.monotonic()-tic
            # The spectra are saved as separate arrays, with channel+source indexes.
            arrays={}
            for ch,rs in spectra.items():
                for ix,result in enumerate(rs):
                    arrays[f'{ch}_src{ix}_poles']=result['poles']
                    arrays[f'{ch}_src{ix}_weights']=result['weights']
            np.savez_compressed(folder/'spectra.npz',**arrays)
            # These two files retain compatibility with the separate native
            # zero-T Green-function evaluator. In a degenerate manifold the
            # first vector is merely a representative; all vectors are saved.
            geometry=g.metadata()
            geometry.update(Nf=nf,V=V,solver='native susceptibility V scan',
                            ground_manifold_dim=int(psis.shape[1]))
            (folder/'geometry.json').write_text(json.dumps(geometry,indent=2,
                default=safe_json),encoding='utf-8')
            if args.save_state:
                np.savez_compressed(folder/'ground_states.npz',states=solver.states,
                    psi=psis,E0=e0,Nf=nf)
                np.savez_compressed(folder/'ground_state.npz',psi=psis[:,0],
                    basis_states=solver.states,e0=e0,Nf=nf)

            (folder/'sectors.json').write_text(json.dumps(records,indent=2,default=safe_json),encoding='utf-8')
            result={'parameters':ident,'V':V,'row':row,'definitions':definitions,
                    'charge_leading_vector':lead['orbital_vector'],
                    'note':'chi_static is NaN on zero-mode/incomplete/unstable poles; χregular excludes these poles.'}
            csvpath.write_text(json.dumps(result,indent=2,default=safe_json),encoding='utf-8')
            rows_by_v[V]=row
            print(f'V={V:+g} E0={e0:.9f}, gs={psis.shape[1]}, '
                  f'LCsame={row["LC_same_chi"]:.4g}, LCopp={row["LC_opposite_chi"]:.4g} '
                  f'({row["seconds"]:.1f}s)',flush=True)
        except Exception as e:
            failed.append({'V':V,'error':repr(e)})
            print('FAILED V=',V,repr(e),flush=True)
            if not args.continue_on_error:raise
        finally:
            if rows_by_v:
                all_rows=[rows_by_v[v] for v in sorted(rows_by_v)]
                fieldnames=list(all_rows[0]);out=root/'susceptibility.csv'
                with out.open('w',encoding='utf-8',newline='') as fp:
                    w=csv.DictWriter(fp,fieldnames=fieldnames)
                    w.writeheader();w.writerows(all_rows)
    (root/'failures.json').write_text(json.dumps(failed,indent=2),encoding='utf-8')
    if rows_by_v: plot_results(root,[rows_by_v[v] for v in sorted(rows_by_v)])
    print(f'Wrote {len(rows_by_v)} V points to {root}',flush=True)


def plot_results(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=sorted(rows,key=lambda x:x['V'])
    xx=np.array([r['V'] for r in rows],float)
    groups=[('charge',['CO_E_even','CO_E_odd','CO_AB','CO_leading']),
            ('charge_components',['CO_E_even_x','CO_E_even_y','CO_E_odd_x','CO_E_odd_y']),
            ('loop_current',['LC_same','LC_opposite','LC_square','LC_hexagon'])]
    if 'SC_pair_chi' in rows[0]:groups.append(('pairing',['SC_pair']))

    def draw(ax,name):
        key=f'{name}_chi'
        if key not in rows[0]:return
        yy=np.array([r.get(key,np.nan) for r in rows],float)
        reg=np.array([r.get(f'{name}_chi_regular',np.nan) for r in rows],float)
        bad=np.array([r.get(f'{name}_status')!='ok' for r in rows])
        ax.plot(xx,yy,'-o',label=name,linewidth=1.7,markersize=4)
        # A flagged point is NOT a finite static χ. Its finite regular part is
        # shown only as an open X and is never connected to valid points.
        if np.any(bad):
            ax.scatter(xx[bad],reg[bad],marker='x',s=60,
                       label=name+' regular part (flagged)')

    for group,names in groups:
        fig,ax=plt.subplots(figsize=(8,4.8))
        for name in names:draw(ax,name)
        ax.set_xlabel('V / t (intra-triangle density interaction)')
        ax.set_ylabel(r'Static susceptibility $\chi_O(0)$')
        ax.set_title(f'{group}: n={rows[0]["n"]:.6g} per primitive cell')
        ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout()
        fig.savefig(root/f'chi_{group}.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.8))
    for name in ['CO_E_even','CO_E_odd','LC_same','LC_opposite']:draw(ax,name)
    ax.set_xlabel('V / t');ax.set_ylabel(r'Static $\chi_O(0)$')
    ax.set_title(f'Zero-T ED responses, n={rows[0]["n"]:.6g}')
    ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(root/'chi_overview.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.8))
    zero_lines=0
    for name in ['CO_E_even','CO_E_odd','CO_AB','CO_leading','LC_same','LC_opposite','LC_square','LC_hexagon']:
        key=f'{name}_zero_weight'
        if key not in rows[0]:continue
        z=np.array([r.get(key,np.nan) for r in rows],float)
        if np.isfinite(z).any() and np.nanmax(z)>1e-10:
            ax.plot(xx,z,'o-',label=name);zero_lines+=1
    if zero_lines:
        ax.set_xlabel('V / t')
        ax.set_ylabel(r'Weight of $P_0(O-\overline{O})P_0$')
        ax.set_title('Degenerate-ground-manifold zero-mode response')
        ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout()
        fig.savefig(root/'chi_zero_mode_weight.png',dpi=180)
    plt.close(fig)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cluster',choices=['ring3','3x1','2x2'],default='ring3')
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--n',type=float,default=None,help='particles per six-site primitive cell; must map to integer Nf')
    g.add_argument('--nf',type=int,default=None,help='exact integer total number of particles')
    p.add_argument('--t',type=float,default=1.)
    p.add_argument('--t1',type=float,default=.5)
    p.add_argument('--t2',type=float,default=.5)
    p.add_argument('--V',type=float,nargs='+',default=None,help='Explicit V scan values')
    p.add_argument('--Vmin',type=float,default=-2.)
    p.add_argument('--Vmax',type=float,default=2.)
    p.add_argument('--Vstep',type=float,default=.5)
    p.add_argument('--symmetry',choices=['point','translation','none'],default='point')
    p.add_argument('--low',type=int,default=5,help='lowest eigenvectors kept per symmetry block')
    p.add_argument('--depth',type=int,default=60,help='Lanczos Krylov depth for susceptibilities')
    p.add_argument('--pole-tol',type=float,default=1e-8)
    p.add_argument('--degeneracy-tol',type=float,default=1e-7)
    p.add_argument('--max-basis',type=int,default=70_000)
    p.add_argument('--no-extra-loops',action='store_true',help='skip square and hexagon channels')
    p.add_argument('--pair',action='store_true',help='also compute leading spinless bond-pair susceptibility using Nf±2')
    p.add_argument('--mu',type=float,default=None,help='chemical potential for pairing source (default Nf±1 midgap)')
    p.add_argument('--max-adjacent-basis',type=int,default=90_000)
    p.add_argument('--save-state',action='store_true')
    p.add_argument('--recompute',action='store_true')
    p.add_argument('--continue-on-error',action='store_true')
    p.add_argument('--output',default='results/susceptibility_scan')
    return p


if __name__=='__main__':run(parser().parse_args())
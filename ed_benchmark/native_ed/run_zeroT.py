#!/usr/bin/env python3
"""Native NumPy/SciPy symmetry-resolved zero-T Ruby-lattice ED (NO QuSpin).

Use --no-observables to validate energy and block dimensions first.
All independent momentum/point-group blocks are searched; never assume Gamma.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator
from geometry import RubyPBC
from core import FixedHamiltonian,lowest,full_hermiticity_error
from symmetry import symmetry_projectors,gauge_back
from observables import build_observables,op_sum,current_terms,lanczos_spectral,max_current_in_manifold


def native_json(x):
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.floating):return float(x)
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,(complex,np.complexfloating)):
        return {'real':float(x.real),'imag':float(x.imag)}
    raise TypeError(type(x).__name__)


def write_json(path,item):
    Path(path).write_text(json.dumps(item,indent=2,default=native_json),encoding='utf-8')


def solve_all(g,nf,V,symmetry='point',low=3,max_dim=900000,
              method='sparse',no_twisted=False):
    it=symmetry_projectors(g,nf,symmetry,max_basis=max_dim,
                           use_signed=(not no_twisted))
    H=None;records=[];winners=[];dim_total=0;state_list=None;gauge=None
    for lab,Q,states,phases in it:
        if H is None:
            H=FixedHamiltonian(g,nf,V,states,method=method)
            state_list=states;gauge=phases
        d=Q.shape[1]
        dim_total+=d
        if d==0:
            records.append({'labels':{k:v for k,v in lab.items() if k!='generators'},'dim':0,'eigenvalues':[]})
            continue
        if method=='sparse':
            HB=(Q.conj().T@(H.operator@Q)).tocsr()
            diff=HB-HB.getH()
            if diff.nnz and np.max(np.abs(diff.data))>1e-8:raise AssertionError('Block Hamiltonian is non-Hermitian')
        else:
            def matvec(v,Q=Q):return Q.conj().T@H.operator.matvec(Q@v)
            HB=LinearOperator((d,d),matvec=matvec,dtype=complex)
        energies,vecs=lowest(HB,min(low,d))
        rec={'labels':{k:v for k,v in lab.items() if k!='generators'},
             'dim':int(d),'eigenvalues':energies.tolist()}
        records.append(rec)
        for k,en in enumerate(energies):
            winners.append((float(en),lab,Q,vecs[:,k]))
    winners.sort(key=lambda x:x[0])
    if dim_total!=len(state_list):raise AssertionError(f'Symmetry blocks cover {dim_total}/{len(state_list)}')
    return H,records,winners,state_list,gauge


def calc_chi(g,H,states,psi,e0,depth,data):
    tri=[p for p in g.cycles() if len(p)==3]
    out={}
    for key,weights in [('LC_same',[1.]*len(tri)),
                        ('LC_opposite',[1. if x[0]%6<3 else -1. for x in tri])]:
        terms=[]
        for lp,w in zip(tri,weights):
            terms.extend([(z*w/len(tri),op) for z,op in current_terms(g,lp)])
        action=op_sum(psi,states,g.Ns,terms)
        source=action-np.vdot(psi,action)*psi
        pole,weight=lanczos_spectral(H.operator,source,e0,m=depth)
        nz=pole>1e-8
        out[key+'_poles']=pole
        out[key+'_weights']=weight
        out[key+'_chi_static']=float(np.sum(2*weight[nz]/pole[nz]))
        out[key+'_zero_mode']=bool(np.any((pole<=1e-8)&(weight>1e-8)))
    # Charge response along the leading connected-density eigenmode;
    # not a claim that this channel is necessarily an ordered phase.
    eigen, vectors=np.linalg.eigh(data['density_connected'])
    pattern=np.asarray(vectors[:,-1].real)
    occup=np.array([[(int(b)>>(g.Ns-1-i))&1 for i in range(g.Ns)]
                   for b in states],dtype=np.float64)
    amp=occup@pattern
    action=amp*psi
    source=action-np.vdot(psi,action)*psi
    pole,weight=lanczos_spectral(H.operator,source,e0,m=depth)
    nz=pole>1e-8
    out['CO_charge_mode']=pattern
    out['CO_poles']=pole
    out['CO_weights']=weight
    out['CO_chi_static']=float(np.sum(2*weight[nz]/pole[nz]))
    out['CO_zero_mode']=bool(np.any((pole<=1e-8)&(weight>1e-8)))
    return out


def main(a):
    g=RubyPBC(a.cluster,t=a.t,t1=a.t1,t2=a.t2)
    dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    tic=time.monotonic()
    H,records,winners,states,gauge=solve_all(
        g,a.nf,a.V,a.symmetry,a.low,a.max_basis,a.method,a.no_twisted)
    if not winners:raise RuntimeError('No eigenstates found')
    E0=winners[0][0]
    result={'cluster':a.cluster,'Ns':g.Ns,'Nf':a.nf,'V':a.V,'t':g.t,
            'symmetry':a.symmetry,'method':a.method,'E0':E0,
            'global_ground_sector':{k:v for k,v in winners[0][1].items() if k!='generators'},
            'low_energies':list(map(float,[w[0] for w in winners[:min(len(winners),40)]])),
            'degenerate_within_tolerance':int(sum(x[0]-E0<=a.degeneracy_tol for x in winners)),
            'gauge_signed_rotation_in_ground_block':bool(winners[0][1].get('signed_rotation')),'temperature':0.0,
            'elapsed_seconds':None}
    metadata=g.metadata()
    metadata.update({'Nf':a.nf,'V':a.V,'solver':'native NumPy/SciPy ED',
                     'bit_order':'orbital index 0 is most significant bit',
                     'site_gauge':None,
                     'current':'particle current in hbar=1 convention',
                     'rho_flux':'arg product rho_ij, not electromagnetic flux'})
    write_json(dest/'geometry.json',metadata)
    write_json(dest/'sectors.json',records)
    result['gauge_signed_rotation_sectors']=int(sum(bool(r['labels'].get('signed_rotation')) for r in records))
    if a.observables:
        if H.n>a.max_observable_basis:
            raise MemoryError(f'Observables require full basis {H.n} > limit {a.max_observable_basis}; use --no-observables or raise cap')
        e,lab,Q,v=winners[0]
        psi=gauge_back(np.asarray(Q@v).ravel(),states,g.Ns,gauge)
        psi=psi/np.linalg.norm(psi)
        if abs(np.vdot(psi,psi)-1)>1e-8:raise AssertionError('State norm error')
        # Residual check on physical Hamiltonian, including gauge undo.
        res=np.linalg.norm(H.operator@psi-E0*psi)
        result['ground_residual']=float(res)
        data=build_observables(g,psi,states,pairs=a.pair)
        if a.chi_lanczos:
            data.update(calc_chi(g,H,states,psi,E0,a.chi_lanczos,data))
        np.savez_compressed(dest/'equal_time.npz',**data)
        if a.save_psi:np.savez_compressed(dest/'ground_state.npz',psi=psi,basis_states=states,e0=E0,Nf=a.nf)
        result['charge_corr_max']=float(np.max(data['charge_eigs']))
        result['current_corr_max']=float(np.max(data['current_eigs']))
        if a.pair:result['pair_corr_max']=float(np.max(data['pair_eigs']))
        if a.max_current:
            manifold=[w for w in winners if w[0]-E0<=a.degeneracy_tol]
            if len(manifold)>1:
                V=np.column_stack([gauge_back(np.asarray(Q@vec).ravel(),states,g.Ns,gauge)
                                   for _,_,Q,vec in manifold])
                curr=max_current_in_manifold(g,V,states,[x[0]-E0 for x in manifold],a.degeneracy_tol)
                result['max_current']=[{'loop':item[1],'length':item[0],
                         'signed_J':item[2],'maximum_abs_J':item[3],
                         'coefficients':[[z.real,z.imag] for z in item[4]]}
                         for item in curr]
                result['manifold_energies']=[x[0] for x in manifold]
            else:result['max_current']='No resolved near-degenerate subspace'
    result['elapsed_seconds']=time.monotonic()-tic
    write_json(dest/'summary.json',result)
    print(json.dumps(result,indent=2,default=native_json))
    print('Saved:',dest)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cluster',choices=['ring3','3x1','2x2'],default='ring3')
    p.add_argument('--nf',type=int,default=6)
    p.add_argument('--V',type=float,default=1.0)
    p.add_argument('--t',type=float,default=1.0)
    p.add_argument('--t1',type=float,default=.5)
    p.add_argument('--t2',type=float,default=.5)
    p.add_argument('--symmetry',choices=['point','translation','none'],default='point')
    p.add_argument('--low',type=int,default=3)
    p.add_argument('--no-twisted',action='store_true')
    p.add_argument('--no-observables',dest='observables',action='store_false')
    p.add_argument('--pair',action='store_true')
    p.add_argument('--chi-lanczos',type=int,default=0)
    p.add_argument('--max-current',action='store_true')
    p.add_argument('--degeneracy-tol',type=float,default=1e-7)
    p.add_argument('--save-psi',action='store_true')
    p.add_argument('--method',choices=['sparse','matrixfree'],default='sparse')
    p.add_argument('--max-basis',type=int,default=900000)
    p.add_argument('--max-observable-basis',type=int,default=70000)
    p.add_argument('--out',default='results/ring3_N6_V1')
    return p

if __name__=='__main__':main(parser().parse_args())
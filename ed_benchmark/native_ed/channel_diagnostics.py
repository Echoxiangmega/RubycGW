#!/usr/bin/env python3
"""Extract comparable *within-channel* Ruby ED order diagnostics.

Reads existing native_ed point directories, writes channel_diagnostics.csv.
Does NOT assign thermodynamic CO/LC/SC phase labels or compare unlike
operator normalizations to choose a dominant phase.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def charge_momenta(conn, reps, momenta):
    """6x6 projected charge-covariance eigenvalues at allowed crystal momenta.

    F[:,alpha]=exp(2*pi*i*k.R)/sqrt(Nc) on orbital alpha. For translation-
    invariant C, F^dag C F resolves q. Evecs are orbital charge patterns.
    """
    nc=len(reps);Ns=6*nc
    if conn.shape!=(Ns,Ns):raise ValueError('Wrong charge covariance shape')
    ret=[]
    for k in momenta:
        F=np.zeros((Ns,6),dtype=complex)
        for cell,R in enumerate(reps):
            phi=2*np.pi*float(np.dot(k,R))
            for alpha in range(6):F[6*cell+alpha,alpha]=np.exp(1j*phi)/np.sqrt(nc)
        block=F.conj().T@conn@F
        block=(block+block.conj().T)/2
        eigenvalues,eigenvectors=np.linalg.eigh(block)
        ret.append({'k':list(k),'value':float(eigenvalues[-1]),
                    'orbital_mode':[[float(z.real),float(z.imag)] for z in eigenvectors[:,-1]]})
    return ret


def read_point(folder):
    folder=Path(folder)
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    spec=json.loads((folder/'geometry.json').read_text(encoding='utf-8'))
    datafile=folder/'equal_time.npz'
    if not datafile.is_file():return None
    with np.load(datafile,allow_pickle=False) as data:
        keys=set(data.files)
        lc_len=np.asarray(data['loop_length'])
        loops=np.asarray(data['loops'])
        C=np.asarray(data['current_corr'],complex)
        mean=np.asarray(data['current_mean'],complex)
        C=(C-np.outer(mean.conj(),mean))
        C=(C+C.conj().T)/2
        tri=np.flatnonzero(lc_len==3)
        if len(tri)!=2*len(spec['reps']):raise AssertionError('Unexpected triangle count')
        # All stored elementary plaquettes oriented CCW; physical same is
        # equal signed loop current on A and B triangles.
        s=np.array([1. if loops[p,0]%6 < 3 else -1. for p in tri])
        u=np.ones(len(tri),complex)/len(tri)
        v=s.astype(complex)/len(tri)
        T=C[np.ix_(tri,tri)]
        same=float(np.vdot(u,T@u).real)
        opposite=float(np.vdot(v,T@v).real)
        tri_largest=float(np.linalg.eigvalsh(T)[-1]/len(tri))
        charge=np.asarray(data['density_connected'],complex)
        charge=(charge+charge.conj().T)/2
        qc=charge_momenta(charge,spec['reps'],spec['momenta'])
        qmax=max(qc,key=lambda q:q['value'])
        row=dict(cluster=spec['name'],Ns=spec['Ns'],Nf=spec['Nf'],
                 n=float(spec['Nf'])/len(spec['reps']),V=spec['V'],
                 t=spec['t']['t'],t1=spec['t']['t1'],t2=spec['t']['t2'],
                 E0=summary['E0'],ground_degeneracy=summary.get('degenerate_within_tolerance'),
                 CO_qmax=qmax['value'],CO_q1=qmax['k'][0],CO_q2=qmax['k'][1],
                 CO_charge_eigenvalue=float(np.linalg.eigvalsh(charge)[-1]),
                 LC_same_S=same,LC_opposite_S=opposite,
                 LC_same_to_opposite_ratio=same/opposite if opposite>1e-15 else float('nan'),
                 LC_triangle_max_eigenvalue_per_triangle=tri_largest,
                 loop_flux_finite_count=int(np.isfinite(data['loop_rho_flux']).sum()),
                 pair_lambda_max=float(np.max(data['pair_eigs'])) if 'pair_eigs' in keys else float('nan'),
                 LC_same_chi=float(data['LC_same_chi_static']) if 'LC_same_chi_static' in keys else float('nan'),
                 LC_opposite_chi=float(data['LC_opposite_chi_static']) if 'LC_opposite_chi_static' in keys else float('nan'),
                 CO_mode_chi=float(data['CO_chi_static']) if 'CO_chi_static' in keys else float('nan'),
                 LC_same_zero_mode=bool(data['LC_same_zero_mode']) if 'LC_same_zero_mode' in keys else None,
                 LC_opposite_zero_mode=bool(data['LC_opposite_zero_mode']) if 'LC_opposite_zero_mode' in keys else None,
                 result_dir=str(folder))
    return row,qc


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True,help='A point folder or a scan root with summary.csv')
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    root=Path(args.input)
    if (root/'summary.json').exists():folders=[root]
    elif (root/'summary.csv').exists():
        with (root/'summary.csv').open(encoding='utf-8',newline='') as f:
            folders=[root/row['result_dir'] for row in csv.DictReader(f)]
    else:raise ValueError(f'Cannot locate point or scan: {root}')
    rows=[];momenta={}
    for folder in folders:
        try:got=read_point(folder)
        except FileNotFoundError:continue
        if got is None:continue
        row,qc=got
        rows.append(row);momenta[str(folder)]=qc
    if not rows:raise RuntimeError('No equal_time.npz files to analyze (run without --no-observables).')
    out=Path(args.out) if args.out else (root/'channel_diagnostics.csv' if root.is_dir() and (root/'summary.csv').exists() else root/'channel_diagnostics.csv')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    jsonout=out.with_suffix('.q_modes.json')
    jsonout.write_text(json.dumps(momenta,indent=2),encoding='utf-8')
    print('Wrote',out,'points',len(rows));print('Charge-q details:',jsonout)
    print('WARNING: These are normalized channel diagnostics, not automatic thermodynamic phase labels.')

if __name__=='__main__':main()
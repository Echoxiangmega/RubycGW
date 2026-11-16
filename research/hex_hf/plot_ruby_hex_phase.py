#!/usr/bin/env python3
"""Phase plot for symmetry-reduced Ruby-HF results, with translation-aware order tests.

Unlike the initial plotter, this version does NOT identify a BOW from
std(Re chi_b) across distinct primitive bond species. Different bonds already
have different expectation values in a translation-invariant normal metal.
For modulation we subtract the primitive-translation average for EACH of the
12 primitive bond species. SD/ISD are labels for overlap with a chosen Ruby
seed form factor, not rigorous irreducible-representation assignments.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from ruby_hex_hf import RubyHexHF

AXES = {'Vtri':'V_tri','Vt1':'V_t1','Vt2':'V_t2'}
LABELS = [
    'No converged solution',
    'No detected symmetry breaking',
    'Finite-Q CDW',
    'SD-like real BOW',
    'ISD-like real BOW',
    'Other real BOW',
    'LC (same circulation)',
    'LC (opposite circulation)',
    'LC (mixed pattern)',
    'LC + finite-Q CDW',
    'Intra-cell charge order',
    'Intra-cell bond order',
    'CDW + SD-like BOW',
    'CDW + ISD-like BOW',
    'CDW + other BOW',
    'Modulated BOW + LC',
]
COLORS = [
    '#a8abb1','#e8ebed','#c8a889','#6aadd3','#b67d55','#b8bda4',
    '#be8ebd','#8268b4','#b59b70','#87a899','#dfc5b3','#9ba6c5',
    '#b6d3d3','#b9b3a6','#b3b7aa','#a88fbd',
]
SHORT = ['?','N','CDW','SD','ISD','BOW','LC+','LC-','LC','LC+CDW','q0-CDW','q0-BOW', 'CDW+SD','CDW+ISD','CDW+BOW','BOW+LC']


def order_diagnostics(m, z, row, a):
    chi = np.asarray(z['chi'], dtype=complex)
    n = np.asarray(z['n'], dtype=float)
    if chi.size != m.B or n.size != m.N:
        raise ValueError('Saved state dimensions do not match its period')
    # Bonds are enumerated primitive cell by primitive cell: c*12 + edge.
    # Every translation-equivalent primitive edge is compared to its mean.
    cr = np.real(chi).reshape(m.ncell, 12)
    dchi = cr - cr.mean(axis=0, keepdims=True)
    bow_rms = float(np.sqrt(np.mean(dchi**2)))
    form, _ = m._real_star_formfactor()
    form = form.reshape(m.ncell, 12)
    form_q = form - form.mean(axis=0, keepdims=True)
    denom = float(np.sqrt(np.sum(dchi**2) * np.sum(form_q**2)))
    form_overlap = float(np.sum(dchi * form_q) / denom) if denom > 1e-18 else 0.
    # Site densities can also differ inside the primitive unit cell without
    # breaking primitive translations. Measure genuine supercell modulation.
    nr = n.reshape(m.ncell, 6)
    dn = nr - nr.mean(axis=0, keepdims=True)
    charge_q_rms = float(np.sqrt(np.mean(dn**2)))
    J = 2*np.imag(m.tb * chi)
    current_rms = float(np.sqrt(np.mean(J**2)))
    charge = charge_q_rms > a.charge_threshold
    bow = bow_rms > a.bond_threshold
    lc = current_rms > a.current_threshold
    plus = abs(float(row['J_plus'])); minus = abs(float(row['J_minus']))
    if lc:
        if charge:
            code=9
        elif bow:
            code=15
        elif plus > 2*minus and plus > a.current_threshold:
            code=6
        elif minus > 2*plus and minus > a.current_threshold:
            code=7
        else:
            code=8
    elif charge and bow:
        if form_overlap >= a.min_form_overlap: code=12
        elif form_overlap <= -a.min_form_overlap: code=13
        else: code=14
    elif charge:
        code=2
    elif bow:
        if form_overlap >= a.min_form_overlap:
            code=3
        elif form_overlap <= -a.min_form_overlap:
            code=4
        else:
            code=5
    else:
        code=1
        # The primitive translation test deliberately ignores q=0 orders.
        # If the *bare* model has C6, deviations from it at q=0 are spontaneous
        # and should be classified rather than marked normal.
        if (m.symmetry != 'c6' and
            abs(m.tmap['t1']-m.tmap['t2']) < 1e-12 and
            abs(m.Vmap['t1']-m.Vmap['t2']) < 1e-12):
            perm,bperm,sgn = m.rotation_c6
            c6_n = float(np.max(np.abs(n[perm]-n)))
            rotated_chi = np.where(sgn>0,chi,chi.conj())
            c6_chi = float(np.max(np.abs(chi[bperm] - rotated_chi)))
            if c6_n > a.charge_threshold: code=10
            elif c6_chi > a.bond_threshold: code=11
    d={
        'current_rms_recomputed':current_rms,
        'charge_q_rms':charge_q_rms,
        'real_bond_q_rms':bow_rms,
        'sd_isd_form_overlap':form_overlap,
        'has_finite_q_cdw':int(charge),
        'has_real_bow_modulation':int(bow),
        'has_loop_current':int(lc),
    }
    return code,d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',nargs='+',help='Directories with scan.json and point_*/best/state.npz')
    p.add_argument('--x',choices=AXES,default='Vtri')
    p.add_argument('--y',choices=AXES,default='Vt1')
    p.add_argument('--output',default='ruby_hex_phase_fixed')
    p.add_argument('--current-threshold',type=float,default=1e-4)
    p.add_argument('--charge-threshold',type=float,default=1e-4)
    p.add_argument('--bond-threshold',type=float,default=2e-3,
                   help='RMS real-bond modulation across primitive translations')
    p.add_argument('--min-form-overlap',type=float,default=.45,
                   help='Absolute normalized overlap needed for SD/ISD-like naming')
    p.add_argument('--annotate',choices=['none','phase','sector'],default='none')
    a=p.parse_args()
    if a.x==a.y: raise ValueError('x and y must differ')
    allrows=[]; config=None; rows=[]
    for folder in a.directories:
        path=Path(folder)
        for row in json.loads((path/'scan.json').read_text()):
            setting={k:row.get(k) for k in (
                'period','Nk1','Nk2','temperature','filling_per_primitive',
                'mu_fixed','t','t1','t2')}
            if config is None:config=setting
            elif setting!=config:
                raise ValueError(f'Incompatible runs! First={config}, current={setting}')
            allrows.append(row)
            if row['valid']:
                state=path / f'point_{int(row["point"]):04d}' / 'best' / 'state.npz'
                if not state.exists():raise FileNotFoundError(state)
                rows.append((row,state))
    if not allrows:raise RuntimeError('No scan rows supplied')
    kx,ky=AXES[a.x],AXES[a.y]
    xs=sorted(set(round(float(row[kx]),10) for row in allrows))
    ys=sorted(set(round(float(row[ky]),10) for row in allrows))
    labels=np.zeros((len(ys),len(xs)),dtype=int)
    best={}
    for row,state in rows:
        key=(round(float(row[kx]),10),round(float(row[ky]),10))
        pot=float(row['F'] if row['mu_fixed'] is None else row['Omega'])
        if key not in best or pot < best[key][0]:best[key]=(pot,row,state)
    records=[]
    for x in xs:
        for y in ys:
            b=best.get((x,y))
            if b is None:
                records.append({'x':x,'y':y,'phase':LABELS[0],'class_index':0})
                continue
            _,row,state=b
            m=RubyHexHF(row['period'],row['symmetry'],(1,1),
                t=row['t'],t1=row['t1'],t2=row['t2'],
                Vtri=row['V_tri'],Vt1=row['V_t1'],Vt2=row['V_t2'])
            with np.load(state) as z:
                code,diag=order_diagnostics(m,z,row,a)
            labels[ys.index(y),xs.index(x)]=code
            records.append({
                'x':x,'y':y,'phase':LABELS[code],'class_index':code,
                'symmetry_constraint':row['symmetry'],
                'F':row['F'],'Omega':row['Omega'],
                'J_plus':row['J_plus'],'J_minus':row['J_minus'],
                'charge_mod_max_legacy':row.get('charge_mod_max'),
                'real_bond_std_legacy_not_order':row.get('real_bond_std'),
                **diag,
            })
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in records for k in r))
    with open(str(out)+'.csv','w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(records)
    fig,ax=plt.subplots(figsize=(max(6,1.08*len(xs)+1.5),max(4.8,0.8*len(ys)+1.5)),layout='constrained')
    ax.imshow(labels,origin='lower',cmap=ListedColormap(COLORS),
        norm=BoundaryNorm(np.arange(-.5,len(COLORS)+.5),len(COLORS)),aspect='auto')
    ax.set_xticks(np.arange(len(xs)),[f'{v:g}' for v in xs])
    ax.set_yticks(np.arange(len(ys)),[f'{v:g}' for v in ys])
    ax.set_xlabel(a.x);ax.set_ylabel(a.y)
    sector=', '.join(sorted(set(r['symmetry'] for r in allrows)))
    ax.set_title(f'Ruby {config["period"]} HF (constraints: {sector})')
    if a.annotate!='none':
        for (x,y),(_,row,_) in best.items():
            ix=xs.index(x);iy=ys.index(y)
            txt=SHORT[labels[iy,ix]] if a.annotate=='phase' else row['symmetry'].upper()
            ax.text(ix,iy,txt,fontsize=8,ha='center',va='center')
    present=sorted(set(labels.ravel()))
    legend=[Patch(facecolor=COLORS[i],edgecolor='#667085',label=LABELS[i]) for i in present]
    ax.legend(handles=legend,loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
    fig.savefig(str(out)+'.png',dpi=190,bbox_inches='tight')
    fig.savefig(str(out)+'.svg',bbox_inches='tight')
    print('Saved',out.with_suffix('.png'),out.with_suffix('.csv'))
    print('NOTE: SD/ISD labels indicate overlap with Ruby seed form factors, not kagome irreps.')

if __name__=='__main__':main()
#!/usr/bin/env python3
"""Plot independent zero-T ED channel diagnostics (NOT categorical phase labels).

Requires matplotlib only for plotting; native ED calculations require NumPy/SciPy.
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def edges(v):
    v=np.asarray(sorted(v),float)
    if len(v)==1:return np.array([v[0]-.5,v[0]+.5])
    mid=(v[1:]+v[:-1])/2
    return np.r_[v[0]-(mid[0]-v[0]),mid,v[-1]+(v[-1]-mid[-1])]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='Scan root or channel_diagnostics.csv')
    parser.add_argument('--output',default=None,help='Folder for PNG heatmaps')
    args=parser.parse_args()
    input_path=Path(args.input)
    csv_path=input_path/'channel_diagnostics.csv' if input_path.is_dir() else input_path
    if not csv_path.is_file():raise FileNotFoundError(csv_path)
    with csv_path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Empty diagnostics CSV')
    import matplotlib.pyplot as plt
    root=Path(args.output) if args.output else csv_path.parent/'diagnostic_figures'
    root.mkdir(parents=True,exist_ok=True)
    VV=sorted(set(float(r['V']) for r in rows))
    nn=sorted(set(float(r['n']) for r in rows))
    indexV={v:i for i,v in enumerate(VV)};indexn={v:i for i,v in enumerate(nn)}
    keys=[('CO_qmax','Leading charge covariance at allowed q'),
          ('LC_same_S','Physical same-direction triangle LC: connected S'),
          ('LC_opposite_S','Physical opposite-direction triangle LC: connected S'),
          ('LC_log_same_over_opposite','log10(LC same / LC opposite)'),
          ('LC_same_chi','Static same LC susceptibility'),
          ('LC_opposite_chi','Static opposite LC susceptibility'),
          ('CO_mode_chi','Static charge susceptibility (leading mode)'),
          ('pair_lambda_max','Spinless nearest-neighbor pair eigenvalue')]
    count=0
    for key,title in keys:
        A=np.full((len(nn),len(VV)),np.nan)
        for r in rows:
            if key=='LC_log_same_over_opposite':
                a=float(r['LC_same_S']);b=float(r['LC_opposite_S'])
                val=np.log10(a/b) if a>1e-14 and b>1e-14 else np.nan
            else:
                s=r.get(key,'')
                val=float(s) if s not in ('','None','nan') else np.nan
            A[indexn[float(r['n'])],indexV[float(r['V'])]]=val
        if not np.isfinite(A).any():continue
        fig,ax=plt.subplots(figsize=(8,5))
        cmap=plt.get_cmap('RdBu_r' if key=='LC_log_same_over_opposite' else 'viridis').copy()
        cmap.set_bad('0.85')
        vmax=None;vmin=None
        if key=='LC_log_same_over_opposite':
            bound=np.nanmax(np.abs(A));vmin=-max(bound,1e-12);vmax=max(bound,1e-12)
        im=ax.pcolormesh(edges(VV),edges(nn),np.ma.masked_invalid(A),cmap=cmap,
                         shading='flat',vmin=vmin,vmax=vmax)
        ax.set_xlabel('Intra-triangle V / t')
        ax.set_ylabel('Filling n = Nf / number of primitive cells')
        ax.set_title(title)
        ax.set_yticks(nn)
        ax.set_xticks(VV if len(VV)<=12 else VV[::max(1,len(VV)//12)])
        fig.colorbar(im,ax=ax,shrink=.9)
        fig.tight_layout()
        fig.savefig(root/f'{key}.png',dpi=190)
        plt.close(fig)
        count+=1
    print(f'Wrote {count} diagnostic heatmaps to {root}')
    print('These are channel strength maps; they are not assigned CO/LC/SC phases.')

if __name__=='__main__':main()
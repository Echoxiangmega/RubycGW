#!/usr/bin/env python3
"""Plot a 4x4 static E-type CO response matrix at a selected V, no ED rerun."""
import argparse
from pathlib import Path
import json
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Susceptibility scan root directory')
    p.add_argument('--V',type=float,required=True,help='Scanned V to inspect')
    p.add_argument('--regular',action='store_true',help='Use regular part if zero-mode flagged')
    args=p.parse_args()
    root=Path(args.input)
    matches=[]
    for f in root.glob('V_*/responses.json'):
        r=json.loads(f.read_text(encoding='utf-8'))
        if abs(float(r['V'])-args.V)<1e-8:matches.append(f.parent)
    if len(matches)!=1:
        raise ValueError(f'Expected one completed V={args.V} point, found {len(matches)}')
    folder=matches[0]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(folder/'co_e_matrix.npz') as z:
        status=str(z['status'])
        if status!='ok' and not args.regular:
            raise ValueError('This matrix has status '+status+'; use --regular for only the regular part')
        chi=np.array(z['chi_regular' if args.regular else 'chi_static'])
        labels=[str(s).replace('CO_E_','') for s in z['source_names']]
    vmax=np.nanmax(abs(chi)) or 1.
    fig,ax=plt.subplots(figsize=(6.0,5.2))
    im=ax.imshow(chi,cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    ax.set_xticks(range(4),labels,rotation=35,ha='right')
    ax.set_yticks(range(4),labels)
    for i in range(4):
        for j in range(4):
            shown='0' if abs(chi[i,j]) < 1e-10 else f'{chi[i,j]:.3g}'
            ax.text(j,i,shown,ha='center',va='center',
                    color='white' if abs(chi[i,j])>vmax*.65 else 'black')
    ax.set_title(f'E-type CO χ(0), V={args.V:g}; {status}'+
                 (' [regular only]' if args.regular else ''))
    fig.colorbar(im,ax=ax,shrink=.82)
    fig.tight_layout()
    dest=folder/('chi_E_matrix_regular.png' if args.regular else 'chi_E_matrix.png')
    fig.savefig(dest,dpi=200)
    print('Saved',dest)
    plt.close(fig)


if __name__=='__main__':main()
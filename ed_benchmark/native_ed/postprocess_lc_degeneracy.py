#!/usr/bin/env python3
"""Diagnose LC zero modes from a completed native ED V scan without rerunning ED.

Always plots and exports chi_regular and ground zero-weight from the CSV.
If each V point also has ground_states.npz (scan performed with --save-state),
calculates the exact projected current matrix, its eigenspectrum and Jmax.
Otherwise Jmax remains NaN: the zero-mode weight does not determine Jmax.
This utility never silently changes the original susceptibility.csv.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from geometry import RubyPBC
from scan_susceptibility import projected_current_manifold


def run(root):
    root=Path(root)
    path=root/'susceptibility.csv'
    with path.open(encoding='utf-8',newline='') as f:
        base=list(csv.DictReader(f))
    if not base:
        raise ValueError('Empty susceptibility.csv')
    meta=json.loads((root/'scan_metadata.json').read_text(encoding='utf-8'))
    g=RubyPBC(meta['cluster'],t=meta.get('t',1.),t1=meta['t1'],t2=meta['t2'])
    records=[]; found=0
    for row in sorted(base,key=lambda row:float(row['V'])):
        V=float(row['V'])
        out={'V':V,'n':float(row['n']), 'ground_dim':int(row['ground_dim']),
             'ground_incomplete':row['ground_incomplete'],
             'source_wavefunctions':'missing'}
        folder=root/f'V_{V:+.9f}'
        for name in ('LC_same','LC_opposite'):
            for src,dst in (('chi_regular','chi_regular'),('zero_weight','zero_weight'),
                            ('status','status')):
                value=row.get(f'{name}_{src}',np.nan)
                out[f'{name}_{dst}']=(value if src=='status' else float(value))
            out[f'{name}_Jmax_abs']=np.nan
            out[f'{name}_Jmax_per_triangle']=np.nan
        states_file=folder/'ground_states.npz'
        if states_file.exists():
            with np.load(states_file) as data:
                psis=np.asarray(data['psi']).copy()
                states=np.asarray(data['states']).copy()
            if psis.shape[1]!=out['ground_dim']:
                raise ValueError(f'ground_dim mismatch for V={V}')
            currents=projected_current_manifold(g,psis,states)
            for name,part in currents.items():
                if abs(part['zero_weight']-out[f'{name}_zero_weight'])>2e-6:
                    raise AssertionError(f'zero-weight mismatch at V={V}, {name}')
                out[f'{name}_Jmax_abs']=part['jmax_abs']
                out[f'{name}_Jmax_per_triangle']=part['jmax_per_triangle']
            out['source_wavefunctions']='ground_states.npz'
            found+=1
        records.append(out)
    outfile=root/'lc_degenerate_diagnostics.csv'
    with outfile.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    plot(records,root)
    print(f'Processed {len(records)} points; Jmax recovered at {found} points. Results in {root}')
    if not found:
        print('No saved ground multiplets: chi_regular and W0 plotted; Jmax cannot be reconstructed from W0 alone.')
    return records


def plot(records,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    X=np.array([r['V'] for r in records])
    dd=np.array([r['ground_dim']>1 for r in records])
    def draw(ax,values,label,color):
        yy=np.asarray(values,float)
        for i in range(len(X)-1):
            if np.isfinite(yy[i]) and np.isfinite(yy[i+1]):
                ax.plot(X[i:i+2],yy[i:i+2],color=color,
                        ls='--' if dd[i] or dd[i+1] else '-',lw=1.8)
        for mask,face in [(~dd,color),(dd,'white')]:
            sel=mask & np.isfinite(yy)
            if np.any(sel):
                ax.scatter(X[sel],yy[sel],s=46,facecolors=face,
                           edgecolors=color,zorder=3)
        ax.plot([],[],color=color,marker='o',label=label)
    def figplot(ykeys,title,ylabel,filename):
        fig,ax=plt.subplots(figsize=(8.2,4.9))
        for name,c in [('LC_same','tab:orange'),('LC_opposite','tab:green')]:
            draw(ax,[r.get(f'{name}_{ykeys}',np.nan) for r in records],name,c)
        ax.set_title(title);ax.set_xlabel('V / t');ax.set_ylabel(ylabel)
        ax.grid(alpha=.25);ax.legend();fig.tight_layout()
        fig.savefig(root/filename,dpi=180)
        plt.close(fig)
    figplot('chi_regular','Finite-energy LC susceptibility, excluding ground zero modes',
            r'$\chi_{\mathrm{regular}}(0)$ (NOT full static susceptibility)','lc_regular_vs_V.png')
    figplot('zero_weight','Current zero-mode weight inside ground multiplet',
            r'$W_0 = \mathrm{Tr}[(P_0OP_0-\bar O)^2]/d$','lc_zero_mode_vs_V.png')
    if any(np.isfinite(r['LC_same_Jmax_abs']) or np.isfinite(r['LC_opposite_Jmax_abs'])
           for r in records):
        figplot('Jmax_abs','Maximum current achievable in ground multiplet',
                r'$\max_{\psi\in P_0} |\langle\psi|O_J|\psi\rangle|$',
                'lc_Jmax_vs_V.png')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Existing scan folder')
    a=p.parse_args()
    run(a.input)

if __name__=='__main__':main()
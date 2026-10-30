#!/usr/bin/env python3
"""Ruby ED benchmark: full nonlocal LC correlations versus V and V=0.

From native_ed directory (PowerShell or CMD, one-line command):
  python plot_lc_structure.py --input results\\chi_ring3_n2 --V 0 1 2 3

Uses a saved complete ground multiplet when available; otherwise solves only
requested points, including V=0 by default. A scan does not generally store
all triangle-triangle correlations, so scalar susceptibility.csv is insufficient.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from geometry import RubyPBC
from plot_current_pattern import load_scan_metadata,load_or_compute_ground
from lc_structure import (lc_correlations,momentum_structure,ground_jmax,
                          pairwise_nonlocal,summarize_shells)


def _json_default(x):
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.floating):return float(x)
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.bool_):return bool(x)
    raise TypeError(f'Cannot encode {type(x)}')


def _write_table(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def _point_cache_path(folder):return folder/'lc_structure.npz'


def _store_point(folder,g,states,psis,V,source):
    folder.mkdir(parents=True,exist_ok=True)
    res=lc_correlations(g,states,psis)
    mom=momentum_structure(g,res)
    J=ground_jmax(g,res)
    pairs=pairwise_nonlocal(g,res)
    shells=summarize_shells(pairs)
    sub=res['sublattice'];loops=res['loops']
    np.savez_compressed(_point_cache_path(folder),
        V=V,Nf=int(states[0]).bit_count(),Ns=g.Ns,
        cluster=g.name,t=g.t['t'],t1=g.t['t1'],t2=g.t['t2'],
        q=mom['q'],loops=np.array(loops,dtype=int),
        triangle_cell=res['cells'],triangle_sublattice=sub,
        mean_current=res['mean'],ground_dim=res['ground_dim'],
        C_connected=res['C'],C_zero=res['C_zero'],C_finite=res['C_finite'],
        projected_I_matrices=res['projected_current_matrices'],
        S_q=mom['S'],S_zero=mom['S_zero'],S_finite=mom['S_finite'],
        S_nonlocal=mom['S_nonlocal'],eigmax_q=mom['eigenvalue_max'],
        eigenvector_max_q=mom['eigenvector_max'],
        Jmax_same=J['same']['Jmax_abs'],Jmax_opposite=J['opposite']['Jmax_abs'],
        Jmax_per_triangle_same=J['same']['Jmax_per_triangle'],
        Jmax_per_triangle_opposite=J['opposite']['Jmax_per_triangle'])
    _write_table(folder/'triangle_pair_correlations.csv',pairs)
    _write_table(folder/'triangle_distance_shells.csv',shells)
    for key,mat in (('full',res['C']),('zero',res['C_zero']),('finite',res['C_finite'])):
        if key=='full':
            plt.figure(figsize=(6.5,5.3))
            signs=np.where(sub==0,1.,-1.)
            staggered=signs[:,None]*np.real(mat)*signs[None,:]
            limit=max(np.max(np.abs(staggered)),1e-12)
            plt.imshow(staggered,cmap='RdBu_r',vmin=-limit,vmax=limit,origin='lower')
            plt.colorbar(label='Connected staggered current correlation')
            plt.xlabel('Triangle q (A/B alternating)');plt.ylabel('Triangle p')
            plt.title(f'Opposite-LC connected correlations, V={V:g}')
            plt.tight_layout();plt.savefig(folder/'lc_correlations.png',dpi=180);plt.close()
    if shells:
        fig,ax=plt.subplots(figsize=(7.3,4.4))
        x=[r['distance'] for r in shells]
        for key,label in [('staggered_corr_mean','All'),('staggered_zero_mean','Ground zero-mode'),
                          ('staggered_finite_mean','Finite-energy')]:
            ax.plot(x,[r[key] for r in shells],'-o',label=label)
        ax.set_xlabel('Triangle-center minimum-image distance')
        ax.set_ylabel('Connected staggered inter-triangle <I_p I_q>_c')
        ax.set_title(f'Nonlocal opposite-LC correlations V={V:g}')
        ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(folder/'lc_correlations_distance.png',dpi=180);plt.close(fig)
    manifest={'V':V,'geometry':g.name,'t':g.t,'Nf':int(states[0]).bit_count(),
              'ground_dim':int(res['ground_dim']),'wavefunctions':source,
              'q':mom['q'].tolist(),'connected_gamma_same':mom['S_same_gamma'],
              'connected_gamma_opposite':mom['S_opposite_gamma'],
              'nonlocal_gamma_same':mom['S_nonlocal_same_gamma'],
              'nonlocal_gamma_opposite':mom['S_nonlocal_opposite_gamma'],
              'zero_mode_gamma_opposite':mom['S_zero_opposite_gamma'],
              'finite_gamma_opposite':mom['S_finite_opposite_gamma'],
              'Jmax_same':J['same']['Jmax_abs'], 'Jmax_opposite':J['opposite']['Jmax_abs'],
              'note':'S_nonlocal excludes p=q self correlations and may be negative. '
                     'Nonzero Jmax at V=0 or exact degeneracy is not proof of interaction-driven LC order.'}
    (folder/'summary.json').write_text(json.dumps(manifest,indent=2,default=_json_default),encoding='utf-8')
    return manifest


def _load_point(path,g,V,nf):
    with np.load(path,allow_pickle=False) as z:
        for k,val in [('Ns',g.Ns),('Nf',nf),('V',V),('t',g.t['t']),
                      ('t1',g.t['t1']),('t2',g.t['t2'])]:
            if not np.isclose(float(z[k]),val,atol=1e-8,rtol=0):
                raise ValueError(f'Incompatible LC cache {path}: {k}')
        if str(z['cluster'])!=g.name:raise ValueError('Wrong cached supercell')
        out={k:z[k].copy() for k in z.files}
    return out


def _row(g,V,z):
    q=z['q'];i0=np.argmin(np.linalg.norm(q,axis=1))
    S=z['S_q'];S0=z['S_zero'];Sfinite=z['S_finite'];Sn=z['S_nonlocal']
    u=np.array([1.,1.])/np.sqrt(2);v=np.array([1.,-1.])/np.sqrt(2)
    projection=lambda a,x:float(np.vdot(x,a[i0]@x).real)
    ans={'V':float(V),'Nf':int(z['Nf']),'n':int(z['Nf'])/g.num_cells,
         'ground_dim':int(z['ground_dim']),
         'S_same_gamma':projection(S,u),'S_opposite_gamma':projection(S,v),
         'S_same_nonlocal_gamma':projection(Sn,u),
         'S_opposite_nonlocal_gamma':projection(Sn,v),
         'S_opposite_zero_gamma':projection(S0,v),
         'S_opposite_finite_gamma':projection(Sfinite,v),
         'Jmax_same':float(z['Jmax_same']),'Jmax_opposite':float(z['Jmax_opposite']),
         'Jmax_per_triangle_same':float(z['Jmax_per_triangle_same']),
         'Jmax_per_triangle_opposite':float(z['Jmax_per_triangle_opposite']),
         'S_max_all_q':max(np.linalg.eigvalsh(s)[-1] for s in S),
         'm2_opp_conn':projection(S,v)/len(z['triangle_cell']),
         'm2_opp_nonlocal':projection(Sn,v)/len(z['triangle_cell'])}
    for j,qq in enumerate(q):
        key='q_'+str(j)
        ans[key+'_coord']=','.join(f'{x:.6g}' for x in qq)
        ans[key+'_max']=float(np.linalg.eigvalsh(S[j])[-1])
        ans[key+'_nonlocal_max']=float(np.linalg.eigvalsh(Sn[j])[-1])
    return ans


def _draw_summary(rows,q,folder):
    rows=sorted(rows,key=lambda z:z['V'])
    xs=np.array([r['V'] for r in rows]); base=next(z for z in rows if abs(z['V'])<1e-12)
    deg=np.array([r['ground_dim']>1 for r in rows]);
    def draw(ax,column,label,baseline=False,offset=0):
        values=np.array([r[column] for r in rows],float)
        if baseline:values=values-base[column]
        lines=ax.plot(xs,values,'-o',lw=1.75,ms=4,label=label)
        c=lines[0].get_color()
        if np.any(deg):ax.scatter(xs[deg],values[deg],facecolors='white',edgecolors=c,s=55,zorder=5)
        if not baseline:ax.axhline(base[column],color=c,linestyle=':',alpha=.55)
        return values
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    ax=axes[0,0]
    draw(ax,'S_same_gamma','Same LC, Gamma')
    draw(ax,'S_opposite_gamma','Opposite LC, Gamma')
    ax.set_ylabel('Full connected S_LC(Gamma)')
    ax=axes[0,1]
    draw(ax,'S_opposite_nonlocal_gamma','Opposite LC, p!=q',baseline=True)
    draw(ax,'S_opposite_gamma','Opposite LC, all p,q',baseline=True)
    ax.set_ylabel('Interaction-induced Delta S relative to V=0')
    ax=axes[1,0]
    draw(ax,'S_opposite_zero_gamma','Zero-mode contribution')
    draw(ax,'S_opposite_finite_gamma','Finite-energy contribution')
    ax.set_ylabel('Opposite LC variance components')
    ax=axes[1,1]
    draw(ax,'Jmax_per_triangle_same','Same LC, maximal per triangle')
    draw(ax,'Jmax_per_triangle_opposite','Opposite LC, maximal per triangle')
    ax.set_ylabel('Maximum current per triangle in ground multiplet')
    for ax in axes.ravel():
        ax.set_xlabel('V / t');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle(f'Ruby ED current benchmark: V=0 reference; fixed n={rows[0]["n"]:.6g}',fontsize=15)
    fig.tight_layout();fig.savefig(folder/'lc_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.2))
    for j,qq in enumerate(q):
        label=f'q=({qq[0]:.4g},{qq[1]:.4g})'
        draw(ax,f'q_{j}_max',label)
    ax.set_xlabel('V / t');ax.set_ylabel('Largest eigenvalue of 2x2 S_AB(q)')
    ax.set_title('All allowed momentum-resolved loop-current structure factors')
    ax.grid(alpha=.25);ax.legend(fontsize=9);fig.tight_layout()
    fig.savefig(folder/'lc_Sq_vs_V.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    draw(ax,'S_opposite_nonlocal_gamma','Opposite LC nonlocal')
    draw(ax,'S_same_nonlocal_gamma','Same LC nonlocal')
    ax.set_title('Inter-triangle only: diagonal p=q removed (may be negative)')
    ax.set_xlabel('V/t');ax.set_ylabel('S_nonlocal(Gamma)')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(folder/'lc_nonlocal_vs_V.png',dpi=180);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Existing fixed-n scan root or an individual V point')
    p.add_argument('--V',type=float,nargs='+',help='Select V values; default all completed V points')
    p.add_argument('--output',help='Default: <scan>/lc_structure; never overwrites susceptibility.csv')
    p.add_argument('--without-v0',action='store_true',help='Disable the V=0 reference (no Delta S plot)')
    p.add_argument('--cache-only',action='store_true',help='Only use previously computed lc_structure.npz; never run ED')
    p.add_argument('--force-resolve',action='store_true',help='Ignore saved ground multiplets and recalculate selected V')
    p.add_argument('--low',type=int,help='Low eigenvectors per symmetry block when recomputing ground states')
    p.add_argument('--max-basis',type=int,default=70000)
    a=p.parse_args(argv)
    inp=Path(a.input);is_point=(inp/'responses.json').is_file()
    if is_point:
        point_json=json.loads((inp/'responses.json').read_text(encoding='utf-8'))
        vs=[float(point_json['V'])]
        root=inp.parent
    else:
        root=inp
        vs=a.V if a.V is not None else [float(q.parent.name[2:]) for q in root.glob('V_*/responses.json')]
    if not vs:raise ValueError('No V points found; use --V 1 2 3')
    if not a.without_v0:vs=[0.]+list(vs)
    vs=sorted(set(round(float(v),12) for v in vs))
    meta_path=root/'scan_metadata.json'
    if meta_path.exists():info=json.loads(meta_path.read_text(encoding='utf-8'))
    elif is_point:info=point_json['parameters']
    else:raise FileNotFoundError(f'No scan_metadata.json in {root}')
    g=RubyPBC(info['cluster'],t=info.get('t',1.),t1=info.get('t1',.5),t2=info.get('t2',.5))
    nf=int(info['nf'])
    out=Path(a.output) if a.output else root/'lc_structure'
    out.mkdir(parents=True,exist_ok=True)
    rows=[];cache=None
    for v in vs:
        store=out/f'V_{v:+.9f}'
        cache_path=_point_cache_path(store)
        if cache_path.exists() and not a.force_resolve:
            z=_load_point(cache_path,g,v,nf)
            origin='cached LC structure factor'
        else:
            if a.cache_only:raise FileNotFoundError(f'Missing LC structure cache for V={v}: {cache_path}')
            folder,config,V,row=load_scan_metadata(root,v)
            cfg=dict(info,max_basis=a.max_basis)
            if a.low is not None:cfg['low']=a.low
            g,states,psis,E0,origin,cache=load_or_compute_ground(folder,cfg,v,cache,a.force_resolve)
            if row and int(row.get('ground_dim',psis.shape[1]))!=psis.shape[1]:
                raise ValueError(f'Ground-manifold dimension mismatch at V={v}')
            _store_point(store,g,states,psis,v,origin)
            z=_load_point(cache_path,g,v,nf)
            if row:
                for name in ('same','opposite'):
                    label='LC_'+name
                    if label+'_ground_Jmax_abs' in row and not np.isclose(
                            z['Jmax_'+name],float(row[label+'_ground_Jmax_abs']),atol=1e-7,rtol=1e-7):
                        raise AssertionError(f'{label} Jmax inconsistent with existing ED scan')
                    if label+'_zero_weight' in row:
                        ref=float(row[label+'_zero_weight'])
                        M=z['projected_I_matrices'];sgn=np.ones(len(M)) if name=='same' else np.where(z['triangle_sublattice']==0,1.,-1.)
                        O=np.einsum('p,pab->ab',sgn/np.sqrt(len(M)),M)
                        val=np.vdot(O-np.trace(O)/len(O)*np.eye(len(O)),O-np.trace(O)/len(O)*np.eye(len(O))).real/len(O)
                        if not np.isclose(ref,val,atol=1e-7,rtol=1e-7):
                            raise AssertionError(f'{label} zero-mode W0 mismatches existing ED result')
        rows.append(_row(g,v,z))
        print(f'V={v:g}: S_opp(Gamma)={rows[-1]["S_opposite_gamma"]:.7g}, '
              f'S_opp(nonlocal)={rows[-1]["S_opposite_nonlocal_gamma"]:.7g}, '
              f'Jmax(per triangle)={rows[-1]["Jmax_per_triangle_opposite"]:.7g}; '
              f'ground_dim={rows[-1]["ground_dim"]} ({origin})',flush=True)
    _write_table(out/'lc_structure_vs_V.csv',rows)
    if not a.without_v0:
        ref=next(z for z in rows if abs(z['V'])<1e-12)
        for r in rows:
            for col in ('S_same_gamma','S_opposite_gamma','S_same_nonlocal_gamma',
                        'S_opposite_nonlocal_gamma','Jmax_per_triangle_same',
                        'Jmax_per_triangle_opposite'):
                r['Delta_'+col]=r[col]-ref[col]
        _write_table(out/'lc_structure_vs_V.csv',rows)
        q=g.permitted_momenta();_draw_summary(rows,np.asarray(q),out)
    elif rows:
        # No artificial V=0 reference when --without-v0 requested.
        print('No comparison plots: --without-v0 set; raw q arrays and CSV are still saved.',flush=True)
    (out/'lc_manifest.json').write_text(json.dumps({
        'cluster':g.name,'hopping':g.t,'Nf':nf,'n':nf/g.num_cells,'Vs':vs,
        'baseline':'V=0 same finite PBC torus and particle number' if not a.without_v0 else 'none',
        'connected':'Full ground-multiplet ensemble; nondiagonal triangle correlations included',
        'normalization':'I_triangle=sum_CCW_edge J_ij/3; S(q)=F^dag C F, F=e^(iq.R)/sqrt(Nc)',
        'warning':'Nonzero V=0 Jmax is compatible with single-particle chirality; compare S_nonlocal, '
                  'Delta S and size trends before inferring loop-current order.'
    },indent=2,default=_json_default),encoding='utf-8')
    print('Saved:',out,flush=True)
    return rows


if __name__=='__main__':main()
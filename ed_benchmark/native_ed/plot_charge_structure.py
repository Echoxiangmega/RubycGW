#!/usr/bin/env python3
"""Plot Ruby ED S_ab(q), q-mode eigenvectors and real-space charge patterns.

Given an existing susceptibility scan, use cached charge-pattern covariances
whenever possible; otherwise load a saved ground multiplet or re-solve only
requested V points. Requires NumPy/SciPy/Matplotlib; no QuSpin.

Example (Windows CMD/PowerShell, from native_ed):
 python plot_charge_structure.py --input results\\chi_ring3_n2 --V 1 2 3
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from geometry import RubyPBC
from plot_current_pattern import load_scan_metadata,load_or_compute_ground
from plot_charge_pattern import projected_site_densities
from charge_structure import (ground_ensemble_charge,export_structure_one,
                              plot_scan_summary)


def try_cache(root,point,V):
    """Previous plot_charge_pattern runs provide exact C and projected P0 n_i P0."""
    roots=[root/'charge_patterns',root/'charge_pattern',root]
    paths=[r/f'V_{V:+.9f}'/'charge_pattern.npz' for r in roots]
    paths.append(point/'charge_pattern.npz')
    for p in paths:
        if p.is_file():
            with np.load(p) as z:
                C=np.array(z['charge_covariance_ensemble'],copy=True)
                n=np.array(z['n_ensemble'],copy=True)
                M=np.array(z['projected_site_density_matrices'],copy=True) if 'projected_site_density_matrices' in z else None
            return n,C,M,str(p)
    return None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Existing scan root with scan_metadata.json, or individual responses.json directory')
    p.add_argument('--V',nargs='+',type=float,help='V points to analyze (not needed for single-point input)')
    p.add_argument('--output',default=None,help='New output directory; default: <scan>/charge_structure')
    p.add_argument('--cache-only',action='store_true',help='Only use previously exported charge_pattern.npz, no ED')
    p.add_argument('--force-resolve',action='store_true',help='Ignore old cache and re-solve the selected V points')
    p.add_argument('--low',type=int,default=None,help='Override number of low eigenvectors per block if re-solving')
    p.add_argument('--max-basis',type=int,default=70000)
    args=p.parse_args(argv)
    root=Path(args.input)
    is_point=(root/'responses.json').exists()
    vs=[None] if is_point else args.V
    if not vs:raise ValueError('For a scan root specify --V 1 2 3 ...')
    output=Path(args.output) if args.output else (root.parent if is_point else root)/'charge_structure'
    output.mkdir(parents=True,exist_ok=True)
    records=[];cache=None
    for value in vs:
        point,info,V,row=load_scan_metadata(root,value)
        g=RubyPBC(info['cluster'],t=info.get('t',1.),t1=info.get('t1',.5),t2=info.get('t2',.5))
        saved=None if args.force_resolve else try_cache(root if not is_point else root.parent,point,V)
        if saved is not None:
            density,C,M,source=saved
            if len(density)!=g.Ns:raise ValueError('Cached covariance has wrong number of sites')
        else:
            if args.cache_only:raise FileNotFoundError(f'Missing charge_pattern.npz for V={V}')
            cfg=dict(info,max_basis=args.max_basis)
            if args.low is not None:cfg['low']=args.low
            g,states,psis,e0,source,cache=load_or_compute_ground(point,cfg,V,cache,
                                                                 force_resolve=args.force_resolve)
            density,C=ground_ensemble_charge(states,psis,g.Ns)
            M,_=projected_site_densities(states,psis,g.Ns)
        summary=export_structure_one(g,C,density,V,output/f'V_{V:+.9f}',projected_density=M)
        # Cross-check the adaptive q chosen in the initial ED susceptibility run.
        if row and 'CO_leading_equal_time_lambda' in row:
            ref=float(row['CO_leading_equal_time_lambda'])
            if not np.isclose(ref,summary['leading_eigenvalue'],rtol=1e-7,atol=1e-8):
                raise AssertionError(f'S(q) differs from original ED covariance mode: {ref} vs {summary["leading_eigenvalue"]}')
        with np.load(output/f'V_{V:+.9f}'/'charge_structure.npz') as z:
            qs=z['momenta'].copy();ev=z['eigenvalues'].copy()
        records.append({'V':V,'source':source,'leading_q_index':summary['leading_q_index'],
            'leading_q':summary['leading_q'],'momenta':qs,
            'eigmax_each_q':ev[:,-1]})
        print(f'V={V:g}: leading q={summary["leading_q"]}, Smax={summary["leading_eigenvalue"]:.8g} ({source})',flush=True)
    if records:
        plot_scan_summary(records,output)
        with (output/'analysis_manifest.json').open('w',encoding='utf-8') as fh:
            json.dump({'cluster':g.name,'hopping':g.t,'points':[{'V':r['V'],'source':r['source'],
                  'leading_q':r['leading_q']} for r in records],
                  'note':'Static one-point density and connected equal-time S(q) are distinct.'},fh,indent=2)
    print('Saved:',output,flush=True)
    return records


if __name__=='__main__':main()
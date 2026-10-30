#!/usr/bin/env python3
"""Resumable zero-T V, Nf scan using only NumPy/SciPy (native solver)."""
import argparse
import csv
import json
import subprocess
import sys
from itertools import product
from pathlib import Path


def main(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    rows=[];failed=[]
    for nf,V in product(args.nf,args.V):
        folder=root/f'{args.cluster}_N{nf}_V{V:+.9f}'
        report=folder/'summary.json'
        if not report.exists():
            cmd=[sys.executable,str(Path(__file__).with_name('run_zeroT.py')),
                 '--cluster',args.cluster,'--nf',str(nf),'--V',str(V),
                 '--t1',str(args.t1),'--t2',str(args.t2),
                 '--symmetry',args.symmetry,'--low',str(args.low),
                 '--out',str(folder)]
            if args.pair:cmd.append('--pair')
            if args.chi_lanczos:cmd+=['--chi-lanczos',str(args.chi_lanczos)]
            if args.no_observables:cmd+=['--no-observables']
            if args.save_psi:cmd+=['--save-psi']
            print('Running:', ' '.join(cmd),flush=True)
            status=subprocess.run(cmd).returncode
            if status:
                failed.append({'Nf':nf,'V':V,'returncode':status})
                if not args.continue_on_error:break
                continue
        meta=json.loads(report.read_text(encoding='utf-8'))
        keys=['cluster','Ns','Nf','V','E0','charge_corr_max','current_corr_max','pair_corr_max','degenerate_within_tolerance','elapsed_seconds']
        rows.append({k:meta.get(k) for k in keys})
        with (root/'summary.csv').open('w',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)
    (root/'scan_metadata.json').write_text(json.dumps({'requested_Nf':args.nf,'requested_V':args.V,
        't1':args.t1,'t2':args.t2,'successful':len(rows),'failures':failed},indent=2))
    print('Saved',root,'finished',len(rows),'failed',len(failed))


def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--cluster',choices=['ring3','3x1','2x2'],default='ring3')
    p.add_argument('--nf',nargs='+',type=int,default=[6,7,8,9])
    p.add_argument('--V',nargs='+',type=float,default=[0,.5,1,2,3])
    p.add_argument('--t1',type=float,default=.5)
    p.add_argument('--t2',type=float,default=.5)
    p.add_argument('--symmetry',choices=['none','translation','point'],default='point')
    p.add_argument('--low',type=int,default=3)
    p.add_argument('--pair',action='store_true')
    p.add_argument('--chi-lanczos',type=int,default=0)
    p.add_argument('--no-observables',action='store_true')
    p.add_argument('--save-psi',action='store_true')
    p.add_argument('--continue-on-error',action='store_true')
    p.add_argument('--output',default='results/scan')
    return p.parse_args()
if __name__=='__main__':main(parse())
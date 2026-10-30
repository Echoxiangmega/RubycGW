#!/usr/bin/env python3
"""Reproducible resumable zero-temperature V,Nf batch for QuSpin ED.

Produces a per-point immutable result folder and compact CSV dashboard.
No thermodynamic 'phase' is assigned solely from finite-size correlation peaks.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cluster',choices=['2x2','3x1','ring3'],default='ring3')
    p.add_argument('--nf',type=int,nargs='+',default=[6,9])
    p.add_argument('--V',type=float,nargs='+',default=[0.,.5,1.,1.5,2.])
    p.add_argument('--output',default='results/scan')
    p.add_argument('--low',type=int,default=3)
    p.add_argument('--chi-lanczos',type=int,default=0)
    p.add_argument('--pair',action='store_true')
    p.add_argument('--no-point',action='store_true')
    p.add_argument('--save-psi',action='store_true')
    p.add_argument('--continue-on-error',action='store_true')
    return p.parse_args()


def run(a):
    root=Path(a.output)
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    failures=[]
    for nf,V in itertools.product(a.nf,a.V):
        sub=root/f'{a.cluster}_N{nf}_V{V:+.9f}'
        report=sub/'summary.json'
        if not report.exists():
            cmd=[sys.executable,str(Path(__file__).with_name('run_zeroT.py')),
                '--cluster',a.cluster,'--nf',str(nf),'--V',repr(V),
                '--low',str(a.low),'--chi-lanczos',str(a.chi_lanczos),
                '--out',str(sub)]
            if a.pair:cmd.append('--pair')
            if a.no_point:cmd.append('--no-point')
            if a.save_psi:cmd.append('--save-psi')
            print('Running:', ' '.join(cmd),flush=True)
            rc=subprocess.run(cmd,check=False).returncode
            if rc:
                failures.append({'Nf':nf,'V':V,'returncode':rc})
                if not a.continue_on_error:
                    print('Stopped on failure:',failures[-1]);break
                continue
        row=json.loads(report.read_text())
        rows.append({k:row.get(k) for k in ('cluster','Ns','Nf','V','E0',
                          'charge_corr_max','current_corr_max','pair_corr_max',
                          'degenerate_within_tolerance')})
        # always write partial dashboard: checkpoint even on later failure
        fields=['cluster','Ns','Nf','V','E0','charge_corr_max',
                'current_corr_max','pair_corr_max','degenerate_within_tolerance']
        with (root/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader();writer.writerows(rows)
    (root/'scan_metadata.json').write_text(json.dumps({
        'cluster':a.cluster,'nf_requested':a.nf,'V_requested':a.V,
        'records':len(rows),'failures':failures,
        'note':'Correlations need normalization, finite-size checks and channel diagnostics; no phase assignment automatically.'
    },indent=2))
    print(f'Completed {len(rows)} points; failures: {len(failures)}. See {root / "summary.csv"}')


if __name__=='__main__':run(parse())
#!/usr/bin/env python3
"""Resumable native zero-T ED scan over V and integer Nf at fixed (t,t1,t2).

Each parameter set gets a separate folder. Do not reuse an existing numerical
result generated with other hopping, symmetry, pair, or response options.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from itertools import product
from pathlib import Path


def _tag(x):
    # Six decimals are sufficient for file labels; identity is checked by
    # actual float metadata and NOT inferred from this truncated label.
    return f'{x:+.6f}'


def _matching(root, args, nf, V):
    path = root / 'summary.json'
    if not path.exists():
        return False
    try:
        summary=json.loads(path.read_text(encoding='utf-8'))
        t=summary.get('t', {})
        checks=[summary.get('cluster')==args.cluster,
                summary.get('Nf')==nf,
                math.isclose(float(summary.get('V',float('nan'))), V, abs_tol=1e-12),
                summary.get('symmetry')==args.symmetry,
                all(math.isclose(float(t.get(k,float('nan'))), z, abs_tol=1e-12)
                    for k,z in (('t',args.t),('t1',args.t1),('t2',args.t2)))]
        if not all(checks):return False
        if not args.no_observables and not (root/'equal_time.npz').exists():return False
        if args.pair and not 'pair_corr_max' in summary:return False
        if args.chi_lanczos:
            if not (root/'equal_time.npz').exists():return False
            import numpy as np
            with np.load(root/'equal_time.npz') as obj:
                if not {'LC_same_chi_static','LC_opposite_chi_static','CO_chi_static'} <= set(obj.files):return False
        if args.save_psi and not (root/'ground_state.npz').exists():return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


def main(args):
    nc={'ring3':3,'3x1':3,'2x2':4}[args.cluster]
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    metadata_file=root/'scan_metadata.json'
    if metadata_file.exists():
        prior=json.loads(metadata_file.read_text(encoding='utf-8'))
        for name,desired in [('cluster',args.cluster),('t',args.t),('t1',args.t1),
                             ('t2',args.t2),('symmetry',args.symmetry)]:
            recorded=prior.get(name)
            if recorded is None or (not math.isclose(recorded,desired,abs_tol=1e-12)
                                    if isinstance(desired,(int,float)) else recorded!=desired):
                raise RuntimeError(f'{root} already belongs to another scan ({name}={recorded}). '
                                   'Use a new --output folder to keep independent parameter sets separate.')
    rows=[];failed=[]
    keys=['cluster','t','t1','t2','Ns','Nf','n','V','E0',
          'charge_corr_max','current_corr_max','pair_corr_max',
          'degenerate_within_tolerance','elapsed_seconds','result_dir']
    for nf,V in product(args.nf,args.V):
        if not (0<=nf<=6*nc):raise ValueError(f'Nf={nf} is not allowed for {args.cluster}')
        sub=f'{args.cluster}_t{_tag(args.t)}_t1{_tag(args.t1)}_t2{_tag(args.t2)}_N{nf}_V{_tag(V)}'
        folder=root/sub
        report=folder/'summary.json'
        # Refuse to reuse a mismatched result. Keep old data intact.
        if report.exists() and not _matching(folder,args,nf,V):
            raise RuntimeError(f'{folder} contains a result with different parameters or options. '
                               'Choose another --output or remove that point after checking its files.')
        if not report.exists():
            cmd=[sys.executable,str(Path(__file__).with_name('run_zeroT.py')),
                 '--cluster',args.cluster,'--nf',str(nf),'--V',repr(V),
                 '--t',repr(args.t),'--t1',repr(args.t1),'--t2',repr(args.t2),
                 '--symmetry',args.symmetry,'--low',str(args.low),
                 '--out',str(folder)]
            if args.pair:cmd.append('--pair')
            if args.chi_lanczos:cmd+=['--chi-lanczos',str(args.chi_lanczos)]
            if args.no_observables:cmd.append('--no-observables')
            if args.save_psi:cmd.append('--save-psi')
            if args.no_twisted:cmd.append('--no-twisted')
            if args.method!='sparse':cmd+=['--method',args.method]
            print('Running:', ' '.join(cmd),flush=True)
            status=subprocess.run(cmd,check=False).returncode
            if status:
                failed.append({'Nf':nf,'V':V,'returncode':status})
                if not args.continue_on_error:break
                continue
        meta=json.loads(report.read_text(encoding='utf-8'))
        row={k:meta.get(k) for k in keys}
        row.update(t=args.t,t1=args.t1,t2=args.t2,n=nf/nc,result_dir=sub)
        rows.append(row)
        # Write checkpoint after each completed point, even if a later one fails.
        with (root/'summary.csv').open('w',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)
    (root/'scan_metadata.json').write_text(json.dumps({
        'cluster':args.cluster,'requested_Nf':args.nf,'requested_V':args.V,
        't':args.t,'t1':args.t1,'t2':args.t2,'symmetry':args.symmetry,
        'pair':args.pair,'chi_lanczos':args.chi_lanczos,
        'successful':len(rows),'failures':failed,
        'note':'No automatic thermodynamic phase labels; use channel_diagnostics.py and finite-size checks.'
    },indent=2),encoding='utf-8')
    print('Saved',root,'finished',len(rows),'failed',len(failed))


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cluster',choices=['ring3','3x1','2x2'],default='ring3')
    p.add_argument('--nf',nargs='+',type=int,default=[6,7,8,9])
    p.add_argument('--V',nargs='+',type=float,default=[0,.5,1,2,3])
    p.add_argument('--t',type=float,default=1.)
    p.add_argument('--t1',type=float,default=.5)
    p.add_argument('--t2',type=float,default=.5)
    p.add_argument('--symmetry',choices=['none','translation','point'],default='point')
    p.add_argument('--low',type=int,default=3)
    p.add_argument('--pair',action='store_true')
    p.add_argument('--chi-lanczos',type=int,default=0)
    p.add_argument('--no-observables',action='store_true')
    p.add_argument('--no-twisted',action='store_true')
    p.add_argument('--save-psi',action='store_true')
    p.add_argument('--method',choices=['sparse','matrixfree'],default='sparse')
    p.add_argument('--continue-on-error',action='store_true')
    p.add_argument('--output',default='results/scan')
    return p.parse_args()

if __name__=='__main__':main(parse())
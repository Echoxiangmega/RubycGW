#!/usr/bin/env python3
"""One command to collect and analyze a complete quarter-filled V scan.

Run after the 12-seed Slurm array finishes. By default refuse to call an
incomplete calculation a phase diagram. --allow-incomplete permits previews.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
SEEDS=("normal","sd","isd","lc_same","lc_opposite","random",
       "ps_co","mix_co_lc_opposite","sdw","pf_s","pf_dplus","pf_dminus")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--mode',choices=('continuation','independent'),default='continuation')
    p.add_argument('--direction',choices=('up','down'),default='up')
    p.add_argument('--V-min',type=float)
    p.add_argument('--V-max',type=float)
    p.add_argument('--V-step',type=float)
    p.add_argument('--tie-sigma',type=float,default=2.)
    p.add_argument('--allow-incomplete',action='store_true')
    p.add_argument('--output',default=None,help='Analysis folder; defaults to ROOT/analysis')
    a=p.parse_args(argv)
    selected=[a.V_min,a.V_max,a.V_step]
    if any(x is not None for x in selected) and not all(x is not None for x in selected):
        p.error('Provide all of --V-min/--V-max/--V-step, or none')
    if a.V_step is not None and (a.V_step<=0 or a.V_max<a.V_min):
        p.error('Require V-step>0 and V-max>=V-min')
    root=Path(a.root).expanduser().resolve()
    master=root/'vscan_master.csv'
    subprocess.run([sys.executable,str(HERE/'collect_supercell_vscan.py'),
                    '--root',str(root),'--output',str(master)],check=True)
    groups=defaultdict(lambda:defaultdict(set))
    with master.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('scan_mode')!=a.mode or r.get('scan_direction')!=a.direction:
                continue
            try:energy=float(r['mc_E'])
            except (KeyError,TypeError,ValueError):continue
            if not math.isfinite(energy):continue
            size=(int(float(r['L1'])),int(float(r['L2'])))
            key=round(float(r['scan_V']),8)
            groups[size][key].add(r['branch_seed'])
    if not groups:raise RuntimeError(f'No {a.mode}/{a.direction} scan rows')
    expected=None
    if a.V_step is not None:
        count=int(math.floor((a.V_max-a.V_min)/a.V_step+1e-10))
        expected={round(a.V_min+i*a.V_step,8) for i in range(count+1)}
        if max(expected)<a.V_max-1e-9:expected.add(round(a.V_max,8))
    gaps=[]
    for size,by_v in sorted(groups.items()):
        vals=expected if expected is not None else set(by_v)
        for V in sorted(vals):
            missing=sorted(set(SEEDS)-by_v.get(V,set()))
            if missing:gaps.append(f'L{size[0]}x{size[1]} V={V:g}: {", ".join(missing)}')
    if gaps:
        print('[analysis] Missing expected branches:',file=sys.stderr)
        for gap in gaps[:30]:print('  '+gap,file=sys.stderr)
        if len(gaps)>30:print(f'  ... {len(gaps)-30} more V points',file=sys.stderr)
        if not a.allow_incomplete:
            raise RuntimeError('Incomplete scan; rerun failed array tasks or pass --allow-incomplete for a preview')
    cmd=[sys.executable,str(HERE/'analyze_supercell_vscan.py'),
         '--root',str(root),'--master',str(master),
         '--mode',a.mode,'--direction',a.direction,
         '--tie-sigma',str(a.tie_sigma),'--energy-reference','normal']
    if a.output is not None:cmd+=['--output',a.output]
    subprocess.run(cmd,check=True)
    print(f'[analysis] One-command collection and analysis complete: {root}')


if __name__=='__main__':
    try:main()
    except RuntimeError as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        raise SystemExit(2)

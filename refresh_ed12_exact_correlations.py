#!/usr/bin/env python3
"""Refresh paired-transition ED correlations while reusing saved Green functions.

For archives written by diagnose_ed12_four_layer.py (short-key schema).
No GW, cGW, or exact one-particle Lehmann calculation is repeated.
"""
import argparse
from pathlib import Path
import numpy as np
from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('benchmark',type=Path)
    p.add_argument('--out',type=Path,default=Path('results/ed12_four_layer_paired.npz'))
    args=p.parse_args()
    with np.load(args.benchmark,allow_pickle=False) as d:
        payload={k:d[k] for k in d.files}
    d=payload
    params=RubyParameters(V=float(d['V']),**{k:float(d[k]) if k in d else v
                           for k,v in [('ti',.4),('t1',.2),('t2',.2)]})
    ex=ExactSmallRubyThermal(int(d['L1']),int(d['L2']),params)
    ex.diagonalize(params.V)
    C=[];chi=[];means=[]
    for ic,ch in enumerate(d['channels']):
        c,x,mean,_=ex.correlation_tau(ex.pseudospin_operator(str(ch),d['q']),d['tau'],
                                    float(d['mu_ed']),float(d['T']))
        C.append(c);chi.append(x);means.append(mean)
        print(f'{ch}: chi {d["exact_chi0"][ic].real:.10f} -> {x:.10f}; '
              f'midpoint delta={(c[len(c)//2]-d["exact_C_tau"][len(c)//2,ic]).real:.3e}; '
              f'KMS error={np.max(abs(c-c[::-1])):.3e}',flush=True)
    payload.update(exact_C_tau=np.array(C).T,exact_chi0=np.array(chi),exact_means=np.array(means),
                   exact_correlation_method=np.array('paired-lower-state-lehmann'),
                   correlation_refresh_source=np.array(str(args.benchmark)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,**payload)

if __name__=='__main__':main()

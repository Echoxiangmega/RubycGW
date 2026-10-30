#!/usr/bin/env python3
"""Regenerate χ(V) figures from a completed or interrupted scan, no ED rerun."""
import argparse
import csv
from pathlib import Path
from scan_susceptibility import plot_results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Root containing susceptibility.csv')
    a=p.parse_args()
    root=Path(a.input)
    with (root/'susceptibility.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError('CSV contains no completed V points')
    for r in rows:
        for key,value in list(r.items()):
            # Preserve booleans from CSV. bool('False') is True in Python.
            if value in ('True','False'):
                r[key]=(value=='True')
            elif key in ('V','n') or key.endswith('_chi') or key.endswith('_chi_regular') or key.endswith('_zero_weight'):
                r[key]=float(value)
    plot_results(root,rows)
    print('Wrote susceptibility figures to',root)

if __name__=='__main__':main()
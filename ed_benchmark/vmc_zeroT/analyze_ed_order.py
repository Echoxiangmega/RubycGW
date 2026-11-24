#!/usr/bin/env python3
"""Analyze saved supercell ED ground states without solving the Hamiltonian."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"ed_benchmark"/"native_ed"))
import geometry
import observables
from benchmark_observables import benchmark_observables


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path,help="Benchmark directory, ed directory, or final_state.npz")
    p.add_argument("--L1",type=int)
    p.add_argument("--L2",type=int)
    for key in ("t","t1","t2"):
        p.add_argument("--"+key,type=float)
    p.add_argument("--degeneracy-tol",type=float)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    src=a.input
    if src.is_dir():
        src=src/("ed/final_state.npz" if (src/"ed/final_state.npz").is_file() else "final_state.npz")
    root=src.parent.parent if src.parent.name=="ed" else src.parent
    settings=json.loads((root/"settings.json").read_text(encoding="utf-8")) if (root/"settings.json").is_file() else {}
    summary_path=src.parent/"summary.json"
    summary=json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    def get(k,default=None):
        val=getattr(a,k,None)
        return val if val is not None else settings.get(k,default)
    t=[get(k) for k in ("t","t1","t2")]
    if any(x is None for x in t):
        p.error("Missing hopping parameters: supply --t --t1 --t2 or preserve settings.json")
    l1,l2=get("L1",1),get("L2",1)
    g=geometry.make_ring3_supercell(l1,l2,t=t[0],t1=t[1],t2=t[2])
    with np.load(src,allow_pickle=False) as z:
        gs=np.asarray(z["psi_ground_manifold"],complex)
        states=np.asarray(z["native_states"],np.uint64)
        eigen=np.asarray(z["eigenvalues"],float)
    if gs.ndim!=2 or gs.shape[0]!=len(states) or gs.shape[1]==0:
        raise ValueError("Invalid ED ground-manifold checkpoint")
    tol=get("degeneracy_tol",settings.get("ed_degeneracy_tol",1e-8))
    if (summary.get("ground_manifold_truncated") or len(eigen)<=gs.shape[1]
        or np.count_nonzero(eigen-eigen[0]<=tol)!=gs.shape[1]):
        raise RuntimeError("Checkpoint does not establish a complete ED ground manifold")
    if np.any(states>>np.uint64(g.Ns)):
        raise ValueError("Saved basis exceeds geometry site count")
    # Check basis against the saved fixed-particle-number convention.
    import core
    nf=int(int(states[0]).bit_count())
    if not np.array_equal(states,core.make_states(g.Ns,nf)):
        raise ValueError("Saved basis does not match reconstructed geometry")
    data=benchmark_observables(g,states,gs,observables,ground_manifold=True).as_dict()
    result={**summary,"Ns":g.Ns,"Nf":nf,"ground_degeneracy":gs.shape[1],
            "E":float(eigen[0]),"analysis_source":str(src),**data}
    out=a.output or src.parent/"order_diagnostics.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,allow_nan=False))
    print(f"Saved: {out}")


if __name__=="__main__":
    main()

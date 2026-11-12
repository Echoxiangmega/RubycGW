#!/usr/bin/env python3
"""Sweep increasing interaction V at fixed filling, warm-starting GW and cGW.

Example PowerShell:
    python research/scan_three_tori_v.py --geometry hex18 --filling 2
      --V-values 0.5 1 1.5 2 3 4 5
      --K 0.333333333333 0.333333333333
      --Q 0.333333333333 0.333333333333
      --T 0.125 --methods ed gw cgw postgw cluster
      --gw-channels density pseudospin bj --post-mmax 0
      --skip-cluster-chi --out results/hex18_V_scan

Each V uses the same three-torus benchmark, and gets its own results.
Each Fierz mode independently retains its most recent *converged* GW
state and completed covariant static vertices. Failed points never
replace valid seeds. scan_results.csv and scan_chi.csv are written after
each V, allowing examination even if a later point fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from benchmark_three_tori import main as benchmark_main, parse_args as benchmark_args


def _tag(v):
    return f"{v:.12g}".replace("-","m").replace("+","p").replace(".","p")


def _number(x):
    if x in (None,""):
        return None
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except (TypeError,ValueError):
        return None


def _write_csv(path,rows,cols):
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",newline="",encoding="utf-8") as handle:
        wr=csv.DictWriter(handle,fieldnames=cols)
        wr.writeheader()
        for row in rows:
            wr.writerow({col:row.get(col) for col in cols})
    tmp.replace(path)


def _fingerprint(args):
    """Require the same physical model, grid and solver knobs on resume."""
    settings=dict(vars(args))
    settings.pop("V",None)
    settings.pop("out",None)
    return json.dumps(settings,sort_keys=True,default=str)


def _save_seeds(directory,gw_memory,vertex_memory,fingerprint):
    """Atomic pickle-free NPZ for converged cross-V self-energies and vertices."""
    arrays={}
    meta=dict(fingerprint=fingerprint,gw=[],vertex=[])
    for mode,(V,state) in sorted(gw_memory.items()):
        if not bool(state.converged):
            continue
        prefix=f"gw_{mode}"
        arrays[prefix+"_static"]=np.asarray(state.Sigma_static,complex)
        arrays[prefix+"_dynamic"]=np.asarray(state.Sigma_c,complex)
        meta["gw"].append(dict(
            mode=mode,V=float(V),mu=float(state.mu),
            static=prefix+"_static",dynamic=prefix+"_dynamic"
        ))
    for mode,(V,gammas) in sorted(vertex_memory.items()):
        for (channel,index),gamma in sorted(gammas.items()):
            key=f"vertex_{len(meta['vertex'])}"
            arrays[key]=np.asarray(gamma,complex)
            meta["vertex"].append(dict(
                mode=mode,V=float(V),channel=str(channel),
                component=int(index),key=key
            ))
    arrays["metadata"]=np.asarray(json.dumps(meta,sort_keys=True))
    temp=directory/"warm_seeds.npz.tmp"
    with temp.open("wb") as fh:
        np.savez_compressed(fh,**arrays)
    temp.replace(directory/"warm_seeds.npz")


def _load_seeds(directory,fingerprint):
    """Reject stale/unsafe checkpoint metadata; never use pickle."""
    with np.load(directory/"warm_seeds.npz",allow_pickle=False) as z:
        meta=json.loads(str(z["metadata"].item()))
        if meta.get("fingerprint")!=fingerprint:
            raise ValueError("warm-state fingerprint differs from current scan")
        gw={}
        chi={}
        for item in meta["gw"]:
            mode=str(item["mode"])
            static=np.asarray(z[item["static"]],complex).copy()
            dynamic=np.asarray(z[item["dynamic"]],complex).copy()
            mu=float(item["mu"])
            if not np.isfinite(mu) or not all(
                    np.all(np.isfinite(a)) for a in (static,dynamic)):
                raise ValueError("checkpoint GW state is nonfinite")
            gw[mode]=(float(item["V"]),SimpleNamespace(
                mode=mode,converged=True,mu=mu,
                Sigma_static=static,Sigma_c=dynamic
            ))
        for item in meta["vertex"]:
            mode=str(item["mode"])
            data=np.asarray(z[item["key"]],complex).copy()
            if not np.all(np.isfinite(data)):
                raise ValueError("checkpoint cGW vertex nonfinite")
            if mode not in chi:
                chi[mode]=(float(item["V"]),{})
            if chi[mode][0]!=float(item["V"]):
                raise ValueError("same-mode cGW vertices belong to different V")
            chi[mode][1][(str(item["channel"]),int(item["component"]))]=data
    return gw,chi


def _point_tables(v,out,info):
    with (out/"comparison.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    bymethod={}
    edchi={}
    for r in rows:
        bymethod.setdefault(r["method"],[]).append(r)
        if r["method"]=="ED" and r["quantity"].startswith("chi["):
            ch=r["quantity"].split("[",1)[1].split("]",1)[0]
            edchi[ch]=_number(r["value"])

    status=info.get("method_status",{})
    allmethods=set(bymethod)|set(status)|set(info.get("G_methods",[]))
    wanted=set(info.get("gw_interaction_channels",[]))
    for mode in wanted:
        allmethods.update(("GW["+mode+"]","cGW["+mode+"]",
                           "post-GW["+mode+"]"))
    green_errors={}
    for method,data in bymethod.items():
        green_errors[method]={
            r["quantity"]:_number(r["value"]) for r in data
            if r["quantity"] in (
                "G_rel_Frobenius_K","G_tau_curve_relative_L2_error"
            )
        }
    detail=[]
    agg=[]
    for method in sorted(allmethods):
        mode=(method.split("[",1)[1][:-1]
              if "[" in method and method.endswith("]") else None)
        prior_gw=info.get("warm_start",{}).get("previous_GW_V",{}).get(mode)
        prior_vertex=info.get("warm_start",{}).get("previous_cGW_V",{}).get(mode)
        reports=bymethod.get(method,[])
        state=status.get(method,{}).get("status")
        if state is None:
            state=("converged" if any(
                r["status"] in ("ok","reference","nonvariational-estimate")
                for r in reports) else "not_requested")
        source=info.get("diagnostics",{}).get(method,{})
        measured_g=method
        if method.startswith("cGW["):
            measured_g="GW["+method[4:]
        recorded=green_errors.get(measured_g,{})
        error_chi=[]
        for r in reports:
            q=r["quantity"]
            if "chi[" not in q and "chi_GG[" not in q:
                continue
            ch=q.split("[",1)[1].split("]",1)[0]
            val=_number(r["value"])
            ed=edchi.get(ch)
            err=(abs(val-ed)/max(abs(ed),1e-12)
                 if val is not None and ed is not None else None)
            detail.append(dict(
                V=float(v),method=method,channel=ch,value=val,
                ED_reference=ed,relative_error=err,status=r["status"]
            ))
            if err is not None and method!="ED":
                error_chi.append(err)
        agg.append(dict(
            V=float(v),method=method,status=state,
            previous_GW_V=prior_gw if method.startswith("GW[") else None,
            previous_cGW_V=prior_vertex if method.startswith("cGW[") else None,
            residual=_number(source.get("residual")),
            screening_smin=_number(source.get("screening_smin")),
            G_iomega_error=recorded.get("G_rel_Frobenius_K"),
            G_tau_error=recorded.get("G_tau_curve_relative_L2_error"),
            mean_chi_error=sum(error_chi)/len(error_chi) if error_chi else None,
            runtime_s=_number(info.get("method_total_runtime_seconds",{})
                                   .get(method,info.get("timings",{}).get(method))),
            output_dir=str(out),
            notes=status.get(method,{}).get("reason",""),
        ))
    return agg,detail


def run_scan(argv=None):
    p=argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--V-values",nargs="+",type=float,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cold-each-point",action="store_true")
    p.add_argument("--cold-vertex-each-point",action="store_true")
    p.add_argument("--resume",action="store_true",
                   help="reuse complete V points and certified GW/Gamma NPZ restart states")
    scan,other=p.parse_known_args(argv)
    values=sorted(scan.V_values)
    if len(set(values))!=len(values) or not all(map(math.isfinite,values)):
        p.error("V must consist of distinct finite numbers")
    if any(x in ("--V","--V-values","--out") for x in other):
        p.error("use --V-values for interactions and --out for output")
    if not other:
        p.error("supply common flags such as --geometry and --filling")
    out=scan.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    probe=benchmark_args(["--V",str(values[0]),"--out",str(out/"check"),*other])
    if probe.gw_ramp_steps:
        p.error("omit --gw-ramp-steps: the scanner already uses previous-V warm starts")
    if probe.strict_method_failures:
        p.error("omit --strict-method-failures: the scan records per-point failures")
    fingerprint=_fingerprint(probe)
    old_points={}
    manifest_path=out/"scan_manifest.json"
    if scan.resume and manifest_path.is_file():
        prior=json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("fingerprint")!=fingerprint or prior.get("V_sorted")!=values:
            p.error("resume manifest incompatible with this parameter set or V list")
        old_points={float(item["V"]):item for item in prior.get("points",[])}
    print("[V-scan] V sorted ascending:",values,flush=True)
    gw_memory={}
    vertex_memory={}
    all_rows=[]
    all_chi=[]
    manifest=[]
    for i,V in enumerate(values):
        directory=out/f"V_{i:03d}_{_tag(V)}"
        args=benchmark_args(["--V",str(V),"--out",str(directory),*other])
        print(f"[V-scan] point {i+1}/{len(values)}: V={V:g}",flush=True)
        recovered=False
        old=old_points.get(float(V))
        if (old and old.get("status")=="complete"
                and (directory/"comparison.csv").is_file()
                and (directory/"summary.json").is_file()
                and (directory/"warm_seeds.npz").is_file()):
            try:
                oldgw,oldchi=_load_seeds(directory,fingerprint)
                old_info=json.loads(
                    (directory/"summary.json").read_text(encoding="utf-8")
                )
                table,chis=_point_tables(V,directory,old_info)
                gw_memory=oldgw
                vertex_memory=oldchi
                all_rows.extend(table)
                all_chi.extend(chis)
                manifest.append(dict(old,resumed=True))
                recovered=True
                print(f"[V-scan] resumed V={V:g} from saved certified state",flush=True)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                print(f"[V-scan] restart failed; recomputing V={V:g}: {exc}",flush=True)
        if not recovered:
            try:
                result=benchmark_main(
                    args,
                    initial_gw={} if scan.cold_each_point else gw_memory,
                    initial_vertices={} if scan.cold_vertex_each_point else vertex_memory,
                    return_scan_state=True,
                )
                for mode,pair in result["converged_GW"].items():
                    gw_memory[mode]=pair
                for mode,pair in result["converged_vertices"].items():
                    vertex_memory[mode]=pair
                info=result["summary"]
                table,chis=_point_tables(V,directory,info)
                all_rows.extend(table)
                all_chi.extend(chis)
                _save_seeds(directory,gw_memory,vertex_memory,fingerprint)
                manifest.append(dict(
                    V=float(V),status="complete",
                    GW_last_converged={
                        mode:float(x[0]) for mode,x in gw_memory.items()
                    },
                    cGW_last_converged={
                        mode:float(x[0]) for mode,x in vertex_memory.items()
                    },
                    method_status=info.get("method_status",{}),
                    directory=str(directory),
                ))
            except Exception as exc:
                why=f"{type(exc).__name__}: {exc}"
                print(f"[V-scan] V={V:g} ERROR: {why}",flush=True)
                manifest.append(dict(V=float(V),status="failed",
                                     error=why,directory=str(directory)))
                all_rows.append(dict(
                    V=float(V),method="scan_point",status="failed",
                    notes=why,output_dir=str(directory)
                ))
        _write_csv(out/"scan_results.csv",all_rows,[
            "V","method","status","previous_GW_V","previous_cGW_V",
            "residual","screening_smin","G_iomega_error","G_tau_error",
            "mean_chi_error","runtime_s","output_dir","notes"
        ])
        _write_csv(out/"scan_chi.csv",all_chi,[
            "V","method","channel","value","ED_reference","relative_error","status"
        ])
        payload=dict(
            filling=float(probe.filling),geometry=probe.geometry,
            T=float(probe.T),K=list(map(float,probe.K)),
            Q=list(map(float,probe.Q)) if probe.Q is not None else None,
            gw_channels=list(probe.gw_channels),V_sorted=values,
            fingerprint=fingerprint,
            points=manifest,
            cold_each_point=bool(scan.cold_each_point),
            cold_vertex_each_point=bool(scan.cold_vertex_each_point),
        )
        temp=out/"scan_manifest.json.tmp"
        temp.write_text(json.dumps(payload,indent=2),encoding="utf-8")
        temp.replace(out/"scan_manifest.json")
    print(f"[V-scan] saved {out/'scan_results.csv'}, scan_chi.csv and scan_manifest.json",flush=True)
    return manifest


if __name__=="__main__":
    run_scan()

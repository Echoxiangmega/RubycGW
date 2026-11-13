#!/usr/bin/env python3
"""Plot density, real bond order, and physical bond currents on one Ruby hexagon.

This reads the *_orders.npz produced by
research/scan_hex18_cluster_order_vn.py. It NEVER recomputes ED+GW.
The plotted geometry is the exact pictured six-triangle true_hex torus,
provably equivalent (by site permutation) to the native RubycGW hex18
supercell. PBC-wrapped hopping bonds are drawn to their correct images.

Run for a single branch:
    python research/plot_hex18_cluster_order.py --orders PATH_orders.npz

Run for an entire scan:
    python research/plot_hex18_cluster_order.py --scan-root results/hex18_ordered_vn
To plot only reliably selected minimum-F branches, add
    --selection winners
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.cm import ScalarMappable

from rubycgw.model import RubyParameters
from rubycgw.supercell import SUPERCELL_MATRIX, SUPERCELL_REPRESENTATIVES
from research.scan_true_hex18_v import RubyTorus

# Explicitly verified in research/verify_hex18_equivalence.py:
# drawing_primitive = U @ native_primitive + orbital_shift.
_U = np.asarray([[0, 1], [-1, 0]], dtype=int)
_ORBITAL_SHIFT = np.asarray([[0, 0]] * 3 + [[-1, -1]] * 3, dtype=int)
_KIND_COLORS = {"t": "#be5255", "t1": "#4884bc", "t2": "#319b80"}


def native_to_picture(torus):
    """Native site index -> pictured true_hex site index, exact PBC quotient."""
    if torus.name != "true_hex":
        raise ValueError("true_hex pictured geometry is required")
    if not np.array_equal(_U @ SUPERCELL_MATRIX, torus.M):
        raise ValueError("picture/native supercell periods are inconsistent")
    mapping = np.full(18, -1, dtype=int)
    for sector, representative in enumerate(SUPERCELL_REPRESENTATIVES):
        for orbital in range(6):
            target = _U @ representative + _ORBITAL_SHIFT[orbital]
            candidates = []
            for cell, repr_pic in enumerate(torus.reps):
                winds = np.linalg.solve(torus.M, target - repr_pic)
                if np.allclose(winds, np.rint(winds), rtol=0, atol=1e-11):
                    candidates.append(cell)
            if len(candidates) != 1:
                raise ValueError("ambiguous native-to-picture site mapping")
            mapping[6 * sector + orbital] = 6 * candidates[0] + orbital
    if set(mapping.tolist()) != set(range(18)):
        raise ValueError("native-to-picture mapping is not a permutation")
    return mapping


def _load_orders(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        need = {"density", "bond_i", "bond_j", "bond_real", "bond_current"}
        if not need.issubset(data.files):
            raise ValueError(
                f"{path} does not contain required order data: "
                f"{sorted(need.difference(data.files))}"
            )
        density = np.asarray(data["density"], dtype=float).reshape(-1)
        ii = np.asarray(data["bond_i"], dtype=int).reshape(-1)
        jj = np.asarray(data["bond_j"], dtype=int).reshape(-1)
        real = np.asarray(data["bond_real"], dtype=float).reshape(-1)
        current = np.asarray(data["bond_current"], dtype=float).reshape(-1)
    if density.shape != (18,) or len(ii) != 36 or not (
        len(jj) == len(real) == len(current) == 36
    ):
        raise ValueError("expected 18 site densities and 36 physical bond values")
    if (not np.all(np.isfinite(density)) or
        not np.all(np.isfinite(real)) or not np.all(np.isfinite(current))):
        raise ValueError("non-finite site or bond observables")
    if np.any(ii < 0) or np.any(ii >= 18) or np.any(jj < 0) or np.any(jj >= 18):
        raise ValueError("bond indices outside 0..17")
    records = {}
    for i, j, b, c in zip(ii, jj, real, current):
        key = tuple(sorted((int(i), int(j))))
        if key in records:
            raise ValueError("duplicate physical bond in orders file")
        records[key] = dict(i=int(i), j=int(j), bond=float(b), current=float(c))
    if len(records) != 36:
        raise ValueError("expected exactly 36 distinct physical bonds")
    meta_path = path.with_name(path.name.removesuffix("_orders.npz") + ".json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return density, records, metadata


def picture_bonds(torus, native_map, records):
    """Return 36 physically oriented edges with exact PBC image shifts.

    Every edge is a 2-site hopping of the pictured true_hex torus. The
    stored current is converted to the *picture's* oriented edge direction.
    """
    inverse = np.argsort(native_map)
    pictured = []
    used = set()
    for edge in torus.bonds:
        ni, nj = int(inverse[edge.i]), int(inverse[edge.j])
        key = tuple(sorted((ni, nj)))
        if key not in records:
            raise ValueError(f"pictured bond {key} missing in ED+GW output")
        if key in used:
            raise ValueError(f"duplicate pictured edge {key}")
        used.add(key)
        item = records[key]
        direction = 1. if (ni, nj) == (item["i"], item["j"]) else -1.
        start = np.asarray(torus.pos[edge.i], float)
        end = np.asarray(torus.pos[edge.j], float) + torus.S @ np.asarray(edge.wind, float)
        pictured.append(dict(
            i=int(edge.i), j=int(edge.j),
            native_i=int(ni), native_j=int(nj),
            kind=str(edge.kind),
            start=start, end=end,
            wrapped=bool(any(edge.wind)),
            bond=float(item["bond"]),
            current=float(direction * item["current"]),
        ))
    if len(pictured) != 36:
        raise ValueError("wrong number of pictured physical bonds")
    return pictured


def _draw_skeleton(ax,torus,edges,*,annotate=False):
    for edge in edges:
        a, b = edge["start"], edge["end"]
        ax.plot([a[0],b[0]], [a[1],b[1]],
                color="#9da7b2",alpha=.30,lw=1.05,
                linestyle="--" if edge["wrapped"] else "-",
                zorder=1)
        if edge["wrapped"]:
            ax.scatter([b[0]],[b[1]],marker="o",s=20,
                       facecolors="white",edgecolors="#9da7b2",
                       linewidths=.8,zorder=2)
    ax.scatter(torus.pos[:,0],torus.pos[:,1],s=18,
               facecolors="#f3f5f8",edgecolors="#7d8792",
               linewidths=.5,zorder=3)
    if annotate:
        reverse = np.argsort(native_to_picture(torus))
        for pic,(x,y) in enumerate(torus.pos):
            ax.text(x,y,str(reverse[pic]),ha="center",va="center",
                    color="#1d2939",fontsize=7,zorder=8)
    ax.set_aspect("equal",adjustable="box")
    ax.axis("off")
    pts = np.vstack(
        [torus.pos]+[np.asarray([x["end"] for x in edges])]
    )
    lo=pts.min(axis=0);hi=pts.max(axis=0)
    span=max(float((hi-lo).max()),1.)
    ax.set_xlim(lo[0]-.11*span, hi[0]+.11*span)
    ax.set_ylim(lo[1]-.11*span, hi[1]+.11*span)


def _draw_density(ax,torus,mapping,density,edges,*,annotate=False):
    _draw_skeleton(ax,torus,edges,annotate=False)
    values=density[np.argsort(mapping)]
    delta=values-values.mean()
    scale=max(float(np.max(np.abs(delta))),1e-10)
    size=125+460*np.abs(delta)/scale
    img=ax.scatter(torus.pos[:,0],torus.pos[:,1],
                   s=size,c=delta,cmap="RdBu_r",
                   vmin=-scale,vmax=scale,
                   edgecolors="#17263a",linewidths=.7,zorder=5)
    if annotate:
        inv=np.argsort(mapping)
        for pic,(x,y) in enumerate(torus.pos):
            ax.annotate(str(inv[pic]),(x,y),
                        ha="center",va="center",fontsize=7,
                        color="black",zorder=7)
    ax.set_title("A  Site density",fontsize=12,weight="semibold")
    ax.figure.colorbar(img,ax=ax,orientation="horizontal",
                       shrink=.8,pad=.04,label=r"$n_i-\overline n$")


def _draw_bonds(ax,torus,edges,*,reference="kind-mean",annotate=False):
    _draw_skeleton(ax,torus,edges,annotate=annotate)
    baseline={
        k:float(np.mean([e["bond"] for e in edges if e["kind"]==k]))
        for k in ("t","t1","t2")
    }
    vals=np.asarray([
        e["bond"]-(baseline[e["kind"]] if reference=="kind-mean" else 0.)
        for e in edges
    ],float)
    scale=max(float(np.max(np.abs(vals))),1e-10)
    cmap=plt.get_cmap("RdBu_r")
    norm=TwoSlopeNorm(vcenter=0,vmin=-scale,vmax=scale)
    for e,val in zip(edges,vals):
        a,b=e["start"],e["end"]
        ax.plot([a[0],b[0]],[a[1],b[1]],
                lw=.8+5.1*abs(val)/scale,
                color=cmap(norm(val)),
                alpha=.95,
                linestyle="--" if e["wrapped"] else "-",
                solid_capstyle="round",zorder=4)
    sm=ScalarMappable(norm=norm,cmap=cmap)
    sm.set_array(vals)
    label=(r"$2\mathrm{Re}\langle c_i^\dagger c_j\rangle"
           r"-\langle B\rangle_{\mathrm{type}}$"
           if reference=="kind-mean"
           else r"$2\mathrm{Re}\langle c_i^\dagger c_j\rangle$")
    ax.figure.colorbar(sm,ax=ax,orientation="horizontal",
                       shrink=.8,pad=.04,label=label)
    ax.set_title("B  Real bond order",fontsize=12,weight="semibold")


def _draw_current(ax,torus,edges,*,cutoff=.03,annotate=False):
    _draw_skeleton(ax,torus,edges,annotate=annotate)
    values=np.asarray([e["current"] for e in edges])
    scale=float(np.max(np.abs(values)))
    threshold=float(cutoff)*scale
    count=0
    if scale>1e-11:
        for e in edges:
            value=e["current"]
            if abs(value)<max(threshold,1e-11):
                continue
            a=e["start"];b=e["end"]
            start,end=(a,b) if value>=0 else (b,a)
            vec=end-start
            start=start+.20*vec
            end=end-.20*vec
            color=_KIND_COLORS[e["kind"]]
            ax.add_patch(FancyArrowPatch(
                start,end,arrowstyle="-|>",
                mutation_scale=9+8*abs(value)/scale,
                linewidth=.8+3.4*abs(value)/scale,
                color=color,alpha=.95,
                zorder=6,shrinkA=0,shrinkB=0,
            ))
            count+=1
    ax.set_title("C  Bond current",fontsize=12,weight="semibold")
    handles=[
        Line2D([0],[0],color=_KIND_COLORS[k],lw=2.5,label=k)
        for k in ("t","t1","t2")
    ]
    if count:
        ax.legend(handles=handles,loc="lower center",ncol=3,
                  frameon=False,bbox_to_anchor=(.5,-.09),fontsize=8)
    else:
        ax.text(.5,.06,"No resolved bond current",
                transform=ax.transAxes,ha="center",
                fontsize=9,color="#667085")
    ax.text(.02,-.045,
            f"max |J| = {scale:.3g} (hopping units)",
            transform=ax.transAxes,fontsize=8,color="#475467")


def plot_order_triptych(orders,*,output=None,formats=("png",),dpi=200,
                       annotate=False,bond_reference="kind-mean",
                       current_cutoff=.03):
    """Render all three panels from one saved ED+GW ordered-state file.

    Returns file paths. Results are static images, not a re-run of the
    self-consistency solver.
    """
    orders=Path(orders)
    if not orders.exists():
        raise FileNotFoundError(orders)
    if bond_reference not in ("kind-mean","zero"):
        raise ValueError("bond reference must be 'kind-mean' or 'zero'")
    if not 0<=float(current_cutoff)<1:
        raise ValueError("current cutoff must lie in [0,1)")
    if any(fmt not in ("png","svg","pdf") for fmt in formats):
        raise ValueError("formats must be png, svg or pdf")
    density,records,metadata=_load_orders(orders)
    # Geometric connectivity does not depend on the magnitudes of hopping.
    torus=RubyTorus("true_hex",t=1.,t1=.5,t2=.5)
    mapping=native_to_picture(torus)
    edges=picture_bonds(torus,mapping,records)
    fig,axes=plt.subplots(1,3,figsize=(18.6,6.2))
    _draw_density(axes[0],torus,mapping,density,edges,
                  annotate=annotate)
    _draw_bonds(axes[1],torus,edges,reference=bond_reference,
                annotate=annotate)
    _draw_current(axes[2],torus,edges,cutoff=current_cutoff,
                  annotate=annotate)
    parts=["Periodic Ruby hex18"]
    for label,key in (("V","V"),("n","filling"),("mode","mode"),
                      ("seed","candidate"),("phase","phase_family")):
        if key in metadata:
            value=metadata[key]
            parts.append(f"{label}={value:.4g}" if isinstance(value,(float,int))
                         else f"{label}={value}")
    title="  |  ".join(parts)
    if len(title)>125:
        title=title[:122]+"..."
    fig.suptitle(title,fontsize=12,y=.985)
    fig.subplots_adjust(left=.015,right=.985,top=.88,bottom=.11,wspace=.06)
    destination=(Path(output) if output is not None else
                 orders.with_name(orders.name.removesuffix(".npz")+"_realspace.png"))
    destination.parent.mkdir(parents=True,exist_ok=True)
    paths=[]
    for fmt in dict.fromkeys(formats):
        target=destination.with_suffix("."+fmt)
        fig.savefig(target,dpi=int(dpi),bbox_inches="tight",facecolor="white")
        paths.append(target)
    plt.close(fig)
    return paths


def _collect_scan(root,*,selection,modes=None,Vs=None,fillings=None):
    root=Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if selection=="winners":
        summary=root/"phase_summary.csv"
        if not summary.is_file():
            raise FileNotFoundError(summary)
        files=[]
        with summary.open(newline="",encoding="utf-8") as stream:
            for entry in csv.DictReader(stream):
                mode=str(entry["mode"])
                V=float(entry["V"])
                n=float(entry["filling"])
                if modes and mode not in modes:
                    continue
                if Vs and not any(abs(V-v)<1e-9 for v in Vs):
                    continue
                if fillings and not any(abs(n-x)<1e-9 for x in fillings):
                    continue
                winner=entry.get("winner","")
                if not winner or entry.get("phase")=="unresolved":
                    continue
                # Keep folder matching independent of float display conventions.
                matches=list(root.glob(
                    "n_*/V_*/"+mode+"__"+winner+"_orders.npz"
                ))
                for path in matches:
                    rowpath=path.with_name(path.name.removesuffix("_orders.npz")+".json")
                    if rowpath.is_file():
                        data=json.loads(rowpath.read_text(encoding="utf-8"))
                        if abs(float(data.get("V",-999))-V)<1e-9 and abs(
                            float(data.get("filling",-999))-n
                        )<1e-9:
                            files.append(path)
        return list(dict.fromkeys(files))
    if selection!="converged":
        raise ValueError("selection must be winners/converged")
    files=[]
    for path in sorted(root.glob("n_*/V_*/*_orders.npz")):
        rowpath=path.with_name(path.name.removesuffix("_orders.npz")+".json")
        if not rowpath.is_file():
            continue
        data=json.loads(rowpath.read_text(encoding="utf-8"))
        mode=str(data.get("mode",""))
        V=float(data.get("V",float("nan")))
        n=float(data.get("filling",float("nan")))
        if data.get("status")!="converged":
            continue
        if modes and mode not in modes:
            continue
        if Vs and not any(abs(V-v)<1e-9 for v in Vs):
            continue
        if fillings and not any(abs(n-x)<1e-9 for x in fillings):
            continue
        files.append(path)
    return files


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument("--orders",type=Path,help="one *_orders.npz result")
    src.add_argument("--scan-root",type=Path,
                     help="scan_hex18_cluster_order_vn.py result directory")
    p.add_argument("--selection",choices=("converged","winners"),
                   default="converged",
                   help="scan-root: all converged candidates or only reliable free-energy winners")
    p.add_argument("--modes",nargs="+",choices=("density","pseudospin","bj"))
    p.add_argument("--V",nargs="+",type=float)
    p.add_argument("--fillings",nargs="+",type=float)
    p.add_argument("--output",type=Path,
                   help="single-file output pathname; omitted uses source folder")
    p.add_argument("--formats",nargs="+",choices=("png","svg","pdf"),
                   default=("png",))
    p.add_argument("--bond-reference",choices=("kind-mean","zero"),
                   default="kind-mean",
                   help="show deviations from each hopping type's mean, or raw bond expectation")
    p.add_argument("--current-cutoff",type=float,default=.03,
                   help="do not draw currents smaller than this fraction of max |J|")
    p.add_argument("--annotate",action="store_true",
                   help="label sites with NATIVE (not pictured) orbital indices")
    p.add_argument("--dpi",type=int,default=200)
    args=p.parse_args(argv)
    if args.orders is not None:
        files=[args.orders]
    else:
        if args.output is not None:
            p.error("--output is for --orders only; scan-root writes beside each branch")
        files=_collect_scan(
            args.scan_root,selection=args.selection,
            modes=args.modes,Vs=args.V,fillings=args.fillings
        )
    if not files:
        print("[hex18-plot] no matching converged branches (or no reliable winner).")
        return []
    all_outputs=[]
    for file in files:
        outputs=plot_order_triptych(
            file,output=args.output,formats=args.formats,dpi=args.dpi,
            annotate=args.annotate,bond_reference=args.bond_reference,
            current_cutoff=args.current_cutoff
        )
        print("[hex18-plot]",str(file),"->",", ".join(map(str,outputs)))
        all_outputs.extend(outputs)
    return all_outputs


if __name__=="__main__":
    main()

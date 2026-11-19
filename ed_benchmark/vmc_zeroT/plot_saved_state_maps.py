#!/usr/bin/env python3
"""Plot site density, bond order, and bond current from saved ED/VMC states.

This reader is designed for outputs from
ed_benchmark/vmc_zeroT/run_projection_comparison.py.

Supported state files
---------------------
ED:
    <result_root>/ed/final_state.npz
    keys: psi_ground_manifold, native_states, ...

Raw/PAV/VAP:
    <result_root>/<method_dir>/final_state.npz
    keys: psi_native, native_states, ...

For a degenerate ED manifold, --mode average evaluates the equal-weight
incoherent ground-manifold density matrix. Order-selecting modes can instead
diagonalize LC, charge, or bond-order operators inside the exact saved ground
subspace and plot the resulting pure branch.

Outputs
-------
site_density.png/.svg
bond_order.png/.svg
bond_current.png/.svg
site_density.csv
bond_observables.csv
summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NATIVE = REPO / "ed_benchmark" / "native_ed"
sys.path.insert(0, str(NATIVE))

from geometry import RubyPBC, make_ring3_supercell, A, ORBITALS  # noqa: E402
from core import edge_triplets  # noqa: E402
import observables as native_observables  # noqa: E402
from benchmark_observables import ruby_star_formfactor  # noqa: E402


SITE_CMAP = "coolwarm"
BOND_CMAP = "RdBu_r"
CURRENT_CMAP = "RdBu_r"


def load_geometry(result_root: Path) -> RubyPBC:
    path = result_root / "geometry.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing geometry.json: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    cluster = meta.get("name", meta.get("cluster"))
    if cluster is None:
        raise KeyError("geometry.json must contain 'name' or 'cluster'")

    # geometry.metadata() stores hopping amplitudes under the nested 't' dict.
    hopping = meta.get("t", {})
    if isinstance(hopping, dict):
        t = float(hopping.get("t", 1.0))
        t1 = float(hopping.get("t1", 0.5))
        t2 = float(hopping.get("t2", 0.5))
    else:
        # Backward compatibility with older flat metadata.
        t = float(meta.get("t", 1.0))
        t1 = float(meta.get("t1", 0.5))
        t2 = float(meta.get("t2", 0.5))
    if str(cluster).startswith("ring3sc_"):
        tag=str(cluster).split("ring3sc_",1)[1]
        try:
            l1s,l2s=tag.lower().split("x",1)
            return make_ring3_supercell(
                int(l1s),int(l2s),t=t,t1=t1,t2=t2
            )
        except Exception as exc:
            raise ValueError(
                f"Could not parse ring3 supercell geometry name {cluster!r}"
            ) from exc
    return RubyPBC(cluster, t=t, t1=t1, t2=t2)


def load_saved_state(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        if "psi_native" in files:
            psi = np.asarray(z["psi_native"], np.complex128)
            states = np.asarray(z["native_states"], np.uint64)
            kind = "vector"
        elif "psi_ground_manifold" in files:
            psi = np.asarray(z["psi_ground_manifold"], np.complex128)
            states = np.asarray(z["native_states"], np.uint64)
            kind = "manifold"
        else:
            raise RuntimeError(
                f"Unrecognized state file {path}: expected psi_native or "
                "psi_ground_manifold"
            )
        metadata = {
            key: np.asarray(z[key]).tolist()
            for key in z.files
            if key not in ("psi_native", "psi_ground_manifold", "native_states")
        }
    return {"kind": kind, "psi": psi, "states": states, "metadata": metadata}


def normalize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.complex128)
    if x.ndim == 1:
        nrm = float(np.linalg.norm(x))
        if nrm < 1e-15:
            raise ValueError("Zero-norm state")
        return x / nrm
    if x.ndim != 2:
        raise ValueError(f"Expected state vector/manifold, got shape {x.shape}")
    out = x.copy()
    norms = np.linalg.norm(out, axis=0)
    if np.any(norms < 1e-15):
        raise ValueError("Ground manifold contains a zero-norm vector")
    out /= norms[None, :]
    return out


def _occupations_bool(states: np.ndarray, ns: int) -> np.ndarray:
    """Boolean occupation table; ~18 MB for the 24-site Nf=8 benchmark."""
    shifts=np.arange(ns-1,-1,-1,dtype=np.uint64)
    return ((states[:,None] >> shifts[None,:]) & np.uint64(1)).astype(bool)


def _projected_site_matrices(
    manifold: np.ndarray, states: np.ndarray, ns: int, occ: np.ndarray | None=None
) -> np.ndarray:
    """M_i[a,b]=<psi_a|n_i|psi_b> inside an orthonormal manifold."""
    if occ is None:
        occ=_occupations_bool(states,ns)
    d=manifold.shape[1]
    out=np.empty((ns,d,d),np.complex128)
    for i in range(ns):
        x=manifold[occ[:,i],:]
        out[i]=x.conj().T@x
    return out


def _ensemble_density_covariance(
    manifold: np.ndarray, states: np.ndarray, ns: int, occ: np.ndarray | None=None
):
    """Basis-independent equal-weight density and connected covariance."""
    if occ is None:
        occ=_occupations_bool(states,ns)
    prob=np.mean(np.abs(manifold)**2,axis=1)
    density=np.asarray(prob@occ,dtype=float)
    nn=np.empty((ns,ns),float)
    # Avoid a D x Ns float64 temporary at D=735471.
    for j in range(ns):
        nn[:,j]=np.asarray((prob*occ[:,j])@occ,dtype=float)
    conn=nn-np.outer(density,density)
    conn=0.5*(conn+conn.T)
    return density,conn,occ


def _real_sources(z: np.ndarray) -> list[np.ndarray]:
    """Orthonormal real quadratures spanning Re/Im of a complex density mode."""
    out=[]
    for raw in (np.real(z),np.imag(z)):
        v=np.asarray(raw,float).copy()
        for q in out:
            v-=np.dot(v,q)*q
        n=float(np.linalg.norm(v))
        if n>1e-10:
            out.append(v/n)
    return out


def _leading_charge_source(g: RubyPBC, covariance: np.ndarray) -> dict:
    """Leading connected-density mode over all allowed finite-torus momenta."""
    best=None
    for q in g.permitted_momenta():
        F=np.zeros((g.Ns,6),np.complex128)
        for cell,R in enumerate(g.reps):
            phase=np.exp(2j*np.pi*float(np.dot(q,np.asarray(R,float))))/np.sqrt(g.num_cells)
            for a in range(6):
                F[6*cell+a,a]=phase
        block=F.conj().T@covariance@F
        block=0.5*(block+block.conj().T)
        w,U=np.linalg.eigh(block)
        if best is None or float(w[-1].real)>best["structure_factor"]:
            best={
                "q":[float(x) for x in q],
                "structure_factor":float(w[-1].real),
                "orbital_vector":np.asarray(U[:,-1],np.complex128),
                "site_vector":np.asarray(F@U[:,-1],np.complex128),
            }
    if best is None:
        raise RuntimeError("No permitted momentum for charge selection")
    return best


def _select_extremal_hermitian(
    manifold: np.ndarray,
    matrix: np.ndarray,
    *,
    ensemble_reference: float=0.0,
    criterion: str="max_shift",
):
    """Choose a normalized state inside the manifold from a Hermitian operator."""
    M=0.5*(np.asarray(matrix,complex)+np.asarray(matrix,complex).conj().T)
    w,U=np.linalg.eigh(M)
    if criterion=="max_positive":
        idx=len(w)-1
    elif criterion=="max_negative":
        idx=0
    elif criterion=="max_abs":
        idx=int(np.argmax(np.abs(w)))
    elif criterion=="max_shift":
        idx=int(np.argmax(np.abs(w-ensemble_reference)))
    else:
        raise ValueError(f"Unknown extremum criterion {criterion}")
    coeff=np.asarray(U[:,idx],np.complex128)
    psi=np.asarray(manifold@coeff,np.complex128)
    psi/=np.linalg.norm(psi)
    return psi,{
        "operator_eigenvalues":[float(x) for x in w],
        "selected_index":int(idx),
        "selected_eigenvalue":float(w[idx]),
        "ensemble_reference":float(ensemble_reference),
        "selected_shift":float(w[idx]-ensemble_reference),
        "coefficients":[[float(z.real),float(z.imag)] for z in coeff],
    }


def _select_lc_branch(
    g: RubyPBC, states: np.ndarray, manifold: np.ndarray, channel: str
):
    triangles=[lp for lp in g.cycles() if len(lp)==3]
    if len(triangles)!=2*g.num_cells:
        raise RuntimeError("Unexpected elementary-triangle count")
    signs=np.ones(len(triangles),float)
    if channel=="lc_opposite":
        signs=np.array([1.0 if lp[0]%6<3 else -1.0 for lp in triangles],float)
    terms=[]
    # Match benchmark_observables: average over all elementary triangles.
    for lp,s in zip(triangles,signs):
        terms.extend([
            (coef*s/len(triangles),ops)
            for coef,ops in native_observables.current_terms(g,lp)
        ])
    actions=np.column_stack([
        native_observables.op_sum(manifold[:,j],states,g.Ns,terms)
        for j in range(manifold.shape[1])
    ])
    M=manifold.conj().T@actions
    psi,meta=_select_extremal_hermitian(
        manifold,M,criterion="max_positive"
    )
    meta.update({
        "selection_available":True,
        "order_mode":channel,
        "definition":"largest signed eigenvalue of the benchmark- normalized triangle loop-current operator within the saved ED ground manifold",
        "n_triangles":len(triangles),
    })
    return psi,meta


def _select_charge_ab_branch(
    g: RubyPBC, states: np.ndarray, manifold: np.ndarray
):
    occ=_occupations_bool(states,g.Ns)
    siteM=_projected_site_matrices(manifold,states,g.Ns,occ)
    weights=np.tile([1.,1.,1.,-1.,-1.,-1.],g.num_cells)
    weights/=np.linalg.norm(weights)
    O=np.einsum("i,iab->ab",weights,siteM)
    ensemble=float(np.trace(O).real/manifold.shape[1])
    psi,meta=_select_extremal_hermitian(
        manifold,O,ensemble_reference=ensemble,criterion="max_shift"
    )
    meta.update({
        "selection_available":True,
        "order_mode":"charge_ab",
        "definition":"extremal A-versus-B triangle density operator within the saved ED ground manifold",
        "source_weights":[float(x) for x in weights],
    })
    return psi,meta


def _select_charge_leading_branch(
    g: RubyPBC, states: np.ndarray, manifold: np.ndarray
):
    density,cov,occ=_ensemble_density_covariance(
        manifold,states,g.Ns
    )
    lead=_leading_charge_source(g,cov)
    siteM=_projected_site_matrices(manifold,states,g.Ns,occ)
    best=None
    for component,weights in enumerate(_real_sources(lead["site_vector"])):
        O=np.einsum("i,iab->ab",weights,siteM)
        ensemble=float(np.dot(weights,density))
        psi,meta=_select_extremal_hermitian(
            manifold,O,ensemble_reference=ensemble,criterion="max_shift"
        )
        score=abs(meta["selected_shift"])
        if best is None or score>best[0]:
            best=(score,psi,meta,component,weights)
    if best is None:
        raise RuntimeError("Leading charge mode has no real Hermitian quadrature")
    _,psi,meta,component,weights=best
    meta.update({
        "selection_available":True,
        "order_mode":"charge_leading",
        "definition":"strongest real quadrature of the leading connected charge structure-factor mode, extremized within the saved ED ground manifold",
        "leading_q":lead["q"],
        "leading_structure_factor":lead["structure_factor"],
        "leading_orbital_vector":[
            [float(z.real),float(z.imag)] for z in lead["orbital_vector"]
        ],
        "selected_quadrature":"real" if component==0 else "imag",
        "source_weights":[float(x) for x in weights],
    })
    return psi,meta


def _select_bow_star_branch(
    g: RubyPBC, states: np.ndarray, manifold: np.ndarray
):
    f=ruby_star_formfactor(g)
    terms=[]
    norm=float(len(g.edges))
    for coeff,e in zip(f,g.edges):
        a=float(coeff)/norm
        terms.append((a,[("+",e.i),("-",e.j)]))
        terms.append((a,[("+",e.j),("-",e.i)]))
    actions=np.column_stack([
        native_observables.op_sum(manifold[:,j],states,g.Ns,terms)
        for j in range(manifold.shape[1])
    ])
    M=manifold.conj().T@actions
    ensemble=float(np.trace(M).real/manifold.shape[1])
    psi,meta=_select_extremal_hermitian(
        manifold,M,ensemble_reference=ensemble,criterion="max_shift"
    )
    meta.update({
        "selection_available":True,
        "order_mode":"bow_star",
        "definition":"extremal Ruby star bond-order operator within the saved ED ground manifold",
    })
    return psi,meta


def select_representation(
    payload: dict, mode: str, column: int | None, g: RubyPBC
) -> dict:
    states=payload["states"]
    psi=normalize_columns(payload["psi"])

    if payload["kind"]=="vector":
        return {
            "mode":"vector",
            "psi":psi,
            "states":states,
            "label":"single saved state",
            "selection":{
                "selection_available":False,
                "order_mode":mode,
                "reason":"Saved file contains only one state; no ground-manifold mixing is possible.",
                "ground_dim":1,
            },
        }

    d=psi.shape[1]
    gram=psi.conj().T@psi
    if np.linalg.norm(gram-np.eye(d))>1e-8:
        raise ValueError(
            "Saved ED ground manifold is not orthonormal; refusing "
            "basis-dependent order selection."
        )
    if mode=="average":
        return {
            "mode":"average",
            "psi":psi,
            "states":states,
            "label":f"equal-weight ED ground-manifold average (d={d})",
            "selection":{
                "selection_available":True,
                "order_mode":"average",
                "ground_dim":int(d),
                "definition":"basis-independent equal-weight ground-manifold mixture",
            },
        }

    if mode=="vector":
        if column is None:
            column=0
        if not 0<=int(column)<d:
            raise IndexError(
                f"--column {column} outside ground-manifold dimension {d}"
            )
        return {
            "mode":"vector",
            "psi":psi[:,int(column)],
            "states":states,
            "label":f"ED ground-manifold column {int(column)}",
            "selection":{
                "selection_available":True,
                "order_mode":"vector",
                "ground_dim":int(d),
                "column":int(column),
                "warning":"A basis vector inside an exactly degenerate manifold is not basis-independent.",
            },
        }

    if d==1:
        return {
            "mode":"vector",
            "psi":psi[:,0],
            "states":states,
            "label":f"unique ED ground state; {mode} cannot polarize it",
            "selection":{
                "selection_available":False,
                "order_mode":mode,
                "ground_dim":1,
                "reason":"Unique ground state cannot acquire a distinct broken-order branch by mixing within the exact ground manifold.",
            },
        }

    if mode in ("lc_same","lc_opposite"):
        chosen,meta=_select_lc_branch(g,states,psi,mode)
    elif mode=="charge_ab":
        chosen,meta=_select_charge_ab_branch(g,states,psi)
    elif mode=="charge_leading":
        chosen,meta=_select_charge_leading_branch(g,states,psi)
    elif mode=="bow_star":
        chosen,meta=_select_bow_star_branch(g,states,psi)
    else:
        raise ValueError(f"Unknown mode {mode}")

    meta["ground_dim"]=int(d)
    return {
        "mode":"vector",
        "psi":chosen,
        "states":states,
        "label":f"ED ground-manifold branch selected by {mode} (d={d})",
        "selection":meta,
    }


def site_positions(g: RubyPBC):
    xy = np.zeros((g.Ns, 2), float)
    cells = np.zeros((g.Ns, 2), int)
    for site in range(g.Ns):
        ci, orb = divmod(site, 6)
        cell = np.asarray(g.reps[ci], int)
        cells[site] = cell
        xy[site] = A @ cell + ORBITALS[orb]
    return xy, cells


def edge_segments(g: RubyPBC, xy: np.ndarray, cells: np.ndarray):
    """Short physical representative for each independent PBC hopping bond."""
    starts = np.zeros((len(g.edges), 2), float)
    ends = np.zeros((len(g.edges), 2), float)
    crossed = np.zeros(len(g.edges), bool)
    for n, e in enumerate(g.edges):
        starts[n] = xy[e.i]
        ci = e.i // 6
        ends[n] = (
            A @ (cells[6 * ci] + np.asarray(e.delta, int))
            + ORBITALS[e.j % 6]
        )
        crossed[n] = np.linalg.norm(ends[n] - xy[e.j]) > 1e-8
    return starts, ends, crossed


def mixture_probabilities(rep: dict) -> np.ndarray:
    if rep["mode"] == "vector":
        return np.abs(rep["psi"]) ** 2
    return np.mean(np.abs(rep["psi"]) ** 2, axis=1)


def compute_site_density(g: RubyPBC, states: np.ndarray, rep: dict):
    prob = mixture_probabilities(rep)
    density = np.zeros(g.Ns, float)
    for i in range(g.Ns):
        bit = np.uint64(1 << (g.Ns - 1 - i))
        density[i] = float(prob @ ((states & bit) != 0))
    return density


def edge_rho_single(g: RubyPBC, states: np.ndarray, psi: np.ndarray):
    """rho_e=<c_i^dag c_j> for the stored oriented edge i,j."""
    rho = np.zeros(len(g.edges), np.complex128)
    for n, e in enumerate(g.edges):
        row, col, sign = edge_triplets(g, states, n)
        filled_j = (
            states[col] & np.uint64(1 << (g.Ns - 1 - e.j))
        ) != 0
        rho[n] = np.vdot(
            psi[row[filled_j]],
            sign[filled_j] * psi[col[filled_j]],
        )
    return rho


def compute_edge_observables(g: RubyPBC, states: np.ndarray, rep: dict):
    if rep["mode"] == "vector":
        rho = edge_rho_single(g, states, rep["psi"])
    else:
        manifold = rep["psi"]
        rho = sum(
            (edge_rho_single(g, states, manifold[:, j])
             for j in range(manifold.shape[1])),
            start=np.zeros(len(g.edges), np.complex128),
        ) / manifold.shape[1]

    bond = 2.0 * np.real(rho)
    current = np.empty(len(g.edges), float)
    for n, e in enumerate(g.edges):
        z = rho[n]
        current[n] = float(
            (1j * g.t[e.kind] * (z - np.conjugate(z))).real
        )
    return rho, bond, current


def export_site_csv(path, g, xy, density):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "site", "cell_index", "orbital_in_cell",
            "x", "y", "density",
        ])
        for site in range(g.Ns):
            cell, orb = divmod(site, 6)
            w.writerow([
                site, cell, orb,
                float(xy[site, 0]), float(xy[site, 1]),
                float(density[site]),
            ])


def export_bond_csv(path, g, starts, ends, crossed, rho, bond, current):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "edge_idx", "site_i", "site_j", "kind",
            "x_i", "y_i", "x_j_image", "y_j_image", "crosses_pbc",
            "rho_real", "rho_imag", "bond_order", "current_i_to_j",
        ])
        for n, e in enumerate(g.edges):
            w.writerow([
                n, e.i, e.j, e.kind,
                float(starts[n, 0]), float(starts[n, 1]),
                float(ends[n, 0]), float(ends[n, 1]),
                bool(crossed[n]),
                float(rho[n].real), float(rho[n].imag),
                float(bond[n]), float(current[n]),
            ])


def configure_axes(ax, xy, starts, ends):
    pts = np.vstack((xy, starts, ends))
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    pad = 0.5
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_base_bonds(ax, g, starts, ends, crossed, alpha=0.3):
    for n, e in enumerate(g.edges):
        ax.plot(
            [starts[n, 0], ends[n, 0]],
            [starts[n, 1], ends[n, 1]],
            lw=1.0,
            color="0.50",
            alpha=alpha,
            linestyle="--" if crossed[n] else "-",
            zorder=1,
        )


def draw_sites(ax, xy, values=None, cmap=None, norm=None, size=120):
    if values is None:
        ax.scatter(
            xy[:, 0], xy[:, 1], s=size, facecolors="white",
            edgecolors="black", linewidths=1.0, zorder=5,
        )
        artist = None
    else:
        artist = ax.scatter(
            xy[:, 0], xy[:, 1], s=size,
            c=values, cmap=cmap, norm=norm,
            edgecolors="black", linewidths=1.0, zorder=5,
        )
    for i, p in enumerate(xy):
        ax.text(
            p[0], p[1], str(i),
            ha="center", va="center", fontsize=7.5, zorder=6,
        )
    return artist


def save_both(fig, base: Path):
    fig.savefig(base.with_suffix(".png"), dpi=230, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_site_density(g, xy, starts, ends, crossed, density, out, title):
    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    draw_base_bonds(ax, g, starts, ends, crossed)
    nbar = float(np.mean(density))
    dev = max(float(np.max(np.abs(density - nbar))), 1e-12)
    norm = TwoSlopeNorm(vmin=nbar-dev, vcenter=nbar, vmax=nbar+dev)
    sc = draw_sites(
        ax, xy, density, SITE_CMAP, norm,
        size=190 + 900 * np.abs(density - nbar) / max(dev, 1e-12),
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.025)
    cb.set_label(r"$\langle n_i\rangle$")
    ax.set_title(f"{title}\nsite density", fontsize=13)
    configure_axes(ax, xy, starts, ends)
    fig.tight_layout()
    save_both(fig, out / "site_density")


def plot_bond_order(g, xy, starts, ends, crossed, bond, out, title):
    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    maxabs = max(float(np.max(np.abs(bond))), 1e-12)
    norm = TwoSlopeNorm(vmin=-maxabs, vcenter=0.0, vmax=maxabs)
    cmap = plt.get_cmap(BOND_CMAP)
    for n, e in enumerate(g.edges):
        ax.plot(
            [starts[n, 0], ends[n, 0]],
            [starts[n, 1], ends[n, 1]],
            color=cmap(norm(bond[n])),
            lw=0.8 + 4.2 * abs(bond[n]) / maxabs,
            linestyle="--" if crossed[n] else "-",
            solid_capstyle="round",
            zorder=2,
        )
    draw_sites(ax, xy, size=105)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.025)
    cb.set_label(r"$2\,\mathrm{Re}\langle c_i^\dagger c_j\rangle$")
    ax.set_title(f"{title}\nbond order", fontsize=13)
    configure_axes(ax, xy, starts, ends)
    fig.tight_layout()
    save_both(fig, out / "bond_order")


def plot_bond_current(
    g, xy, starts, ends, crossed, current, out, title,
    threshold_fraction=0.02,
):
    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    draw_base_bonds(ax, g, starts, ends, crossed, alpha=0.22)
    maxabs = max(float(np.max(np.abs(current))), 1e-14)
    norm = TwoSlopeNorm(vmin=-maxabs, vcenter=0.0, vmax=maxabs)
    cmap = plt.get_cmap(CURRENT_CMAP)

    for n, e in enumerate(g.edges):
        j = float(current[n])
        if abs(j) < threshold_fraction * maxabs:
            continue
        start = starts[n].copy()
        end = ends[n].copy()
        if j < 0:
            start, end = end, start
        tail = 0.23 * start + 0.77 * end
        head = 0.77 * start + 0.23 * end
        ax.add_patch(
            patches.FancyArrowPatch(
                tail, head,
                arrowstyle="-|>",
                mutation_scale=10 + 8 * abs(j) / maxabs,
                linewidth=1.0 + 3.0 * abs(j) / maxabs,
                color=cmap(norm(j)),
                linestyle="--" if crossed[n] else "-",
                zorder=4,
            )
        )

    draw_sites(ax, xy, size=105)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.025)
    cb.set_label(r"$J_{i\to j}$")
    ax.set_title(
        f"{title}\nbond current; max |J|={maxabs:.5g}",
        fontsize=13,
    )
    configure_axes(ax, xy, starts, ends)
    fig.tight_layout()
    save_both(fig, out / "bond_current")


def plot_site_bond_current_triptych(
    g, xy, starts, ends, crossed, density, bond, current, out, title,
    threshold_fraction=0.02,
):
    """Save site density, bond order and bond current in one 1x3 figure."""
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.3))

    # Site density.
    ax=axes[0]
    draw_base_bonds(ax,g,starts,ends,crossed)
    nbar=float(np.mean(density))
    dev=max(float(np.max(np.abs(density-nbar))),1e-12)
    nnorm=TwoSlopeNorm(vmin=nbar-dev,vcenter=nbar,vmax=nbar+dev)
    sc=draw_sites(
        ax,xy,density,SITE_CMAP,nnorm,
        size=150+650*np.abs(density-nbar)/max(dev,1e-12),
    )
    cb=fig.colorbar(sc,ax=ax,shrink=0.72,pad=0.02)
    cb.set_label(r"$\langle n_i\rangle$")
    ax.set_title("site density",fontsize=12)
    configure_axes(ax,xy,starts,ends)

    # Bond order.
    ax=axes[1]
    bmax=max(float(np.max(np.abs(bond))),1e-12)
    bnorm=TwoSlopeNorm(vmin=-bmax,vcenter=0.0,vmax=bmax)
    bcmap=plt.get_cmap(BOND_CMAP)
    for n,e in enumerate(g.edges):
        ax.plot(
            [starts[n,0],ends[n,0]],
            [starts[n,1],ends[n,1]],
            color=bcmap(bnorm(bond[n])),
            lw=0.8+3.8*abs(bond[n])/bmax,
            linestyle="--" if crossed[n] else "-",
            solid_capstyle="round",zorder=2,
        )
    draw_sites(ax,xy,size=90)
    sm=ScalarMappable(norm=bnorm,cmap=bcmap)
    sm.set_array([])
    cb=fig.colorbar(sm,ax=ax,shrink=0.72,pad=0.02)
    cb.set_label(r"$2\,\mathrm{Re}\langle c_i^\dagger c_j\rangle$")
    ax.set_title("bond order",fontsize=12)
    configure_axes(ax,xy,starts,ends)

    # Bond current.
    ax=axes[2]
    draw_base_bonds(ax,g,starts,ends,crossed,alpha=0.22)
    jmax=max(float(np.max(np.abs(current))),1e-14)
    jnorm=TwoSlopeNorm(vmin=-jmax,vcenter=0.0,vmax=jmax)
    jcmap=plt.get_cmap(CURRENT_CMAP)
    for n,e in enumerate(g.edges):
        j=float(current[n])
        if abs(j)<threshold_fraction*jmax:
            continue
        start=starts[n].copy()
        end=ends[n].copy()
        if j<0:
            start,end=end,start
        tail=0.23*start+0.77*end
        head=0.77*start+0.23*end
        ax.add_patch(
            patches.FancyArrowPatch(
                tail,head,arrowstyle="-|>",
                mutation_scale=9+7*abs(j)/jmax,
                linewidth=0.9+2.7*abs(j)/jmax,
                color=jcmap(jnorm(j)),
                linestyle="--" if crossed[n] else "-",
                zorder=4,
            )
        )
    draw_sites(ax,xy,size=90)
    sm=ScalarMappable(norm=jnorm,cmap=jcmap)
    sm.set_array([])
    cb=fig.colorbar(sm,ax=ax,shrink=0.72,pad=0.02)
    cb.set_label(r"$J_{i\to j}$")
    ax.set_title(f"bond current; max |J|={jmax:.4g}",fontsize=12)
    configure_axes(ax,xy,starts,ends)

    fig.suptitle(title,fontsize=14,y=0.99)
    fig.tight_layout(rect=(0,0,1,0.965))
    fig.savefig(out/"site_bond_current_triptych.png",dpi=230,bbox_inches="tight")
    fig.savefig(out/"site_bond_current_triptych.svg",bbox_inches="tight")
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--result-root", required=True,
        help="Projection-comparison result directory containing geometry.json."
    )
    p.add_argument(
        "--state-file", default=None,
        help=(
            "Saved final_state.npz. If omitted, use "
            "<result-root>/ed/final_state.npz."
        ),
    )
    p.add_argument(
        "--mode",
        choices=[
            "average", "vector",
            "lc_same", "lc_opposite",
            "charge_leading", "charge_ab",
            "bow_star",
        ],
        default="average",
        help=(
            "For a degenerate ED manifold, either average/select a raw basis "
            "vector or choose the pure ground-manifold branch that extremizes "
            "the requested LC/charge/bond-order operator."
        )
    )
    p.add_argument(
        "--column", type=int, default=None,
        help="Ground-manifold column used with --mode vector."
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Default: <result-root>/ed_maps or sibling <state-dir>_maps."
    )
    p.add_argument(
        "--current-threshold", type=float, default=0.02,
        help="Hide arrows below this fraction of max |current|."
    )
    return p


def main():
    a = parser().parse_args()
    root = Path(a.result_root).resolve()
    state_file = (
        Path(a.state_file).resolve()
        if a.state_file is not None
        else root / "ed" / "final_state.npz"
    )
    if not state_file.exists():
        raise FileNotFoundError(
            f"State file not found: {state_file}\n"
            "For ED, rerun run_projection_comparison.py with --ed-only "
            "after updating the repository."
        )

    if a.output_dir is not None:
        out = Path(a.output_dir).resolve()
    elif state_file.parent.name == "ed":
        out = root / f"ed_maps_{a.mode}"
    else:
        out = state_file.parent.parent / (state_file.parent.name + f"_maps_{a.mode}")
    out.mkdir(parents=True, exist_ok=True)

    g = load_geometry(root)
    payload = load_saved_state(state_file)
    rep = select_representation(payload, a.mode, a.column, g)
    states = np.asarray(rep["states"], np.uint64)
    if len(states) != len(rep["psi"]):
        raise RuntimeError(
            f"Basis/state mismatch: {len(states)} states versus "
            f"wavefunction leading dimension {len(rep['psi'])}"
        )

    xy, cells = site_positions(g)
    starts, ends, crossed = edge_segments(g, xy, cells)
    density = compute_site_density(g, states, rep)
    rho, bond, current = compute_edge_observables(g, states, rep)

    export_site_csv(out / "site_density.csv", g, xy, density)
    export_bond_csv(
        out / "bond_observables.csv",
        g, starts, ends, crossed, rho, bond, current,
    )

    title = f"{g.name}: {rep['label']}"
    plot_site_density(
        g, xy, starts, ends, crossed, density, out, title
    )
    plot_bond_order(
        g, xy, starts, ends, crossed, bond, out, title
    )
    plot_bond_current(
        g, xy, starts, ends, crossed, current, out, title,
        threshold_fraction=float(a.current_threshold),
    )
    plot_site_bond_current_triptych(
        g,xy,starts,ends,crossed,density,bond,current,out,title,
        threshold_fraction=float(a.current_threshold),
    )

    summary = {
        "cluster": g.name,
        "Ns": int(g.Ns),
        "state_file": str(state_file),
        "representation": rep["label"],
        "selection": rep.get("selection", {}),
        "mean_density": float(np.mean(density)),
        "density_min": float(np.min(density)),
        "density_max": float(np.max(density)),
        "max_abs_bond_order": float(np.max(np.abs(bond))),
        "max_abs_current": float(np.max(np.abs(current))),
        "note": (
            "For a unique spinless time-reversal-even ED ground state, "
            "all one-point bond currents should vanish up to numerical noise."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if rep["mode"]=="vector":
        np.savez_compressed(
            out/"selected_state.npz",
            psi_native=np.asarray(rep["psi"],np.complex128),
            native_states=np.asarray(states,np.uint64),
            selection_mode=np.asarray(a.mode),
        )

    print(f"[saved-state maps] state={state_file}")
    print(f"[saved-state maps] representation={rep['label']}")
    print(f"[saved-state maps] output={out}")
    print(
        f"[saved-state maps] n=[{summary['density_min']:.6g},"
        f"{summary['density_max']:.6g}], "
        f"max|B|={summary['max_abs_bond_order']:.6g}, "
        f"max|J|={summary['max_abs_current']:.6g}"
    )


if __name__ == "__main__":
    main()

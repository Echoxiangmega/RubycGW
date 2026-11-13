"""Order seeds and real-space observables for the periodic hex18 Ruby torus.

All seed matrices act on the actual 18-orbital folded lattice. Sources are
initial *self-energy perturbations only*; physical h0 is never altered.
The SD/ISD names label trial real-bond motifs, not a proven kagome geometry.
"""
from __future__ import annotations

import numpy as np

from .model import ruby_hoppings
from .pseudospin import primitive_pseudospin_vertex
from .supercell import supercell_hoppings


QK = (1. / 3., 1. / 3.)
Q0 = (0., 0.)
FAMILIES = ("symmetric", "co_q0_A0", "co_qK_A0",
            "co_qK_A1", "co_qK_A2", "co_qK_odd_A0",
            "lc_q0_same", "lc_q0_opposite",
            "lc_qK_same_cos", "lc_qK_opposite_cos",
            "lc_qK_same_sin", "lc_qK_opposite_sin",
            "bond_q0_c3", "bond_qK_c3",
            "bond_qK_sd", "bond_qK_isd",
            "mixed_qK", "random_complex")
DEFAULT_FAMILIES = (
    "symmetric", "co_q0_A0", "co_qK_A0", "co_qK_odd_A0",
    "lc_q0_same", "lc_q0_opposite",
    "lc_qK_same_cos", "lc_qK_opposite_cos",
    "bond_qK_sd", "bond_qK_isd", "mixed_qK"
)


def _phase(geom, cell, q=Q0, harmonic="cos"):
    angle=2*np.pi*float(np.dot(np.asarray(q,float),geom.cells[cell]))
    if harmonic=="cos":
        return float(np.cos(angle))
    if harmonic=="sin":
        return float(np.sin(angle))
    raise ValueError("harmonic must be cos/sin")


def _hermitian_normalize(a):
    a=np.asarray(a,complex)
    a=.5*(a+a.conj().T)
    scale=float(np.max(np.abs(a)))
    if scale<=1e-14:
        return a
    return a/scale


def _diag_co(geom, *, q=Q0, parity=1., orient=0):
    mat=np.zeros((geom.nsites,geom.nsites),complex)
    pattern=np.roll(np.asarray([2.,-1.,-1.]),int(orient))
    for cell in range(geom.ncell):
        w=_phase(geom,cell,q)
        for side in (0,1):
            sign=(1. if side==0 else float(parity))
            for j in range(3):
                site=cell*6+3*side+j
                mat[site,site]+=w*sign*pattern[j]
    return mat


def _lc(geom,*,q=Q0,kind="same",harmonic="cos"):
    name="z_same" if kind=="same" else "z_opposite"
    vertex=primitive_pseudospin_vertex(name)
    mat=np.zeros((geom.nsites,geom.nsites),complex)
    for cell in range(geom.ncell):
        sl=slice(6*cell,6*(cell+1))
        mat[sl,sl]=_phase(geom,cell,q,harmonic)*vertex
    return mat


def _physical_hoppings(geom,params):
    """Unique undirected oriented hoppings, including PBC wraps."""
    if geom.name!="hex18":
        raise ValueError("order seed/diagnostics require hex18 torus")
    # Retain physically distinct links; duplicate directed reversals are removed.
    edges={}
    for i,j,shift,t in supercell_hoppings(params):
        i,j=int(i),int(j)
        if i==j:
            continue
        key=(min(i,j),max(i,j))
        if key not in edges:
            edges[key]=(i,j,complex(t))
    return list(edges.values())


def _real_bond(geom,params,*,q=QK,kind="sd"):
    mat=np.zeros((geom.nsites,geom.nsites),complex)
    for i,j,t in _physical_hoppings(geom,params):
        ci=i//6
        cj=j//6
        tri=(i//3==j//3)
        if kind=="sd":
            w=1. if tri else -0.5
        elif kind=="isd":
            w=-1. if tri else 0.5
        elif kind=="c3":
            # Directionally anisotropic, with the three intra-triangle
            # bond orientations represented by 2,-1,-1.
            if not tri:
                continue
            local=tuple(sorted((i%3,j%3)))
            w={(0,1):2.,(0,2):-1.,(1,2):-1.}[local]
        else:
            raise ValueError("unknown real-bond seed")
        wave=0.5*(_phase(geom,ci,q)+_phase(geom,cj,q))
        mat[i,j]+=w*wave
        mat[j,i]+=w*wave
    return mat


def seed_matrix(geom,params,name,*,random_seed=1):
    """Return a Hermitian initial self-energy direction, max element = 1."""
    if name not in FAMILIES:
        raise ValueError(f"unknown candidate {name!r}; choices: {FAMILIES}")
    if name=="symmetric":
        return np.zeros((geom.nsites,geom.nsites),complex)
    if name.startswith("co_"):
        q=QK if "_qK_" in name else Q0
        parity=-1. if "_odd_" in name else 1.
        orient=int(name.rsplit("_A",1)[-1]) if "_A" in name else 0
        return _hermitian_normalize(
            _diag_co(geom,q=q,parity=parity,orient=orient)
        )
    if name.startswith("lc_"):
        parts=name.split("_")
        q=QK if parts[1]=="qK" else Q0
        harmonic=parts[3] if len(parts)==4 else "cos"
        return _hermitian_normalize(
            _lc(geom,q=q,kind=parts[2],harmonic=harmonic)
        )
    if name.startswith("bond_"):
        _,qt,kind=name.split("_")
        return _hermitian_normalize(
            _real_bond(geom,params,q=QK if qt=="qK" else Q0,kind=kind)
        )
    if name=="mixed_qK":
        return _hermitian_normalize(
            _diag_co(geom,q=QK)+_lc(geom,q=QK,kind="same")
        )
    if name=="random_complex":
        rng=np.random.default_rng(int(random_seed))
        mat=np.zeros((geom.nsites,geom.nsites),complex)
        for i,j,t in _physical_hoppings(geom,params):
            z=rng.normal()+1j*rng.normal()
            mat[i,j]=z
            mat[j,i]=z.conjugate()
        mat[np.diag_indices(geom.nsites)]=rng.normal(
            size=geom.nsites
        )
        return _hermitian_normalize(mat)
    raise AssertionError(name)


def _channel_fourier(rho,geom,name,q):
    K=primitive_pseudospin_vertex(name)
    val=0j
    for cell in range(geom.ncell):
        sl=slice(6*cell,6*(cell+1))
        phase=np.exp(-2j*np.pi*float(np.dot(q,geom.cells[cell])))
        val+=phase*np.trace(K@rho[sl,sl])
    return complex(val/geom.ncell)


def order_observables(rho,geom,params,*,threshold=1e-3):
    """Return channel amplitudes, physical bond currents and order label.

    rho_ij=<c_j^dag c_i>. Intra/intertriangle physical hoppings are
    enumerated separately. Current signs follow the chosen oriented edge.
    """
    rho=np.asarray(rho,complex)
    rho=.5*(rho+rho.conj().T)
    if rho.shape!=(geom.nsites,geom.nsites):
        raise ValueError("density matrix must have shape (18,18)")
    local=np.diag(rho).real
    channels=("x_even","x_odd","y_even","y_odd",
              "z_same","z_opposite")
    out={}
    for qlabel,q in (("q0",Q0),("qK",QK)):
        for name in channels:
            value=_channel_fourier(rho,geom,name,q)
            out[f"{name}_{qlabel}"]=value
    bond_rows=[]
    for i,j,t in _physical_hoppings(geom,params):
        expectation=complex(rho[j,i])
        bond_rows.append(dict(
            i=i,j=j,t_real=float(t.real),t_imag=float(t.imag),
            real_bond=float(2*expectation.real),
            current=float(2*np.imag(t*expectation)),
            triangle=int(i//3==j//3),
        ))
    # Direction-dependent real bond order, measured on intra-triangle
    # edges after subtracting their average. This is an order diagnostic,
    # not a normalized susceptibility.
    b_by_tri=[]
    for c in range(geom.ncell):
        for side in range(2):
            base=6*c+3*side
            vals=np.asarray([
                2*rho[base+j,base+i].real
                for i,j in ((0,1),(1,2),(2,0))
            ])
            b_by_tri.extend(vals.tolist())
    bond_arr=np.asarray(b_by_tri,float).reshape(geom.ncell,2,3)
    bond_anis=float(np.max(abs(bond_arr-bond_arr.mean(axis=2,keepdims=True))))
    amps={k:float(abs(v)) for k,v in out.items()}
    co=max((v for k,v in amps.items() if k.startswith(("x_","y_"))),default=0.)
    lc=max((v for k,v in amps.items() if k.startswith("z_")),default=0.)
    dominant=max(amps,key=amps.get) if amps else "none"
    active_co=co>float(threshold)
    active_lc=lc>float(threshold)
    active_bond=bond_anis>float(threshold)
    if active_co and active_lc:
        family="mixed_CO_LC"
    elif active_co:
        family="CO"
    elif active_lc:
        family="LC"
    elif active_bond:
        family="real_bond_order"
    else:
        family="symmetric"
    return dict(
        density=local,bond_rows=bond_rows,
        amplitudes=out,amplitude_abs=amps,
        co_amplitude=float(co),lc_amplitude=float(lc),
        bond_anisotropy=bond_anis,
        density_range=float(local.max()-local.min()),
        dominant_channel=dominant,
        dominant_amplitude=float(amps.get(dominant,0.)),
        phase_family=family,
    )


__all__=["QK","Q0","FAMILIES","DEFAULT_FAMILIES",
         "seed_matrix","order_observables"]

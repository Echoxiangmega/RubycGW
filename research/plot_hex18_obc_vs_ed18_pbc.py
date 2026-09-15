#!/usr/bin/env python3
"""Compare six-triangle Hex18 OBC and the existing 18-site index-three PBC ED.

Usage
-----
python plot_hex18_obc_vs_ed18_pbc.py \
  --hex results/hex18_low_energy_modes.npz \
  --pbc results/ed18_pbc_low_energy_modes.npz \
  --out-prefix results/hex18_obc_vs_pbc

The script writes four figures:
  *_spectrum.png   low-energy branches; marker size = uniform-current weight
  *_gaps.png       first gap and dominant uniform-current gap
  *_scaled_gaps.png V times those gaps (strong-coupling 1/V diagnostic)
  *_chirality.png  dominant current spectral weight and projected Jmax
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hex", type=Path, required=True)
    p.add_argument("--pbc", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, default=Path("results/hex18_obc_vs_pbc"))
    p.add_argument("--n-branches", type=int, default=8)
    p.add_argument("--subspace", type=int, default=4, help="PJP subspace size to compare")
    return p.parse_args()


def _load(path):
    return np.load(path, allow_pickle=False)


def _hex_uniform_weights(d):
    comps = [str(x) for x in d["components"]]
    iz = comps.index("z")
    return np.asarray(d["spectral_weights"][:, :, iz, 0], dtype=float)


def _hex_current_summary(d):
    gaps = np.asarray(d["gaps"], dtype=float)
    w = _hex_uniform_weights(d)
    idx=[]; gap=[]; wt=[]; deff=[]
    for iv in range(len(gaps)):
        n = 1 + int(np.argmax(w[iv, 1:]))
        idx.append(n); gap.append(gaps[iv,n]); wt.append(w[iv,n])
        good = (np.arange(gaps.shape[1]) > 0) & (gaps[iv] > 1e-12)
        S = float(np.sum(w[iv,good]))
        chi = float(2*np.sum(w[iv,good]/gaps[iv,good]))
        deff.append(2*S/chi if chi > 0 else np.nan)
    return np.asarray(idx), np.asarray(gap), np.asarray(wt), np.asarray(deff)


def _projected_jmax(d, M):
    sizes = np.asarray(d["subspace_sizes"], dtype=int)
    if M not in sizes:
        # nearest available size, stated in terminal output
        j = int(np.argmin(np.abs(sizes-M)))
    else:
        j = int(np.flatnonzero(sizes==M)[0])
    return int(sizes[j]), np.asarray(d["projected_jmax"][:,j], dtype=float)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main():
    a = _args()
    h = _load(a.hex); p = _load(a.pbc)
    Vh=np.asarray(h["V"],dtype=float); Vp=np.asarray(p["V"],dtype=float)
    gh=np.asarray(h["gaps"],dtype=float); gp=np.asarray(p["gaps"],dtype=float)
    wh=_hex_uniform_weights(h); wp=np.asarray(p["spectral_weights"][:,:,2,0,0],dtype=float)  # z, even, Gamma
    ih,cgh,cwh,deffh=_hex_current_summary(h)
    ip=np.asarray(p["uniform_current_index"],dtype=int)
    cgp=np.asarray(p["uniform_current_gap"],dtype=float)
    cwp=np.asarray(p["uniform_current_weight"],dtype=float)
    deffp=np.asarray(p["uniform_current_delta_eff"],dtype=float)

    # 1. Low-energy spectra with current weight encoded by marker area.
    fig,ax=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    nb=min(a.n_branches,gh.shape[1]-1,gp.shape[1]-1)
    for n in range(1,nb+1):
        ax[0].plot(Vh,gh[:,n],lw=0.9,alpha=0.65)
        ax[0].scatter(Vh,gh[:,n],s=12+42*wh[:,n],alpha=0.8)
        ax[1].plot(Vp,gp[:,n],lw=0.9,alpha=0.65)
        ax[1].scatter(Vp,gp[:,n],s=12+42*wp[:,n],alpha=0.8)
    ax[0].set_title("Hex18 OBC")
    ax[1].set_title("ED18 PBC torus")
    for x in ax:
        x.set_xlabel("V"); x.set_ylabel(r"$E_n-E_0$")
        x.grid(alpha=0.2)
    fig.suptitle("Low-energy spectrum (marker area = uniform-current spectral weight)")
    _save(fig,Path(str(a.out_prefix)+"_spectrum.png"))

    # 2. First vs current-carrying gap.
    fig,ax=plt.subplots(figsize=(7,4.8))
    ax.plot(Vh,gh[:,1],"o-",label="OBC first gap")
    ax.plot(Vh,cgh,"o--",label="OBC uniform-current gap")
    ax.plot(Vp,gp[:,1],"s-",label="PBC first gap")
    ax.plot(Vp,cgp,"s--",label="PBC uniform-current gap")
    ax.plot(Vh,deffh,":",label=r"OBC $2S/\chi$")
    ax.plot(Vp,deffp,":",label=r"PBC $2S/\chi$")
    ax.set_xlabel("V"); ax.set_ylabel("gap")
    ax.set_title("Lowest and uniform-current gaps")
    ax.grid(alpha=0.2); ax.legend(fontsize=8)
    _save(fig,Path(str(a.out_prefix)+"_gaps.png"))

    # 3. V * gaps: tests whether the large-V splitting is t^2/V-like.
    fig,ax=plt.subplots(figsize=(7,4.8))
    ax.plot(Vh,Vh*gh[:,1],"o-",label=r"OBC $V\Delta_1$")
    ax.plot(Vh,Vh*cgh,"o--",label=r"OBC $V\Delta_J$")
    ax.plot(Vp,Vp*gp[:,1],"s-",label=r"PBC $V\Delta_1$")
    ax.plot(Vp,Vp*cgp,"s--",label=r"PBC $V\Delta_J$")
    ax.set_xlabel("V"); ax.set_ylabel(r"$V\Delta$")
    ax.set_title(r"Strong-coupling scaling diagnostic")
    ax.grid(alpha=0.2); ax.legend(fontsize=8)
    _save(fig,Path(str(a.out_prefix)+"_scaled_gaps.png"))

    # 4. Chirality strength.
    Mh,jh=_projected_jmax(h,a.subspace); Mp,jp=_projected_jmax(p,a.subspace)
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.4))
    ax[0].plot(Vh,cwh,"o-",label="Hex18 OBC")
    ax[0].plot(Vp,cwp,"s-",label="ED18 PBC")
    ax[0].set_xlabel("V"); ax[0].set_ylabel(r"$|\langle n_*|J_u|0\rangle|^2$")
    ax[0].set_title("Dominant uniform-current spectral weight")
    ax[0].grid(alpha=0.2); ax[0].legend()
    ax[1].plot(Vh,jh,"o-",label=f"OBC M={Mh}")
    ax[1].plot(Vp,jp,"s-",label=f"PBC M={Mp}")
    ax[1].set_xlabel("V"); ax[1].set_ylabel(r"max eig$(P_MJ_uP_M)$")
    ax[1].set_title("Best chiral combination in low-energy subspace")
    ax[1].grid(alpha=0.2); ax[1].legend()
    _save(fig,Path(str(a.out_prefix)+"_chirality.png"))

    print("\nPBC first-excitation fingerprints:")
    labels=[str(x) for x in p["first_multiplet_label"]]
    mult=np.asarray(p["first_multiplicity"],dtype=int)
    for V,label,m in zip(Vp,labels,mult):
        print(f"  V={V:g}: {label}  multiplicity={m}")
    print("\nDominant current state indices:")
    for V,n,g,w in zip(Vp,ip,cgp,cwp):
        print(f"  PBC V={V:g}: n*={n}, gap={g:.8e}, weight={w:.8e}")


if __name__ == "__main__":
    main()

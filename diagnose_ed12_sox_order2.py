#!/usr/bin/env python3
"""Strict bare-SOX O(V^2) diagnostic for the 12-site current response ledger.

The expensive weak-coupling ED/cGW scan is not repeated. This driver evaluates
only the second-order-exchange skeleton derivative on the V=0 one-body
background, where its coefficient is unambiguous. If an existing
``ledger_fits.json`` / sibling ``ledger_scan.npz`` is supplied, the SOX result is
compared directly with the missing exact-minus-cGW quadratic coefficient.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.sox_diagnostic import (
    gw_direct_order2_response_coefficient_free,
    solve_free_mu,
    sox_response_coefficient_free,
)

CHANNELS = ("z_same", "z_opposite")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--filling", type=float, default=3.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--nquad", type=int, default=128)
    p.add_argument("--channels", nargs="+", default=CHANNELS)
    p.add_argument(
        "--ledger",
        type=Path,
        default=Path("results/ed12_order2_ledger/ledger_fits.json"),
        help="existing O(V^2) ledger_fits.json; omitted comparison if absent",
    )
    p.add_argument(
        "--fit-windows", nargs="+", type=float, default=(0.02, 0.03, 0.05),
        help="refit exact/cGW b from sibling ledger_scan.npz without rerunning ED/GW",
    )
    p.add_argument("--out", type=Path, default=Path("results/ed12_sox_order2"))
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def _fit_abc(V, y, vmax):
    V = np.asarray(V, float); y = np.asarray(y, float)
    m = (V > 0) & (V <= float(vmax) + 1e-15)
    x = V[m]; z = y[m]
    if len(x) < 3:
        return None
    A = np.column_stack([x, x*x, x*x*x])
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    return dict(a=float(c[0]), b=float(c[1]), c=float(c[2]), n=int(len(x)))


def _load_ledger(path, channels, fit_windows):
    path = Path(path)
    out = dict(saved=None, refits={})
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            out["saved"] = json.load(f)
    scan = path.with_name("ledger_scan.npz")
    if not scan.exists():
        return out
    with np.load(scan, allow_pickle=False) as d:
        V = np.asarray(d["V"], float)
        ch_saved = [str(x) for x in np.asarray(d["channels"])]
        exact = np.asarray(d["exact_vertex"], float)
        full = np.asarray(d["full_vertex"], float)
    for vmax in fit_windows:
        wf = {}
        for ch in channels:
            if ch not in ch_saved:
                continue
            ic = ch_saved.index(ch)
            fe = _fit_abc(V, exact[:, ic], vmax)
            fc = _fit_abc(V, full[:, ic], vmax)
            if fe is None or fc is None:
                continue
            wf[ch] = dict(exact=fe, cgw=fc, missing=float(fe["b"] - fc["b"]))
        if wf:
            out["refits"][f"{float(vmax):.12g}"] = wf
    return out


def _plot(path, channels, result, ledger, dpi):
    saved = ledger.get("saved")
    if not saved:
        return
    x = np.arange(len(channels), dtype=float)
    exact = np.array([saved[ch]["exact_vertex"]["b"] for ch in channels])
    cgw = np.array([saved[ch]["full_vertex"]["b"] for ch in channels])
    sox = np.array([result[ch]["b_sox"] for ch in channels])
    pred = cgw + sox
    missing = exact - cgw

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    width = 0.24
    axes[0].bar(x-width, exact, width, label="exact ED b")
    axes[0].bar(x, cgw, width, label="cGW b")
    axes[0].bar(x+width, pred, width, label="cGW + bare SOX")
    axes[0].set_xticks(x, channels)
    axes[0].set_ylabel(r"quadratic coefficient $b$")
    axes[0].set_title(r"$\Delta\chi=aV+bV^2+\cdots$")
    axes[0].axhline(0, linewidth=.8, color="0.4")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=.25)

    axes[1].bar(x-width/2, missing, width, label="exact - cGW")
    axes[1].bar(x+width/2, sox, width, label="bare SOX")
    axes[1].set_xticks(x, channels)
    axes[1].set_ylabel(r"missing / added $b$")
    axes[1].set_title("Does SOX fill the missing O(V^2) response?")
    axes[1].axhline(0, linewidth=.8, color="0.4")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=int(dpi)); plt.close(fig)


def main():
    args = _args()
    if any(ch not in CHANNELS for ch in args.channels):
        raise ValueError("this diagnostic is restricted to z_same/z_opposite")
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0)
    exact = ExactSmallRubyThermal(args.L1, args.L2, params)
    h0 = np.asarray(exact.h0, complex)
    v_unit = np.asarray(exact.Vunit, complex)
    target = float(args.filling) * int(args.L1) * int(args.L2)
    mu0 = solve_free_mu(h0, target, args.T)
    grid = MatsubaraGrid(
        nk1=1, nk2=1, nw=int(args.nw), nOmega=int(args.nomega), T=float(args.T)
    )
    ledger = _load_ledger(args.ledger, args.channels, args.fit_windows)

    print("=== strict free-background O(V^2) exchange diagnostic ===")
    print(f"cluster={args.L1}x{args.L2}, T={args.T:g}, nw={args.nw}, mu0={mu0:.12f}")
    result = {}
    for ch in args.channels:
        K = exact.pseudospin_operator(ch, (0.0, 0.0))
        sox, gamma = sox_response_coefficient_free(
            h0, mu0, v_unit, K, grid, n_quad=int(args.nquad)
        )
        direct = gw_direct_order2_response_coefficient_free(h0, mu0, v_unit, K, grid)
        imag = max(abs(sox.imag), *(abs(x.imag) for x in direct.values()))
        if imag > 1e-8:
            raise RuntimeError(f"unexpected imaginary O(V^2) response for {ch}: {imag:.3e}")
        result[ch] = dict(
            b_sox=float(sox.real),
            b_gw_direct=float(direct["total"].real),
            b_mt_strict=float(direct["mt"].real),
            b_al1_strict=float(direct["al1"].real),
            b_al2_strict=float(direct["al2"].real),
            b_al_strict=float(direct["al"].real),
            gamma_hermiticity=float(np.max(
                np.abs(gamma[:,0,0] - np.swapaxes(gamma[::-1,0,0].conj(), -1, -2))
            )),
        )
        print(f"\n[{ch}]")
        print(f"  bare SOX b                  {sox.real:+.9e}")
        print(f"  strict GW direct MT b       {direct['mt'].real:+.9e}")
        print(f"  strict GW direct AL1 b      {direct['al1'].real:+.9e}")
        print(f"  strict GW direct AL2 b      {direct['al2'].real:+.9e}")
        print(f"  strict GW direct MT+AL b    {direct['total'].real:+.9e}")

        saved = ledger.get("saved")
        if saved and ch in saved:
            f = saved[ch]
            be = float(f["exact_vertex"]["b"])
            bc = float(f["full_vertex"]["b"])
            miss = float(f.get("b_missing_exact_minus_cgw", be-bc))
            pred = bc + sox.real
            resid = be - pred
            fit_direct = float(f["MT1"]["b"] + f["AL1"]["b"])
            frac = sox.real / miss if abs(miss) > 1e-14 else np.nan
            result[ch].update(
                ledger_b_exact=be, ledger_b_cgw=bc, ledger_b_missing=miss,
                ledger_b_cgw_plus_sox=float(pred),
                ledger_b_exact_minus_cgw_sox=float(resid),
                missing_fraction_explained=float(frac),
                direct_fit_minus_strict=float(fit_direct-direct["total"].real),
            )
            print("  -- comparison to saved cGW/ED ledger --")
            print(f"  exact - cGW missing b        {miss:+.9e}")
            print(f"  SOX / missing                {frac:+.6f}")
            print(f"  predicted (cGW+SOX) b        {pred:+.9e}")
            print(f"  exact-(cGW+SOX) residual     {resid:+.9e}")
            print(f"  fitted(MT+AL)-strict direct  {fit_direct-direct['total'].real:+.3e}")

    if ledger.get("refits"):
        print("\n=== no-rerun fit-window check from ledger_scan.npz ===")
        for vmax, wf in ledger["refits"].items():
            print(f"\nVmax={vmax}")
            for ch in args.channels:
                if ch not in wf:
                    continue
                miss = wf[ch]["missing"]
                sox = result[ch]["b_sox"]
                print(
                    f"  {ch:12s} missing={miss:+.9e}  SOX={sox:+.9e}  "
                    f"residual={miss-sox:+.3e}"
                )

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        parameters=dict(
            L1=args.L1, L2=args.L2, filling=args.filling, T=args.T,
            ti=args.ti, t1=args.t1, t2=args.t2, nw=args.nw,
            nomega=args.nomega, nquad=args.nquad, mu0=mu0,
        ),
        result=result,
        ledger_refits=ledger.get("refits", {}),
    )
    with (outdir/"sox_order2.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _plot(outdir/"sox_order2.png", args.channels, result, ledger, args.dpi)
    print(f"\nwrote {outdir/'sox_order2.json'}")
    if ledger.get("saved"):
        print(f"wrote {outdir/'sox_order2.png'}")


if __name__ == "__main__":
    main()

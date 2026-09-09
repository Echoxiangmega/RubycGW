#!/usr/bin/env python3
"""Inspect +/- twist-pair convergence in TABC GW output.

Reads results/twist_averaged_finite_source/twist_averaged_finite_source.npz
(or a user-supplied path) and reports whether GW convergence failures occur
systematically in time-reversal-related twist pairs theta <-> -theta.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "npz",
        nargs="?",
        type=Path,
        default=Path("results/twist_averaged_finite_source/twist_averaged_finite_source.npz"),
    )
    p.add_argument("--show-pairs", action="store_true")
    return p.parse_args()


def _wrap_phi(x):
    return (np.asarray(x, dtype=float) + 0.5) % 1.0 - 0.5


def _key(phi, ndigits=12):
    p = _wrap_phi(phi)
    return tuple(np.round(p, ndigits).tolist())


def main():
    args = _args()
    d = np.load(args.npz, allow_pickle=True)
    twists = np.asarray(d["twists"], dtype=float)
    phi = _wrap_phi(twists / (2.0 * np.pi))
    conv = np.asarray(d["gw_converged"], dtype=bool)
    residual = np.asarray(d["gw_residual"], dtype=float)
    Jgw = np.asarray(d["J_gw_twist"], dtype=float)
    Jed = np.asarray(d["J_ed_twist"], dtype=float)
    Gerr = np.asarray(d["Gerr_twist"], dtype=float)
    sources = [str(x) for x in np.asarray(d["sources"]).tolist()]
    Vvals = np.asarray(d["V"], dtype=float)
    hvals = np.asarray(d["h"], dtype=float)

    lookup = {_key(p): i for i, p in enumerate(phi)}
    partner = np.empty(len(phi), dtype=int)
    for i, p in enumerate(phi):
        k = _key(-p)
        if k not in lookup:
            raise RuntimeError(f"missing -theta partner for twist {i}: phi={p}")
        partner[i] = lookup[k]

    # Unique unordered pairs.
    pairs = []
    seen = set()
    for i, j in enumerate(partner):
        pair = tuple(sorted((int(i), int(j))))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    print(f"file={args.npz}")
    print(f"twists={len(phi)}, +/- pairs={len(pairs)}")

    for isrc, src in enumerate(sources):
        for iv, V in enumerate(Vvals):
            print(f"\n### source={src}, V={V:g} ###")
            for ih, h in enumerate(hvals):
                c = conv[isrc, iv, ih]
                r = residual[isrc, iv, ih]
                j = Jgw[isrc, iv, ih]
                je = Jed[isrc, iv, ih]
                ge = Gerr[isrc, iv, ih]

                both = one = none = 0
                same_side = opposite_side = 0
                one_pairs = []
                for a, b in pairs:
                    ca, cb = bool(c[a]), bool(c[b])
                    if ca and cb:
                        both += 1
                    elif ca or cb:
                        one += 1
                        one_pairs.append((a, b))
                    else:
                        none += 1

                    # Diagnostic only: for a one-converged pair, record whether
                    # the converged member has positive/negative theta1 relative
                    # to its partner.  This can reveal a systematic half-grid bias.
                    if ca != cb:
                        good = a if ca else b
                        bad = b if ca else a
                        if phi[good, 0] * phi[bad, 0] < 0:
                            opposite_side += 1
                        else:
                            same_side += 1

                failed = ~c
                rc = r[c]
                rf = r[failed]
                jc = j[c]
                jf = j[failed]
                gec = ge[c]
                gef = ge[failed]

                def stats(x):
                    x = np.asarray(x, dtype=float)
                    x = x[np.isfinite(x)]
                    if x.size == 0:
                        return "n/a"
                    return f"mean={np.mean(x):+.6g}, std={np.std(x):.3g}, min={np.min(x):+.6g}, max={np.max(x):+.6g}"

                print(
                    f"h={h:g}: conv={int(np.sum(c))}/{len(c)}; "
                    f"pairs both/one/none={both}/{one}/{none}"
                )
                print(f"  converged Jgw: {stats(jc)}")
                print(f"  failed    Jgw: {stats(jf)}")
                print(f"  converged residual: {stats(rc)}")
                print(f"  failed    residual: {stats(rf)}")
                print(f"  converged Gerr: {stats(gec)}")
                print(f"  failed    Gerr: {stats(gef)}")
                print(f"  ED all twists: {stats(je)}")

                if args.show_pairs and one_pairs:
                    print("  one-converged +/-theta pairs:")
                    for a, b in one_pairs:
                        print(
                            f"    ({a:2d},{b:2d}) "
                            f"phi_a=({phi[a,0]:+.5f},{phi[a,1]:+.5f}) "
                            f"phi_b=({phi[b,0]:+.5f},{phi[b,1]:+.5f}) | "
                            f"conv=({int(c[a])},{int(c[b])}) "
                            f"J=({j[a]:+.6f},{j[b]:+.6f}) "
                            f"r=({r[a]:.3e},{r[b]:.3e})"
                        )


if __name__ == "__main__":
    main()

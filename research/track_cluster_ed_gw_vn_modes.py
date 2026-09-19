#!/usr/bin/env python3
"""Track retained JF eigenmodes across V and extract lambda=1 crossings.

The input is vn_instability_summary.npz from scan_cluster_ed_gw_vn_instability.py.
For every filling, q point, and response sector, retained eigenmodes are matched
between neighbouring V values by maximum absolute eigenvector overlap.  This
avoids the false discontinuities produced by independently selecting

    argmin |1-lambda|

at every parameter point.

Outputs
-------
vn_tracked_modes.npz
    One row per tracked retained mode and parameter point, including branch id,
    lambda, signed mass 1-lambda, channel weights, and overlap to the preceding
    point on the same branch.

vn_mode_crossings.csv
    Every reliable real-axis crossing of lambda=1 found on a tracked branch.

vn_instability_crossings.png
    A V-filling scatter plot of the crossing candidates.  These are reference-
    branch instabilities, not automatically thermodynamic first-order boundaries.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


WEIGHT_KEYS = (
    "co_even", "co_odd", "lc_same", "lc_opposite", "uniform",
)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("--jf-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--min-overlap", type=float, default=0.35,
        help="minimum |<v_i|v_j>| required to continue an existing branch",
    )
    p.add_argument(
        "--imag-tol", type=float, default=5e-3,
        help="maximum |Im lambda| at both endpoints for a static real-axis crossing",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="skip the crossing-candidate phase-diagram plot",
    )
    return p.parse_args()


def _tag(x: float) -> str:
    return f"{float(x):g}"


def _jf_path(jfdir, meta, V, filling):
    return jfdir / (
        f"lambda_allq_ori{meta['orientation']}_L{meta['Lx']}x{meta['Ly']}_"
        f"V{_tag(V)}_Vp{_tag(meta['Vprime'])}_Vx{_tag(meta['Vcross'])}_"
        f"fill{_tag(filling)}.npz"
    )


def _scalar(d, key, cast=float):
    return cast(np.asarray(d[key]).reshape(()))


def _load_summary(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: np.asarray(z[k]) for k in z.files}
    meta = dict(
        orientation=_scalar(d, "orientation", int),
        Lx=_scalar(d, "Lx", int),
        Ly=_scalar(d, "Ly", int),
        Vprime=_scalar(d, "Vprime", float),
        Vcross=_scalar(d, "Vcross", float),
    )
    return d, meta


def _load_modes(path):
    with np.load(path, allow_pickle=False) as z:
        required = (
            "full_spectrum_schema", "mode_q_index", "mode_sector", "mode_lambda",
            "mode_vectors", "mode_vectors_saved",
        )
        missing = [k for k in required if k not in z]
        if missing:
            raise RuntimeError(
                f"{path} is an old JF output without full-spectrum tracking data: "
                + ", ".join(missing)
            )
        if not bool(np.asarray(z["mode_vectors_saved"]).reshape(())):
            raise RuntimeError(f"{path} has no saved mode vectors")
        return {
            "q": np.asarray(z["mode_q_index"], dtype=int),
            "sector": np.asarray(z["mode_sector"]).astype(str),
            "lambda": np.asarray(z["mode_lambda"], dtype=complex),
            "vector": np.asarray(z["mode_vectors"], dtype=np.complex64),
            "co_even": np.asarray(z["mode_co_even_weight"], dtype=float),
            "co_odd": np.asarray(z["mode_co_odd_weight"], dtype=float),
            "lc_same": np.asarray(z["mode_lc_same_weight"], dtype=float),
            "lc_opposite": np.asarray(z["mode_lc_opposite_weight"], dtype=float),
            "uniform": np.asarray(z["mode_uniform_weight"], dtype=float),
        }


def _groups(m):
    out = {}
    for q in sorted(set(map(tuple, np.asarray(m["q"], dtype=int)))):
        qmask = np.all(m["q"] == np.asarray(q, dtype=int), axis=1)
        sectors = sorted(set(m["sector"][qmask].tolist()))
        for sector in sectors:
            idx = np.where(qmask & (m["sector"] == sector))[0]
            out[(tuple(q), str(sector))] = idx
    return out


def _normalize_rows(v):
    v = np.asarray(v, dtype=complex)
    n = np.linalg.norm(v, axis=1)
    n = np.maximum(n, 1e-300)
    return v / n[:, None]


def _overlap_matrix(a, b):
    a = _normalize_rows(a)
    b = _normalize_rows(b)
    return np.abs(a.conj() @ b.T)


def _channel(weights):
    vals = {k: float(weights[k]) for k in WEIGHT_KEYS}
    ch = max(vals, key=vals.get)
    co = vals["co_even"] + vals["co_odd"]
    lc = vals["lc_same"] + vals["lc_opposite"]
    if co >= max(lc, vals["uniform"]):
        group = "CO"
    elif lc >= max(co, vals["uniform"]):
        group = "LC"
    else:
        group = "uniform/mixed"
    return ch, group


def _append_record(records, *, filling, V, q, sector, branch_id, idx, modes, overlap):
    weights = {k: float(modes[k][idx]) for k in WEIGHT_KEYS}
    ch, group = _channel(weights)
    lam = complex(modes["lambda"][idx])
    records.append(
        dict(
            filling=float(filling),
            V=float(V),
            q1=int(q[0]),
            q2=int(q[1]),
            sector=str(sector),
            branch_id=int(branch_id),
            lambda_value=lam,
            mass=1.0 - lam,
            overlap_previous=float(overlap),
            dominant_channel=ch,
            channel_group=group,
            **weights,
        )
    )


def _track_group(points, *, filling, q, sector, min_overlap, records, next_branch):
    active = {}
    first = True
    for V, modes, idx in points:
        cur_vectors = np.asarray(modes["vector"][idx], dtype=complex)
        if first:
            for local, global_idx in enumerate(idx):
                bid = next_branch
                next_branch += 1
                active[bid] = cur_vectors[local]
                _append_record(
                    records, filling=filling, V=V, q=q, sector=sector,
                    branch_id=bid, idx=global_idx, modes=modes, overlap=np.nan,
                )
            first = False
            continue

        prev_ids = list(active)
        prev_vectors = np.stack([active[bid] for bid in prev_ids], axis=0)
        O = _overlap_matrix(prev_vectors, cur_vectors)
        rows, cols = linear_sum_assignment(-O)

        assigned_cur = set()
        new_active = {}
        for rr, cc in zip(rows, cols):
            ov = float(O[rr, cc])
            if ov < float(min_overlap):
                continue
            bid = prev_ids[int(rr)]
            global_idx = int(idx[int(cc)])
            assigned_cur.add(int(cc))
            new_active[bid] = cur_vectors[int(cc)]
            _append_record(
                records, filling=filling, V=V, q=q, sector=sector,
                branch_id=bid, idx=global_idx, modes=modes, overlap=ov,
            )

        for cc, global_idx in enumerate(idx):
            if cc in assigned_cur:
                continue
            bid = next_branch
            next_branch += 1
            new_active[bid] = cur_vectors[cc]
            _append_record(
                records, filling=filling, V=V, q=q, sector=sector,
                branch_id=bid, idx=int(global_idx), modes=modes, overlap=np.nan,
            )
        active = new_active

    return next_branch


def _crossings(records, imag_tol):
    by_branch = {}
    for r in records:
        by_branch.setdefault(int(r["branch_id"]), []).append(r)

    out = []
    for bid, rr in by_branch.items():
        rr = sorted(rr, key=lambda x: x["V"])
        for a, b in zip(rr[:-1], rr[1:]):
            la = complex(a["lambda_value"])
            lb = complex(b["lambda_value"])
            if abs(la.imag) > imag_tol or abs(lb.imag) > imag_tol:
                continue
            ra = 1.0 - la.real
            rb = 1.0 - lb.real
            if ra == 0.0:
                t = 0.0
            elif rb == 0.0:
                t = 1.0
            elif ra * rb > 0.0:
                continue
            else:
                den = rb - ra
                if den == 0.0:
                    continue
                t = -ra / den
            if not (0.0 <= t <= 1.0):
                continue
            Vc = a["V"] + t * (b["V"] - a["V"])
            weights = {
                k: (1.0 - t) * float(a[k]) + t * float(b[k])
                for k in WEIGHT_KEYS
            }
            ch, group = _channel(weights)
            pair_overlap = float(b["overlap_previous"])
            out.append(
                dict(
                    filling=float(a["filling"]),
                    V_cross=float(Vc),
                    q1=int(a["q1"]),
                    q2=int(a["q2"]),
                    sector=str(a["sector"]),
                    branch_id=int(bid),
                    channel_group=group,
                    dominant_channel=ch,
                    V_left=float(a["V"]),
                    V_right=float(b["V"]),
                    lambda_left=float(la.real),
                    lambda_right=float(lb.real),
                    overlap=float(pair_overlap),
                    **weights,
                )
            )
    return out


def _save_records(path, records, crossings, meta, Vs, fillings):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        V_values=np.asarray(Vs, dtype=float),
        fillings=np.asarray(fillings, dtype=float),
        orientation=int(meta["orientation"]),
        Vprime=float(meta["Vprime"]),
        Vcross=float(meta["Vcross"]),
        Lx=int(meta["Lx"]),
        Ly=int(meta["Ly"]),
        filling=np.asarray([r["filling"] for r in records], dtype=float),
        V=np.asarray([r["V"] for r in records], dtype=float),
        q_index=np.asarray([[r["q1"], r["q2"]] for r in records], dtype=int),
        sector=np.asarray([r["sector"] for r in records]),
        branch_id=np.asarray([r["branch_id"] for r in records], dtype=int),
        lambda_value=np.asarray([r["lambda_value"] for r in records], dtype=complex),
        mass=np.asarray([r["mass"] for r in records], dtype=complex),
        overlap_previous=np.asarray([r["overlap_previous"] for r in records], dtype=float),
        dominant_channel=np.asarray([r["dominant_channel"] for r in records]),
        channel_group=np.asarray([r["channel_group"] for r in records]),
        co_even=np.asarray([r["co_even"] for r in records], dtype=float),
        co_odd=np.asarray([r["co_odd"] for r in records], dtype=float),
        lc_same=np.asarray([r["lc_same"] for r in records], dtype=float),
        lc_opposite=np.asarray([r["lc_opposite"] for r in records], dtype=float),
        uniform=np.asarray([r["uniform"] for r in records], dtype=float),
        crossing_filling=np.asarray([r["filling"] for r in crossings], dtype=float),
        crossing_V=np.asarray([r["V_cross"] for r in crossings], dtype=float),
        crossing_q_index=np.asarray(
            [[r["q1"], r["q2"]] for r in crossings], dtype=int
        ).reshape((-1, 2)),
        crossing_sector=np.asarray([r["sector"] for r in crossings]),
        crossing_branch_id=np.asarray([r["branch_id"] for r in crossings], dtype=int),
        crossing_channel_group=np.asarray([r["channel_group"] for r in crossings]),
        crossing_dominant_channel=np.asarray([r["dominant_channel"] for r in crossings]),
        crossing_overlap=np.asarray([r["overlap"] for r in crossings], dtype=float),
    )


def _save_crossing_csv(path, crossings):
    fields = [
        "filling", "V_cross", "channel_group", "dominant_channel",
        "q1", "q2", "sector", "branch_id", "overlap",
        "V_left", "V_right", "lambda_left", "lambda_right",
        "co_even", "co_odd", "lc_same", "lc_opposite", "uniform",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in crossings:
            w.writerow({k: r[k] for k in fields})


def _plot_crossings(path, crossings):
    fig, ax = plt.subplots()
    groups = ("CO", "LC", "uniform/mixed")
    markers = {"CO": "o", "LC": "s", "uniform/mixed": "x"}
    for group in groups:
        rr = [r for r in crossings if r["channel_group"] == group]
        if not rr:
            continue
        ax.scatter(
            [r["V_cross"] for r in rr],
            [r["filling"] for r in rr],
            marker=markers[group],
            label=group,
        )
    ax.set_xlabel("V")
    ax.set_ylabel("filling n")
    ax.set_title(r"Tracked reference-branch crossings: $\lambda_\nu(q)=1$")
    if crossings:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = _args()
    if not args.summary.exists():
        raise FileNotFoundError(args.summary)
    summary, meta = _load_summary(args.summary)
    Vs = np.asarray(summary["V_values"], dtype=float)
    fillings = np.asarray(summary["fillings"], dtype=float)
    jfdir = args.jf_dir or (args.summary.parent / "jf_allq")
    outdir = args.out or (args.summary.parent / "tracked_modes")
    outdir.mkdir(parents=True, exist_ok=True)

    records = []
    next_branch = 0
    missing = []

    for filling in fillings:
        loaded = []
        for V in np.sort(Vs):
            path = _jf_path(jfdir, meta, float(V), float(filling))
            if not path.exists():
                missing.append(str(path))
                continue
            modes = _load_modes(path)
            loaded.append((float(V), modes))

        group_keys = set()
        for _, modes in loaded:
            group_keys.update(_groups(modes).keys())

        for q, sector in sorted(group_keys):
            points = []
            for V, modes in loaded:
                idx = _groups(modes).get((q, sector))
                if idx is None or len(idx) == 0:
                    continue
                points.append((V, modes, idx))
            if not points:
                continue
            next_branch = _track_group(
                points,
                filling=float(filling),
                q=q,
                sector=sector,
                min_overlap=float(args.min_overlap),
                records=records,
                next_branch=next_branch,
            )

    crossings = _crossings(records, float(args.imag_tol))
    tracked_path = outdir / "vn_tracked_modes.npz"
    csv_path = outdir / "vn_mode_crossings.csv"
    _save_records(tracked_path, records, crossings, meta, Vs, fillings)
    _save_crossing_csv(csv_path, crossings)
    if not bool(args.no_plot):
        _plot_crossings(outdir / "vn_instability_crossings.png", crossings)

    print(f"tracked mode rows: {len(records)}")
    print(f"lambda=1 crossings: {len(crossings)}")
    print(f"saved: {tracked_path}")
    print(f"saved: {csv_path}")
    if missing:
        print(f"warning: {len(missing)} JF files were missing and skipped")


if __name__ == "__main__":
    main()

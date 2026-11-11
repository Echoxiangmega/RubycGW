"""Human-readable wide and ranked ED/GW/cGW/post-GW comparison files.

Keep the original long comparison.csv untouched as a machine-readable log.
Never produce a combined 'best method' score: susceptibility, G and energy
refer to different physics/levels of approximation.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

CHANNELS = ("x_even", "y_even", "z_same", "z_opposite")
PREFERRED_METHODS = (
    "ED", "GW[density]", "cGW[density]", "post-GW[density]",
    "GW[pseudospin]", "cGW[pseudospin]", "post-GW[pseudospin]",
    "GW[bj]", "cGW[bj]", "post-GW[bj]", "cluster-ED+GW",
)


def _num(value):
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _csv(path, records, columns):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in records:
            writer.writerow({
                k: "" if item.get(k) is None else item.get(k)
                for k in columns
            })


def make_comparison_scorecards(rows, output_dir):
    """Create one-row-per-method CSV and independently ranked G/chi CSVs.

    The four-channel chi mean is published only when all four physical
    channels converged; no absent or unconverged channel is treated as zero.
    The cGW one-particle G is identical to its GW background, but it is
    intentionally not counted as an independent Green-method ranking entry.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_method = {}
    for record in rows:
        by_method.setdefault(record["method"], {})[record["quantity"]] = record
    if not by_method:
        raise ValueError("comparison table has no methods")
    ed = by_method.get("ED", {})
    refs = {
        c: _num(ed.get(f"chi[{c}](Q)", {}).get("value"))
        for c in CHANNELS
    }
    ordered = [m for m in PREFERRED_METHODS if m in by_method]
    ordered.extend(m for m in by_method if m not in ordered)
    wide = []
    for method in ordered:
        quantities = by_method[method]
        status_row = quantities.get("method_status")
        status = status_row.get("status") if status_row else "ok"
        iw = _num(quantities.get("G_rel_Frobenius_K", {}).get("value"))
        tau = _num(quantities.get("G_tau_curve_relative_L2_error", {}).get("value"))
        g_shared = method.startswith("cGW[")
        if g_shared:
            parent = by_method.get("GW[" + method[4:], {})
            iw = _num(parent.get("G_rel_Frobenius_K", {}).get("value"))
            tau = _num(parent.get("G_tau_curve_relative_L2_error", {}).get("value"))
        errors = {}
        values = {}
        for ch in CHANNELS:
            key = f"chi_GG[{ch}](Q)" if method.startswith("GW[") else f"chi[{ch}](Q)"
            item = quantities.get(key, {})
            valid = item.get("status", "ok") not in (
                "failed", "unconverged", "skipped"
            )
            value = _num(item.get("value")) if valid else None
            error = _num(item.get("relative_error")) if valid else None
            if error is None and value is not None and refs[ch] not in (None, 0.0):
                error = abs(value - refs[ch]) / abs(refs[ch])
            values[ch] = value
            errors[ch] = None if error is None else error * 100
        errlist = [errors[ch] for ch in CHANNELS]
        mean = sum(errlist)/4 if all(e is not None for e in errlist) else None
        xy = sum(errlist[:2])/2 if all(e is not None for e in errlist[:2]) else None
        lc = sum(errlist[2:])/2 if all(e is not None for e in errlist[2:]) else None
        runtime = _num(quantities.get("total_method_runtime_seconds", {}).get("value"))
        if runtime is None:
            runtime = max((t for x in quantities.values()
                           if (t := _num(x.get("runtime_seconds"))) is not None),
                          default=None)
        if method == "ED":
            a = _num(quantities.get("ground_state_E0", {}).get("stage_runtime_seconds"))
            b = _num(quantities.get("G_reference_initialized", {}).get("stage_runtime_seconds"))
            if a is not None and b is not None:
                runtime = a+b
        row = {
            "method": method, "status": status,
            "G_iomega_error_pct": None if iw is None else iw*100,
            "G_tau_error_pct": None if tau is None else tau*100,
            "chi_mean_4_error_pct": mean,
            "chi_xy_mean_error_pct": xy,
            "chi_current_mean_error_pct": lc,
            "runtime_s": runtime,
            "runtime_min": None if runtime is None else runtime/60,
            "G_shared_with_GW": bool(g_shared),
            "notes": (status_row.get("notes", "") if status_row
                      else "cGW G is identical to underlying GW" if g_shared
                      else "post-GW chi was not computed" if method.startswith("post-GW")
                      else ""),
        }
        for ch in CHANNELS:
            row[f"chi_{ch}_error_pct"] = errors[ch]
            row[f"chi_{ch}_value"] = values[ch]
        row["chi_same_minus_opposite"] = (
            values["z_same"] - values["z_opposite"]
            if values["z_same"] is not None and values["z_opposite"] is not None
            else None
        )
        wide.append(row)
    columns = [
        "method", "status", "G_iomega_error_pct", "G_tau_error_pct",
        "chi_mean_4_error_pct", "chi_xy_mean_error_pct",
        "chi_current_mean_error_pct",
        *(f"chi_{c}_error_pct" for c in CHANNELS),
        *(f"chi_{c}_value" for c in CHANNELS),
        "chi_same_minus_opposite", "runtime_s", "runtime_min",
        "G_shared_with_GW", "notes",
    ]
    _csv(output/"comparison_wide.csv", wide, columns)
    green = sorted((
        r for r in wide if r["G_iomega_error_pct"] is not None
        and r["method"] != "ED" and not r["G_shared_with_GW"]
    ), key=lambda r: r["G_iomega_error_pct"])
    chi = sorted((
        r for r in wide if r["chi_mean_4_error_pct"] is not None
        and r["method"] != "ED"
    ), key=lambda r: r["chi_mean_4_error_pct"])
    _csv(
        output/"ranking_green.csv",
        [dict(rank=i, **r) for i,r in enumerate(green,1)],
        ["rank","method","status","G_iomega_error_pct",
         "G_tau_error_pct","runtime_min","notes"],
    )
    _csv(
        output/"ranking_susceptibility.csv",
        [dict(rank=i, **r) for i,r in enumerate(chi,1)],
        ["rank","method","status","chi_mean_4_error_pct",
         "chi_xy_mean_error_pct","chi_current_mean_error_pct",
         *(f"chi_{c}_error_pct" for c in CHANNELS),
         "runtime_min","notes"],
    )
    with (output/"comparison_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Ruby benchmark | method comparison\n\n")
        handle.write("Errors are percentages relative to ED. Smaller is better. "
                     "The lists rank different observables independently.\n\n")
        for heading, group, field in (
            ("Green function, Matsubara", green, "G_iomega_error_pct"),
            ("Static susceptibility, four-channel mean", chi, "chi_mean_4_error_pct"),
        ):
            handle.write(f"## {heading}\n\n")
            handle.write("| Rank | Method | Error (%) | G(tau) error (%) | Time (min) |\n")
            handle.write("| ---: | --- | ---: | ---: | ---: |\n")
            for pos,item in enumerate(group,1):
                def fmt(x):
                    return "—" if x is None else f"{x:.2f}"
                handle.write(
                    f"| {pos} | {item['method']} | {fmt(item[field])} | "
                    f"{fmt(item['G_tau_error_pct'])} | {fmt(item['runtime_min'])} |\n"
                )
            handle.write("\n")
        handle.write(
            "Caveats: cGW shares its GW one-particle Green function. "
            "post-GW and skipped cluster response have no chi values. "
            "ED E0 is zero-temperature, while GW energies are finite-temperature "
            "estimates; these energies must not be combined into a ground-state "
            "accuracy ranking. The ED thermal Green and susceptibility need "
            "their own convergence/error-bar checks.\n"
        )
        if refs["z_same"] is not None and refs["z_opposite"] is not None:
            split=refs["z_same"]-refs["z_opposite"]
            handle.write(
                f"ED chi(z_same)-chi(z_opposite)={split:.8g}; "
                "check any accidental degeneracy of these channels in GW/cGW.\n"
            )
    return wide

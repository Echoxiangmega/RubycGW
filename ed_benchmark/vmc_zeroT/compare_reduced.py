#!/usr/bin/env python3
"""Compare ED, full-orbital VAP, and three symmetry-tied parameterized families."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


METHOD_ORDER = [
    "ED",
    "VAP-Slater-Jastrow",
    "Full-VAP-Slater-Jastrow",
    "Param-Slater",
    "Param-Slater-Jastrow",
    "Param-VAP-Slater-Jastrow",
]


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fval(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-vap", required=True, help="Directory containing vap_best.csv")
    ap.add_argument("--parametric", required=True, help="Directory containing parametric_best.csv")
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    full = Path(a.full_vap).resolve()
    red = Path(a.parametric).resolve()
    full_rows = read_csv(full / "vap_best.csv") if (full / "vap_best.csv").exists() else []
    red_rows = read_csv(red / "parametric_best.csv")

    by = {}
    for r in full_rows + red_rows:
        m = r.get("method")
        if m in METHOD_ORDER:
            if m not in by or fval(r.get("E"), 1e300) < fval(by[m].get("E"), 1e300):
                by[m] = r

    required = ["ED", "Param-Slater", "Param-Slater-Jastrow", "Param-VAP-Slater-Jastrow"]
    missing = [m for m in required if m not in by]
    if missing:
        raise RuntimeError(f"Missing required methods in supplied results: {missing}")
    order = [m for m in METHOD_ORDER if m in by]

    edE = fval(by["ED"]["E"])
    outrows = []
    for m in order:
        r = by[m]
        E = fval(r.get("E"))
        row = {
            "method": m,
            "seed": r.get("seed", ""),
            "E": E,
            "E_per_site": fval(r.get("E_per_site")),
            "delta_E_vs_ED": E-edE,
            "relative_E_error": abs((E-edE)/edE),
            "fidelity_ground_subspace": fval(r.get("fidelity_ground_subspace"), 1.0 if m=="ED" else None),
            "CO_qmax": fval(r.get("CO_qmax")),
            "S_BOW_star": fval(r.get("S_BOW_star")),
            "S_LC_same": fval(r.get("S_LC_same")),
            "S_LC_opposite": fval(r.get("S_LC_opposite")),
            "LC_ratio_opposite_over_same": (
                fval(r.get("S_LC_opposite")) / max(fval(r.get("S_LC_same")), 1e-15)
            ),
            "target_sector_weight": fval(r.get("target_sector_weight")),
            "projected_sector_residual": fval(r.get("projected_sector_residual")),
            "n_parameters": fval(r.get("n_parameters")),
            "n_one_body_parameters": fval(r.get("n_one_body_parameters")),
            "n_jastrow_parameters": fval(r.get("n_jastrow_parameters")),
            "time_total_s": fval(r.get("time_total_s")),
            "source": str(full if m=="VAP-Slater-Jastrow" else red),
        }
        outrows.append(row)

    out = Path(a.output).resolve() if a.output else red / "comparison_ed_fullvap_parametric.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(outrows[0])
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(outrows)
    out.with_suffix(".json").write_text(json.dumps(outrows, indent=2), encoding="utf-8")

    print("\n=== ED / full-orbital VAP / symmetry-tied comparison ===")
    for r in outrows:
        print(
            f"{r['method']:30s} seed={r['seed']:11s} "
            f"E={r['E']:.12g} dE={r['delta_E_vs_ED']:.3e} "
            f"F={r['fidelity_ground_subspace'] if r['fidelity_ground_subspace'] is not None else float('nan'):.6f} "
            f"SLC+={r['S_LC_same']:.5g} SLC-={r['S_LC_opposite']:.5g} "
            f"npar={r['n_parameters']} time={r['time_total_s']}"
        )
    print("Saved", out)


if __name__ == "__main__":
    main()

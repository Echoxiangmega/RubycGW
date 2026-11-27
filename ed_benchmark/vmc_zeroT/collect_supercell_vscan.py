#!/usr/bin/env python3
"""Collect branch_scan.csv files from Raw-only V scans into one master table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--output",default=None)
    a=p.parse_args()

    root=Path(a.root).expanduser().resolve()
    files=sorted(root.glob("L*x*/seed_*/**/branch_scan.csv"))
    if not files:
        raise FileNotFoundError(f"No branch_scan.csv files below {root}")

    rows=[]
    physical_settings={}
    for path in files:
        settings_path=path.parent/"scan_settings.json"
        if settings_path.exists():
            settings=json.loads(settings_path.read_text(encoding="utf-8"))
            size=(int(settings["L1"]),int(settings["L2"]))
            physics=(int(settings["Nf"]),float(settings["t"]),
                     float(settings["t1"]),float(settings["t2"]))
            old=physical_settings.setdefault(size,(physics,settings_path))
            if old[0]!=physics:
                raise ValueError(
                    f"Cannot compare different Nf/t/t1/t2 at L{size[0]}x{size[1]}: "
                    f"{old[1]} has {old[0]}, {settings_path} has {physics}. "
                    "Use separate output roots for distinct Hamiltonians."
                )
        with path.open(newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row=dict(row)
                row["branch_scan_file"]=str(path)
                rows.append(row)

    def fnum(r,key,default=0.0):
        try:
            return float(r.get(key,default))
        except Exception:
            return default

    rows.sort(key=lambda r:(
        int(float(r.get("L1",0))),
        int(float(r.get("L2",0))),
        str(r.get("branch_seed","")),
        str(r.get("scan_mode","")),
        str(r.get("scan_direction","")),
        fnum(r,"scan_V"),
    ))

    keys=list(dict.fromkeys(k for r in rows for k in r))
    out=(
        Path(a.output).expanduser().resolve()
        if a.output is not None
        else root/"vscan_master.csv"
    )
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out}")


if __name__=="__main__":
    main()

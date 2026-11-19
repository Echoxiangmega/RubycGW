#!/usr/bin/env python3
"""Check the larger Ruby finite tori used for VMC size scaling."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

from geometry import RubyPBC

EXPECTED = {
    "ring3": 18,
    "2x2": 24,
    "ring7": 42,
    "3x3": 54,
    "ring12": 72,
    "ring13": 78,
    "4x4": 96,
    "ring19": 114,
    "ring21": 126,
    "5x5": 150,
    "ring28": 168,
    "ring31": 186,
    "6x6": 216,
}

MAIN = {"2x2","3x3","4x4","5x5","6x6"}


def main():
    print("name       Ns  Ncell  det(M)  C3  mirrors  C6@t1=t2")
    print("-"*62)
    for name, ns in EXPECTED.items():
        g=RubyPBC(name,t=1.0,t1=0.5,t2=-0.5)
        assert g.Ns==ns,(name,g.Ns,ns)
        assert g.num_cells==ns//6
        assert abs(round(np.linalg.det(g.M)))==g.num_cells
        assert len(g.permitted_momenta())==g.num_cells

        pm=g.point_maps()
        c3=pm["C3"] is not None
        nmir=len(pm["mirrors"])
        assert c3,f"{name} lost C3"
        if name in MAIN:
            assert nmir>=3,f"{name} expected >=3 mirrors, got {nmir}"

        geq=RubyPBC(name,t=1.0,t1=0.5,t2=0.5)
        c6=geq.point_maps()["C6"] is not None
        if name in MAIN:
            assert c6,f"{name} should retain ordinary C6 when t1=t2"

        print(
            f"{name:9s} {g.Ns:4d} {g.num_cells:6d} "
            f"{abs(round(np.linalg.det(g.M))):7d} "
            f"{str(c3):>4s} {nmir:8d} {str(c6):>9s}"
        )

    print("\nlarge-supercell geometry test: PASS")


if __name__=="__main__":
    main()

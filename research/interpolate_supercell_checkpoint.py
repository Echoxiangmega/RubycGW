#!/usr/bin/env python3
"""Create an 18-site GW warm-start checkpoint on a different k mesh.

Example
-------
python interpolate_supercell_checkpoint.py ^
  --checkpoint results\supercell18\checkpoints\V1.730000_n3.000000_nk3x3_nw47_no10_T0.05.npz ^
  --nk1 4 --nk2 4

The generated checkpoint is marked non-converged.  Use it explicitly with
``run_supercell_gw.py --restart-from ...`` and self-consistently converge the
new k mesh before using the result physically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rubycgw.checkpoint_interpolate import write_interpolated_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Source 18-site supercell GW checkpoint (.npz).",
    )
    parser.add_argument("--nk1", required=True, type=int, help="Target k mesh along k1.")
    parser.add_argument("--nk2", required=True, type=int, help="Target k mesh along k2.")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output checkpoint path. Default: ordinary checkpoint filename for "
            "the target mesh next to the source checkpoint."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing target warm-start checkpoint.",
    )
    args = parser.parse_args()

    source = Path(args.checkpoint)
    output, meta = write_interpolated_checkpoint(
        source,
        args.nk1,
        args.nk2,
        output_path=args.output,
        overwrite=args.force,
    )

    print("=" * 80)
    print("18-site Ruby GW cross-kmesh warm start")
    print(f"source      : {source}")
    print(
        f"k mesh      : {int(meta['interpolated_from_nk1'])}x"
        f"{int(meta['interpolated_from_nk2'])} -> "
        f"{int(meta['nk1'])}x{int(meta['nk2'])}"
    )
    print(f"V           : {float(meta['V']):.8g}")
    print(f"mu copied   : {float(meta['mu']):.10f}")
    print(f"output      : {output}")
    print("state       : WARM START ONLY; converged=False")
    print("next step   : run the target k mesh self-consistently with")
    print(f"  --restart-from {output}")
    print("NOTE: Sigma_H, density and mu are copied; full Sigma_GW(iw,k) is")
    print("      periodically Fourier-interpolated. The target-mesh GW equations")
    print("      themselves are unchanged and must be reconverged.")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create an 18-site GW warm-start checkpoint at a different temperature.

Example
-------
python interpolate_supercell_checkpoint_T.py ^
  --checkpoint results\supercell18\checkpoints\V1.730000_n3.000000_nk4x4_nw47_no10_T0.05.npz ^
  --T 0.06

The generated checkpoint is marked non-converged.  Use it explicitly with
``run_supercell_gw.py --restart-from ...`` and self-consistently converge the
new temperature before using the result physically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rubycgw.checkpoint_temperature import write_temperature_interpolated_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Source 18-site supercell GW checkpoint (.npz).",
    )
    parser.add_argument("--T", required=True, type=float, help="Target temperature.")
    parser.add_argument(
        "--nw",
        type=int,
        default=None,
        help="Optional target fermionic half-window. Default: keep source nw.",
    )
    parser.add_argument(
        "--tail-pairs",
        type=int,
        default=4,
        help="Number of largest-|omega| +/- pairs used to estimate Sigma_infinity.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output checkpoint path. Default: ordinary checkpoint filename for "
            "the target temperature next to the source checkpoint."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing target warm-start checkpoint.",
    )
    args = parser.parse_args()

    source = Path(args.checkpoint)
    output, meta = write_temperature_interpolated_checkpoint(
        source,
        target_T=args.T,
        target_nw=args.nw,
        output_path=args.output,
        overwrite=args.force,
        tail_pairs=args.tail_pairs,
    )

    print("=" * 80)
    print("18-site Ruby GW temperature warm start")
    print(f"source      : {source}")
    print(
        f"temperature : {float(meta['temperature_interpolated_from_T']):.8g} -> "
        f"{float(meta['T']):.8g}"
    )
    print(
        f"nw          : {int(meta['temperature_interpolated_from_nw'])} -> "
        f"{int(meta['nw'])}"
    )
    print(f"k mesh      : {int(meta['nk1'])}x{int(meta['nk2'])}")
    print(f"V           : {float(meta['V']):.8g}")
    print(f"mu copied   : {float(meta['mu']):.10f}")
    print(f"tail pairs  : {int(meta['temperature_interpolation_tail_pairs'])}")
    print(f"output      : {output}")
    print("state       : WARM START ONLY; converged=False")
    print("next step   : run the target temperature self-consistently with")
    print(f"  --restart-from {output}")
    print("NOTE: Sigma_H, density and mu are copied. Sigma_GW is interpolated")
    print("      in physical Matsubara frequency after separating an estimated")
    print("      static high-frequency limit; the dynamic tail is continued as 1/omega.")
    print("=" * 80)


if __name__ == "__main__":
    main()

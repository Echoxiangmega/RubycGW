# 18-site GW checkpoint / restart

`run_supercell_gw.py` now saves every converged **zero-source** 18-site GW state
as a compressed NumPy checkpoint.  This avoids repeating the full continuation
from `V=0.70` every time a larger interaction is requested.

The default persistent checkpoint directory is

```text
results/supercell18/checkpoints
```

A checkpoint stores the restart state

- `Sigma_H`
- `Sigma_GW`
- `mu`
- site density

plus metadata (`V`, filling, `T`, hopping parameters and all momentum/frequency
grid sizes).  `V` is deliberately allowed to change on restart; the other model
and numerical settings must match.

## First broken-symmetry run

Run the ordinary continuation once.  For example,

```bash
python run_supercell_gw.py --V 1.0 --primitive-filling 3 --nk1 3 --nk2 3 --nw 55 --nomega 12 --gw-max-iter 1000
```

At the source-onset point the driver performs the finite source sequence and
removes the source.  Every subsequently converged `h=0` point is checkpointed.
Finite-source states are intentionally not put in the persistent branch
database.

## Later calculations

Use the closest compatible zero-source checkpoint automatically:

```bash
python run_supercell_gw.py --V 1.35 --primitive-filling 3 --nk1 3 --nk2 3 --nw 55 --nomega 12 --restart-from auto
```

If a `V=1.25` checkpoint exists, the new run starts from that state and only
continues from `1.25` to the requested target.  It does not repeat the ramp from
`0.70`.

A specific checkpoint can also be selected:

```bash
python run_supercell_gw.py --V 1.35 --primitive-filling 3 --nk1 3 --nk2 3 --nw 55 --nomega 12 --restart-from results/supercell18/checkpoints/V1.250000_n3.000000_nk3x3_nw55_no12_T0.05.npz
```

## Compatibility

Direct restart requires identical

- `ti`, `t1`, `t2`
- primitive-cell filling
- temperature
- `nk1`, `nk2`
- `nw`, `nOmega`

because the self-energy arrays otherwise describe a different numerical
problem.  A different `V` is expected and is the purpose of continuation.

If `--restart-from auto` cannot find a compatible checkpoint, the driver falls
back to the ordinary V ramp.

# Periodic 18-site Ruby hexagon: relation to the existing hex18 benchmark

**Important correction.** The existing `rubycgw.supercell` / `research/scan_three_tori_v.py --geometry hex18` already describes the **same physical index-three hexagonal torus** as the six-triangle ring drawing. It is **not** a different torus. A comparison that treats two sets of primitive coordinates as if they were the same basis incorrectly suggests otherwise.

The native code uses the primitive-coordinate period matrix (columns are supercell translations)

```text
M_native = [[ 1, 1],
            [-1, 2]]
```

The ring drawing and `scan_true_hex18_v.py --torus true_hex` use

```text
M_drawing = [[-1, 2],
             [-1,-1]]
```

The exact change of primitive basis is

```text
U = [[ 0, 1],
     [-1, 0]]
M_drawing = U @ M_native
```

Additionally, triangle-B orbitals 3,4,5 are assigned the drawing-origin shift `(-1,-1)` while triangle-A orbitals 0,1,2 are unshifted. With these transforms, every primitive hopping bond maps exactly, and a site permutation `P` gives

```text
H_drawing(k_supercell=0) = P @ H_native(k_supercell=0) @ P.T
```

All 18 intratriangle density interactions also match. Hence their exact fixed-N ED spectra are identical for identical `ti,t1,t2,V`; there is no new independent hex18 finite-size geometry.

## Files

- `research/scan_true_hex18_v.py`: standalone fixed-particle-number ED V scan, with a drawing of the six-triangle periodic ring. Does not run GW/cGW/post-GW and should not be used to claim independent finite-size confirmation of the native hex18.
- `research/verify_hex18_equivalence.py`: checks primitive bond conversion, 18-site hopping matrices, and the interaction bond mapping.

## Run

From the repository root:

```bash
python research/verify_hex18_equivalence.py
python research/scan_true_hex18_v.py --self-test
python research/scan_true_hex18_v.py --torus true_hex --Nf 6 --t 1 --t1 0.5 --t2 0.5 --Vmin 0 --Vmax 2 --Vstep 0.25 --output results/true_hex18_ed
```

For the **original ED, GW, cGW, and post-GW benchmark**, continue to use:

```bash
python research/scan_three_tori_v.py --geometry hex18 --filling 2 --V-values 0.5 1 1.5 2 --K 0.333333333333 0.333333333333 --Q 0.333333333333 0.333333333333 --T 0.125 --methods ed gw cgw --out results/native_hex18_scan
```

The momenta differ between coordinate systems: the native `(1/3,1/3)` corresponds to one of the drawing `(1/3,2/3)` / `(2/3,1/3)` momenta after the basis transformation.

**Do not compare** `--torus skew18` from the standalone drawing-coordinate script with `--geometry hex18` from the original repository as if both names meant the same primitive-coordinate period matrix. They do not.

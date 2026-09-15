# 18-site period-three supercell GW

This workflow follows the finite-momentum density instability that appears in the primitive six-site normal-state GW calculation near

\[
Q=(1/3,1/3).
\]

It is deliberately separate from `filling_scan.py`: first obtain a stable broken-symmetry GW background, then formulate current response on top of that background.

## 1. Why 18 sites are sufficient

Choose supercell translations

\[
T_1=a_1-a_2,\qquad T_2=a_1+2a_2.
\]

Their area is three primitive cells because

\[
\det\begin{pmatrix}1&1\\-1&2\end{pmatrix}=3.
\]

Moreover

\[
Q\cdot T_1=0,\qquad Q\cdot T_2=1,
\]

so the primitive `Q=(1/3,1/3)` modulation is periodic in this enlarged cell. The basis is therefore

```text
3 primitive-cell sectors x 6 Ruby sublattices = 18 sites.
```

Bloch momentum is retained, but it now lives in the reduced supercell Brillouin zone. The original finite-Q order is folded to supercell `q=0` and appears as inequivalent densities among the three internal primitive cells.

The basis order is

```text
I = 6*sector + sublattice
sector = 0,1,2
sublattice = 0,...,5
```

with representatives `R_s=(s,0)`.

## 2. Exact noninteracting folding check

`rubycgw.supercell.build_supercell_h0` constructs the 18x18 Bloch Hamiltonian directly from the original Ruby bond list. Unit tests verify that its 18 eigenvalues at every supercell momentum equal the union of the six primitive bands at

\[
k_p,\qquad k_p+Q,\qquad k_p+2Q,
\]

where

\[
k_p=A^{-T}k_{sc},\qquad
A=\begin{pmatrix}1&1\\-1&2\end{pmatrix}.
\]

The supercell therefore changes only the representation, not the underlying one-body Ruby model.

## 3. GW equations

For supercell orbital indices `A,B=0,...,17`,

\[
G^{-1}_{AB}(k,i\omega_n)
=(i\omega_n+\mu)\delta_{AB}
-h^0_{AB}(k)-\Sigma^H_{AB}-\Sigma^{GW}_{AB}(k,i\omega_n),
\]

\[
P_{AB}(Q)=\frac{T}{N_k}\sum_k G_{AB}(k+Q)G_{BA}(k),
\]

\[
W(Q)=[I-V(q)P(Q)]^{-1}V(q),
\]

\[
\Sigma^{GW}_{AB}(k)
=-\frac{T}{N_q}\sum_Q G_{AB}(k+Q)W_{BA}(Q).
\]

The same FFT momentum backend, fixed-filling analytic tail subtraction and raw fixed-point residual are used as in the six-site solver.

`--primitive-filling` is quoted per original six-site cell. Internally the 18-site target is three times larger. Thus half filling is

```text
primitive filling = 3
supercell filling = 9
```

## 4. Temporary charge source

The canonical complex primitive soft mode is represented as

\[
v=(1,\omega,\omega^2,-1,-\omega,-\omega^2),\qquad
\omega=e^{2\pi i/3}.
\]

The real pattern in the three primitive-cell sectors is

\[
\begin{array}{c|rrrrrr}
&s0&s1&s2&s3&s4&s5\\\hline
R_0&1&-1/2&-1/2&-1&1/2&1/2\\
R_1&-1/2&-1/2&1&1/2&1/2&-1\\
R_2&-1/2&1&-1/2&1/2&-1&1/2
\end{array}
\]

The driver can add

\[
H_{source}=-h\sum_I p_I n_I
\]

and then remove `h` adiabatically. The default source sequence is

```text
0.01 -> 0.005 -> 0.001 -> 0
```

applied at the first V point at or above `--source-onset-V` (default 0.78). If the final zero-source solution retains nonzero charge amplitude, the broken symmetry is spontaneous rather than pinned by the external source.

## 5. Charge-order diagnostic

Define the complex 18-site folded mode `z`. The driver reports

\[
\Phi=\frac{2\langle z|\delta n\rangle}{\langle z|z\rangle}.
\]

If

\[
\delta n=A\,\mathrm{Re}[z e^{i\theta}],
\]

then

\[
\Phi=Ae^{i\theta}.
\]

Therefore `abs(Phi)` measures the period-three charge amplitude independently of which translated member of the threefold family is selected.

## 6. Optimized continuation numerics

The current driver combines fast fixed-filling solves, inexact inner tolerances, branch continuation and safeguarded Anderson acceleration. The GW equations themselves are unchanged.

### Cached analytic tail subtraction

For fixed filling, the static reference

\[
h_{ref}(k)=h_0(k)+\Sigma_H
\]

is diagonalized once per outer GW iterate. The eigensystem is cached while the chemical potential is varied.

### Safeguarded Newton chemical-potential solve

For fixed self-energies define

\[
F(\mu)=N(\mu)-N_{target}.
\]

The solver now evaluates the analytic derivative of the same tail-subtracted numerical filling. Since

\[
G^{-1}=i\omega+\mu-h_0-\Sigma_H-\Sigma_{GW},
\]

at fixed self-energy

\[
\frac{\partial G}{\partial\mu}=-G^2.
\]

The analytic reference-tail contribution is differentiated as well, so Newton steps require no extra Dyson inversion beyond evaluating the new trial `mu`. A sign bracket is accumulated as a safeguard; bisection is used only if Newton leaves the bracket or the derivative is unusable.

With `--verbose-iterations`, `mu_eval` reports the number of Dyson evaluations used by the current chemical-potential solve.

### Inexact inner filling solve

It is wasteful to force

\[
|N-N_{target}|<10^{-8}
\]

while the outer GW residual is still of order `1e-1`. Intermediate iterations therefore use

\[
\tau_\mu=
\max\left(\tau_{strict},
\min\left(10^{-4},10^{-2}\epsilon_{GW}\right)\right),
\]

where `epsilon_GW` is the outer raw self-energy residual and `tau_strict=--mu-tol` (default `1e-8`). Thus a far-from-converged iterate typically solves filling only to `1e-4`; the tolerance automatically tightens to `1e-5`, `1e-6`, ... as the GW fixed point is approached.

Before any GW state is returned, the chemical potential is refined once more at the strict requested `--mu-tol`, and the raw GW residual is recomputed on that exact state.

### Loose ramp, strict target

Intermediate continuation points and nonzero-source steps use

```text
--ramp-tol 1e-6
```

while the requested final `V` at `h=0` uses

```text
--gw-tol 1e-8
```

### Conservative Anderson acceleration

The solver no longer switches to aggressive Anderson mixing merely because a fixed number of iterations has passed. It first performs ordinary linear basin finding. Anderson becomes eligible only when either

\[
\epsilon_{GW}\lesssim 0.1
\]

or a stable sequence has reduced the residual substantially relative to the starting value.

When Anderson starts, the effective initial damping is at most about `0.3`. Recent states are used in a Type-II Anderson least-squares update with block scaling between the Hartree and dynamic GW self-energy sectors.

Crucially, an Anderson proposal is checked using the *actual next GW map*. If it raises the residual by more than about 10%, the proposal is rejected, the state is rolled back to its parent, Anderson history is cleared and a conservative linear recovery step is taken. This prevents trajectories such as

```text
0.3 -> 0.6 -> 1.1 -> 0.3
```

from consuming many iterations.

### Retry continuation

If an attempt remains finite but does not meet the requested tolerance, the next linear/Pulay fallback starts from the best state reached at the same `(V,h)` instead of discarding it.

### V predictor

After two consecutive converged zero-source states are identified as the same charge-order branch, the next `V` may be initialized by a damped secant extrapolation of `Sigma_H`, `Sigma_GW` and `mu`. The predictor is disabled automatically across a normal-to-broken branch change.

## 7. First validation run

Start with

```bash
python run_supercell_gw.py \
  --V 1.0 \
  --primitive-filling 3 \
  --nk1 3 --nk2 3 \
  --nw 55 --nomega 12 \
  --verbose-iterations
```

The default V path is

```text
0.70 -> 0.75 -> 0.78 -> 0.80 -> 0.85 -> 0.90 -> 1.00
```

and the source is inserted at `V=0.78` then removed before continuation proceeds.

After the latest optimization, a useful first diagnostic is that `mu_eval` should usually be small while the GW residual is large, with `mu_tol` printed near `1e-4`. Only close to the fixed point should `mu_tol` tighten toward `1e-8`.

Once zero-source checkpoints exist, use

```bash
python run_supercell_gw.py \
  --V 1.35 \
  --primitive-filling 3 \
  --nk1 3 --nk2 3 \
  --nw 55 --nomega 12 \
  --restart-from auto
```

so the scan starts from the nearest compatible zero-source checkpoint rather than repeating the ramp from `V=0.7`.

## 8. Momentum symmetry and k-point reduction

The factor-three supercell construction already reduces the Brillouin-zone volume by three. Further irreducible-k reduction is possible only for symmetries preserved by the *chosen broken-symmetry domain*.

A finite-Q density order can break primitive translations and may also break point-group operations that map `Q` to a different member of its star or rotate one charge-order domain into another. Therefore one must first determine the little group/stabilizer of the converged 18-site density pattern before applying point-group k reduction.

For a real charge-density state with no loop current, time reversal is normally still present, so `k` and `-k` remain related. However, the present GW implementation evaluates momentum convolutions on the full regular grid using FFT. An irreducible-wedge implementation would still have to reconstruct the full grid before the FFT convolution, so the gain at the current `3x3` supercell mesh is modest. It also becomes invalid once a time-reversal-breaking loop-current state is allowed.

For these reasons the current optimized driver keeps the full reduced-BZ mesh. Symmetry reduction is a later optimization, best considered after the charge-domain symmetry has been classified and if larger `nk` becomes the dominant cost.

## 9. Outputs

The default output directory is

```text
results/supercell18/<timestamp>/
```

Main files:

```text
supercell_scan.csv
    every Anderson/linear/Pulay attempt, residual, requested tolerance,
    retry-seed status, source, mu, |Phi|, screening smin and soft supercell Q

density_profile.csv
    all 18 site densities for every converged V/source substep

charge_order_vs_V.png
    zero-source |Phi| versus V

screening_smin_vs_V.png
    zero-source screening minimum versus V

settings.json
    complete numerical settings and resolved continuation schedules
```

Converged zero-source continuation states are checkpointed for later restart. Failed attempts are never promoted to the next V/source point, but a finite low-residual failed attempt may seed the next retry at the same point.

## 10. What this does not yet calculate

This module currently stops at the broken-symmetry GW background. It does **not** yet solve the 18-site cGW current vertex. Once the charge branch is verified and can be continued to the desired V, the next step is to embed the physical opposite/same loop-current probes into the 18-site supercell and compute the corresponding response matrix on this background.

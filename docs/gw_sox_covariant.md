# Covariant SOX and the GW / cGW / cGW+SOX comparison

This note documents the production extension of the strict 12-site bare-SOX
`O(V^2)` diagnostic.  The goal is to support later phase-diagram scans with a
stable hierarchy of methods:

1. **GG**: bubble built from a converged ordinary SC-GW Green function;
2. **cGW**: the tail-consistent covariant H/F/MT/AL response on that SC-GW background;
3. **cGW+SOX**: the H/F/MT/AL/SOX covariant response on a self-consistent GW+SOX background;
4. **ED**: added only in same-torus exact benchmarks.

A fixed-GW-background SOX response is retained as a diagnostic, but it is not a
production method label.  It isolates the explicit SOX kernel while keeping the
one-particle background fixed.

## 1. SOX self-energy

For the spinless density-density interaction in the repository convention, the
bare second-order exchange skeleton in a full real-space site basis is

\[
\Sigma^{\mathrm{SOX}}_{ij}(\tau)
=
\sum_{kl}
 v_{il}v_{kj}
 G_{ik}(\tau)G_{kl}(-\tau)G_{lj}(\tau).
\]

The sign is the exchange sign used by the existing strict second-Born/SOX
diagnostic.  Expanding ordinary GW generates the second-order direct/ring
self-energy, but not this crossed exchange topology.

For a periodic problem, the implementation first converts each k-resolved field
to the finite real-space torus corresponding to the chosen k mesh,

\[
G_{R\alpha,R'\beta}(\tau)
=
\frac{1}{N_k}\sum_k
 e^{ik\cdot(R-R')}G_{\alpha\beta}(k,\tau),
\]

and applies the same transform to the bare interaction.  The SOX contraction is
then evaluated in that real-space basis and transformed back to k.  This avoids
maintaining a second set of delicate crossed-diagram momentum-routing formulae.

## 2. Covariant derivative

For a static q=0 source, the positive tangent convention used throughout cGW is

\[
X = G\Gamma G = -\frac{\partial G}{\partial h}.
\]

Differentiating the SOX skeleton gives exactly three terms,

\[
\begin{aligned}
D\Sigma^{\mathrm{SOX}}_{ij}[X](\tau)
=\sum_{kl}v_{il}v_{kj}\Big[
&X_{ik}(\tau)G_{kl}(-\tau)G_{lj}(\tau)\\
+&G_{ik}(\tau)X_{kl}(-\tau)G_{lj}(\tau)\\
+&G_{ik}(\tau)G_{kl}(-\tau)X_{lj}(\tau)
\Big].
\end{aligned}
\]

The production cGW+SOX equation is therefore

\[
\Gamma
=K
+D\Sigma_H[X]
+D\Sigma_F[X]
+D\Sigma_c[X]
+D\Sigma_{\mathrm{SOX}}[X],
\]

where `D Sigma_c` is the already validated MT+AL derivative of the retarded
`W-V` GW part.  In the implementation this remains a linear matrix-free equation
and is solved with the same restarted GMRES machinery as production cGW.

## 3. Why the reversed line needs a different Matsubara phase

For `0 < tau < beta`,

\[
G(+\tau)=T\sum_n e^{-i\omega_n\tau}G(i\omega_n),
\qquad
G(-\tau)=T\sum_n e^{+i\omega_n\tau}G(i\omega_n).
\]

The same rule applies to `X`.  Treating the middle SOX line with the same phase
as the two forward lines gives the wrong finite-frequency routing.  The general
kernel therefore reconstructs the positive- and negative-tau lines separately.

## 4. Tail-consistent tau reconstruction

A direct transform of a finite fermionic Matsubara box is not accurate enough
for SOX.  A static reference is used,

\[
G_{\rm ref}^{-1}(k,i\omega_n)
=i\omega_n+\mu-h_{\rm static}(k),
\]

with the natural GW+SOX choice

\[
h_{\rm static}=h_0+\Sigma_H+\Sigma_F.
\]

The interacting Green function is reconstructed as

\[
G(\pm\tau)
=G_{\rm ref}(\pm\tau)
+T\sum_{n\in\mathrm{box}}
 e^{\mp i\omega_n\tau}
 [G(i\omega_n)-G_{\rm ref}(i\omega_n)].
\]

For the directional derivative, the same subtraction must be differentiated.
Let `Gamma_static` denote the high-frequency static part of the q=0 vertex and

\[
X_{\rm ref}=G_{\rm ref}\Gamma_{\rm static}G_{\rm ref}.
\]

Then

\[
X(\pm\tau)
=X_{\rm ref}(\pm\tau)
+T\sum_{n\in\mathrm{box}}
 e^{\mp i\omega_n\tau}
 [X(i\omega_n)-X_{\rm ref}(i\omega_n)].
\]

The code estimates `Gamma_static` from symmetric outer Matsubara points.  That
operation is linear in `Gamma`, so it does not break the linearity required by
GMRES.

## 5. Self-consistent GW+SOX background

The background solver uses

\[
\Sigma
=\Sigma_H+\Sigma_{\rm corr},
\qquad
\Sigma_{\rm corr}
=\Sigma_F+\Sigma_c+\Sigma_{\rm SOX}.
\]

`Sigma_F`, `Sigma_c`, and `Sigma_SOX` are stored separately in the result even
though Dyson's equation uses their sum.  The fixed-point residual is evaluated
for `Sigma_H` and the total non-Hartree `Sigma_corr`, so a converged result is a
fixed point of the actual GW+SOX map rather than an ordinary GW solution with a
post-processing SOX correction.

This distinction matters:

- **fixed-GW + SOX diagnostic**: ordinary GW `G,W` are frozen and only the SOX
  contribution is added to the covariant vertex equation;
- **cGW+SOX production**: both the background and the response kernel include SOX.

Only the second is labelled `cGW+SOX` in the new phase-scan output.

## 6. Raw versus tail-completed susceptibility

The historical static response is the represented finite-box contraction

\[
\chi^{\rm box}_{ab}
=-\frac{T}{N_k}\sum_{n,k}
\mathrm{Tr}[K_a G\Gamma_b G].
\]

ED static susceptibilities contain the complete Matsubara sum.  The new scan and
benchmark interfaces therefore save both the raw box result and

\[
\chi^{\rm full}_{ab}
=\chi^{\rm box}_{ab}+\Delta\chi^{\rm ref-tail}_{ab}.
\]

Use `*_completed` when comparing GG/cGW/cGW+SOX to ED or to a thermodynamic
static derivative.  The raw value is retained for regression against older
outputs.

## 7. Validation status

The strict free-background implementation remains the normalization anchor.  In
the 12-site weak-coupling ledger it gives

- `z_same`: `b_SOX = -2.591702289`;
- `z_opposite`: `b_SOX = -2.152210216`.

These reproduce the previously identified missing negative `O(V^2)` current
response to within a few percent.  The new periodic kernel is regression-tested
against that strict implementation, and its three-line derivative is also
checked directly against a finite difference of the SOX self-energy.

The interrupted longer validation run reported the following additional checks:

- at `V=0.1`, the fixed-GW-background `z_same` response changed from cGW
  `2.96585` to `2.93332` after adding the SOX kernel;
- a direct `+h/-h` fully self-consistent source check at `V=1` agreed with the
  covariant GW+SOX response to relative error below `1e-6`;
- increasing the SOX Gauss-Legendre quadrature from 440 to 660 nodes did not
  change the `V=1` result;
- enlarging the Matsubara cutoff at `V=0.5` and `V=1.0` changed the three tested
  static responses by less than `0.81%`;
- in that setup, the missing static reference tail was about `0.0230`, large
  enough that raw finite-box values should not be compared directly with ED.

These numbers are validation records from that run, not a new claim that bare
SOX is accurate at strong coupling.

## 8. Important finite-coupling limitation

The same validation run found a clear limitation at `V=1`: adding SOX strongly
reduced the over-enhanced cGW current response and restored the ordering with
`x_even` above the two current channels, but bare SOX then suppressed the current
channels too much.  Updating the one-particle background self-consistently did
not remove that undershoot.

The interpretation is therefore deliberately limited:

- the weak-coupling comparison identifies crossed second-order exchange as the
  dominant missing `O(V^2)` topology;
- bare SOX is **not** yet established as a quantitatively reliable
  intermediate/strong-coupling approximation;
- phase diagrams should show GG, cGW and cGW+SOX side by side rather than
  replacing cGW by cGW+SOX;
- if strong-coupling conclusions depend sensitively on their difference, the
  next method question is a screened-exchange/SOSEX or more complete vertex
  extension rather than simply increasing numerical cutoffs.

## 9. Entry points

Core modules:

- `rubycgw/sox_covariant.py`: periodic SOX self-energy, tau reconstruction and
  covariant derivative;
- `rubycgw/gw_sox.py`: self-consistent GW+SOX background;
- `rubycgw/production_cgw_sox.py`: tail-consistent q=0 cGW+SOX vertex and
  raw/completed static-response helpers.

Scans and benchmarks:

```bash
python scan_primitive_gw_cgw_sox_vs_V.py \
  --V 0.1 0.25 0.5 0.75 1.0 \
  --channels x_even z_same z_opposite
```

This stores GG, cGW and self-consistent cGW+SOX.

```bash
python benchmark_ed_gw_cgw_sox.py \
  --L1 2 --L2 1 \
  --V 0.05 0.1 0.25 0.5 1.0 \
  --channels x_even z_same z_opposite
```

This adds ED on the exactly matching finite torus.  For ED comparison use the
`*_completed` arrays in `benchmark.npz`.

# Primitive-cell finite-q covariant GW

This note documents the static finite-external-momentum response implemented in `rubycgw/finite_q_cgw.py` and `scan_primitive_cgw_q.py`.

## 1. Why no supercell is needed

To diagnose an instability of a translationally invariant normal state, the external ordering wavevector can be kept explicitly in the response function.  The self-consistent GW background remains the six-site primitive-cell solution

\[
G(k,i\omega_n),\qquad W(Q,i\Omega_m),
\]

and the response carries a separate external primitive momentum \(p\).  A supercell is only needed later if one wants to construct a self-consistent broken-symmetry state whose ordering wavevector enlarges the real-space unit cell.

The finite-q cGW calculation therefore keeps all one-particle orbital matrices at size \(6\times6\).

## 2. Response convention

For a static external field with primitive momentum \(p\), define the off-diagonal Green-function derivative

\[
X(k;p)
\equiv
\frac{\delta G(k+p,k)}{\delta h_p}
=
G(k+p)\,\Gamma(k;p)\,G(k).
\]

No finite field is applied numerically.  `K` is the bare functional-derivative vertex and the code solves the linear covariant equation

\[
(I-\mathcal L_p)\Gamma_p=K_p.
\]

The production split-GW self-energy is

\[
\Sigma_{GW}=\Sigma_F+\Sigma_c,
\qquad
\Sigma_c=-G*(W-V),
\]

so the corresponding finite-q vertex decomposition is

\[
\Gamma
=K
+\Gamma_H
+\Gamma_F
+\Gamma_{MT,c}
+\Gamma_{AL1}
+\Gamma_{AL2}.
\]

`MT,c` uses \(W-V\); the AL terms use full \(W\) because \(\delta(W-V)=\delta W\).

## 3. Finite-q H/F/MT terms

Once

\[
X(k;p)=G(k+p)\Gamma(k;p)G(k)
\]

is formed, the static Fock and MT derivatives retain the same internal momentum convolutions as at q=0.  The external momentum is already carried by \(X\).

For the present Ruby model the density interaction is only on the six intracell triangle bonds, so \(V(q)\) is independent of q.  The Hartree finite-q convention is nevertheless implemented explicitly.

## 4. Finite-q AL momentum routing

This is the genuinely new part compared with the q=0 solver.

Using the bosonic momentum convention of the existing GW code, the screened-interaction tangent entering a fermionic response carrying \(+p\) is

\[
\delta W(Q-p,Q)
=
W(Q-p)\,\delta P(Q-p,Q)\,W(Q).
\]

The polarization tangent has two terms,

\[
\delta P=L_1+L_2,
\]

with

\[
[L_1(Q;p)]_{ab}
=
\frac{T}{N_k}\sum_{k,n}
X_{ab}(k+Q-p;p)\,G_{ba}(k),
\]

and

\[
[L_2(Q;p)]_{ab}
=
\frac{T}{N_k}\sum_{k,n}
G_{ab}(k+Q)\,X_{ba}(k;p).
\]

Thus

\[
M_{1,2}(Q;p)
=
W(Q-p)L_{1,2}(Q;p)W(Q),
\]

and the AL self-energy tangents use the same final internal-Q convolution as the q=0 code,

\[
\Gamma_{AL1/2}(k;p)
=
-\frac{T}{N_k}\sum_Q
G(k+Q)\odot M_{1/2}(Q;p)^T.
\]

At \(p=0\), every expression above reduces term by term to the established q=0 implementation.  This equality is enforced by regression tests.

## 5. Momentum mesh

The interacting background \(G\) and \(\Sigma\) are known on a discrete periodic k mesh.  The first production finite-q implementation therefore accepts external momenta on that same mesh only:

\[
p=\left(\frac{i_1}{n_{k1}},\frac{i_2}{n_{k2}}\right).
\]

This is not a supercell restriction; it is simply the momentum resolution of the chosen numerical integration grid.  It avoids uncontrolled interpolation of an interacting frequency-dependent self-energy.

For example, to evaluate exactly

\[
Q=(1/3,1/3),
\]

both `nk1` and `nk2` must be divisible by 3.

## 6. Pseudospin vertices at finite q

The local pseudospin operators are defined per primitive cell in the cell-periodic convention:

\[
\tau_x=2n_0-n_1-n_2,
\qquad
\tau_y=\sqrt3(n_2-n_1),
\]

with the corresponding B-triangle physical-frame convention, while \(\tau_z\) is the Pauli-normalized loop chirality.

Because these are intracell operators, their bare orbital matrices are independent of external p in this convention.  The finite momentum resides in the cell-to-cell modulation \(e^{ip\cdot R}\).  Thus the same `primitive_pseudospin_vertex(channel)` can be used at every p.

This convention probes a finite-q modulation of a fixed primitive-cell form factor.  If a future calculation requires a different real-space origin choice for A/B triangle centers or bond centers, the API also allows a full k-dependent bare vertex field.

## 7. Susceptibility

For local left/right pseudospin form factors,

\[
\chi_{\mu\nu}(p)
=-\frac{T}{N_k}\sum_{k,n}
\mathrm{Tr}
\left[
K_\mu\,G(k+p)\,\Gamma_\nu(k;p)\,G(k)
\right].
\]

A single diagonal response \(\chi_{xx}(p)\) requires one finite-q cGW solve.  A full N-channel matrix requires N driven vertices at each p.

When both p and -p are available, the exact equilibrium relation

\[
\chi_{\mu\nu}(p)=\chi_{\nu\mu}(-p)^*
\]

provides a useful numerical consistency check.  `hermitianize_q_pair` averages this pair.

## 8. Driver examples

One point:

```bash
python scan_primitive_cgw_q.py \
  --V 1.0 --filling 2 \
  --nk1 6 --nk2 6 \
  --chi x_even,x_even \
  --q 0.3333333333333333 0.3333333333333333
```

Whole q mesh:

```bash
python scan_primitive_cgw_q.py \
  --V 1.0 --filling 2 \
  --nk1 6 --nk2 6 \
  --chi x_even,x_even \
  --all-q
```

Cheap first pass:

```bash
python scan_primitive_cgw_q.py \
  --V 1.0 --filling 2 \
  --nk1 12 --nk2 12 \
  --chi x_even,x_even \
  --all-q --stage gg
```

Full pseudospin competition:

```bash
python scan_primitive_cgw_q.py \
  --V 1.0 --filling 2 \
  --nk1 6 --nk2 6 \
  --channels x_even y_even z_same z_opposite \
  --all-q --stage full
```

The driver solves SC-GW once, then reuses the same background for every external q.  By default the converged vertex from the previous q is used as the initial GMRES field for the next q.

## 9. Validation hierarchy

The finite-q implementation is checked in increasing order of difficulty:

1. **q=0 identity:** H, F, MT, AL1 and AL2 match the production q=0 kernel term by term.
2. **FFT/direct identity:** for nonzero external q, the optimized FFT and transparent direct momentum loops agree on random complex arrays.
3. **V=0 limit:** the cGW vertex reduces to the bare vertex and the result equals the finite-q bubble.
4. **q/-q relation:** equilibrium response is checked against the conjugate reversed-momentum relation.
5. **Supercell folding benchmark:** Q=(1/3,1/3) can be compared against the 18-site q_sc=0 formulation after momentum-grid convergence.  Because the rectangular primitive and supercell integration meshes are not identical point sets at finite size, this is a convergence benchmark rather than an elementwise finite-grid identity.

## 10. Recommended workflow

For a new parameter point:

1. converge the six-site primitive SC-GW background;
2. scan `--stage gg` on a relatively fine q mesh to locate candidate wavevectors cheaply;
3. run `split-mt` or `full` cGW only around the candidate q region;
4. increase `nk`, `nw`, and `nOmega` to verify the location and channel of the soft response;
5. only after an ordering wavevector is established, build a commensurate supercell if a self-consistent broken-symmetry phase is needed.

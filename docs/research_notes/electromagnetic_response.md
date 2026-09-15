# Electromagnetic covariant response

## Scope

This module differentiates the **same self-consistent GW approximation used by the 18-site production solver** with respect to a periodic Peierls phase on the elementary Ruby triangles. It provides

\[
K_\phi=\frac{\partial H_0}{\partial\phi},\qquad
\Gamma_\phi,\qquad
G_\phi=\frac{\partial G}{\partial\phi},\qquad
\Sigma_\phi,\qquad
P_\phi,\qquad
W_\phi,
\]

and an independent centered finite-difference check from fully self-consistent GW calculations at \(\pm\delta\phi\).

This is a genuine electromagnetic lattice response because the perturbation is introduced through a Peierls phase. It is **not yet the strict uniform bulk orbital magnetization**. A uniform magnetic field in a periodic crystal requires a long-wavelength transverse \(A(q)\) construction, or an equivalent magnetic-unit-cell limit.

## Peierls source

For a directed triangle bond,

\[
t_{ij}\rightarrow t_{ij}e^{i c_{ij}\phi},\qquad c_{ji}=-c_{ij}.
\]

The total algebraic Peierls phase around a selected triangle is \(\phi w_X\), and it is distributed equally over the three edges. Therefore each directed edge carries phase \(\phi w_X/3\).

The implemented channels are

- `A`: \((w_A,w_B)=(1,0)\),
- `B`: \((0,1)\),
- `opposite`: \((1,1)/\sqrt2\), conjugate to the project's physical-opposite current channel,
- `same`: \((1,-1)/\sqrt2\), conjugate to the project's physical-same current channel.

The bare electromagnetic vertex is

\[
K_\phi=\left.\frac{\partial H_0(\phi)}{\partial\phi}\right|_{\phi=0}.
\]

For one elementary triangle this is the eta/current matrix multiplied by \(t_i/3\), with the channel normalization above.

## Covariant GW derivative

The production supercell solver evaluates

\[
\Sigma=\Sigma_H+\Sigma_F+\Sigma_c,
\qquad
\Sigma_c=-G*(W-V),
\]

where the bare-\(V\) Fock piece is treated as a static equal-time term and only the retarded \(W-V\) part is Matsubara truncated.

For a fixed chemical potential,

\[
G_\phi=G\Gamma_\phi G,
\]

with

\[
\Gamma_\phi
=K_\phi
+\Gamma_H+\Gamma_F+\Gamma_{\mathrm{MT},c}
+\Gamma_{\mathrm{AL1}}+\Gamma_{\mathrm{AL2}}.
\]

The implementation reuses the matrix-free GMRES solver in `rubycgw.supercell_cgw`, so the same Hartree/Fock/MT/AL decomposition used for the current susceptibility is used for the electromagnetic derivative.

From \(G_\phi\),

\[
P_\phi=\frac{\partial P[G]}{\partial\phi},
\qquad
W_\phi=W P_\phi W.
\]

The code implements both direct and FFT versions of \(P_\phi\).

## Fixed filling

The checkpoints used in the phase scan are normally fixed-filling solutions. Then \(\mu\) also changes under the external perturbation:

\[
G_\phi
=G\left(K_\phi+\Sigma_\phi-\mu_\phi I\right)G.
\]

Because the cGW equation is linear, the implementation solves two response problems:

1. the Peierls source \(K_\phi\),
2. the chemical-potential source \(-I\).

If their density responses are \(dN_\phi\) and \(dN_\mu\), respectively, the fixed-filling condition gives

\[
\mu_\phi=-\frac{dN_\phi}{dN_\mu}.
\]

The two vertex solutions are then combined linearly. `--fixed-mu` disables this step.

## Finite-difference validation

For a selected checkpoint and channel, the validation path builds

\[
H_0(+\delta\phi),\qquad H_0(-\delta\phi),
\]

starts both calculations from the zero-field checkpoint, and fully reconverges GW. It then evaluates

\[
X_\phi^{\rm FD}
=\frac{X(+\delta\phi)-X(-\delta\phi)}{2\delta\phi}
\]

for

\[
X\in\{G,\Sigma_H,\Sigma_{GW},P,W,\mu,n\}.
\]

The effective finite-difference vertex is reconstructed as

\[
\Gamma_\phi^{\rm FD}
=K_\phi-\mu_\phi^{\rm FD}I
+\Sigma_{H,\phi}^{\rm FD}
+\Sigma_{GW,\phi}^{\rm FD}.
\]

The command-line driver reports absolute, RMS, and relative max errors for every quantity.

### Expected convergence

For a well-converged calculation, the centered-difference truncation error should decrease approximately as

\[
O(\delta\phi^2)
\]

until one of the following dominates:

- the GW fixed-point tolerance,
- the cGW GMRES tolerance,
- the finite fermionic Matsubara box.

The last point matters because the production GW density/Fock map uses analytic tail subtraction, while the cGW equal-time response uses the absolutely convergent \(G\Gamma G\) sum over the stored Matsubara box. If the finite-difference error stops improving when \(\delta\phi\) is reduced, increase `nw` before changing the response equations.

## Usage

Covariant response only:

```bash
python analyze_em_response.py results/supercell18/checkpoints/FILE.npz \
  --channel same \
  --npz em_same.npz
```

Covariant response plus fully self-consistent finite difference:

```bash
python analyze_em_response.py results/supercell18/checkpoints/FILE.npz \
  --channel same \
  --finite-difference 1e-4 \
  --json em_same_validation.json
```

For a cutoff study, repeat the validation for several `delta_phi` values and several checkpoints with increasing `nw`.

## Tests

`tests/test_electromagnetic.py` contains three levels:

1. the analytic Peierls vertex is checked against a Hamiltonian finite difference;
2. the full covariant response is checked against finite difference at \(V=0\), where agreement should be nearly machine precision;
3. a small-\(V\), self-consistent interacting GW calculation checks that the full Hartree/Fock/MT/AL response tracks the reconverged finite difference.

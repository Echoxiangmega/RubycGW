# Chirality-pseudospin susceptibilities

This note documents the unified cGW response interface for the local Ruby-triangle pseudospin discussed in the chirality effective theory.

## 1. Local pseudospin operators

For triangle A=(0,1,2), in the low-energy E doublet,

\[
\tau_x = 2n_0-n_1-n_2,
\qquad
\tau_y = \sqrt{3}(n_2-n_1),
\]

while the loop-current operator projects to \(\sqrt{3}\tau_z\). The code therefore defines the Pauli-normalized z vertex by dividing the legacy eta/current vertex by \(\sqrt{3}\).

The x/y components are TR-even E-type intra-triangle charge/orbital polarization. A nonzero q=0 expectation value breaks C3 while preserving time reversal (an intra-unit-cell charge/orbital nematic). The z component is TR-odd loop chirality.

The B triangle has the opposite geometric handedness. The physical B pseudospin frame is obtained by swapping the algebraic |+> and |-> labels. Hence Bx is unchanged while By and Bz flip sign relative to the algebraic triangle convention. With this choice, A and B use the same physical Pauli convention.

## 2. Available channels

Local channels:

```text
Ax Ay Az Bx By Bz
```

Normalized A/B combinations:

```text
x_even x_odd
y_even y_odd
z_even z_odd
```

For z,

```text
z_even = z_same     = physical same circulation
z_odd  = z_opposite = physical opposite circulation
```

Aliases `same` and `opposite` are accepted.

Because z is Pauli-normalized,

\[
K_{z,\mathrm{same}}=-K_{\eta,\mathrm{same}}/\sqrt{3},
\qquad
K_{z,\mathrm{opposite}}=-K_{\eta,\mathrm{opposite}}/\sqrt{3}.
\]

Therefore diagonal z susceptibilities are one third of the legacy eta-current susceptibilities. This is intentional and makes chi_xx, chi_yy, and chi_zz directly comparable.

## 3. Direct q=0 supercell cGW

The production pseudospin script does **not** add a finite pseudospin perturbation and does **not** rerun SC-GW in a perturbed Hamiltonian. It loads a converged zero-field SC-GW fixed point and evaluates the covariant derivative directly.

For one primitive-cell pseudospin operator \(K_\mu\), the normalized primitive-q=0 operator embedded in the 18-site supercell is

\[
K_{\mu,q0}
=
\frac{1}{\sqrt{3}}
\begin{pmatrix}
K_\mu&0&0\\
0&K_\mu&0\\
0&0&K_\mu
\end{pmatrix}.
\]

The factor \(1/\sqrt{3}\) is the same orthonormal harmonic normalization used by the old sector-basis transformation. The cGW equation is solved directly for this operator direction,

\[
(I-\mathcal L)\Gamma_{\mu,q0}=K_{\mu,q0}.
\]

Here \(K\) is the bare vertex entering the functional derivative. Calling it a response/source vertex does **not** mean that a finite field is added numerically. No finite \(h\), no \(+h/-h\) difference, and no new broken-symmetry SC-GW solution is involved.

The static susceptibility is

\[
\chi_{\mu\nu}(q=0)
=-\frac{T}{N_k}\sum_{k,n}
\mathrm{Tr}\left[
K_{\mu,q0}G\Gamma_{\nu,q0}G
\right].
\]

Thus a diagonal response such as \(\chi_{xx}\) needs only one cGW solve for \(\Gamma_x\). A cross response \(\chi_{xy}\) also needs only one cGW solve, for the right derivative vertex \(\Gamma_y\); \(K_x\) appears only in the final contraction. A full N-channel susceptibility matrix requires N cGW solves.

The old sector-local vertices `s0/s1/s2` remain in the library only as a basis and regression check. They are no longer used as three prerequisite cGW solves by the production pseudospin driver.

## 4. Unified driver

List channels:

```bash
python analyze_pseudospin_susceptibility.py --V 1.4 --list-channels
```

Compute chi_xx for the A/B-even E-type order at primitive q=0:

```bash
python analyze_pseudospin_susceptibility.py \
  --V 1.4 --primitive-filling 2 \
  --chi x_even,x_even \
  --harmonic q0 \
  --checkpoint PATH_TO_CHECKPOINT
```

Compute chi_yy:

```bash
python analyze_pseudospin_susceptibility.py \
  --V 1.4 --primitive-filling 2 \
  --chi y_even,y_even \
  --harmonic q0 \
  --checkpoint PATH_TO_CHECKPOINT
```

Compute loop-current pseudospin susceptibility in the physical opposite channel:

```bash
python analyze_pseudospin_susceptibility.py \
  --V 1.4 --primitive-filling 2 \
  --chi z_opposite,z_opposite \
  --harmonic q0 \
  --checkpoint PATH_TO_CHECKPOINT
```

Cross response:

```bash
python analyze_pseudospin_susceptibility.py \
  --V 1.4 --primitive-filling 2 \
  --chi x_even,y_even \
  --harmonic q0 \
  --checkpoint PATH_TO_CHECKPOINT
```

Compute a complete matrix among several channels:

```bash
python analyze_pseudospin_susceptibility.py \
  --V 1.4 --primitive-filling 2 \
  --channels x_even y_even z_same z_opposite \
  --harmonic q0 \
  --checkpoint PATH_TO_CHECKPOINT
```

At the present stage this production driver intentionally accepts only `--harmonic q0`. Finite-Q pseudospin response is deferred.

## 5. cGW content

Every requested direct q=0 bare matrix K is fed into the same supercell covariant-GW kernel as the loop-current response,

\[
\Gamma
=K+\Gamma_H+\Gamma_F+\Gamma_{MT,c}+\Gamma_{AL1}+\Gamma_{AL2},
\]

or equivalently

\[
(I-\mathcal L)\Gamma=K.
\]

`split-mt` includes Hartree, static Fock, and MT with \(W-V\). `full` also includes AL1 and AL2. Therefore x/y and z are treated on exactly the same cGW footing; only the bare operator vertex differs.

On a TR-symmetric zero-field background, static mixed responses between TR-even x/y and TR-odd z should vanish up to numerical error. On a spontaneously TR-broken zero-field branch they need not vanish.

## 6. Recommended comparison

At the same checkpoint and Pauli normalization compare

\[
\chi_{xx},\quad \chi_{yy},\quad \chi_{zz}.
\]

This tests whether the leading response of the E doublet is toward TR-even intra-unit-cell charge/orbital nematicity or toward TR-odd loop current. For z also resolve `z_same` and `z_opposite` to determine the physical circulation pattern.

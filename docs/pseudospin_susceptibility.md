# Chirality-pseudospin susceptibilities

This note documents the unified cGW response interface for the local Ruby-triangle pseudospin discussed in the chirality effective theory.

## 1. Local pseudospin operators

For triangle A=(0,1,2), in the low-energy E doublet,

\[
\tau_x = 2n_0-n_1-n_2,
\qquad
\tau_y = \sqrt{3}(n_2-n_1),
\]

while the loop-current operator projects to \(\sqrt{3}\tau_z\).  The code therefore defines the Pauli-normalized z vertex by dividing the legacy eta/current vertex by \(\sqrt{3}\).

The x/y components are TR-even E-type intra-triangle charge/orbital polarization.  A nonzero q=0 expectation value breaks C3 while preserving time reversal (an intra-unit-cell charge/orbital nematic).  The z component is TR-odd loop chirality.

The B triangle has the opposite geometric handedness.  The physical B pseudospin frame is obtained by swapping the algebraic |+> and |-> labels.  Hence Bx is unchanged while By and Bz flip sign relative to the algebraic triangle convention.  With this choice, A and B use the same physical Pauli convention.

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

Therefore diagonal z susceptibilities are one third of the legacy eta-current susceptibilities.  This is intentional and makes chi_xx, chi_yy, and chi_zz directly comparable.

## 3. Unified driver

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

The same three sector harmonics as the current-response code are available:

```text
q0, Qc, Qs
```

with primitive Q=(1/3,1/3) folded to supercell q_sc=0.  Use `--harmonic all` to print all three blocks.

## 4. cGW equation

Every requested bare matrix K is fed into the same q_sc=0 covariant-GW equation as the legacy current source,

\[
(I-L)\Gamma=K,
\]

with Hartree, Fock, MT, AL1, and AL2 pieces controlled by `--stage`.  Thus x/y and z are treated on exactly the same response footing; only the bare source operator differs.

For a set of source vertices \(K_a\),

\[
\chi_{ab}
=-\frac{T}{N_k}\sum_{k,n}
\mathrm{Tr}[K_a G \Gamma_b G].
\]

On a TR-symmetric background, static mixed responses between TR-even x/y and TR-odd z should vanish up to numerical error.  On a spontaneously TR-broken zero-source branch they need not vanish.

## 5. Recommended comparison

For the pseudospin mechanism, compare at the same checkpoint and normalization:

\[
\chi_{xx},\quad \chi_{yy},\quad \chi_{zz}.
\]

This tests whether the leading instability of the E doublet is toward TR-even intra-unit-cell charge/orbital nematicity or toward TR-odd loop current.  For z also resolve `z_same` and `z_opposite` to determine the physical circulation pattern.

# Orbital moment post-processing

## Scope

`RubycGW` now includes a checkpoint post-processor for the **local orbital
moment carried by the elementary Ruby triangles**.  It is intentionally
separate from the strict periodic bulk orbital magnetization.

The current implementation answers:

> Given an already computed interacting GW checkpoint, what circulating bond
> current and local plaquette orbital moment are encoded in its one-body Green
> function?

It does **not** yet solve the electromagnetic covariant response
\(\partial G/\partial A\).  That is the next layer needed for a strict bulk
orbital magnetization or orbital susceptibility of a symmetric state.

## From a checkpoint to the one-body density matrix

The checkpoint stores \(\Sigma_H\), \(\Sigma_{GW}(k,i\omega_n)\), \(\mu\), and
the tail-corrected density.  The post-processor reconstructs

\[
G^{-1}(k,i\omega_n)
=
i\omega_n+\mu-H_0(k)-\Sigma_H-\Sigma_{GW}(k,i\omega_n).
\]

For the one-body density matrix we use

\[
\rho_{ab}
=
\langle c_b^\dagger c_a\rangle
=
\frac{T}{N_k}\sum_{k,n}G_{ab}(k,i\omega_n)
+\frac{1}{2}\delta_{ab}.
\]

The diagonal entries are replaced by the tail-corrected densities saved in the
checkpoint.  The off-diagonal entries that determine bond currents have a
\(1/\omega_n^2\) high-frequency tail and are directly convergent in the
symmetric Matsubara box.

## Directed bond current

For a hopping matrix element \(t_{IJ}\), attach a Peierls phase

\[
t_{IJ}\rightarrow t_{IJ}e^{i\phi_{IJ}}.
\]

The quantity conjugate to that phase is

\[
j^\phi_{I\rightarrow J}
=
\left\langle\frac{\partial H}{\partial\phi_{IJ}}\right\rangle
=
-2\,{\rm Im}
\left[
t_{IJ}\langle c_I^\dagger c_J\rangle
\right].
\]

`RubycGW` reports this first in code units.  It has units of the model energy.

The triangle orientations are exactly the eta conventions already used by the
project:

\[
A:\ 0\rightarrow1\rightarrow2\rightarrow0,
\qquad
B:\ 3\rightarrow4\rightarrow5\rightarrow3.
\]

The three bond currents are reported separately.  Their mean is the conserved
loop component,

\[
j^\phi_{\rm loop}
=
\frac{j^\phi_1+j^\phi_2+j^\phi_3}{3},
\]

and the maximum deviation from this mean is reported as a continuity/numerical
diagnostic.

## Local plaquette moment

For a conserved current around a plaquette \(p\),

\[
m_p
=
\frac12\sum_{\langle IJ\rangle\in p}
I_{IJ}
(\mathbf r_I\times\mathbf r_J)_z
=
I_{\rm loop}A_p.
\]

The code therefore defines

\[
m_p^{\rm code}
=
j^\phi_{\rm loop} A_p^{\rm signed}.
\]

The real-space embedding is the same one used by
`plot_supercell_order_realspace.py`.  The project eta loops on triangles A and
B have opposite geometric handedness, so their signed areas have opposite
signs.  This automatically converts the algebraic eta convention into a
physical real-space orbital-moment sign.

For every primitive sector \(s\), the tool reports

\[
m_{\rm net}(s)=m_A(s)+m_B(s),
\]

and

\[
m_{\rm staggered}(s)=m_A(s)-m_B(s).
\]

The first is the local net orbital moment.  The second is useful for an
orbital-altermagnetic pattern with opposite local moments.

## Physical-unit conversion

If the model energy unit \(E_0\) in eV and the lattice length unit \(a\) in
Angstrom are supplied, the code uses the electron Peierls convention

\[
\phi_{IJ}
=
-\frac{e}{\hbar}\int_I^J\mathbf A\cdot d\mathbf l
\]

and converts

\[
m_p
=
\frac{e}{\hbar}E_0a^2m_p^{\rm code}.
\]

The result is reported in \(\mu_B\).  The repository model is spinless, so the
default multiplicity is one.  A different multiplicity can be supplied only
when it is physically justified.

## Command-line use

Code units only:

```bash
python analyze_orbital_moment.py checkpoints/example.npz
```

Save triangle-resolved output:

```bash
python analyze_orbital_moment.py checkpoints/example.npz \
  --csv orbital_moment.csv \
  --json orbital_moment.json
```

Also convert to SI current and Bohr magnetons:

```bash
python analyze_orbital_moment.py checkpoints/example.npz \
  --energy-unit-ev 1.0 \
  --lattice-constant-angstrom 5.0
```

A checkpoint marked nonconverged is rejected by default.  For numerical
diagnostics only, use `--allow-nonconverged`.

A checkpoint with a nonzero temporary source is also rejected because the
current checkpoint format does not store the source operator needed to
reconstruct the correct \(H_0\).

## What a symmetric checkpoint gives

A time-reversal-symmetric zero-source state should give

\[
j^\phi_{\rm loop}\simeq0,\qquad m_p\simeq0
\]

up to numerical error.  This is expected: a direct expectation value of the
current is not the same as a magnetic response.

To calculate the response of a symmetric state, one must instead evaluate a
covariant derivative such as

\[
G_A
=
G(\gamma_A+\Sigma_A)G,
\]

with

\[
\gamma_A=\frac{\partial H_0}{\partial A},
\qquad
\Sigma_A=\frac{\partial\Sigma}{\partial A}.
\]

For GW this produces the tangent/covariant hierarchy discussed in the project,

\[
G_A\rightarrow P_A\rightarrow W_A\rightarrow\Sigma_A\rightarrow G_A.
\]

That response implementation is deliberately not mixed into the present local
moment tool.

## Relation to bulk orbital magnetization

The local triangle moment above is a real-space loop-current observable.  It is
not claimed to be the exact modern-theory bulk orbital magnetization of a
periodic interacting system.

For the latter, the relevant interacting Green-function and vertex literature
includes:

1. R. Nourafkan, G. Kotliar, and A.-M. S. Tremblay,
   *Orbital magnetization of correlated electrons with arbitrary band
   topology*, **Phys. Rev. B 90, 125132 (2014)**,
   DOI: 10.1103/PhysRevB.90.125132.
2. F. Aryasetiawan, K. Karlsson, and T. Miyake,
   *Green's function theory of orbital magnetic moment of interacting electrons
   in solids*, **Phys. Rev. B 93, 161104(R) (2016)**,
   DOI: 10.1103/PhysRevB.93.161104.
3. R. Bianco and R. Resta,
   *Orbital Magnetization as a Local Property*,
   **Phys. Rev. Lett. 110, 087202 (2013)**,
   DOI: 10.1103/PhysRevLett.110.087202.
   This is a noninteracting projector marker and should not be generalized by
   simply replacing the occupied-state projector with the interacting one-body
   density matrix.
4. H. Li et al.,
   *Linear response functions respecting Ward-Takahashi identity and
   fluctuation-dissipation theorem within the GW approximation*,
   **Phys. Rev. B 107, 085106 (2023)**,
   DOI: 10.1103/PhysRevB.107.085106.
   This provides the covariant-GW functional-derivative logic relevant to a
   future electromagnetic response implementation.

# Ruby-lattice ED benchmark geometry

This directory holds the **spinless Ruby-lattice exact-diagonalization (ED) benchmark** definition. It uses the same six-orbital and hopping convention as the earlier `ruby_selection_rule_check_physical_labels.py`.

## Files

- [Model/numbering/PBC specification (PDF)](model_geometry_and_parameters.pdf)
- [2 x 2 periodic cluster](figures/2x2_pbc_verified.svg) (24 sites)
- [3 x 1 periodic cluster](figures/3x1_pbc_verified.svg) (18 sites)
- [Six-triangle ring periodic cluster](figures/ring3_pbc_verified.svg) (18 sites)
- [Complete independent bond list](pbc_bond_list.csv)

## Hamiltonian and parameters

`H0 = sum_{R,(i,j,dr)} [ tij c^dagger_{R,i} c_{R+dr,j} + h.c. ]`

`Hint = V sum_R (n0*n1 + n0*n2 + n1*n2 + n3*n4 + n3*n5 + n4*n5)`

All operators are **spinless**. The default intra-triangle hopping is `t=+1.0`; inter-triangle hoppings are `t1=t2=+0.5`. There is no inter-triangle density interaction. For fixed-particle-number ED the chemical potential term can be omitted.

The original code uses the sign convention `+t c^dagger_i c_j + h.c.`; **do not silently change it to `-t`**. `V` is the quantity to scan.

## Primitive cell

The six orbitals are **0, 1, 2** (triangle A) and **3, 4, 5** (triangle B). In units of the side length:

`a1 = (-(1+sqrt(3))/2, -(3+sqrt(3))/2)`
`a2 = (-(1+sqrt(3))/2,  (3+sqrt(3))/2)`

Orbitals: `r0=(0,sqrt(3)/2)`, `r1=(-1/2,0)`, `r2=(1/2,0)`, `r3=(0,-1-sqrt(3)/2)`, `r4=(-1/2,-1)`, `r5=(1/2,-1)`.

The independent directed primitive hopping representatives are:

```text
i j  dx dy  type
0 1   0  0  t
0 2   0  0  t
2 1   0  0  t
3 4   0  0  t
3 5   0  0  t
4 5   0  0  t
1 4   0  0  t1
5 0   0 -1  t1
2 3  -1  0  t1
3 1   0 -1  t2
2 5   0  0  t2
0 4  -1  0  t2
```

A representative `(i,j,dx,dy)` means `c^dagger_{R,i} c_{R+(dx,dy),j}`. Exactly **one directed representative per physical undirected bond** is stored, with the Hermitian conjugate implicit. In particular, `dx,dy` may already be negative. The diagrams do **not** omit a separate physical set of negative-displacement bonds.

## PBC clusters

| Key | Supercell vectors | primitive cells | sites | t bonds | t1 bonds | t2 bonds | display wrap bonds |
|---|---|---:|---:|---:|---:|---:|---:|
| `2x2` | `A1=2a1, A2=2a2` | 4 | 24 | 24 | 12 | 12 | 8 |
| `3x1` | `A1=3a1, A2=a2` | 3 | 18 | 18 | 9 | 9 | 8 |
| `ring3` | `A1=a1-a2, A2=a1+2a2` | 3 | 18 | 18 | 9 | 9 | 6 |

Primitive-cell representative order is `[(0,0),(1,0),(0,1),(1,1)]`, `[(0,0),(1,0),(2,0)]`, and `[(0,0),(1,0),(1,1)]` respectively. The site index is `6*cell_index + orbital`.

For `ring3`, the plotted periodic images are selected so that **six whole triangles enclose a physical six-site hexagonal plaquette**, with alternating `t1,t2` edges. The canonical ED site-index assignment is unchanged. The *display* PBC winding can therefore differ from the *canonical* PBC winding; both are saved in `pbc_bond_list.csv`.

## Validated invariants

- Coordination number is four at every site.
- No missing or duplicate undirected hopping.
- Noninteracting real-space PBC spectra agree with primitive six-band Bloch spectra folded into each supercell to `~1e-15` numerical precision in the checked implementation.
- `ring3` encloses six whole triangles and one physical inter-triangle hexagon.

**ED convention:** if building a Hermitian one-body hopping matrix, insert both `H[s,d] += amplitude` and `H[d,s] += conjugate(amplitude)`. If building Fock-space hopping operators, include both directions with the correct fermionic anticommutation sign. Only intra-triangle `t` edges carry `V n_s n_d`.

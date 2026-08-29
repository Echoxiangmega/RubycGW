# API Reference

这一页按模块说明当前公开类和主要函数。shape 中 `Nf=2*nw`, `Nb=2*nOmega+1`, `Nk1=nk1`, `Nk2=nk2`。

## `rubycgw.model`

### `RubyParameters`

```python
RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.2)
```

保存 Ruby hopping 和 NN interaction 参数。

### `ruby_hoppings(params)`

返回 directed hopping list。内部 `_base_bonds()` 的每条 undirected bond 会自动补上 Hermitian conjugate。

### `build_h0(kpts, params)`

输入：`kpts` shape `(...,2)` 的 reduced momentum。

输出：`h0` shape `(...,6,6)`。

用途：构造单粒子 Bloch Hamiltonian。

### `build_interaction(qpts, params)`

输入：`qpts` shape `(...,2)`。

输出：`Vq` shape `(...,6,6)`。

用途：构造 NN density-density interaction matrix `V_ab(q)`。

### `eta_vertices()`

返回

```python
K_A, K_B, K_plus, K_minus
```

每个都是 `6x6` Hermitian matrix。`K_plus` 是 physical opposite，`K_minus` 是 physical same。

## `rubycgw.grids`

### `MatsubaraGrid`

```python
MatsubaraGrid(nk1=4, nk2=4, nw=16, nOmega=8, T=0.05)
```

主要属性：

```text
n_values   fermion Matsubara integer indices
m_values   boson Matsubara integer indices
omega      fermion frequencies
Omega      boson frequencies
nk         nk1*nk2
nf         2*nw
nb         2*nOmega+1
```

`kmesh()` 和 `qmesh()` 返回 shape `(Nk1,Nk2,2)` 的 reduced momentum mesh。

### `shift_fermion_field(field, dq1, dq2, m)`

输入 `field` shape `(Nf,Nk1,Nk2,...)`，返回 `F(k+Q)`。

空间 momentum periodic wrap；frequency shift 超出有限 fermion box 时设为零。

## `rubycgw.gw`

### `GWOptions`

主要参数：

```text
mu              初始 chemical potential
target_filling  None 表示固定 mu，否则每轮调整 mu
max_iter        GW 最大迭代次数
tol             收敛阈值
mixing          self-energy mixing
mu_tol          filling root solver 阈值
mu_max_iter     chemical-potential bisection 最大次数
verbose         是否打印每轮信息
```

### `NonInteractingResult`

字段：`G0`, `mu`, `density`。

### `GWResult`

字段：`G`, `W`, `P`, `Sigma_H`, `Sigma_GW`, `mu`, `density`, `converged`, `iterations`。

### `build_g0_inverse(h0, grid, mu)`

返回 shape `(Nf,Nk1,Nk2,6,6)` 的 `G0^{-1}`。

### `dyson_from_sigma(h0, grid, mu, sigma_h, sigma_gw)`

根据 Dyson equation 返回 interacting `G`。

### `density_from_G(G, grid)`

返回 length-6 的 orbital density array。

### `hartree_self_energy(density, Vq0)`

返回 `6x6` diagonal Hartree self-energy。

### `compute_polarization(G, grid)`

返回 shape `(Nb,Nk1,Nk2,6,6)` 的 `P(Q)`。

### `compute_screened_interaction(P, Vq, grid)`

返回同 shape 的 `W(Q)`。

### `compute_sigma_gw(G, W, grid)`

返回 shape `(Nf,Nk1,Nk2,6,6)` 的 GW self-energy。

### `solve_noninteracting(params, grid, mu=0.0, target_filling=None, ...)`

构造 noninteracting reference。若给 `target_filling`，独立求 `mu0`。

### `solve_gw(params, grid, opts)`

执行完整 self-consistent GW fixed-point loop，返回 `GWResult`。

## `rubycgw.cgw`

### `VertexOptions`

```text
max_iter
tol
mixing
include_hartree
include_mt
include_al
verbose
```

用三个 include flag 可以分别做 `bare`, `MT-only` 或 full cGW 的诊断。

### `VertexResult`

字段：

```text
Gamma
Gamma_H
Gamma_MT
Gamma_AL1
Gamma_AL2
converged
iterations
```

每个 Gamma field shape `(Nf,Nk1,Nk2,6,6)`。

### `gamma_h_q0(G, Gamma, Vq0, grid)`

计算 static uniform Hartree vertex correction。

### `gamma_mt_q0(G, W, Gamma, grid)`

计算 q=(0,0) MT correction。

### `gamma_al_q0(G, W, Gamma, grid)`

返回 `(Gamma_AL1, Gamma_AL2)`。

### `solve_vertex_q0(G, W, Vq0, K, grid, opts)`

在 converged GW background 上解 q=(0,0) linear vertex equation。

## `rubycgw.susceptibility`

### `chi_eta(G, K_left, grid, q1=0, q2=0, m=0, Gamma=None)`

计算

\[
-\frac{T}{N_k}\sum_k\mathrm{Tr}[K_{left}G(k+q)\Gamma(k,q)G(k)].
\]

若 `Gamma=None`，右 vertex 使用 bare `K_left`。因此：

```python
chi_eta(G0, K, grid)              # G0G0
chi_eta(G, K, grid)               # dressed GG
chi_eta(G, K, grid, Gamma=Gamma)  # cGW
```

### `channel_summary(G, K_plus, K_minus, grid)`

快速返回 plus/opposite、minus/same 以及 same-minus-opposite 差值。

## `run_ruby_cgw.py`

这是目前推荐的 end-to-end reference driver。它不是 library API，而是展示正确调用顺序的教程脚本：

```text
solve_noninteracting
    -> chi G0G0
solve_gw
    -> chi GG
solve_vertex_q0(include_al=False)
    -> GW+MT
solve_vertex_q0(include_al=True)
    -> full cGW
```

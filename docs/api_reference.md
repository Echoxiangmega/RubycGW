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

输入：`kpts` shape `(...,2)`；输出：`(...,6,6)` 的 `h0(k)`。

### `build_interaction(qpts, params)`

输入：`qpts` shape `(...,2)`；输出：`(...,6,6)` 的 `V_ab(q)`。

### `eta_vertices()`

返回 `K_A, K_B, K_plus, K_minus`。`K_plus` 是 physical opposite，`K_minus` 是 physical same。

## `rubycgw.grids`

### `MatsubaraGrid`

```python
MatsubaraGrid(nk1=4, nk2=4, nw=16, nOmega=8, T=0.05)
```

主要属性：`n_values`, `m_values`, `omega`, `Omega`, `nk`, `nf`, `nb`。

### `frequency_shift_slices(nf, m)`

返回 `(src,dst)` 两个 slice，使

```python
F_shifted[dst] = F[src]
```

对应 `omega -> omega + Omega_m` 的有效 Matsubara window。hot loop 使用这个函数避免为每个 Q 分配完整 zero-padded array。

### `roll_spatial(field, dq1, dq2)`

只对两个 momentum axes 做 periodic `k -> k+q` shift。

### `shift_fermion_field(field, dq1, dq2, m)`

保留原公开接口，返回完整 shape 的 `F(k+Q)`；超出 fermion box 的 frequency 设为零。主要用于通用调用和测试，性能关键代码现在直接使用前两个 allocation-light helper。

## `rubycgw.gw`

### `GWOptions`

主要参数：`mu`, `target_filling`, `max_iter`, `tol`, `mixing`, `mu_tol`, `mu_max_iter`, `verbose`。

### `NonInteractingResult`

字段：`G0`, `mu`, `density`。

### `GWResult`

字段：`G`, `W`, `P`, `Sigma_H`, `Sigma_GW`, `mu`, `density`, `converged`, `iterations`。

### `build_g0_inverse(h0, grid, mu)`

返回 shape `(Nf,Nk1,Nk2,6,6)` 的 `G0^{-1}`。

### `density_from_G(G, grid)`

返回 length-6 orbital density。

### `compute_polarization(G, grid)`

返回 `(Nb,Nk1,Nk2,6,6)` 的 `P(Q)`。当前实现只对每个 m 的有效 Matsubara slice 做 contraction。

### `compute_screened_interaction(P, Vq, grid)`

返回同 shape 的 `W(Q)`。

### `compute_sigma_gw(G, W, grid)`

返回 `(Nf,Nk1,Nk2,6,6)` 的 `Sigma_GW`，同样只处理有效 frequency window。

### `solve_noninteracting(params, grid, mu=0.0, target_filling=None, ...)`

构造 noninteracting reference。固定 filling 时独立求 `mu0`。

### `solve_gw(params, grid, opts, initial=None)`

执行 self-consistent GW。`initial` 可以传上一个参数点的 `GWResult`：若 fermionic array shape 相同，则旧的 `Sigma_H`, `Sigma_GW`, `mu` 作为新点初值；若 shape 不同自动忽略。

这适合 `V/filling/nOmega` 等 continuation scan。warm start 不改变新点方程，也不会跳过 convergence test。

## `rubycgw.cgw`

### `VertexOptions`

字段：`max_iter`, `tol`, `mixing`, `include_hartree`, `include_mt`, `include_al`, `verbose`。

### `VertexResult`

字段：`Gamma`, `Gamma_H`, `Gamma_MT`, `Gamma_AL1`, `Gamma_AL2`, `converged`, `iterations`。

### `gamma_h_q0`, `gamma_mt_q0`, `gamma_al_q0`

分别计算 q=(0,0) Hartree、MT、AL1/AL2 correction。保留为独立诊断 API。

### `solve_vertex_q0(G, W, Vq0, K, grid, opts, initial_gamma=None)`

解 q=(0,0) vertex equation。`initial_gamma` 可传：

- shape `(Nf,Nk1,Nk2,6,6)` 的前一参数点 full vertex；
- 当前点已经收敛的 MT-only vertex，用来 warm-start full MT+AL solve；
- `6x6` matrix（会 broadcast）。

solver 内部每轮只形成一次 `X=G Gamma G`，然后 Hartree/MT/AL 共用同一个 internal-Q loop。

## `rubycgw.susceptibility`

### `chi_eta(G, K_left, grid, q1=0, q2=0, m=0, Gamma=None)`

计算

\[
-\frac{T}{N_k}\sum_k\mathrm{Tr}[K_{left}G(k+q)\Gamma(k,q)G(k)].
\]

若 `Gamma=None`，右 vertex 使用 bare `K_left`：

```python
chi_eta(G0, K, grid)              # G0G0
chi_eta(G, K, grid)               # dressed GG
chi_eta(G, K, grid, Gamma=Gamma)  # cGW
```

## Driver scripts

### `run_ruby_cgw.py`

完整 staged reference run。full cGW 自动从同一点已经收敛的 MT vertex 开始。

### `convergence_scan.py`

支持

```text
--vertex-stage mt
--vertex-stage full
--vertex-stage both
--no-continuation
--skip-hartree
```

默认开启 compatible continuation，并把 `bare/GW/MT/full` 各阶段 wall time 写入 CSV。

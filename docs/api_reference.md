# API Reference

这一页按模块说明当前公开类和主要函数。shape 中 `Nf=2*nw`, `Nb=2*nOmega+1`, `Nk1=nk1`, `Nk2=nk2`。

## `rubycgw.model`

### `RubyParameters`

```python
RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.2)
```

保存 Ruby hopping 和 NN interaction 参数。

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

返回 `(src,dst)` 两个 slice，对应 `omega -> omega + Omega_m` 的有效 Matsubara window。

### `roll_spatial(field, dq1, dq2)`

只对两个 momentum axes 做 periodic `k -> k+q` shift。

### `shift_fermion_field(field, dq1, dq2, m)`

返回完整 shape 的 `F(k+Q)`；超出 fermion box 的 frequency 设为零。

## `rubycgw.gw`

### `GWOptions`

主要参数：

```text
mu
target_filling
max_iter
tol
mixing
mu_tol
mu_max_iter
verbose
momentum_backend   # "fft" or "direct"
```

### `NonInteractingResult`

字段：`G0`, `mu`, `density`。

### `GWResult`

字段：

```text
G
W
P
Sigma_H
Sigma_GW
mu
density
converged
iterations
final_error
```

`final_error` 是最后一次 GW self-consistency iteration 的 fixed-point error；若 `converged=True`，应满足 `final_error < opts.tol`。

### `build_g0_inverse(h0, grid, mu)`

返回 shape `(Nf,Nk1,Nk2,6,6)` 的 `G0^{-1}`。

### `density_from_G(...)`

固定 filling 的内部调用采用 analytic reference-Green-function tail subtraction；noninteracting 情况由 Fermi occupation 精确给出。

### `compute_polarization(G, grid, backend="fft")`

返回 `(Nb,Nk1,Nk2,6,6)` 的 `P(Q)`。另保留 `compute_polarization_direct/fft` 用于回归。

### `compute_screened_interaction(P, Vq, grid)`

批量求解

\[
[I-VP]W=V.
\]

### `compute_sigma_gw(G, W, grid, backend="fft")`

返回 `(Nf,Nk1,Nk2,6,6)` 的 `Sigma_GW`。另保留 direct/FFT reference functions。

### `solve_noninteracting(params, grid, mu=0.0, target_filling=None, ...)`

构造 noninteracting reference；固定 filling 时独立求 `mu0`。

### `solve_gw(params, grid, opts, initial=None)`

执行 self-consistent GW。`initial` 可传上一参数点的 `GWResult`；若 shape 相同，则复用 `Sigma_H`, `Sigma_GW`, `mu` 作为初值。返回值包含 `final_error`，便于区分“接近收敛”和“明显 fixed-point instability”。

## `rubycgw.cgw`

### `VertexOptions`

字段：

```text
max_iter
tol
mixing
include_hartree
include_mt
include_al
verbose
momentum_backend
```

### `VertexResult`

字段：`Gamma`, `Gamma_H`, `Gamma_MT`, `Gamma_AL1`, `Gamma_AL2`, `converged`, `iterations`。

### `gamma_h_q0`, `gamma_mt_q0`, `gamma_al_q0`

分别计算 q=0 Hartree、MT 与 AL corrections。

### `solve_vertex_q0(...)`

解 q=0 vertex fixed-point equation；`initial_gamma` 可使用前一参数点或已收敛 MT vertex。

## `rubycgw.susceptibility`

### `chi_eta(G, K_left, grid, q1=0, q2=0, m=0, Gamma=None)`

计算

\[
-\frac{T}{N_k}\sum_k\mathrm{Tr}[K_{left}G(k+q)\Gamma(k,q)G(k)].
\]

若 `Gamma=None`，右 vertex 使用 bare `K_left`。

## Driver scripts

### `run_ruby_cgw.py`

完整 staged reference run，默认 FFT momentum backend。

### `convergence_scan.py`

支持 `mt/full/both` vertex stage、continuation 与 split timings。

### `filling_scan.py`

固定 V 扫描 filling，画

\[
r_{\rm opposite}^{\rm eff}=\chi_{\rm opposite}^{-1},\qquad
r_{\rm same}^{\rm eff}=\chi_{\rm same}^{-1}.
\]

默认 `V=3`, `T=0.05`, filling `0.05...5.95` 共 241 点。强耦合默认采用：

```text
anchor near filling=3
interaction V-ramp
GW adaptive mixing retries: 0.20 -> 0.15 -> 0.10 -> 0.05
two filling continuation branches
```

可用

```bash
--gw-mixing 0.20
--gw-retry-mixings 0.15 0.10 0.05
```

修改 retry schedule。每一个 V-ramp attempt 都写入 `v_ramp.csv`，包含 `mixing`, `iterations`, `final_error`, `mu`, `actual_filling`, `runtime_s`。正式 filling CSV 还保存 `GW_final_error`, `GW_mixing_used`, `GW_attempts`。

若所有 GW retries 都失败，该 filling 跳过 vertex 并把 response 写成 NaN。

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
mixing_method          # "linear" or "pulay"
pulay_history          # default 6
pulay_start            # default 3
pulay_regularization   # default 1e-10
mu_tol
mu_max_iter
verbose
momentum_backend       # "fft" or "direct"
```

`mixing_method="linear"` 使用普通 under-relaxation。

`mixing_method="pulay"` 使用 recent residual history 做 Pulay/DIIS extrapolation；`mixing` 此时表示对 extrapolated self-energy 的 damping。Pulay 只改变 self-consistency 的数值求解路径，不改变 GW 方程。

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
mixing_method
min_screening_singular_value
min_screening_m
min_screening_Omega
min_screening_q1
min_screening_q2
```

`final_error` 现在是未乘 mixing 的 raw self-energy fixed-point residual：

\[
\max\left(
\|\Sigma_H^{out}-\Sigma_H\|_\infty,
\|\Sigma_{GW}^{out}-\Sigma_{GW}\|_\infty
\right).
\]

因此不同 mixing/method 的 `final_error` 可以直接比较；若 `converged=True`，应满足 `final_error < opts.tol`。

screening fields 对应

\[
s_{\min}=\min_Q\sigma_{\min}[I-V(\mathbf q)P(Q)]
\]

以及该最小值出现的 bosonic index/frequency 和 reduced momentum。

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

### `screening_diagnostic(P, Vq, grid)`

返回 `I-VP` 的全 Q 最小奇异值及其 `(m, Omega, q1, q2)` 位置。该 helper 当前位于 `rubycgw.gw` 模块。

### `compute_sigma_gw(G, W, grid, backend="fft")`

返回 `(Nf,Nk1,Nk2,6,6)` 的 `Sigma_GW`。另保留 direct/FFT reference functions。

### `solve_noninteracting(params, grid, mu=0.0, target_filling=None, ...)`

构造 noninteracting reference；固定 filling 时独立求 `mu0`。

### `solve_gw(params, grid, opts, initial=None)`

执行 self-consistent GW。`initial` 可传上一参数点的 `GWResult`；若 shape 相同，则复用 `Sigma_H`, `Sigma_GW`, `mu` 作为初值。

未收敛时返回的 `G` 与 `Sigma` 保持同一 iterate 的一致性，不再用 raw map output 覆盖 `Sigma`；这对 continuation/retry 更安全。

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
GW attempts: linear:0.20 -> linear:0.10 -> pulay:0.70
two filling continuation branches
```

相关参数：

```bash
--gw-mixing 0.20
--gw-retry-mixings 0.10
--gw-pulay-mixing 0.70
--gw-pulay-history 6
--gw-pulay-start 3
--no-gw-pulay
```

每一个 V-ramp attempt 都写入 `v_ramp.csv`，包含 method/mixing、iterations、raw `final_error`、chemical potential、actual filling，以及 `min_screening_singular_value` 和其 Q 位置。正式 filling CSV 保存同类 GW diagnostics。

若所有 GW attempts 都失败，该 filling 跳过 vertex 并把 response 写成 NaN。

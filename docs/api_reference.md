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

对应 `omega -> omega + Omega_m` 的有效 Matsubara window。

### `roll_spatial(field, dq1, dq2)`

只对两个 momentum axes 做 periodic `k -> k+q` shift。direct reference backend 使用它。

### `shift_fermion_field(field, dq1, dq2, m)`

保留原公开接口，返回完整 shape 的 `F(k+Q)`；超出 fermion box 的 frequency 设为零。

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
momentum_backend   # "fft" (default) or "direct"
```

`momentum_backend="fft"` 只对二维 periodic momentum convolution 使用 FFT；Matsubara sums 和 frequency cutoff convention 不变。

### `NonInteractingResult`

字段：`G0`, `mu`, `density`。

### `GWResult`

字段：`G`, `W`, `P`, `Sigma_H`, `Sigma_GW`, `mu`, `density`, `converged`, `iterations`。

### `build_g0_inverse(h0, grid, mu)`

返回 shape `(Nf,Nk1,Nk2,6,6)` 的 `G0^{-1}`。

### `density_from_G(G, grid)`

返回 length-6 orbital density。

### `compute_polarization(G, grid, backend="fft")`

返回 `(Nb,Nk1,Nk2,6,6)` 的 `P(Q)`。

另外保留：

```python
compute_polarization_direct(G, grid)
compute_polarization_fft(G, grid)
```

用于回归。

### `compute_screened_interaction(P, Vq, grid)`

返回同 shape 的 `W(Q)`。当前使用 batch `np.linalg.solve` 一次处理全部 Q。

### `compute_sigma_gw(G, W, grid, backend="fft")`

返回 `(Nf,Nk1,Nk2,6,6)` 的 `Sigma_GW`。

另外保留：

```python
compute_sigma_gw_direct(G, W, grid)
compute_sigma_gw_fft(G, W, grid)
```

### `solve_noninteracting(params, grid, mu=0.0, target_filling=None, ...)`

构造 noninteracting reference。固定 filling 时独立求 `mu0`。

### `solve_gw(params, grid, opts, initial=None)`

执行 self-consistent GW。`initial` 可以传上一个参数点的 `GWResult`：若 fermionic array shape 相同，则旧的 `Sigma_H`, `Sigma_GW`, `mu` 作为新点初值；若 shape 不同自动忽略。

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
momentum_backend   # "fft" (default) or "direct"
```

### `VertexResult`

字段：`Gamma`, `Gamma_H`, `Gamma_MT`, `Gamma_AL1`, `Gamma_AL2`, `converged`, `iterations`。

### `gamma_h_q0(G, Gamma, Vq0, grid)`

计算 q=(0,0) Hartree correction。

### `gamma_mt_q0(G, W, Gamma, grid, backend="fft")`

计算 MT correction，可显式选择 `fft/direct`。

### `gamma_al_q0(G, W, Gamma, grid, backend="fft")`

返回 `(AL1, AL2)`，同样支持 `fft/direct`。

### `solve_vertex_q0(G, W, Vq0, K, grid, opts, initial_gamma=None)`

解 q=(0,0) vertex equation。`initial_gamma` 可传：

- shape `(Nf,Nk1,Nk2,6,6)` 的前一参数点 full vertex；
- 当前点已经收敛的 MT-only vertex；
- `6x6` matrix（会 broadcast）。

FFT backend 对 MT 与 AL 内部的所有二维 momentum convolutions批量处理；direct backend 保留显式 Q-loop。

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

完整 staged reference run。由于 `GWOptions` 和 `VertexOptions` 默认 `momentum_backend="fft"`，主程序现在默认使用 FFT backend。

### `convergence_scan.py`

支持

```text
--vertex-stage mt
--vertex-stage full
--vertex-stage both
--no-continuation
--skip-hartree
```

默认计算也使用 FFT backend，并把 `bare/GW/MT/full` 各阶段 wall time 写入 CSV。若需要严格 direct-vs-FFT debugging，可在 Python API 中把两个 Options 的 `momentum_backend` 都设为 `"direct"`。

### `filling_scan.py`

固定 interaction 后扫描 six-site unit cell filling，并画物理 loop-current order parameter 的 effective quadratic mass

\[
r_{\rm opposite}^{\rm eff}=\chi_{\rm opposite}^{-1},\qquad
r_{\rm same}^{\rm eff}=\chi_{\rm same}^{-1}.
\]

这里 `+` 是 physical opposite，`-` 是 physical same。该 `r_eff` 是物理 eta effective action 的二阶曲率，不是旧 HS auxiliary field 的 `3V-(V^2/2)chi0`。

默认参数刻意复现旧 HS filling 图的扫描范围：

```text
V = 3.0
T = 0.05
filling = 0.05 ... 5.95
241 points
```

但 many-body grid 默认使用 `6x6`, `nw=60`, `nOmega=12`，并默认 `GW+MT + FFT + continuation`。输出包括：

```text
results/filling/<timestamp>/filling_scan.csv
results/filling/<timestamp>/r_eff_vs_filling.png
results/filling/<timestamp>/chi_vs_filling.png
results/filling/<timestamp>/delta_r_vs_filling.png
results/filling/<timestamp>/settings.json
```

快速粗扫示例：

```bash
python filling_scan.py --num-fillings 61
```

旧 HS 同样的 241 点网格：

```bash
python filling_scan.py
```

full cGW 可用：

```bash
python filling_scan.py --vertex-stage both
```

其中 `both` 先收敛 MT，再以 MT vertex warm-start full cGW。由于 `V=3` 明显强于当前 convergence benchmark 中的弱耦合点，脚本默认使用较保守的 GW/vertex mixing，并把每一点的 convergence flag 和 iteration count 写入 CSV。

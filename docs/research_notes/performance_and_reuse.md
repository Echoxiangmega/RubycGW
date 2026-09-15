# Performance and Reuse Guide

这一页说明 RubycGW 当前主要耗时来自哪里，以及参数扫描时哪些对象可以复用。

## 1. 为什么需要 FFT

旧的 reference backend 对每个 internal momentum `q` 显式循环，再在全部 `k` 上做 contraction。若线性 momentum mesh 为 `nk x nk`，总 momentum 点数为

\[
N_k=n_k^2,
\]

典型 convolution 的成本接近

\[
N_k^2\sim n_k^4.
\]

实际 `4x4 -> 6x6 -> 8x8` 的 wall time 也显示了接近 `n_k^4` 的 scaling。

当前版本已经加入二维 momentum FFT backend，并将它设为默认：

```text
GWOptions(momentum_backend="fft")
VertexOptions(momentum_backend="fft")
```

保留

```text
momentum_backend="direct"
```

作为数值回归路径。

## 2. FFT 改了什么、没有改什么

FFT **只**替换 periodic two-dimensional momentum convolution。Matsubara frequency sum 仍然显式执行，因此之前的 finite-frequency-box convention 完全不变。

对相关式

\[
C(q)=\sum_k A(k+q)B(k),
\]

离散 Fourier transform 给出

\[
\widehat C(r)=\widehat A(r)\widehat B(-r).
\]

代码中

\[
\widehat B(-r)
=\operatorname{conj}\{\mathrm{FFT}[\operatorname{conj}B](r)\},
\]

这里不能误写成普通 Hermitian correlation，因为原始 GW/cGW 方程并没有对第二个 Green function 做 complex conjugation。

FFT backend 已用于：

- `P(Q)=int_k G(k+Q)G(k)`；
- `Sigma_GW(k)=-int_Q G(k+Q)W(Q)^T`；
- MT vertex convolution；
- AL1/AL2 中的两个 loop `L1(Q),L2(Q)`；
- AL correction 最后的 `sum_Q G(k+Q)M(Q)^T` convolution。

`W(Q)` 的 `6x6` matrix solve 也改成 NumPy batched solve，不再逐 Q 调用 Python `np.linalg.solve`。

## 3. direct / FFT 回归

仓库保留 direct 实现并新增 `tests/test_fft_backend.py`。测试在随机 complex arrays 上逐项比较：

```text
P_direct      vs P_fft
Sigma_direct  vs Sigma_fft
MT_direct     vs MT_fft
AL1_direct    vs AL1_fft
AL2_direct    vs AL2_fft
```

目标 tolerance 为 `1e-11`。因此以后进一步优化时，direct backend 继续作为 reference equation implementation。

## 4. 已实现的其他单点优化

### 有效 Matsubara slice

`frequency_shift_slices()` 只处理 `omega+Omega` 仍位于存储区间的 frequency slice，而不是为每个 Q 构造完整 zero-padded fermion box。

### 合并 vertex correction

每一轮 vertex iteration 只形成一次

\[
X(k)=G(k)\Gamma(k)G(k).
\]

Hartree、MT、AL1、AL2 共用这个 `X`。

### MT -> full warm start

若同一点先计算 `GW+MT`，full cGW 以收敛的 MT vertex 为初值：

\[
\Gamma_{full}^{(0)}=\Gamma_{MT}^{converged}.
\]

## 5. 参数点之间的 continuation

`solve_gw(..., initial=previous_gw)` 支持用上一点的 `Sigma_H`, `Sigma_GW`, `mu` 作为新点初值。

`solve_vertex_q0(..., initial_gamma=previous_gamma)` 支持用上一点的 vertex 作为新点初值。

这些只是 initial guess；每一个新点仍重新计算 `P,W,Sigma` 和 vertex correction，并迭代到新点自己的 tolerance。

## 6. 哪些对象可以真正“不重算”

### 扫 V

固定 hopping、temperature、filling 和 grid 时：

- `h0(k)` 不变；
- `G0`, `mu0`, bare `G0G0` susceptibility 不变；
- `K_plus`, `K_minus` 不变；
- momentum/Matsubara grid 不变。

因此 bare reference 只需算一次。interacting `G,W,Sigma,Gamma` 必须对每个 V 重新收敛，但可以用前一个 V continuation。

### 扫 filling

`h0`, `V(q)`, `K_eta`, grid 不变；但 `mu0`, `G0` 和 bare susceptibility 变，因此需要重算 bare reference。相邻 filling 可 continuation interacting solution。

### 扫 nOmega

`G0` 完全不依赖 nOmega，可直接复用。fermionic `G/Sigma/Gamma` shape 也不变，所以这是最适合 continuation 的 convergence scan。

### 扫 nw

fermionic frequency 数组长度改变。当前不做 frequency interpolation / embedding，因此旧 `Sigma/Gamma` 不直接复用。

### 扫 nk

momentum mesh shape 改变。当前不对 self-energy 或 vertex 做 momentum interpolation，因此旧 interacting arrays 不直接复用。

## 7. 推荐两级扫描策略

### Fast screening

对大范围 V/filling/hopping 扫描：

```text
nk = 4x4 or 6x6
nw ~ 48-64
nOmega ~ 12-16
vertex-stage = mt
momentum backend = fft
continuation = on
```

### Production verification

只对以下点使用 full cGW：

- `chi_same - chi_opposite` 接近零的区域；
- 可能发生 selection reversal 的点；
- 文章图中的代表点；
- 扫描得到的 phase boundary 两侧。

这些点再提高 `nk/nw/nOmega` 并检查 full AL correction。

## 8. 下一步性能升级

FFT 后首先重新 benchmark `4x4,6x6,8x8,...` 的 wall-time scaling。理论上 momentum 部分可由接近

\[
N_k^2
\]

降低为接近

\[
N_k\log N_k,
\]

但总 wall time 还包含 Matsubara sums、Dyson inversions、chemical-potential root solve 和 matrix products，因此实际 exponent 需要实测。

若 vertex iteration 数仍然占主要成本，下一步再考虑 GMRES/Krylov 解

\[
(I-\mathcal K)\Gamma=K.
\]

其他可选优化包括：

1. `+/-` 两个 eta channel 加 batch 维度；
2. high-frequency analytic tail，减少所需 `nw`；
3. parameter points 并行；
4. compiled/JAX/Numba kernels；
5. 更大的 MPI production driver。

性能优化始终保留 direct reference backend，确保算法改造不改变物理方程。

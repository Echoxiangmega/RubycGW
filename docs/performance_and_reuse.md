# Performance and Reuse Guide

这一页说明 RubycGW 当前主要耗时来自哪里，以及参数扫描时哪些对象可以复用。

## 1. 当前主要瓶颈

对于 full cGW，最昂贵的是 q=(0,0) vertex equation 中对 internal bosonic `Q=(q,iOmega)` 的循环，特别是 AL1/AL2。经验上当 momentum 点数从 `4x4=16` 增加到 `6x6=36` 时，总运行时间可增长约 5 倍，接近多重 momentum convolution 的预期 scaling。

因此性能优化优先级是：

```text
减少重复 Q-loop
> 减少重复 k+Q allocation
> continuation 减少 fixed-point iterations
> 大扫描使用 MT-only
> 最后再考虑 FFT / Krylov / compiled backend
```

## 2. 已实现的单点优化

### 有效 Matsubara slice

旧实现每次 `k+Q` shift 都构造完整 zero-padded fermion box。当前 hot loops 通过 `frequency_shift_slices()` 只处理 `omega+Omega` 仍位于存储区间的 frequency slice。

### 合并 vertex correction loop

每一轮 vertex iteration 只计算一次

\[
X(k)=G(k)\Gamma(k)G(k).
\]

Hartree、MT、AL1、AL2 共用该 `X`。MT 与 AL 也共用同一 internal-Q loop 和 `X(k+Q)` shift，不再各做一次。

### MT -> full warm start

若同一点先计算 `GW+MT`，full cGW 默认以收敛的 MT vertex 为初值：

\[
\Gamma_{full}^{(0)}=\Gamma_{MT}^{converged}.
\]

由于当前 Ruby 参数下 AL correction 比 MT 小很多，这通常比从 bare `K_eta` 开始更快。

## 3. 参数点之间的 continuation

`solve_gw(..., initial=previous_gw)` 支持用上一点的 `Sigma_H`, `Sigma_GW`, `mu` 作为新点初值。

`solve_vertex_q0(..., initial_gamma=previous_gamma)` 支持用上一点的 vertex 作为新点初值。

这些只是 initial guess；每一个新点仍重新计算 `P,W,Sigma` 和 vertex correction，并迭代到新点自己的 tolerance。

## 4. 哪些对象可以真正“不重算”

### 扫 V

固定 hopping、temperature、filling 和 grid 时：

- `h0(k)` 不变；
- `G0`, `mu0`, bare `G0G0` susceptibility 不变；
- `K_plus`, `K_minus` 不变；
- momentum/Matsubara grid 不变。

因此 bare reference 只需算一次。interacting `G,W,Sigma,Gamma` 必须对每个 V 重新收敛，但可以用前一个 V continuation。

另外 `V(q)` 对统一 NN coupling V 是线性的；未来专门的 V-scan driver 可以预先构造 `V(q)/V` 的几何矩阵，再按当前 V 缩放。

### 扫 filling

`h0`, `V(q)`, `K_eta`, grid 不变；但 `mu0`, `G0` 和 bare susceptibility 变，因此需要重算 bare reference。相邻 filling 可 continuation interacting solution。

### 扫 nOmega

`G0` 完全不依赖 nOmega，可直接复用。fermionic `G/Sigma/Gamma` shape 也不变，所以这是最适合 continuation 的 convergence scan。

### 扫 nw

fermionic frequency 数组长度改变。当前不插值、不 embedding，因此旧 `Sigma/Gamma` 不直接复用。

### 扫 nk

momentum mesh shape 改变。当前不对 self-energy 或 vertex 做 momentum interpolation，因此旧 interacting arrays 不直接复用。

## 5. 推荐两级扫描策略

### Fast screening

对大范围 V/filling/hopping 扫描：

```text
nk = 4x4 or 6x6
nw ~ 48-64
nOmega ~ 12-16
vertex-stage = mt
continuation = on
```

当前测试表明 MT 是主要 vertex correction，而 AL 对 susceptibility 的改变量很小。因此该模式适合找趋势、候选相边界和 same/opposite 竞争区域。

### Production verification

只对以下点使用 full cGW：

- `chi_same - chi_opposite` 接近零的区域；
- 可能发生 selection reversal 的点；
- 文章图中的代表点；
- 扫描得到的 phase boundary 两侧。

这些点再提高 `nk/nw/nOmega` 并检查 full AL correction。

## 6. 仍可继续做的性能升级

当前版本仍是 NumPy reference implementation。下一步可考虑：

1. 缓存/批量化 spatial momentum rolls；
2. 将 momentum convolution 改成 FFT；
3. 用 GMRES/BiCGSTAB 解线性 vertex equation，替代 simple fixed point；
4. 用 Numba/Cython/JAX 或 compiled kernels 减少 Python Q-loop overhead；
5. 对 `+/-` 两个 eta channel 增加 channel batch 维度，一次计算共享的 G/W contraction；
6. MPI 或 multiprocessing 对 independent Q / parameter points 并行。

在这些优化之前，应始终保留现有 reference 路径作为数值回归基准，确保性能优化不改变物理方程。

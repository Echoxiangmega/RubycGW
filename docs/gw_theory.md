# Self-Consistent GW in RubycGW

## 1. 基本对象

所有 fermionic quantity 使用

\[
k=(\mathbf k,i\omega_n),\qquad \omega_n=(2n+1)\pi T,
\]

所有 bosonic quantity 使用

\[
Q=(\mathbf q,i\Omega_m),\qquad \Omega_m=2m\pi T.
\]

代码约定

\[
\int_k\equiv\frac{T}{N_k}\sum_{\mathbf k,n},\qquad
\int_Q\equiv\frac{T}{N_k}\sum_{\mathbf q,m}.
\]

每个 Green's function、self-energy 和 vertex 都是 6x6 sublattice matrix。

## 2. Noninteracting Green's function

\[
G_0^{-1}(k)=(i\omega_n+\mu)I_6-h_0(\mathbf k).
\]

代码对应 `build_g0_inverse()`。

固定 filling 时，`solve_noninteracting()` 会独立寻找 `mu0`，使非相互作用参考体系满足目标总粒子数。

## 3. Density 和 Hartree self-energy

对称 Matsubara box 中，代码使用

\[
n_a=\frac12+\frac{T}{N_k}\sum_{\mathbf k,n}G_{aa}(k).
\]

其中 `1/2` 是有限对称频率截断漏掉的 `1/(i omega_n)` 高频尾贡献。

Hartree self-energy 为

\[
[\Sigma_H]_{ab}=\delta_{ab}\sum_c V_{ac}(0)n_c.
\]

代码对应 `density_from_G()` 和 `hartree_self_energy()`。

## 4. Polarization

RubycGW 当前采用

\[
P_{ab}(Q)=\int_k G_{ab}(k+Q)G_{ba}(k).
\]

代码对应 `compute_polarization()`。其中 `shift_fermion_field()` 实现 `k -> k+Q`，空间动量按网格周期 wrap，Matsubara frequency 超出已存储 box 时设为零。

## 5. Screened interaction

不用显式计算 `V^{-1}`，而是解

\[
W(Q)=V(Q)+V(Q)P(Q)W(Q),
\]

即

\[
[I-V(Q)P(Q)]W(Q)=V(Q).
\]

代码中对每个 `(q,m)` 解一个 6x6 线性方程，对应 `compute_screened_interaction()`。

## 6. GW self-energy

\[
[\Sigma_{GW}(k)]_{ab}
=-\int_Q G_{ab}(k+Q)W_{ba}(Q).
\]

代码对应 `compute_sigma_gw()`。

## 7. Dyson equation

\[
G^{-1}(k)=G_0^{-1}(k)-\Sigma_H-\Sigma_{GW}(k).
\]

代码对应 `dyson_from_sigma()`。

## 8. Self-consistent loop

`solve_gw()` 反复执行

\[
G\rightarrow n\rightarrow\Sigma_H,
\]

\[
G\rightarrow P\rightarrow W\rightarrow\Sigma_{GW},
\]

再由 Dyson equation 更新 `G`。为了避免振荡，对 self-energy 使用 linear mixing：

\[
\Sigma^{(r+1)}=(1-\alpha)\Sigma^{(r)}+\alpha\Sigma_{\rm new}^{(r)}.
\]

固定 filling 时，每轮还会用 bisection 更新 chemical potential。

## 9. 返回对象

`GWResult` 包含：

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
```

这些量会直接作为 cGW vertex calculation 的背景，不需要在求 vertex 时再次更新 GW fixed point。

## 10. 与之前 HS 二阶计算的关系

HS/Landau 二阶计算使用 bare `G0G0` bubble；self-consistent GW 首先把传播子从 `G0` 改成 interacting `G`。只计算

\[
\chi_\eta^{GG}=-\int_k\mathrm{Tr}[K^\eta G K^\eta G]
\]

时，已经包含 self-energy dressing，但还没有包含 external eta source 对 self-energy 的反馈。这个反馈正是 cGW vertex correction 的来源。

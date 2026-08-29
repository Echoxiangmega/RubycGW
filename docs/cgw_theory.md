# Covariant GW Vertex in RubycGW

## 1. 外源与 full vertex

给 eta channel 加 infinitesimal source `h_eta`，定义

\[
\Gamma^\eta=\left.\frac{\delta G^{-1}}{\delta h_\eta}\right|_{h=0}.
\]

由矩阵恒等式 `G G^{-1}=1`，

\[
\frac{\delta G}{\delta h_\eta}=-G\Gamma^\eta G.
\]

在动量空间，若外源携带 q，则

\[
\frac{\delta G(k+q,k)}{\delta h_\eta(q)}
=-G(k+q)\Gamma^\eta(k,q)G(k).
\]

当前代码只实现 static uniform external source，即 `q=(0,0)`。

## 2. 为什么会有 H、MT、AL

Dyson equation 为

\[
G^{-1}=G_0^{-1}-\Sigma_H-\Sigma_{GW}.
\]

对 `h_eta` 求导：

\[
\Gamma^\eta=K^\eta-\frac{\delta\Sigma_H}{\delta h_\eta}
-\frac{\delta\Sigma_{GW}}{\delta h_\eta}.
\]

由于 `Sigma_GW=-GW`，链式法则给出两条反馈路径：导数打在 `G` 上得到 MT；导数打在 `W` 上时，由 `W^{-1}=V^{-1}-P` 得到 `delta W=W delta P W`，再因为 `P=GG`，导数可以打在 polarization bubble 的两条 fermion line 上，分别产生 AL1 和 AL2。

因此完整 q=0 vertex equation 是

\[
\Gamma^\eta=K^\eta+\Gamma_H^\eta+\Gamma_{MT}^\eta
+\Gamma_{AL1}^\eta+\Gamma_{AL2}^\eta.
\]

这些项不是人工选择的 diagram，而是对已经选定的 GW self-energy functional 做 functional derivative 自动得到的。

## 3. Hartree vertex

定义

\[
X(k)=G(k)\Gamma^\eta(k)G(k).
\]

q=0 时

\[
[\Gamma_H^\eta]_{aa}
=\sum_c V_{ac}(0)\frac{T}{N_k}\sum_k X_{cc}(k).
\]

`gamma_h_q0()` 实现这个公式。对 time-reversal symmetric normal state 中的 static eta source，density 是 T-even，eta 是 T-odd，因此这个 density-eta response 应为零。程序中 `max |Gamma_H|` 是重要的 symmetry diagnostic。

## 4. MT vertex

q=0 时

\[
[\Gamma_{MT}^\eta(p)]_{ij}
=-\int_Q [G(p+Q)\Gamma^\eta(p+Q)G(p+Q)]_{ij}W_{ji}(Q).
\]

代码先形成 `X=G Gamma G`，再对每个 internal bosonic Q 用 `shift_fermion_field(X,...)` 得到 `X(p+Q)`，对应 `gamma_mt_q0()`。

## 5. AL1 和 AL2

因为

\[
\delta P=\delta G\,G+G\,\delta G,
\]

有两个独立的 fermion-loop insertion。代码把 density projector 的 orbital contraction 预先化简成两个 6x6 loop matrix：

\[
L_1(Q)_{ef}=\int_k X_{ef}(k+Q)G_{fe}(k),
\]

\[
L_2(Q)_{ef}=\int_k G_{ef}(k+Q)X_{fe}(k).
\]

再形成

\[
M_1(Q)=W(Q)L_1(Q)W(Q),\qquad
M_2(Q)=W(Q)L_2(Q)W(Q).
\]

最后与 `G(p+Q)` 做 elementwise orbital contraction。对应 `_al_loops_q0()` 和 `gamma_al_q0()`。

## 6. Vertex solver

`solve_vertex_q0()` 从

\[
\Gamma^{(0)}=K^\eta
\]

开始 fixed-point iteration：

\[
\Gamma_{\rm rhs}=K^\eta+H+MT+AL1+AL2,
\]

\[
\Gamma^{(r+1)}=(1-\beta)\Gamma^{(r)}+\beta\Gamma_{\rm rhs}^{(r)}.
\]

由于 converged `G` 和 `W` 在这一步固定，vertex equation 对 `Gamma` 本身是线性的。当前用 fixed-point 主要为了透明和方便检查；以后可以改成 GMRES/Krylov 解 `(1-Kernel)Gamma=K_eta`。

## 7. 最终 susceptibility

\[
\chi_\eta(q)
=-\frac{T}{N_k}\sum_k
\mathrm{Tr}[K^\eta G(k+q)\Gamma^\eta(k,q)G(k)].
\]

当前 full cGW 使用 q=(0,0)。`chi_eta()` 若不传 `Gamma`，右 vertex 自动使用 bare `K_eta`，因此同一个函数可以计算 `G0G0`、`GG` 和 full cGW。

## 8. V=0 检查

当 `V=0` 时，`W=0`，所以 H、MT、AL1、AL2 全部消失：

\[
\Gamma^\eta=K^\eta,
\]

并且

\[
\chi_\eta^{cGW}=\chi_\eta^{G_0G_0}.
\]

`tests/test_cgw.py` 将这一点作为 regression test。

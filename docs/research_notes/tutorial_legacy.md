---
title: "RubycGW 完整教程"
author: "RubycGW project"
date: "自动从仓库文档构建"
lang: zh-CN
---

# 1. 项目目标

RubycGW 用来研究 spinless Ruby lattice 上最近邻 density-density interaction 驱动的 loop-current response，并比较两个 triangle 的 physical same / opposite current pattern。它不是从 HS field 的小振幅 Landau 展开开始，而是先求 self-consistent GW Green's function，再对 eta source 做 covariant functional derivative，得到包含 self-energy 和 vertex correction 的 physical susceptibility。

当前程序主要比较四个层级：

\[
G_0G_0\rightarrow GG\rightarrow GW+MT\rightarrow full\ cGW.
\]

它们分别对应 bare band response、self-energy dressing、MT vertex feedback、以及进一步包含 AL1/AL2 的完整 q=(0,0) cGW response。

# 2. 安装与第一次运行

在仓库目录外建立环境：

```bash
conda create -n rubycgw python=3.11
conda activate rubycgw
```

进入仓库后：

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_ruby_cgw.py
```

测试通过后再看物理输出。默认 `4x4` momentum mesh 和较小 Matsubara cutoff 是 debug 参数，不是 production convergence 参数。

# 3. Ruby lattice convention

每个 unit cell 有六个 site，编号 `0,1,2,3,4,5`。两个 triangle 是 `(0,1,2)` 和 `(3,4,5)`。代码保留之前 Ruby selection-rule calculation 的 12 条 NN bond 以及 `ti/t1/t2` 分类。

Fourier transform 使用 reduced reciprocal coordinates：

\[
c_{Ra}=\frac{1}{\sqrt N}\sum_k e^{2\pi i k\cdot R}c_{ka}.
\]

因此

\[
[h_0(k)]_{ab}=\sum_R t_{ab}(R)e^{2\pi i k\cdot R}.
\]

这里 phase 只含 Bravais cell offset `R`，不额外含 intracell coordinate。因此当前完全位于同一个 unit cell 内的 triangle eta operator 对应常数 6x6 bare vertex。

# 4. Interaction

模型使用同一组 NN bond 上的 density-density interaction：

\[
H_V=V\sum_{\langle ia,jb\rangle}n_{ia}n_{jb}.
\]

Fourier 形式为

\[
H_V=\frac{1}{2N}\sum_q n_a(q)V_{ab}(q)n_b(-q).
\]

`build_interaction()` 构造 6x6 Hermitian `V(q)`。程序求 screened interaction 时不显式使用 `V^{-1}`，而是直接解 `W=V+VPW`。

# 5. eta operator 和 same/opposite 标签

一条有向 bond `a -> b` 定义

\[
\eta_{ab}=i(\bar\psi_a\psi_b-\bar\psi_b\psi_a).
\]

没有额外 `1/2`。

A triangle 使用

\[
0\to1\to2\to0,
\]

B triangle 使用

\[
3\to4\to5\to3.
\]

因此有矩阵 `K_A`, `K_B`，并定义

\[
K^+=\frac{K^A+K^B}{\sqrt2},\qquad
K^-=\frac{K^A-K^B}{\sqrt2}.
\]

由于两个 algebraic arrow loops 在实际 Ruby embedding 中几何 handedness 相反，当前 convention 下必须记住

\[
\boxed{\eta_+\leftrightarrow\text{physical opposite}},
\]

\[
\boxed{\eta_-\leftrightarrow\text{physical same}}.
\]

所以程序表格中的 `opposite (+)` 和 `same (-)` 是刻意写出的 physical labels。

# 6. Matsubara grid

fermion frequency：

\[
\omega_n=(2n+1)\pi T,
\]

boson frequency：

\[
\Omega_m=2m\pi T.
\]

`MatsubaraGrid(nk1,nk2,nw,nOmega,T)` 使用

\[
n=-n_w,\ldots,n_w-1,
\]

\[
m=-n_\Omega,\ldots,n_\Omega.
\]

并定义

\[
\int_k=\frac{T}{N_k}\sum_{\mathbf k,n},\qquad
\int_Q=\frac{T}{N_k}\sum_{\mathbf q,m}.
\]

所有 `G`, `Sigma`, `Gamma` 的最后两个维度都是 `6x6` sublattice matrix。

# 7. 为什么 `G Gamma G` 在动量空间是 `G(k+q) Gamma(k,q) G(k)`

外源项写成

\[
h_\eta(q)\bar\psi(k+q)K^\eta\psi(k),
\]

因此 source 给 fermion 注入 external bosonic momentum-frequency q，使 incoming fermion `k` 变成 outgoing fermion `k+q`。

由

\[
GG^{-1}=1
\]

求导得到

\[
\frac{\delta G}{\delta h_\eta}=-G\Gamma^\eta G,
\]

恢复动量后即

\[
\frac{\delta G(k+q,k)}{\delta h_\eta(q)}
=-G(k+q)\Gamma^\eta(k,q)G(k).
\]

代码的 `shift_fermion_field()` 就负责实现 `k -> k+Q`。

# 8. Noninteracting reference

\[
G_0^{-1}(k)=(i\omega_n+\mu)I_6-h_0(\mathbf k).
\]

如果比较固定 filling，bare calculation 不能直接沿用 interacting GW 的 chemical potential。因此 `solve_noninteracting()` 独立寻找 `mu0`，使

\[
\sum_a n_a=n_{target}.
\]

这保证 `G0G0` 和 interacting results 比较的是同一个 filling。

# 9. Self-consistent GW

## 9.1 Density

有限对称 Matsubara box 下，代码用

\[
n_a=\frac12+\frac{T}{N_k}\sum_{\mathbf k,n}G_{aa}(k).
\]

`1/2` 补回有限 symmetric sum 漏掉的 `1/(i omega_n)` 高频尾贡献。

## 9.2 Hartree self-energy

\[
[\Sigma_H]_{ab}=\delta_{ab}\sum_cV_{ac}(0)n_c.
\]

## 9.3 Polarization

\[
P_{ab}(Q)=\frac{T}{N_k}\sum_kG_{ab}(k+Q)G_{ba}(k).
\]

## 9.4 Screened interaction

\[
W(Q)=V(Q)+V(Q)P(Q)W(Q),
\]

即

\[
[I-V(Q)P(Q)]W(Q)=V(Q).
\]

每个 `(q,m)` 只需要解一个 6x6 linear system。

## 9.5 GW self-energy

\[
[\Sigma_{GW}(k)]_{ab}
=-\frac{T}{N_k}\sum_QG_{ab}(k+Q)W_{ba}(Q).
\]

## 9.6 Dyson equation

\[
G^{-1}(k)=G_0^{-1}(k)-\Sigma_H-\Sigma_{GW}(k).
\]

程序不断执行

\[
G\to P\to W\to\Sigma_{GW}\to G
\]

以及 density/Hartree 更新，直到 fixed point 收敛。为了避免振荡，代码对 self-energy 做 linear mixing。

# 10. 从 GW 到 covariant GW

只把 `G0` 换成 dressed `G` 后得到

\[
\chi_\eta^{GG}
=-\frac{T}{N_k}\sum_k\mathrm{Tr}[K^\eta G(k)K^\eta G(k)].
\]

这只包含 self-energy dressing。真正的 physical response 还需要考虑 external eta source 会改变 `G`，进而改变 self-energy 本身。因此定义 full vertex

\[
\Gamma^\eta
=\left.\frac{\delta G^{-1}}{\delta h_\eta}\right|_{h=0}.
\]

对 Dyson equation 求导得到

\[
\Gamma^\eta
=K^\eta-\frac{\delta\Sigma_H}{\delta h_\eta}
-\frac{\delta\Sigma_{GW}}{\delta h_\eta}.
\]

# 11. Hartree、MT、AL 从哪里来

GW self-energy 是 `-GW`。链式法则给出

\[
\delta\Sigma_{GW}=-(\delta G)W-G(\delta W).
\]

第一项是外源插到 self-energy 的 fermion line 上，对应 MT。第二项需要继续对 screened interaction 求导：

\[
\delta W=W(\delta P)W.
\]

又因为 `P=GG`，

\[
\delta P=(\delta G)G+G(\delta G),
\]

所以有两种 insertion，分别是 AL1 和 AL2。

最终

\[
\boxed{
\Gamma^\eta=K^\eta+\Gamma_H^\eta+
\Gamma_{MT}^\eta+
\Gamma_{AL1}^\eta+
\Gamma_{AL2}^\eta }.
\]

这些项不是凭 diagram 名称手工补进去的，而是 functional derivative 自动产生。

# 12. q=(0,0) cGW 当前实现

当前代码专门实现 static uniform eta response。

定义

\[
X(k)=G(k)\Gamma^\eta(k)G(k).
\]

Hartree correction 用 `X` 的 diagonal density response；MT correction 对 internal Q 累加 `X(p+Q) W(Q)`；AL 部分先构造

\[
L_1(Q)_{ef}=\int_kX_{ef}(k+Q)G_{fe}(k),
\]

\[
L_2(Q)_{ef}=\int_kG_{ef}(k+Q)X_{fe}(k),
\]

再计算

\[
M_1=W L_1 W,\qquad M_2=W L_2 W.
\]

这样显式四重 sublattice sum 被压缩为 6x6 matrix multiplication。

# 13. Vertex fixed-point solver

`solve_vertex_q0()` 从 bare vertex 开始：

\[
\Gamma^{(0)}=K^\eta.
\]

每一轮计算

\[
\Gamma_{rhs}=K^\eta+H+MT+AL1+AL2,
\]

再 mixing：

\[
\Gamma^{(r+1)}=(1-\beta)\Gamma^{(r)}+\beta\Gamma_{rhs}^{(r)}.
\]

由于 `G` 和 `W` 在 cGW stage 已经固定，这实际上是关于 `Gamma` 的线性方程。当前 fixed-point 写法主要用于检查；后续可升级为 GMRES。

# 14. 最终 susceptibility

\[
\chi_\eta(q)
=-\frac{T}{N_k}\sum_k
\mathrm{Tr}[K^\eta G(k+q)\Gamma^\eta(k,q)G(k)].
\]

`chi_eta()` 统一实现这个式子。如果不传 `Gamma`，就自动把右 vertex 设为 bare `K_eta`。

因此：

```python
chi_eta(G0, K, grid)              # G0G0
chi_eta(G, K, grid)               # GG
chi_eta(G, K, grid, Gamma=Gamma)  # cGW
```

# 15. 四层结果怎么读

主程序输出：

```text
G0G0 (bare)
GG
GW + MT
full cGW
```

对于每层都给

```text
opposite (+)
same (-)
same-opposite
```

如果

\[
\chi_{same}>\chi_{opposite},
\]

则在 response 意义下 same channel 更软，对应 effective quadratic mass 更小。接近连续 instability 时可把 inverse susceptibility 理解成 effective action 的二阶核：

\[
r_{eff}\propto\chi^{-1}.
\]

分层比较可以判断：selection 是 bare band structure 已经决定，还是被 self-energy、MT、AL 改变。

# 16. 为什么当前取 q=0, Omega=0

`q=0` 表示所有 unit cell 重复同一个 intracell current pattern，不破坏 Bravais translation。`Omega=0` 表示研究 equilibrium static order。

这不是 cGW 方法本身要求 q=0，而是当前物理问题首先聚焦 uniform static loop-current order。以后若要无偏寻找 ordering wave vector，应推广到整个

\[
\chi_\eta(\mathbf q,0)
\]

并找最大 susceptibility 的 q。

# 17. 关键数值检查

必须检查 `h0`、`V(q)`、`K_eta` 的 Hermiticity；固定 filling 是否满足目标粒子数；GW 和 vertex 是否真正达到 tolerance；static susceptibility 的 imaginary part 是否只在 floating-point noise 范围；time-reversal symmetric normal state 中 `Gamma_H^eta(q=0)` 是否接近零。

最关键 regression limit 是 `V=0`：

\[
W=0,\quad\Gamma^\eta=K^\eta,
\]

\[
\chi^{cGW}=\chi^{G_0G_0}.
\]

仓库测试已自动检查这一点。

# 18. Matsubara cutoff 的当前限制

当前 `shift_fermion_field()` 对超出存储 fermion frequency box 的 `G(omega+Omega)` 直接设为零。这让 reference implementation 很透明，但也意味着 production result 必须对 `nw` 做严格 convergence test。

推荐顺序：先固定小 momentum mesh 收敛 `nw`，再收敛 `nOmega`，最后收敛 `nk1=nk2`。不要同时把三个 cutoff 一起改变，否则很难判断误差来源。

# 19. 代码模块地图

`rubycgw/model.py`：Ruby hopping、interaction、eta vertices。

`rubycgw/grids.py`：momentum/Matsubara grids 与 `k+Q` shift。

`rubycgw/gw.py`：noninteracting reference、density、Hartree、P、W、Sigma_GW、Dyson、GW self-consistency。

`rubycgw/cgw.py`：q=0 Hartree/MT/AL vertex correction 与 full vertex solver。

`rubycgw/susceptibility.py`：bare、dressed 和 covariant eta response。

`run_ruby_cgw.py`：完整 staged workflow 示例。

`tests/`：convention、Hermiticity、filling、V=0 regression tests。

# 20. 推荐的后续扩展

第一优先级是做 cutoff 和 momentum-mesh convergence，并保存每层 `G0G0/GG/MT/full cGW` 的结果。随后可以加入 parameter scan，研究 V、filling、T、`ti/t1/t2` 对 same/opposite selection 的影响。

第二优先级是性能优化：缓存 shifted fields、FFT convolution、利用 symmetry、用 GMRES 解 linear vertex equation。

第三优先级是有限 external q，将 `K_eta` 和 `Gamma_eta(k,q)` 推广到一般 response，并扫描整个 Brillouin zone 寻找真正的 ordering wave vector。

# 21. 文档与 PDF

本文件 `docs/tutorial.md` 是完整教程的 source of truth。仓库中的 documentation workflow 会在代码或文档更新时自动用 Pandoc + XeLaTeX 构建 `RubycGW_Tutorial.pdf`，并作为 GitHub Actions artifact 保存。这样以后只需要维护 Markdown，不需要手工编辑 PDF。

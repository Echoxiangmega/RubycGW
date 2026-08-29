# Numerics and Validation

## 1. 当前实现的定位

RubycGW 目前首先是 transparent reference implementation：公式和 momentum routing 优先于速度。默认小网格用于检查方程、符号和 symmetry，不应直接作为 production result。

## 2. Matsubara cutoff

fermion grid 使用

\[
n=-n_w,\ldots,n_w-1,
\]

boson grid 使用

\[
m=-n_\Omega,\ldots,n_\Omega.
\]

当 convolution 需要 `G(i omega_n+i Omega_m)` 超出已存储 fermion box 时，`shift_fermion_field()` 当前将该值设为零。这会产生 cutoff error，因此必须做 `nw` convergence。

推荐顺序：固定很小 `nk`，逐步增加 `nw`，直到 `mu`, `density`, `chi_plus`, `chi_minus` 的变化足够小；随后增加 `nOmega`；最后再增加 momentum mesh。

## 3. GW convergence

`solve_gw()` 的收敛误差取以下几项的最大值：

```text
max |G_new-G_old|
max |Sigma_GW,new-Sigma_GW,old|
max |Sigma_H,new-Sigma_H,mixed|
```

若 fixed-point 振荡，可减小 `GWOptions.mixing`。典型 debug 值为 `0.1-0.3`。

固定 filling 时还要检查输出的 `total filling` 是否达到目标值。

## 4. Vertex convergence

`solve_vertex_q0()` 检查

\[
\max|\Gamma^{(r+1)}-\Gamma^{(r)}|<\text{tol}.
\]

若迭代到 `max_iter` 仍未达到 `tol`，即使 susceptibility 看似稳定，也应先增加 `max_iter` 或减小 `mixing`。

由于 vertex equation 在固定 `G,W` 后是线性的，后续可用 GMRES/Krylov 替代 fixed-point。

## 5. 必须通过的极限和 symmetry checks

### Hermiticity

`h0(k)` 和 `V(q)` 应满足 Hermiticity。`K_A/B/+/-` 也应为 Hermitian matrix。

### V=0

必须满足

\[
W=0,\qquad \Gamma^\eta=K^\eta,
\]

以及

\[
\chi_\eta^{cGW}=\chi_\eta^{G_0G_0}.
\]

`tests/test_cgw.py` 已将这一点固定为 regression test。

### Static susceptibility 的 imaginary part

在 q=(0,0) static response 中，最终 susceptibility 应为实数。`10^{-14}` 到 `10^{-16}` 一类 imaginary part 一般只是 floating-point noise；若 imaginary part 与 real part 同量级，则必须检查 momentum/frequency routing 或收敛。

### Hartree eta vertex

在 time-reversal symmetric normal state，density 是 T-even，而 eta 是 T-odd，因此 static q=0 的 density-eta response 应消失。当前运行会打印 `max |Gamma_H|`；若其接近机器精度，这是很强的 symmetry check。

## 6. 物理分层诊断

推荐始终比较四层结果：

\[
G_0G_0\rightarrow GG\rightarrow GW+MT\rightarrow full\ cGW.
\]

它们分别回答：bare band structure 的 selection；self-energy dressing 的作用；MT vertex feedback 的作用；AL fluctuation feedback 的作用。

如果 same/opposite 的符号只在某一层发生变化，就可以明确指出 selection reversal 来自哪类 many-body correction。

## 7. 推荐的 production convergence 顺序

不要同时把所有 cutoff 一起增大。建议：

```text
A. 固定 nk=4x4, nOmega 小，收敛 nw
B. 固定收敛后的 nw，收敛 nOmega
C. 依次比较 nk=4x4, 8x8, 12x12, ...
D. 检查 mixing 改变是否影响最终 fixed point
E. 对关键 V / filling / T 点重复以上检查
```

## 8. 当前主要性能瓶颈

full cGW 的 AL 部分最昂贵。当前实现已经把显式四重 orbital sum 化为 `L1/L2` 和 `W L W` 的 6x6 matrix contraction，但仍对 internal bosonic momentum/frequency 做显式循环。以后优化方向包括 FFT convolution、缓存 shifted fields、利用 q=0/time-reversal symmetry、以及用 Krylov solver 解 vertex equation。

## 9. 自动 convergence scan

仓库提供 `convergence_scan.py` 自动执行上述 A-C 三步。最简单的完整扫描是：

```bash
python convergence_scan.py --scan all
```

如果 full AL 计算较慢，建议先分开执行：

```bash
python convergence_scan.py --scan nw
python convergence_scan.py --scan nomega
python convergence_scan.py --scan nk
```

每个 grid point 都完整保存 `G0G0`, `GG`, `GW+MT`, `full cGW` 四层结果、GW/vertex convergence flag、chemical potential、vertex norms 和 runtime。默认输出目录为：

```text
results/convergence/YYYYMMDD-HHMMSS/
```

并生成 `convergence.csv` 以及对应的 `convergence_nw.png`, `convergence_nOmega.png`, `convergence_nk.png`。每张图画 full-cGW 的 `chi_opposite`, `chi_same` 和 `chi_same-chi_opposite`。

完整命令行参数和输出字段见 [convergence_scan.md](convergence_scan.md)。

# Filling scan and effective mass

`filling_scan.py` 用来固定 interaction 后扫描 six-site spinless Ruby unit-cell filling，并比较 physical opposite/same loop-current channel。

## 1. 这里画的 r 是什么

当前 cGW 直接给出 physical static susceptibility：

\[
\chi_{\rm opposite}=\chi_+,
\qquad
\chi_{\rm same}=\chi_-.
\]

若 physical loop-current order parameter 为

\[
m_\lambda=\langle\eta_\lambda\rangle,
\]

则 normal state 原点的 effective-action curvature 是

\[
\boxed{r_{\lambda}^{\rm eff}=\chi_{\lambda}^{-1}}.
\]

因此脚本画

\[
\boxed{
r_{\rm opposite}^{\rm eff}=\chi_{\rm opposite}^{-1},
\qquad
r_{\rm same}^{\rm eff}=\chi_{\rm same}^{-1}.
}
\]

仍采用

\[
+\leftrightarrow\text{physical opposite},
\qquad
-\leftrightarrow\text{physical same}.
\]

该 `r_eff` 不是旧 HS auxiliary field 的 `3V-(V^2/2)chi0`。对于连续 instability，较小的 `r_eff` 是较软的 channel。

## 2. 默认扫描参数

默认使用

```text
V = 3.0
T = 0.05
filling = 0.05 ... 5.95
241 points
nk = 6x6
nw = 60
nOmega = 12
vertex-stage = mt
momentum-backend = fft
GW tol = 1e-8
vertex tol = 1e-8
```

GW 首先尝试 fast linear mixing：

```text
linear:0.20 -> linear:0.10
```

若同一个 GW 点仍未收敛，则程序从**同一个最近已收敛 seed** 自动切换到

```text
pulay:0.70
```

其中 `0.70` 是对 Pulay/DIIS extrapolated self-energy 的 damping，而不是重新定义方程。默认 Pulay history 为 6，第三步开始启用。

可自行改变：

```bash
python filling_scan.py \
  --gw-retry-mixings 0.15 0.10 0.05 \
  --gw-pulay-mixing 0.6 \
  --gw-pulay-history 8
```

若需要完全关闭 Pulay 作为诊断：

```bash
python filling_scan.py --no-gw-pulay
```

vertex mixing 默认仍是 `0.20`，目前 vertex 尚未做同样的 Pulay fallback。

## 3. 运行

完整扫描：

```bash
python filling_scan.py
```

快速诊断 half filling：

```bash
python filling_scan.py --fillings 3 --nk 6 --nw 55 --nomega 12
```

粗扫描：

```bash
python filling_scan.py --num-fillings 61
```

full cGW：

```bash
python filling_scan.py --vertex-stage both
```

## 4. Anchor + V-ramp continuation

强耦合下默认从最接近 `--anchor-filling` 的 requested filling 开始；默认 anchor 是 `3.0`。

对 `V=3`，默认 interaction path 为

```text
0.1 -> 0.25 -> 0.5 -> 0.6 -> 0.7 -> 0.75 -> 0.9 -> 1.0
    -> 1.25 -> 1.5 -> 1.75 -> 2.0 -> 2.25 -> 2.5 -> 2.75 -> 3.0
```

每一级首先做 linear mixing；若失败，才切换 Pulay。所有 retry 都重新从**上一级 converged GW solution** 出发，不使用失败 iterate 作为 seed。

例如终端可能显示：

```text
[V-ramp  6/16, try 1/3] V=0.7500 linear mix=0.200 conv=False ...
[V-ramp  6/16, try 2/3] V=0.7500 linear mix=0.100 conv=False ...
[V-ramp  6/16, try 3/3] V=0.7500 pulay  mix=0.700 conv=True  ...
```

到达 target V 后，由同一个 anchor solution 分成两个独立 branch：

```text
anchor -> lower fillings
anchor -> higher fillings
```

## 5. GW residual：现在使用真正的 fixed-point residual

GW 方程可写成

\[
X=(\Sigma_H,\Sigma_{GW}),\qquad F[X]=(\Sigma_H^{out},\Sigma_{GW}^{out}).
\]

当前 `final_error` 定义为未乘 mixing 的 raw residual：

\[
\boxed{
\epsilon_{GW}
=\max\left(
\|\Sigma_H^{out}-\Sigma_H\|_\infty,
\|\Sigma_{GW}^{out}-\Sigma_{GW}\|_\infty
\right).
}
\]

因此 `final_error` 不会因为把 linear mixing 从 `0.20` 改成 `0.05` 就人为变小。对于 converged 点必须满足

\[
\text{final_error}<\text{GW tol}.
\]

## 6. Pulay/DIIS 做了什么

linear mixing 是

\[
X_{n+1}=X_n+\alpha R_n,
\qquad
R_n=F[X_n]-X_n.
\]

Pulay 保存最近若干步 residual，并寻找系数 `c_i` 使

\[
\sum_i c_i R_i
\]

尽可能小，同时满足

\[
\sum_i c_i=1.
\]

代码中 Hartree block 和 dynamic GW block 在 residual inner product 中分别按自身元素数归一化。Pulay 只改变数值求解路径，不改变 GW 方程本身。

## 7. Screening stability diagnostic

每个 GW result 计算

\[
\boxed{
s_{\min}
=
\min_Q\sigma_{\min}\left[I-V(\mathbf q)P(Q)\right].
}
\]

由于

\[
W(Q)=\left[I-V(\mathbf q)P(Q)\right]^{-1}V(\mathbf q),
\]

所以 `s_min` 是判断 screened interaction 是否接近奇异的直接数值指标。程序同时记录出现最小值的

```text
screening_m
screening_Omega
screening_q1
screening_q2
```

解释时要区分：

```text
final_error large, s_min still moderate
    -> 更像 fixed-point solver 问题

s_min -> very small
    -> I-VP 本身接近奇异，可能是 screening/RPA instability
```

## 8. 最软模式 `screening_mode.csv`

对 V-ramp 中每一个**已经收敛**的 GW 点，程序进一步取最小奇异值所在的

\[
Q_*= (\mathbf q_*,i\Omega_{m_*})
\]

并保存两种六子格 mode。

第一种是 `screening_mode`：

\[
\left[I-V(\mathbf q_*)P(Q_*)\right]v_{\rm scr}
\approx0.
\]

它是 `I-VP` 的 right singular vector，更自然地解释为 effective-potential / screening direction，而不是直接等同于 density modulation。

第二种是后续构造 charge-order seed 更有用的 `density_mode`：

\[
\boxed{
v_n\propto P(Q_*)v_{\rm scr}.
}
\]

因为

\[
(I-PV)P v_{\rm scr}
=
P(I-VP)v_{\rm scr},
\]

所以当 screening singularity 被逼近时，`density_mode` 同时逼近 density self-consistency matrix `I-PV` 的 null direction。

两组 mode 均归一化为

\[
\sum_{a=0}^{5}|v_a|^2=1.
\]

SVD 自带任意整体复相位。为了让不同 V 的 mode 可直接比较，代码固定规范：**绝对值最大的 sublattice 分量被旋到实且为正**。

`screening_mode.csv` 每一行对应一个 converged V-ramp 点，包含：

```text
V, filling, method, mixing, iterations, final_error
min_screening_singular_value
screening_m, screening_Omega, screening_q1, screening_q2
density_mode_residual
screening_mode_0_re/im/abs/phase_rad ... screening_mode_5_...
density_mode_0_re/im/abs/phase_rad ... density_mode_5_...
```

其中 `density_mode_residual` 是

\[
\left\|(I-PV)v_n\right\|_\infty
\]

在同一个 `Q_*` 上的值。后续若要建立 finite-Q symmetry-broken GW 初值，应优先使用 `density_mode`。

## 9. 输出

`v_ramp.csv` 每一个 V / retry attempt 都保存：

```text
step
attempt
V
method
mixing
converged
iterations
final_error
mu
actual_filling
min_screening_singular_value
screening_m
screening_Omega
screening_q1
screening_q2
runtime_s
```

正式 `filling_scan.csv` 也保存

```text
GW_final_error
GW_mixing_method_used
GW_mixing_used
GW_attempts
GW_min_screening_singular_value
GW_screening_m
GW_screening_Omega
GW_screening_q1
GW_screening_q2
```

如果所有 GW attempt 都失败，则

```text
GW not converged -> skip vertex -> chi/r_eff = NaN
```

主要输出文件为

```text
results/filling/<timestamp>/filling_scan.csv
results/filling/<timestamp>/v_ramp.csv
results/filling/<timestamp>/screening_mode.csv
results/filling/<timestamp>/settings.json
results/filling/<timestamp>/r_eff_vs_filling.png
results/filling/<timestamp>/chi_vs_filling.png
results/filling/<timestamp>/delta_r_vs_filling.png
```

即使 V-ramp 在较大 V 处失败，前面所有 converged V 的 `screening_mode.csv` 仍会保留。

## 10. 图的解释

如果

\[
r_{\rm same}^{\rm eff}<r_{\rm opposite}^{\rm eff},
\]

则 same 是 leading continuous-instability channel；反之 opposite 更软。

脚本还画

\[
\Delta r=r_{\rm same}^{\rm eff}-r_{\rm opposite}^{\rm eff}.
\]

因此

```text
Delta r < 0 : same softer
Delta r > 0 : opposite softer
Delta r = 0 : quadratic-level crossing
```

这仍是 normal-state quadratic-response criterion；严格 ordered-state 基态比较需要进一步做 symmetry-broken free-energy calculation。

## 11. 终端进度条

正式 filling scan 默认显示：

```text
[########--------------------]  72/241  29.88% | elapsed 14m08s | ETA 33m12s | filling=1.7958
```

V-ramp 初始化会逐级打印 V、method、mixing、iteration count、`final_error`、`smin`、最危险的 Q、chemical potential 和 runtime。关闭 filling 进度条：

```bash
python filling_scan.py --no-progress
```

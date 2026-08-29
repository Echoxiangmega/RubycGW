# Filling scan and effective mass

`filling_scan.py` 用来固定 interaction 后扫描 six-site spinless Ruby unit-cell filling，并比较 physical opposite/same loop-current channel。

## 1. 这里画的 r 是什么

当前 cGW 直接给出 physical static susceptibility：

\[
\chi_{\rm opposite}=\chi_+,
\qquad
\chi_{\rm same}=\chi_-.
\]

若把 physical loop-current order parameter 本身记作

\[
m_\lambda=\langle\eta_\lambda\rangle,
\]

并用与它共轭的 source `h_lambda` 定义 effective action，则 normal state 原点的二阶曲率是 inverse susceptibility：

\[
\boxed{
r_{\lambda}^{\rm eff}=\chi_{\lambda}^{-1}
}.
\]

因此脚本画

\[
\boxed{
r_{\rm opposite}^{\rm eff}=\chi_{\rm opposite}^{-1},
\qquad
r_{\rm same}^{\rm eff}=\chi_{\rm same}^{-1}.
}
\]

这里仍然遵循项目的 physical label convention：

\[
+\leftrightarrow\text{physical opposite},
\qquad
-\leftrightarrow\text{physical same}.
\]

这和旧 HS auxiliary field 的二次系数

\[
3V-\frac{V^2}{2}\chi^{(0)}
\]

不是同一个归一化对象。旧式子是 HS auxiliary field 的 Landau coefficient；当前 `1/chi` 是 physical eta effective action 的 curvature。

对于连续 instability，判断规则非常直接：

\[
\chi_{\rm same}>\chi_{\rm opposite}
\Longleftrightarrow
r_{\rm same}^{\rm eff}<r_{\rm opposite}^{\rm eff},
\]

因此较小的 `r_eff` 对应较软、较先发生连续失稳的 channel。

## 2. 默认扫描范围

为了和之前 Ruby HS filling 图逐点对应，默认使用

```text
V = 3.0
T = 0.05
filling = 0.05 ... 5.95
241 points
```

即

\[
n_j=0.05+j\frac{5.90}{240},\qquad j=0,\ldots,240.
\]

这正好给出步长 `0.0245833...`；旧脚本也采用 `np.linspace(0.05, 5.95, 241)`。

many-body 数值默认参数为

```text
nk = 6x6
nw = 60
nOmega = 12
vertex-stage = mt
momentum-backend = fft
```

`V=3` 明显强于此前 `V=0.1` 的 convergence benchmark，所以脚本默认把 GW mixing 降到 `0.08`，vertex mixing 降到 `0.10`，并把最大迭代次数提高到 300。

## 3. 运行

完整 241 点扫描：

```bash
python filling_scan.py
```

建议第一次先做较粗的 61 点扫描，检查强耦合收敛与整体趋势：

```bash
python filling_scan.py --num-fillings 61
```

若只想检查整数 filling：

```bash
python filling_scan.py --fillings 1 2 3 4 5
```

full cGW：

```bash
python filling_scan.py --vertex-stage both
```

`both` 会先求 GW+MT，再用当前 filling 已收敛的 MT vertex 作为 full MT+AL 的初值。

## 4. Continuation

扫描 filling 时 `h0(k)`, `V(q)`, `K_plus/K_minus` 和 grid 不变，但 chemical potential、`G0`、interacting `G/W/Sigma/Gamma` 都会变化。

因此每个 filling 都重新求解对应方程，但默认使用前一个 filling 的解作为初值：

- 前一点 `mu0` 作为下一点 noninteracting chemical-potential bisection 的 seed；
- 前一点 converged `Sigma_H`, `Sigma_GW`, `mu_GW` warm-start 下一点 GW；
- 前一点 converged eta vertex warm-start 下一点 MT/full vertex。

这不改变每个 filling 的方程，只减少 fixed-point iteration 次数。可用

```bash
python filling_scan.py --no-continuation
```

关闭，用于验证 continuation 没有把程序锁在错误 branch。

## 5. 输出

每次运行自动建立

```text
results/filling/<timestamp>/
```

并输出：

```text
filling_scan.csv
settings.json
r_eff_vs_filling.png
chi_vs_filling.png
delta_r_vs_filling.png
```

`filling_scan.csv` 保存 bare `G0G0`、dressed `GG`、GW+MT、full cGW（若计算）以及 selected stage 的 susceptibility，同时保存

```text
r_eff_opposite_re/im
r_eff_same_re/im
delta_r_same_minus_opposite_re/im
```

以及 chemical potential、actual filling、GW/vertex convergence flag、iteration count 与各阶段 wall time。

## 6. 图的解释

主图是

\[
r_{\rm opposite}^{\rm eff}(n),\qquad
r_{\rm same}^{\rm eff}(n).
\]

如果某个 filling 上

\[
r_{\rm same}^{\rm eff}<r_{\rm opposite}^{\rm eff},
\]

则 same 是 leading continuous-instability channel；反之 opposite 更软。

`delta_r_vs_filling.png` 使用

\[
\Delta r=r_{\rm same}^{\rm eff}-r_{\rm opposite}^{\rm eff}.
\]

所以

```text
Delta r < 0 : same softer
Delta r > 0 : opposite softer
Delta r = 0 : quadratic-level degeneracy/crossing
```

注意这仍是 normal-state quadratic-response criterion。若要严格判断深处 ordered phase 的最终基态，需要进一步做 finite-source / symmetry-broken free-energy comparison。

## 7. 终端进度条

扫描默认显示一个不依赖第三方库的单行进度条。例如：

```text
[########--------------------]  72/241  29.88% | elapsed 14m08s | ETA 33m12s | filling=1.7958
```

它显示已完成点数、百分比、累计时间、按当前平均速度估计的剩余时间以及刚完成的 filling。开始计算下一点前会先清掉动态进度条，因此每个 filling 原有的 `chi/r/iteration/time` 详细输出仍然保留。

若不希望显示进度条，可运行：

```bash
python filling_scan.py --no-progress
```

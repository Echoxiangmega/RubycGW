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

many-body 数值默认参数为

```text
nk = 6x6
nw = 60
nOmega = 12
vertex-stage = mt
momentum-backend = fft
GW mixing = 0.20
vertex mixing = 0.20
GW tol = 1e-8
vertex tol = 1e-8
```

早期 strong-coupling filling scan 曾把 mixing 设成 `GW=0.08`, `vertex=0.10`，结果即使在容易收敛的 V-ramp 点也需要约两百次 iteration。现在由于默认已经有 V-ramp 和 filling continuation 提供 nearby seed，mixing 提高到 `0.20/0.20`，同时保持 `1e-8` tolerance 不变。也就是说加速来自更少的 fixed-point iteration，而不是放宽收敛精度。

若某个强耦合点出现明显振荡，可手动降低：

```bash
python filling_scan.py --gw-mixing 0.10 --vertex-mixing 0.10
```

## 3. 运行

完整 241 点扫描：

```bash
python filling_scan.py
```

粗扫描：

```bash
python filling_scan.py --num-fillings 61
```

只检查整数 filling：

```bash
python filling_scan.py --fillings 1 2 3 4 5
```

full cGW：

```bash
python filling_scan.py --vertex-stage both
```

`both` 会先求 GW+MT，再用当前 filling 已收敛的 MT vertex 作为 full MT+AL 的初值。

## 4. 强耦合下的 anchor + V-ramp continuation

不能从 `filling=0.05, V=3` 直接 cold start：若第一个 GW 点没有收敛，后续点也没有可用 continuation seed，最终会得到毫无物理意义的巨大 fixed-point iterate。

因此当前默认流程改为：

1. 在用户要求的 filling 网格中找到最接近 `--anchor-filling` 的点；默认 anchor 是 `3.0`。
2. 在 anchor filling 固定粒子数，先做 interaction continuation。对 `V=3` 默认路径为

```text
0.1 -> 0.25 -> 0.5 -> 0.75 -> 1.0 -> 1.5 -> 2.0 -> 2.5 -> 3.0
```

3. 每一级都把上一级 converged GW solution 作为下一步初值。
4. 到达 target V 后，再求 anchor 的 eta vertex。
5. 由同一个 anchor solution 分成两个独立 branch：

```text
anchor -> lower fillings
anchor -> higher fillings
```

这样 higher branch 不会使用 lower branch 最末端的状态作为初值。

可改变 anchor：

```bash
python filling_scan.py --anchor-filling 2.0
```

可指定自己的 interaction ramp：

```bash
python filling_scan.py --v-ramp-values 0.1 0.3 0.6 1.0 1.5 2.0 2.5 3.0
```

如需故意测试 target V 的 cold start：

```bash
python filling_scan.py --no-v-ramp
```

`--no-continuation` 会同时关闭参数点 continuation；主要用于诊断，不推荐做强耦合 production scan。

若 V-ramp 在某一级 GW 已不能收敛，程序会在那里停止，而不会继续 cold-start target V。`v_ramp.csv` 会记录最后能够到达的 interaction、iteration count、chemical potential 和 wall time。

## 5. GW / vertex 失败时怎样处理

一个没有收敛的 GW fixed point 不能作为 covariant response 的 background。因此现在的安全规则是：

```text
GW not converged -> skip vertex -> chi/r_eff = NaN
```

不会再把 `10^100`、`10^150` 一类发散的 fixed-point iterate 写成 susceptibility。

如果 GW 收敛但某个 eta vertex 未收敛，则只把该 channel 的 susceptibility 与 `r_eff` 记为 NaN，并保留上一个 converged vertex 作为相邻 filling 的 continuation seed。

这使 CSV 中的有限数值都具有明确的 convergence status。

## 6. 输出

每次运行自动建立

```text
results/filling/<timestamp>/
```

并输出：

```text
filling_scan.csv
v_ramp.csv
settings.json
r_eff_vs_filling.png
chi_vs_filling.png
delta_r_vs_filling.png
```

`filling_scan.csv` 最终始终按 filling 从小到大排列，即使实际计算顺序是从 anchor 向两侧展开。新增字段包括：

```text
scan_branch
vertex_skipped_because_GW_failed
GW_converged
selected_plus_converged
selected_minus_converged
```

同时保存 chemical potential、actual filling、iteration count 与各阶段 wall time。

## 7. 图的解释

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

未收敛点不会进入曲线。

注意这仍是 normal-state quadratic-response criterion。若要严格判断深处 ordered phase 的最终基态，需要进一步做 finite-source / symmetry-broken free-energy comparison。

## 8. 终端进度条

filling scan 默认显示单行进度条，例如：

```text
[########--------------------]  72/241  29.88% | elapsed 14m08s | ETA 33m12s | filling=1.7958
```

V-ramp 本身会逐级打印 `V`, convergence, iteration count, chemical potential 和 time。正式 filling scan 开始后，进度条 ETA 只统计 filling 点，不把一次性的 V-ramp 初始化时间混入平均每点耗时。

若不希望显示进度条：

```bash
python filling_scan.py --no-progress
```

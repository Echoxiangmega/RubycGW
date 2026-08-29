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

GW 首选 mixing 为 `0.20`。如果该固定点没有收敛，程序不会立刻失败，而是从**同一个最近已收敛 seed** 自动重试：

```text
0.20 -> 0.15 -> 0.10 -> 0.05
```

这使弱耦合点保持快速，而强耦合点才自动使用更保守的 under-relaxation。可自行改变 fallback：

```bash
python filling_scan.py --gw-retry-mixings 0.12 0.08 0.04
```

vertex mixing 默认仍是 `0.20`，目前 vertex 尚未做相同的 adaptive retry。

## 3. 运行

完整扫描：

```bash
python filling_scan.py
```

快速诊断 half filling：

```bash
python filling_scan.py --fillings 3 --nk 4 --nw 55 --nomega 12
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

对 `V=3`，默认 interaction path 现在加密为

```text
0.1 -> 0.25 -> 0.5 -> 0.6 -> 0.7 -> 0.75 -> 0.9 -> 1.0
    -> 1.25 -> 1.5 -> 1.75 -> 2.0 -> 2.25 -> 2.5 -> 2.75 -> 3.0
```

每一级先用 `mixing=0.20`。若失败，则同一级 V 自动按 `0.15 -> 0.10 -> 0.05` 重试；每次 retry 都重新从**上一级 converged GW solution** 出发，而不是从失败 iterate 继续。

例如终端可能显示：

```text
[V-ramp  6/16, try 1/4] V=0.7500 mix=0.200 converged=False ... err=...
[V-ramp  6/16, try 2/4] V=0.7500 mix=0.150 converged=False ... err=...
[V-ramp  6/16, try 3/4] V=0.7500 mix=0.100 converged=True  ... err=...
```

只有当该 V 的所有 mixing 都失败时，V-ramp 才停止。

可自定义 V path：

```bash
python filling_scan.py --v-ramp-values 0.1 0.25 0.5 0.6 0.7 0.75 1.0 1.5 2.0 2.5 3.0
```

到达 target V 后，由同一个 anchor solution 分成两个独立 branch：

```text
anchor -> lower fillings
anchor -> higher fillings
```

## 5. GW residual 与失败处理

`GWResult` 现在额外保存

```text
final_error
```

即最后一次 self-consistency iteration 使用的 fixed-point error。对于 converged 点应满足

\[
\text{final_error}<\text{GW tol}.
\]

`v_ramp.csv` 每一个 V / retry attempt 都保存：

```text
step
attempt
V
mixing
converged
iterations
final_error
mu
actual_filling
runtime_s
```

因此可以直接区分：

```text
err ~ 1e-7 : 已很接近，只差少量 iteration
err ~ 1e-3 : 收敛很慢或振荡
err ~ O(1) : fixed-point iteration 明显不稳定
```

正式 filling scan 同样会 adaptive retry GW，并保存

```text
GW_final_error
GW_mixing_used
GW_attempts
```

如果所有 GW retry 都失败，则

```text
GW not converged -> skip vertex -> chi/r_eff = NaN
```

不会把发散 iterate 当作 susceptibility。

## 6. 输出

每次运行自动建立

```text
results/filling/<timestamp>/
```

主要文件为

```text
filling_scan.csv
v_ramp.csv
settings.json
r_eff_vs_filling.png
chi_vs_filling.png
delta_r_vs_filling.png
```

`filling_scan.csv` 最终始终按 filling 从小到大排列，即使实际计算顺序是从 anchor 向两侧展开。

## 7. 图的解释

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

## 8. 终端进度条

正式 filling scan 默认显示：

```text
[########--------------------]  72/241  29.88% | elapsed 14m08s | ETA 33m12s | filling=1.7958
```

V-ramp 初始化会逐级打印 V、mixing、iteration count、`final_error`、chemical potential 和 runtime。关闭 filling 进度条：

```bash
python filling_scan.py --no-progress
```

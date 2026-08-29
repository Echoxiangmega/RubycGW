# Automated Convergence Scan

`convergence_scan.py` 用于系统检查 `nw`、`nOmega` 和 `nk x nk` 三个主要 cutoff。当前推荐顺序仍然是

```text
nw -> nOmega -> nk
```

但脚本现在支持 continuation/warm start 和不同 vertex stage，因此大网格扫描不再必须每个点从零开始。

## 1. 三种 vertex stage

### `--vertex-stage both`

完整分层诊断：

\[
G_0G_0\rightarrow GG\rightarrow GW+MT\rightarrow full\ cGW.
\]

full cGW 会直接从已经收敛的 MT vertex 开始，而不是重新从 bare `K_eta` 开始。这是默认模式，适合最终 convergence test 和关键物理点。

```bash
python convergence_scan.py --scan nomega --vertex-stage both
```

### `--vertex-stage mt`

只算到

\[
GW+MT.
\]

这是推荐的大范围快速扫描模式。当前 Ruby 参数测试中 AL1/AL2 对最终 susceptibility 的修正远小于 MT，但 AL 是最昂贵的步骤。因此探索 `V`、filling、hopping 或粗略 convergence 时可先用：

```bash
python convergence_scan.py --scan nk \
  --vertex-stage mt \
  --base-nw 64 --base-nomega 16 \
  --nk-values 4 6 8
```

最终关键点仍应使用 full cGW 复核，不能把 `GW+MT` 标成 full cGW。

### `--vertex-stage full`

跳过单独 MT-only 输出，只求 full cGW。若前一个扫描点存在 compatible full vertex，则从前一点的 full vertex continuation。

## 2. continuation 会复用什么

只要 fermionic array shape 不变，脚本默认把前一个点的收敛解作为下一个点的初值。

### `nOmega` 扫描

`nw` 和 `nk` 不变，因此可以直接复用：

- noninteracting `G0`（它根本不依赖 `nOmega`）；
- `mu_GW`；
- `Sigma_H`；
- `Sigma_GW(k,iw)`；
- MT/full `Gamma_eta(k,iw)`。

这通常能明显减少 GW 和 vertex iteration 数。

### `nw` 扫描

fermionic frequency array 长度改变，因此当前不直接复用 `Sigma_GW` 或 `Gamma`。以后可以进一步实现 frequency embedding，但当前为了透明性不做插值。

### `nk` 扫描

momentum array shape 改变，因此当前也不直接复用旧 self-energy/vertex。若未来加入 momentum interpolation，应单独验证不会引入系统误差。

如果想检查 warm start 是否改变最终 fixed point，可以关闭 continuation：

```bash
python convergence_scan.py --scan nomega --no-continuation
```

warm start 只改变初值，不改变方程或 convergence tolerance。

## 3. q=0 eta Hartree 项

在 time-reversal symmetric normal state 中，static q=0 density-eta response 数值上已经验证到机器精度为零。默认仍保留 Hartree 作为 symmetry diagnostic；大量扫描确认后可以用

```bash
--skip-hartree
```

略去这一项。它不是主要性能瓶颈，所以第一次研究新参数区时仍建议保留。

## 4. 内部卷积优化

当前版本相比最初 reference code 做了两项不改变公式的优化：

1. 不再为每个 internal `Q` 构造完整 zero-padded `G(k+Q)`；只处理实际有效的 Matsubara slice。
2. 每个 vertex iteration 只计算一次

\[
X(k)=G(k)\Gamma(k)G(k),
\]

并让 Hartree、MT、AL1、AL2 共用同一组 `X(k+Q)` shift 和 internal-Q loop。

特别是旧版本中 MT 和 AL 会分别重新计算/shift `X`，现在这部分重复工作已经移除。

## 5. 常用命令

### 收敛 `nw`

```bash
python convergence_scan.py --scan nw \
  --vertex-stage both \
  --nw-values 16 24 32 48 64 96
```

### 收敛 `nOmega`

`nOmega` 扫描最适合 continuation：

```bash
python convergence_scan.py --scan nomega \
  --base-nw 64 --base-nk 4 \
  --nomega-values 4 6 8 12 16 24
```

### 收敛 `nk`

大 `nk` 的 full AL 很昂贵。建议先用 MT 快扫：

```bash
python convergence_scan.py --scan nk \
  --vertex-stage mt \
  --base-nw 64 --base-nomega 16 \
  --nk-values 4 6 8 10
```

随后只对需要的 `nk` 点用 `--vertex-stage both` 做 full cGW。

## 6. 输出和 timing

CSV 现在除 susceptibility 和 convergence flags 外，还记录：

```text
time_bare_s
time_GW_s
time_MT_s
time_full_s
runtime_s
```

因此可以直接判断瓶颈来自 GW、MT 还是 full AL vertex。

figure 使用当前请求 stage 的 response：`mt` 模式画 `GW+MT`，`full/both` 模式画 full cGW，不会把 MT-only 结果误标为 full cGW。

## 7. 哪些量在参数扫描中不需要重复计算

这对以后相图非常重要。

### 扫 `V`，固定 hopping、T、filling 和 grid

noninteracting Hamiltonian 与 `G0` 完全不依赖 `V`，因此

\[
\chi^{G_0G_0}
\]

只需计算一次。`K_plus/K_minus`、momentum grid 和 bare hopping matrix 也只需建立一次。相邻 `V` 点的 interacting GW 和 vertex 则可以 continuation。

### 扫 filling

`h0`, `V(q)`, grid 和 eta vertices 不变，但 noninteracting chemical potential 和 `G0` 会随 filling 改变，因此 bare response 需要更新。相邻 filling 的 GW/self-energy/vertex 仍可作为初值继续。

### 扫 temperature

Matsubara frequency 数值改变，所以不能把旧解当成严格相同对象，但若 array shape 相同，旧 self-energy/vertex 仍可作为 continuation guess；必须重新收敛。

### 扫 hopping `ti/t1/t2`

`h0` 改变，bare response 必须重算；`V(q)` 和 eta vertices 若 interaction/bond convention 不变可以复用。小步扫描时旧 GW/vertex 仍可作为初值。

### 扫 `nk` 或 `nw`

array shape 改变，当前不直接复用旧 interacting arrays。

## 8. 推荐生产流程

大范围探索：

```text
4x4 or 6x6
nw ~ 48-64
nOmega ~ 12-16
GW+MT only
continuation on
```

找到 same/opposite 接近竞争、相边界或文章关键点后：

```text
increase nk / cutoffs
full cGW (MT + AL1 + AL2)
check no-continuation gives same fixed point
```

这样比所有参数点都暴力 full cGW 更有效，同时保留最终结果的严格复核。

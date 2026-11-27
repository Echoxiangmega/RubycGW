# Ruby hex HF: 固定 Vt=0 扫描 (Vtri, n)

`scan_ruby_hex_vtri_filling.py` 是之前 `ruby_hex_hf.py`（纯 Hartree–Fock，不包括超导配对）的补充二维扫描器。

**模型**：`V_t1=V_t2=0`，但 **hopping** `t1,t2` 仍按命令行输入，不是零。`V_tri` 仅作用在两类三角形内部的密度键。`n` 是 **每六位点原胞的平均 spinless 电子数**，满足 `0<n<6`；由每次自洽的化学势调整实现固定 filling。例如 ring3 的超胞有 3 个原胞，平均粒子数为 `3*n`（一般可非整数，因为它是无限周期体系中的一个超胞，对 k 网格取平均）。

脚本中使用相同的 Hartree 与复数 Fock 序参数，允许实数 CDW、SD/ISD-like BOW、loop current、面内 pseudospin CO 和共存序；每个点通过多初态找收敛的最低 **Helmholtz 自由能 F**。不收敛点不会伪装成某一相。

## 安装与目录

确保下列三个文件处于同一目录：

- `ruby_hex_hf.py` — 之前的平均场引擎
- `plot_ruby_hex_phase.py` — 修正过的平移调制诊断
- `scan_ruby_hex_vtri_filling.py` — 新扫描器

依赖：`numpy scipy matplotlib`。

## 首次小范围试跑

```bash
OPENBLAS_NUM_THREADS=1 python scan_ruby_hex_vtri_filling.py \
  --period ring3 --symmetry none \
  --t 1 --t1 0.5 --t2 0.5 --nk 8 8 \
  --Vtri-min 0 --Vtri-max 1 --Vtri-step 0.25 \
  --n-min 1.25 --n-max 1.75 --n-step 0.25 \
  --temperature 0.04 --max-iter 400 --tol 1e-7 \
  --output results/hex18_Vtri_n
```

## 更宽范围的相图

```bash
OPENBLAS_NUM_THREADS=1 python scan_ruby_hex_vtri_filling.py \
  --period ring3 --symmetry none \
  --t 1 --t1 0.5 --t2 0.5 --nk 12 12 \
  --Vtri-min 0 --Vtri-max 3 --Vtri-step 0.2 \
  --n-min 0.75 --n-max 3.25 --n-step 0.125 \
  --temperature 0.03 --max-iter 700 --tol 1e-8 \
  --random-starts 3 --output results/hex18_Vtri_n_fine
```

`--period hex2x2` 对应 24-site 的 2×2 Bravais 超胞、三个 M 波矢；`ring3` 对应 18-site 六角环和 K 相关调制。默认 `--symmetry none` 不施加点群约束，保留超胞周期，并允许面内 pseudospin CO 打破 C3。`c3` 或 `c6` 会投影掉这些面内 CO 分量，不适用于包含 `x_even/x_odd/y_even/y_odd` 的扫描。结果端会从占据数投影出 `pseudospin_x/y_even/odd` 和 even/odd 总幅度，并用于相分类；这与 seed 名称分开，分类依据是收敛后的实际占据数。

可以用 `--Vtri-values 0 .25 .5 1 1.5 2` 和 `--n-values 1.0 1.25 1.5 1.75 2` 指定非均匀网格，覆盖连续的相关区间。

## 扫描效率与收敛

默认包含 normal / 面内 CO（x/y 的 even/odd）/ CDW / SD / ISD / real BOW / 同反向 LC / random 及独立 LC modes，多初态严格比较最低自由能。扫描不再使用混合序参量 seed。它支持两个方向的 warm start（前一个 V 和相邻 n），以及蛇形参数扫描；但 warm start 只用于初值，不会强制某一序。

快速定位相边界可暂时使用：

```bash
python scan_ruby_hex_vtri_filling.py \
  --symmetry none \
  --seeds normal x_even x_odd y_even y_odd cdw sd isd lc_same lc_opposite \
  --random-starts 0 --no-auto-lc-modes
```

**快速模式可能遗漏有序态**。正式图最好恢复默认 LC modes、多随机起点并提高 k 网格、降低温度后复查近简并区域。

## 输出与恢复

完成每个点即自动写入：

```text
results/hex18_Vtri_n/
    scan_settings.json
    scan.json
    scan.csv
    phase.png
    phase.svg
    charge_q_rms.png
    real_bond_q_rms.png
    current_rms_recomputed.png
    pseudospin_co_even_amplitude.png
    pseudospin_co_odd_amplitude.png
    J_tri_rms.png
    F_per_primitive.png
    point_0000/
        candidates.csv
        best/
            summary.json
            state.npz
            sites.csv
            bonds.csv
    ...
```

原始 `point` 的对应关系可从 `scan.csv` 的 `V_tri`、`filling_per_primitive`、`point` 查找。`best/state.npz` 用于之后画对应的周期 bond/current pattern。`--save-candidates` 可额外保存其他收敛解，默认只完整保存最优态。

中途停止后使用 **完全相同扫描参数** 加 `--resume` 即可；对未收敛的点再次尝试加 `--retry-invalid`。要从已有文件重画相图：

```bash
python scan_ruby_hex_vtri_filling.py --plot-only --output results/hex18_Vtri_n
```

`--plot-only` 仅重画已经保存的分类和强度图，不改变分类阈值，也不重新计算 HF。

## 重要的物理限制

- 相图只是在所选 `ring3` / `hex2x2` 超胞周期与 `c3/c6/none` 对称性约束下的 **平均场结果**。
- 对 SD/ISD 的命名是 Ruby real-bond form-factor 的正负重叠，**不是 kagome SD/ISD 的严格同一群表示**。
- `n=1.5` 不是对任意 `t,t1,t2` 都成立的 van Hove filling；扫描时要用实际 hopping 的非相互作用能带确认。
- 在固定 n 下相变附近，如出现非凸自由能或负压缩率，需额外做 Maxwell/凸包检查以识别相分离。简单逐点自由能最低解不自动排除相分离。
- 新脚本不调用 ED、GW、cGW；与 RubycGW 原始模型的轨道坐标 convention 应在需要时通过 gauge/原胞位移映射确认。

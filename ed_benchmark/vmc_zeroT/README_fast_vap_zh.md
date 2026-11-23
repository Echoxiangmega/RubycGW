# Ruby 超胞快速 VAP（从已有 Raw V-scan checkpoint 开始）

本包包含两个用于实际超算计算的 Python 文件：

- `run_supercell_vap_fast.py`：新入口；直接读取现有 Raw `final_params.npz`，依次运行 Tint、Tint+C3v、可选时间反演 TR-VAP，按阶段和每 10 次迭代保存 checkpoint。
- `projected_fast_kernels.py`：投影态的 rank-3 Woodbury local-energy、投影态持久逆矩阵缓存 Metropolis sampler。两者都根据既有 `supercell_ansatz.py` 的 backflow、Jastrow、Bloch 约定构造。
- `test_projected_kernels.py`：不依赖 NetKet 的 JAX/NumPy 代数回归测试。它验证空间/TR 投影 local energy 及采样 log-probability 更新。

**放置方式**：把两个运行 Python 文件复制到当前 RubycGW 仓库的 `ed_benchmark/vmc_zeroT/`，即 `run_supercell_large.py` 同目录。不要覆盖旧的 `run_supercell_vscan.py` 或 `run_supercell_large.py`。

**现有环境**：请使用之前运行 Raw 的 NetKet、JAX、Flax、Optax、NumPy、SciPy 环境。建议安装的 NetKet 版本与之前 Raw 计算一致。代码未在本地完成实际 NetKet 集成测试（本地无 NetKet），**先在超算上进行少步、严格一致性检查再跑正式任务**。

## 1. 从一个现有 72-site Raw 结果开始

```bash
cd /path/to/RubycGW
RAW=/scratch/vscan/L2x2/seed_lc_opposite/continuation_up/V_p5p5/large_raw_backflow__lc_opposite
OUT=/scratch/fast_vap/L2x2/V_p5p5/seed_lc_opposite

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python ed_benchmark/vmc_zeroT/run_supercell_vap_fast.py \
  --L1 2 --L2 2 --nf 24 \
  --t 1 --t1 0.5 --t2 -0.5 --V 5.5 \
  --seed lc_opposite --raw-stage "$RAW" --output "$OUT" \
  --irreps A1 A2 E --restore-tr \
  --tint-iter 50 --spatial-iter 30 --tr-iter 30 \
  --samples 2048 --spatial-samples 512 --eval-samples 8192 \
  --sampler projected-cache --trainer shared_sr \
  --fast-local --check-roots 3 --check-gradient \
  --checkpoint-every 10 --strict
```

这是**严格验证首轮**（`--strict` 使检查失败时直接报错，不会悄悄使用错误内核）。跑通后如需自动回退到标准 sampler/standard local energy，可去掉 `--strict`。梯度检查很贵，验证通过后，后续扫描可省略 `--check-gradient`（保留默认的 direct-local-energy 根检查）。

如果所有 Raw 扫描点都在同一个根目录，还可以使用 `--scan-root /scratch/vscan` 代替 `--raw-stage`，脚本按原 `run_supercell_vscan.py` 的目录名自动定位：

```
scan_root/L2x2/seed_lc_opposite/continuation_up/V_p5p5/
  large_raw_backflow__lc_opposite/final_params.npz
```

## 2. 更保守的调试命令

先使用 **18-site**：

```bash
python ed_benchmark/vmc_zeroT/run_supercell_vap_fast.py \
  --L1 1 --L2 1 --nf 6 --V 2 \
  --t 1 --t1 0.5 --t2 -0.5 \
  --seed lc_opposite \
  --raw-stage /path/to/existing/18site/large_raw_backflow__lc_opposite \
  --irreps A1 --restore-tr \
  --tint-iter 2 --spatial-iter 2 --tr-iter 2 \
  --samples 128 --spatial-samples 128 --eval-samples 512 \
  --n-chains 8 --check-roots 3 --check-gradient \
  --sampler projected-cache --strict \
  --output /tmp/ruby_fast_vap_smoke
```

如果还未有 18-site 的 Raw checkpoint，可以用 72-site 已有 checkpoint 做前两步测试；脚本不会产生或训练新的 Raw reference。

## 3. 加速项与验证

| 项目 | 本脚本采用的实现 | 与旧 Raw 内核的关系 |
|---|---|---|
| 加速 local energy | 将原有每个对称 image 的 Woodbury `HΨ_g/Ψ_g` 按投影振幅相干加权 | **精确相同物理波函数**，不是近似合并 |
| 加速 sampler | 每条链保存每个 image 的占据轨道矩阵逆；接受跳跃用 rank-3 更新 | 与投影后的 `|Ψ|²` 做 MH，**不是**分别采样 Raw images |
| Shared-SR | 调用已有 `run_supercell_large._make_shared_sr_driver` | 与先前 Raw 扫描的 shared Jacobian 思路一致 |
| QR | 默认仍按原 VAP 使用 QR，减少数值风险 | `--qr-free-spatial` 为可选实验项，TR 始终使用 QR |
| TR | 先得到 18 个空间 image 的振幅和 HΨ，再用复共轭组合 | **不重复做另外 18 个 determinants** |
| 失败处理 | 比较投影高速 local-energy 与完全重新计算 determinant 的标准结果；采样器比较累计 log 概率比值 | 默认回退标准实现；`--strict` 则报错 |

`projected_batch=scan`（默认）按 image 累加，避免把很大的 `18×batch` 木桶式分配全部摊开。如果显存足够，可尝试 `--projected-batch flatten`，两种路径均在自带测试中验证数学等价。

**验收标准**：首轮 `--strict --check-gradient` 应通过；检查 `summary.json` 的 `fast_local_check.passed=true` 和 `sampler_check.passed=true`，对比旧 VAP 在同一参数、相同 seed 与 MC samples 下的 `mc_E`、结构因子及 `time_per_iteration_s`。正式比较速度时，需要扣除首次 JIT 编译、初始化和最终观测量统计时间。不能根据 18 images 就预设加速倍数。

## 4. 结果目录

```
OUT/
  run_config.json
  fast_tint__Gamma/
    final_params.npz
    summary.json
    history.json
  irrep_pav_screen.json  # 仅启用 --screen-irreps 时生成，筛选属于启发式
  fast_spatial__A1/
    final_params.npz
    summary.json
    history.json
  fast_tr__A1/
    final_params.npz
    summary.json
    history.json
  fast_spatial__A2/
  fast_spatial__E/   # 二维 E 表示的中心投影
  fast_tr__A2/
  fast_vap_summary.csv
```

每阶段计算时还保存 `progress_params.npz` 与 `progress.json`，记录已完成的 iteration。`--screen-irreps` 默认关闭，正式相图建议优化 A1、A2 和 E 全部三个 Γ 扇区。需要省时可显式启用 `--screen-irreps`，但 PAV 的初始能量排序不能保证 VAP 后的排序；因此仅用于初筛。再次使用原命令运行，会读取这些中途检查点继续完成目标 iteration 数；阶段全部成功后删除临时进度文件。

**不会修改原有 Raw 文件**，所有结果写入单独指定的 `--output`。`--resume` 默认开启。

## 5. 物理和计算范围

- 超胞的平移由原 Bloch ansatz 内置。Tint 是 6-site 原胞之间的残余平移投影，共 3 images；Tint+C3v 是 18 images；TR 以复共轭恢复。
- 现在支持内部 Γ 扇区的 C3v **A1、A2、E** 三种表示。E 使用二维不可约表示的**中心投影算符**，得到整个 E 子空间中的一个变分态，而非强制挑选一个分量。但这仍只覆盖内部 Γ：对于有限 primitive-cell momentum（K/K′）或者其他超胞总动量，必须另算相关动量扇区。只扫描 Γ 的 A1/A2/E **不能被称为无偏的完整零温相图**。
- 需要真实的 C3v 空间群；非 C3v 兼容的 supercell 几何将拒绝投影。`--L1 2 --L2 2` 适用。
- `TR-VAP` 的一阶环电流为零；相图使用 `S_LC_same`、`S_LC_opposite` 和归一化后的 `S_CO`，并做尺寸标度。
- 这里继续使用原四通道 bond-aware complex backflow，不会改变 Hamiltonian 或引入之前讨论的 `dEV` 新 backflow。加速与新 ansatz 应分开做以保证 benchmark 公平。
- 截止当前包的测试环境：通过了 **A1/A2/E×空间/TR×scan/flatten 的 NumPy/JAX local-energy 测试**，以及 **A1/A2/E×空间/TR 的 cache sampler 概率比值测试**。**未测试 NetKet driver 的端到端集成，不能因此宣称已有超算实测加速比。**

## 6. 运行回归测试（不需要 NetKet）

```bash
cd ed_benchmark/vmc_zeroT
OPENBLAS_NUM_THREADS=1 python test_projected_kernels.py
```

## 7. 常见问题

- `Checkpoint occupations per k differ`：某个 seed 的原 Raw `n_occ(k)` 与现在重新生成的不一致，应保持原填充和动量块占据，不能自动重排。
- `Insufficient non-nodal samples`：强制投影扇区几乎没有重叠，或测试样本落在精确投影节点；改换相关 sector、Raw 分支，必要时增加 `--check-roots` 或采样候选数，而不是强制忽略。
- `projected-cache probability-ratio validation failed`：先改用 `--sampler standard`，保留快速 local-energy 和 Shared-SR。
- `projected local energy != direct`：先使用 `--no-fast-local` 获得标准结果；检查 site maps、Fock parity、NetKet 版本和模型约定。
- 显存不足：减少 `--chunk-size`、`--spatial-samples`、`--n-chains`，并保持 `--projected-batch scan`。


## 8. 未知基态对称扇区：如何使用

不要仅依据上一小尺寸 ED 已知 irrep，假定全部尺寸和相互作用下仍属于相同的表示。超算相图建议先用 `--irreps A1 A2 E --no-screen-irreps` 检查所有内部 Γ 下的普通 C3v 扇区，比较各自 VAP 后的能量与结构因子，确保 E 的二维投影不被忽略。

**限制**：本程序的 `Tint` 使用 3 个内部平移的 Γ character `(+1,+1,+1)`。所以即使 `A1 A2 E` 全扫，也并未包括 K/K′ 等非 Γ 平移 character，也没有枚举不同 supercell 总动量。若基态发生不同动量扇区的能级交叉，当前最低 Γ 能量只能作为**受限变分上界**。需要针对非 Γ 建立平移/小群的适当投影及 TR 伴随扇区，不能套用 Γ 下的 A1/A2/E 字符。

此外，`--screen-irreps` 的 PAV 初筛依赖一个固定初始变分态，可能将尚未优化就偏高的 E 等扇区误筛，因此生产级相界附近推荐关闭；少量起始参考态也可能对某一扇区投影权重极低，此时应改换或扰动 Raw 参考态，而不是认定扇区物理上缺失。
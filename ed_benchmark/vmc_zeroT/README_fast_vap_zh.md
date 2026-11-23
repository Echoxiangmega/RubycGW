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
  --fast-local --check-woodbury --check-roots 3 --check-gradient \
  --checkpoint-every 10 --strict
```

这是**严格验证首轮**（`--strict` 使检查失败时直接报错，不会悄悄使用错误内核）。跑通后如需自动回退到标准 sampler/standard local energy，可去掉 `--strict`。梯度与直接 Woodbury 根检查都比较耗时。验证通过后批量扫描可省略 `--check-gradient` 和 `--check-woodbury`；两者现在默认关闭，不影响 fused Woodbury 加速的使用。

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
| 失败处理 | 比较投影高速 local-energy 与完全重新计算 determinant 的标准结果；采样器比较累计 log 概率比值 | 只有启用 `--check-woodbury` 时才做直接能量对照并在失败时回退标准实现；`--strict` 则报错。默认跳过这项验证 |

`projected_batch=scan`（默认）按 image 累加，避免把很大的 `18×batch` 木桶式分配全部摊开。如果显存足够，可尝试 `--projected-batch flatten`，两种路径均在自带测试中验证数学等价。

**验收标准**：首轮 `--strict --check-woodbury --check-gradient` 应通过；仅在实际启用相应校验时才检查 `summary.json` 的 `fast_local_check.passed=true` 和 `sampler_check.passed=true`，对比旧 VAP 在同一参数、相同 seed 与 MC samples 下的 `mc_E`、结构因子及 `time_per_iteration_s`。正式比较速度时，需要扣除首次 JIT 编译、初始化和最终观测量统计时间。不能根据 18 images 就预设加速倍数。

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

## 9. v2 融合 Woodbury 和 LC structure factor

新版额外依赖同目录的 projected_fused_kernels.py。以下参数现在是默认值：

- --local-kernel fused：只构造一次 Bloch orbitals、backflow W 和 Jastrow C，再依次计算每个 symmetry image 的 rank-3 hopping ratios；原有波函数形式不变。
- --lc-estimator onehop：利用同一批 one-hop ratios 计算复数 J_same_loc 和 J_opposite_loc，再以 <J^2> = <|J_loc|^2> 计算 S_LC；无需构造 J@J。
- --observable-chunk 16：限制最终 S 计算的 batch；可在 8/16/32 中比较。
- --check-currents：选取实际采样构型逐个 hopping 重算整个投影波函数，验证 one-hop 电流方向和 TR 奇偶性。
- --local-kernel legacy 和 --lc-estimator netket：保留以前的两种计算路径，用于公平对照。
- --allow-missing-observables：仅调试使用，正式相图不要启用。若缺失 S，默认保存最终 checkpoint 和错误诊断后报错；下一次运行从保存参数重新测量 S，不会重复全部优化。

首次运行建议先选 72-site Raw checkpoint，分别使用 5+5+5 步的 Tint、Spatial 和 TR，并启用：

    --irreps A1 A2 E --no-screen-irreps --restore-tr
    --local-kernel fused --lc-estimator onehop
    --sampler standard --trainer shared_sr --check-gradient --strict

验证后再测试 --sampler projected-cache。若想比较旧路径，从同一个 Raw checkpoint 在**不同输出目录**重新运行：

    --local-kernel legacy --lc-estimator netket --sampler standard

每阶段 summary.json 和 fast_vap_summary.csv 保存 S_CO、S_CO_over_Ncell、
S_LC_same、S_LC_opposite、connected 版本、一阶电流及统计误差，以及
LC_estimator、lc_direct_check_max_abs_error、local_kernel_family、
time_per_iteration_s、steady_time_per_iteration_s、fast_local_check、sampler_status。
中途 progress_params.npz 只保存参数与进度，不执行昂贵的 S 测量。

无 NetKet 的纯 JAX 代数测试覆盖 A1/A2/E、TR on/off、多 k blocks、不同
symmetry images 下的 hopping 方向；局域能量和 LC local estimator 相对
直接构型求和的最大绝对误差约 3.3e-14。完整 NetKet + SR 的端到端运行、
实际加速比及大体系数值稳定性仍需在超算节点验证。若接近投影节点或
单个 image 的 Slater 矩阵严重病态，务必使用严格的数值检查；失败时切回
legacy，不可直接用错误的结构因子绘制相图。


## 10. 3x3 CPU 性能：区分训练和 eval 耗时

训练进度条中的秒/步 **不包含** 最终 Monte Carlo 重新采样与 S 的计算。
如果采用标准 Metropolis sampler，VAP eval 每个提议需要对投影波函数重新求值；
这没有获得 Raw 的专用 cached Woodbury sampler 加速。--fast-local 和
--local-kernel fused 只加速局域能量计算；--lc-estimator onehop 加速 S_LC，
但不会自动加速标准 sampler。

现在每个阶段会实时打印并写入 summary.json：

- time_eval_sampling_s：vstate.sample() 的真实等待时间；
- time_eval_energy_s：vstate.expect(Hrun) 的时间（fused Woodbury 或 fallback）；
- time_current_direct_check_s：独立完全重算 LC 的耗时（若开启检查）；
- time_eval_observables_s：包含 current check、S_CO、S_LC 的时间；
- time_eval_total_s：上述 eval 总耗时。

默认 --check-currents 仍为 True，现已改为批量而不是每 hopping
一个 model.apply；可调 --current-check-batch 8。完成代表性计算的
直接校验后，批量 V 扫描可使用 --no-check-currents，且默认不进行
Woodbury direct-check。需要重新验证时加 --check-woodbury --strict。

对于 3x3 的多 irrep 扫描，新增：

    --eval-samples-intermediate 512 --eval-samples 4096

这样 Tint 和 Spatial 阶段只用 512 个最终评估样本，TR 阶段仍用
4096 个。若不设置 --eval-samples-intermediate，则保持原有行为：
每个阶段都使用 --eval-samples。粗采样适合筛选、排障，不适合用
低统计量做最终 S 的有限尺寸外推。

第一次验证 sampler 时使用 --sampler standard；
确认 projected-cache 的概率比值检查及统计结果后再选择
--sampler projected-cache。--check-gradient 也可在单个代表性
参数点充分验证后于整个 scan 中关闭，避免对每个 stage 都做昂贵的
标准 NetKet 梯度对照。两项检查都是可选诊断，关闭不会改变波函数。

现有正在运行的 Python 进程不会因为修改磁盘代码而自动采用这些改动。


## 11. 多 seed、多 V：Slurm array 并行运行 Fast-VAP

run_supercell_vap_vscan.py 是**调度程序**，不改变单点 Fast-VAP 的物理求解器。
slurm_vap_3x3_seed_array.sh 复制之前的 CPU 分区 C064M0256G、Conda 初始化
和 8 个 seed 的 Slurm 数组提交方式。

**每项 Slurm task 只负责一个 seed**。同一 task 内依次处理各个 V，
每个 V 都从自己原来的 Raw checkpoint 启动；VAP 参数不会自动跨 V warm-start，
与原始 Raw continuation 的机制不同。

先更新：

    cd "$HOME/RubycGW"
    git pull --ff-only origin main

默认脚本 SEEDS 包含 8 个名字，Slurm array 为 0-7。
测试 4 个 seed、一个 V 时，无需修改脚本：

~~~bash
cd "$HOME/RubycGW"
mkdir -p logs
sbatch --array=0-3 \
  --export=ALL,VAP_SEEDS="sd isd lc_same lc_opposite",V_MIN=3.6,V_MAX=3.6 \
  ed_benchmark/vmc_zeroT/slurm_vap_3x3_seed_array.sh
~~~

原 Raw 扫描的整个 V 区间（大量任务，确认分区时间与资源后再提交）：

~~~bash
sbatch --array=0-7 \
  --export=ALL,V_MIN=0.8,V_MAX=4.0,V_STEP=0.1 \
  ed_benchmark/vmc_zeroT/slurm_vap_3x3_seed_array.sh
~~~

可通过环境变量修改 sampler、sweep 和优化设置：
SAMPLER=standard/projected-cache, TINT_SWEEP, SPATIAL_SWEEP,
TINT_SAMPLES, SPATIAL_SAMPLES, EVAL_INTERMEDIATE, EVAL_SAMPLES,
TINT_ITER, SPATIAL_ITER, TR_ITER, CHUNK_SIZE, QGT_CHUNK 等。

先检查源文件和生成命令，不启动 NetKet：

~~~bash
python ed_benchmark/vmc_zeroT/run_supercell_vap_vscan.py \
  --L1 3 --L2 3 --nf 54 --t 1 --t1 -0.5 --t2 -0.5 \
  --V-values 3.6 --seeds sd isd lc_same lc_opposite --array-index 0 \
  --scan-root "$HOME/RubycGW/results/vscan_3x3_down" \
  --output-root "$HOME/RubycGW/results/vap_3x3_down_array" \
  --seed-amplitude 0.09 --dry-run
~~~

输出目录隔离为：

    results/vap_3x3_down_array/
      L3x3/seed_sd/continuation_down/V_p3p6/
        fast_tint__Gamma/summary.json
        fast_spatial__A1/summary.json
        fast_tr__A1/summary.json
        ...
        fast_vap_summary.csv

每个 seed 分支目录保存 vap_scan_settings.json 与 vap_scan_status.csv，
包括各 V 的 completed、failed 和 missing_raw 状态。
缺少 Raw checkpoint 默认跳过，若该 seed 没有任何可用 checkpoint，
则该 task 返回非零；可改用 --missing-raw error 强制立刻终止。
--keep-going 在某个 VAP 失败后继续其他 V，但最终仍返回非零状态。
已有阶段的复用由 Fast-VAP 的 --resume 实现。
注意已完成的结果不会因为增加 --tint-iter 等参数而自动继续训练；
若要比较不同 sampler/sweep，请使用不同 --output-root。

纯 Python 逻辑回归测试：
    python ed_benchmark/vmc_zeroT/test_supercell_vap_vscan.py

## 12. 关于对称投影后能量比 Raw 高

令 P 是选定的内部 Gamma/点群 irrep 投影算符。由于 [H,P]=0，
Raw 波函数分解到不同对称子空间后，其能量是各个投影子空间能量的
范数平方加权平均。某个被选定的子空间的能量可以比未投影 Raw 高，
也可以更低，并不违反变分原理。

完整 VAP 对固定 irrep 应比较：

- Raw 的 mc_E、mc_E_error（Hamiltonian、V、尺寸必须一致）；
- 当前 irrep 的 initial_mc_E（本阶段训练前的投影态能量）；
- 本阶段优化后的 mc_E、mc_E_error、history.json 能量曲线；
- A1、A2、E 三个 irrep 各自的结果，以及 TR 前后结果。

不要期望每个 irrep 或每个 TR 投影都单调降低能量。对固定投影 irrep，
充分优化的最低可能能量不会高于其同一 irrep 的初始投影态能量；
但是有限步随机 SR 迭代和 MC 误差不保证每次观察到的能量下降。
若某一扇区明显比 Raw 高很多，可能因为选定的 irrep 与低能态不匹配，
也可能是投影权重小、采样混合不充分、优化步数不足、学习率不稳定等。

实际结构因子统计必须来自相同的投影态。关闭 Woodbury check 仅节省
每阶段的 full-determinant 对照，不改变优化波函数和理论能量；
但也失去每个阶段实时发现内核错误的保险。因此建议在少量代表性
V/seed/irrep 上保留 --check-woodbury，然后批量扫描关闭。

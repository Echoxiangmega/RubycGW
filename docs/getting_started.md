# Getting Started

## 1. 本地安装

推荐使用独立的 conda 环境：

```bash
conda create -n rubycgw python=3.11
conda activate rubycgw
python -m pip install -r requirements.txt
```

在仓库根目录先运行测试：

```bash
python -m pytest -q
```

然后运行参考计算：

```bash
python run_ruby_cgw.py
```

## 2. 默认计算做了什么

`run_ruby_cgw.py` 当前按同一个目标 filling 依次计算：

1. `G0G0 (bare)`：非相互作用 Green's function 和 bare eta vertex；
2. `GG`：self-consistent GW 后的 dressed Green's function，但 vertex 仍取 bare `K_eta`；
3. `GW + MT`：在 converged GW 背景上解 Hartree + MT vertex equation；
4. `full cGW`：进一步加入 AL1 和 AL2。

最终分别输出

\[
\chi_+\equiv\chi_{\rm opposite},\qquad
\chi_-\equiv\chi_{\rm same},
\]

以及

\[
\Delta\chi=\chi_{\rm same}-\chi_{\rm opposite}.
\]

正的 `same-opposite` 表示当前参数下 same-current channel 的 susceptibility 更大。

## 3. 最常修改的参数

主程序开头有

```python
params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.10)
grid = MatsubaraGrid(nk1=4, nk2=4, nw=16, nOmega=6, T=0.05)
target_filling = 2.0
```

其中：

- `ti`：两个三角形内部 hopping；
- `t1`, `t2`：其余两类最近邻 hopping；
- `V`：所有 12 条 NN density-density interaction 的强度；
- `nk1`, `nk2`：二维 reduced Brillouin-zone 网格；
- `nw`：fermionic Matsubara index 使用 `n=-nw,...,nw-1`；
- `nOmega`：bosonic index 使用 `m=-nOmega,...,+nOmega`；
- `T`：温度；
- `target_filling`：每个六-site unit cell 的总粒子数。

## 4. 为什么 bare 和 GW 的 chemical potential 不一样

如果固定 filling，非相互作用和 interacting GW 体系通常需要不同的 chemical potential。因此代码先调用

```python
bare = solve_noninteracting(..., target_filling=target_filling)
```

得到 `mu0`，再独立求 self-consistent GW 的 `mu_GW`。这样 `G0G0`、`GG`、`GW+MT`、`full cGW` 的比较才是在同一 filling 下进行。

## 5. 第一次得到结果后不要立刻当成最终物理结果

默认 `4x4`, `nw=16`, `nOmega=6` 是 debug 网格。建议先确认：

- 所有 solver 显示 `converged: True`；
- static susceptibility 的 imaginary part 只在数值误差量级；
- `max |Gamma_H|` 在 eta 的 q=(0,0) 正常态中接近机器精度；
- 增大 `nw`, `nOmega`, `nk1`, `nk2` 后结果稳定。

详细的收敛流程见 [numerics_and_validation.md](numerics_and_validation.md)。

# Automated Convergence Scan

`convergence_scan.py` 用于系统检查 full cGW 结果对三个主要数值 cutoff 的依赖：fermionic Matsubara cutoff `nw`、bosonic Matsubara cutoff `nOmega`、以及 momentum mesh `nk x nk`。

## 1. 为什么要分开扫描

当前 reference implementation 对超出 fermion frequency box 的 `G(omega+Omega)` 设为零，因此 `nw` convergence 是第一优先级。随后再检查 `nOmega`，最后增加 momentum mesh。不要同时改变三个参数，否则无法判断误差来自哪一个 cutoff。

推荐顺序：

```text
nw -> nOmega -> nk
```

## 2. 默认运行

```bash
python convergence_scan.py --scan all
```

默认分别使用：

```text
nw      = 8, 12, 16, 24
nOmega  = 2, 4, 6, 8
nk      = 2, 3, 4, 6
```

基准值为：

```text
base_nw      = 16
base_nOmega  = 6
base_nk      = 4
```

扫描 `nw` 时保持 `nOmega=6, nk=4`；扫描 `nOmega` 时保持 `nw=16, nk=4`；扫描 `nk` 时保持 `nw=16, nOmega=6`。

full cGW 包含 AL1/AL2，因此 `--scan all` 可能较慢。建议先单独扫描一个 cutoff。

## 3. 单独扫描 nw

```bash
python convergence_scan.py --scan nw --nw-values 8 12 16 24 32 48
```

也可以改变固定的 momentum mesh 和 bosonic cutoff：

```bash
python convergence_scan.py --scan nw \
  --base-nk 6 \
  --base-nomega 8 \
  --nw-values 12 16 24 32 48
```

## 4. 单独扫描 nOmega

```bash
python convergence_scan.py --scan nomega \
  --base-nw 32 \
  --base-nk 6 \
  --nomega-values 4 6 8 12 16
```

命令行参数写作 `nomega`，输出图和 CSV 中保留物理记号 `nOmega`。

## 5. 单独扫描 momentum mesh

```bash
python convergence_scan.py --scan nk \
  --base-nw 32 \
  --base-nomega 12 \
  --nk-values 4 6 8 10 12
```

`nk=8` 表示 `8 x 8` uniform reduced-coordinate mesh。

## 6. 每个点实际计算什么

每个 `(nk,nw,nOmega)` 点都完整执行：

\[
G_0G_0\rightarrow GG\rightarrow GW+MT\rightarrow full\ cGW.
\]

因此 CSV 中不仅有最终 full-cGW susceptibility，还保存四个层级的 opposite、same 和 `same-opposite`，以及：

- noninteracting chemical potential `mu0`；
- interacting `mu_GW`；
- actual filling；
- GW / MT / full-cGW convergence flags 和 iteration 数；
- `Gamma_H`, `Gamma_MT`, `Gamma_AL1`, `Gamma_AL2` 的最大模；
- 每个 susceptibility 的 real / imaginary part；
- 每个点的 wall-clock runtime。

## 7. 输出目录

若不指定 `--outdir`，程序创建：

```text
results/convergence/YYYYMMDD-HHMMSS/
```

其中包含：

```text
convergence.csv
convergence_nw.png
convergence_nOmega.png
convergence_nk.png
```

如果只扫描一个 cutoff，只生成对应的 PNG。

也可指定固定输出目录：

```bash
python convergence_scan.py --scan nw --outdir results/my_nw_test
```

## 8. 三条曲线怎么读

每张 convergence figure 使用 full cGW 结果画三条曲线：

\[
\chi_{opposite},\qquad
\chi_{same},\qquad
\Delta\chi=\chi_{same}-\chi_{opposite}.
\]

绝对 susceptibility 尚未完全收敛时，`Delta chi` 的符号可能已经稳定；但 production conclusion 最好同时要求：

1. `chi_same` 和 `chi_opposite` 的相对变化足够小；
2. `Delta chi` 的符号和大小稳定；
3. GW 与两个 full-cGW vertex 的 `converged` flag 都为 `True`；
4. static susceptibility 的 imaginary part 只在 floating-point noise 范围；
5. `Gamma_H` 在 time-reversal symmetric normal state 仍接近机器精度。

## 9. 改变物理参数

脚本也支持直接在命令行改变模型参数，例如：

```bash
python convergence_scan.py --scan nw \
  --V 0.15 --T 0.04 --filling 2.0 \
  --ti 0.4 --t1 0.2 --t2 0.2
```

这使 convergence test 可以针对真正准备做相图的关键参数点进行，而不是只检查默认 debug 点。

## 10. 收敛容差

默认设置为：

```text
GW tol       = 1e-8
GW mixing    = 0.20
vertex tol   = 1e-8
vertex mixing= 0.20
```

可以通过

```text
--gw-max-iter
--gw-tol
--gw-mixing
--vertex-max-iter
--vertex-tol
--vertex-mixing
```

修改。

若需要查看每一次 GW / vertex fixed-point iteration，增加：

```bash
--verbose-iterations
```

通常 convergence scan 建议关闭详细 iteration 输出，只保留每个 grid point 的 summary。

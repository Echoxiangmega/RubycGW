# Ruby Model and Conventions

## 1. 六子晶格与 hopping

RubycGW 保留之前 Ruby selection-rule 代码的编号：每个 unit cell 有六个 site，编号 `0,1,2,3,4,5`。两个三角形分别由 `(0,1,2)` 和 `(3,4,5)` 构成。

代码中 `_base_bonds()` 存放 12 条 undirected NN bond：

```text
(0,1,0,0)  ti
(0,2,0,0)  ti
(2,1,0,0)  ti
(3,4,0,0)  ti
(3,5,0,0)  ti
(4,5,0,0)  ti
(1,4,0,0)  t1
(5,0,0,-1) t1
(2,3,-1,0) t1
(3,1,0,-1) t2
(2,5,0,0)  t2
(0,4,-1,0) t2
```

这里 cell offset `R=(R1,R2)` 表示从当前原胞中的第一个 site 指向平移后原胞中的第二个 site。

## 2. Fourier convention

使用 reduced reciprocal coordinates：

\[
c_{Ra}=\frac{1}{\sqrt N}\sum_k e^{2\pi i k\cdot R}c_{ka}.
\]

因此 hopping matrix 为

\[
[h_0(k)]_{ab}=\sum_R t_{ab}(R)e^{2\pi i k\cdot R}.
\]

注意 phase 只含 Bravais-lattice cell vector `R`，不额外包含 intracell position `r_a`。这也是为什么当前两个 intracell triangle 的 eta bare vertex 可以写成与 k 无关的常数 6x6 矩阵。

## 3. Density interaction

当前模型在同一组 12 条 NN bond 上放相同的 density-density interaction `V`：

\[
H_V=V\sum_{\langle ia,jb\rangle} n_{ia}n_{jb}.
\]

Fourier 形式写作

\[
H_V=\frac{1}{2N}\sum_q n_a(q)V_{ab}(q)n_b(-q).
\]

`build_interaction(qpts, params)` 返回 shape `(...,6,6)` 的 Hermitian matrix `V(q)`。

## 4. eta bond operator

有向 bond `a -> b` 的反对称双线性仍沿用之前 HS 推导中的记号：

\[
\eta_{ab}=i\left(\bar\psi_a\psi_b-\bar\psi_b\psi_a\right).
\]

没有额外的 `1/2`。

三角形 A 采用

\[
0\to1\to2\to0,
\]

因此

\[
\eta_A=\eta_{01}+\eta_{12}+\eta_{20}.
\]

三角形 B 采用

\[
3\to4\to5\to3.
\]

代码中 `eta_vertices()` 返回

```python
K_A, K_B, K_plus, K_minus
```

使得

\[
\eta_\lambda=\bar\psi K^\lambda\psi.
\]

## 5. plus/minus 与物理 same/opposite 的标签

由于两个代数箭头环在实际 Ruby 几何中的 handedness 相反，当前 convention 下

\[
K^+=\frac{K^A+K^B}{\sqrt2}
\]

对应 **physical opposite circulation**，而

\[
K^-=\frac{K^A-K^B}{\sqrt2}
\]

对应 **physical same circulation**。

这是整个项目最容易混淆的标签之一。程序输出始终明确写成：

```text
opposite (+)
same (-)
```

不要把代数 `+/-` 直接解释成物理 same/opposite。

## 6. 有限外部动量 q 时 bare eta vertex

一般外源定义为

\[
\eta_\lambda(q)=\sum_k\bar\psi(k+q)K^\lambda(k,q)\psi(k).
\]

对于当前完全位于同一 unit cell 内的 triangle eta，`K^lambda` 不依赖 k 和 q，有限 q 只体现在 fermion 从 `k` 变成 `k+q`。如果以后把跨原胞 bond 也纳入 eta operator，则 bare vertex 会获得 Bloch phase，例如对 `(R,a)->(R+delta,b)`：

\[
K_{ab}(k,q)=ie^{2\pi i k\cdot\delta},\qquad
K_{ba}(k,q)=-ie^{-2\pi i (k+q)\cdot\delta}.
\]

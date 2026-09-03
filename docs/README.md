# RubycGW Documentation

这一目录是 RubycGW 的长期维护文档。代码中的 docstring 只说明局部接口；这里记录模型约定、方程来源、数值实现、模块/函数功能以及完整使用流程。

## 文档导航

- [getting_started.md](getting_started.md)：安装、测试、第一次运行以及如何修改参数。
- [model_and_conventions.md](model_and_conventions.md)：Ruby lattice 的六子晶格编号、12 条 hopping bond、Fourier convention、`eta_A/B` 与 same/opposite 标签。
- [gw_theory.md](gw_theory.md)：self-consistent GW 的方程、每个数组的含义以及代码中的对应实现。
- [cgw_theory.md](cgw_theory.md)：为什么对外源求导会得到 Hartree、Fock、MT、AL1、AL2，以及 q=(0,0) 版本在代码中的实现。
- [orbital_moment.md](orbital_moment.md)：从 checkpoint 重建 `G`，计算 bond current 与局域 plaquette orbital moment。
- [electromagnetic_response.md](electromagnetic_response.md)：Peierls-flux electromagnetic covariant response、固定 filling 的 `dmu/dphi` 以及与 `+/-delta_phi` 完整 GW 有限差分的验证。
- [api_reference.md](api_reference.md)：模块、类和主要函数的接口、输入输出 shape 和用途。
- [numerics_and_validation.md](numerics_and_validation.md)：Matsubara cutoff、mixing、收敛测试、V=0 极限、时间反演检查和结果可信度判断。
- [convergence_scan.md](convergence_scan.md)：自动 `nw/nOmega/nk` convergence scan、fast MT 模式和 continuation。
- [performance_and_reuse.md](performance_and_reuse.md)：性能瓶颈、warm start、参数扫描时哪些量可以复用，以及推荐的两级扫描策略。
- [tutorial.md](tutorial.md)：将理论与主程序串起来的完整教程，也是 PDF manual 的主体源文件。
- [maintenance.md](maintenance.md)：以后修改代码时需要同步更新哪些文档，以及 PDF 如何自动生成。

## 文档维护原则

`docs/tutorial.md` 是完整理论教程的主体源文件；独立的数值与方法章节用于快速查阅。PDF 不手工编辑，而是由 GitHub Actions 从维护中的 Markdown 文档自动构建，以避免代码和 PDF 版本不一致。

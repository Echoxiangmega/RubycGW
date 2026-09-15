# Documentation Maintenance

## 1. Source of truth

`docs/tutorial.md` 是完整教程和 PDF 的 source of truth。模块化页面用于快速查阅，但如果代码逻辑、公式、默认参数、函数签名或物理 convention 改变，必须同时更新 `tutorial.md`。

## 2. 代码更新时需要检查的文档

如果修改 `model.py`，检查 `model_and_conventions.md`, `api_reference.md`, `tutorial.md`。

如果修改 `grids.py`，检查 `api_reference.md`, `numerics_and_validation.md`, `tutorial.md`。

如果修改 `gw.py`，检查 `gw_theory.md`, `api_reference.md`, `numerics_and_validation.md`, `tutorial.md`。

如果修改 `cgw.py`，检查 `cgw_theory.md`, `api_reference.md`, `numerics_and_validation.md`, `tutorial.md`。

如果修改主程序的 staged workflow 或默认参数，检查 `getting_started.md` 和 `tutorial.md`。

## 3. PDF 自动生成

`.github/workflows/docs.yml` 在以下内容变化时构建 PDF：

```text
docs/**
rubycgw/**
run_ruby_cgw.py
README.md
.github/workflows/docs.yml
```

workflow 使用 Pandoc + XeLaTeX + Noto CJK fonts 生成：

```text
build/RubycGW_Tutorial.pdf
```

然后上传为 GitHub Actions artifact `RubycGW-Tutorial-PDF`。

因此仓库中不需要手工提交二进制 PDF；每次相关代码或文档更新，PDF artifact 都会从最新 `tutorial.md` 重建。

## 4. 本地只维护 Markdown 即可

日常修改不要求本地安装 LaTeX。直接阅读和编辑 `docs/*.md` 即可。需要本地 PDF 时，可以参考 workflow 中的 Pandoc command；Windows 上如果没有 XeLaTeX，优先使用 GitHub Actions 生成的 artifact。

## 5. 更新检查表

每次较大改动至少确认：公式与实现一致；公开函数的参数和 shape 文档一致；same/opposite 标签没有反转；默认参数与 `run_ruby_cgw.py` 一致；新增的数值近似写入 `numerics_and_validation.md`；`python -m pytest -q` 仍通过。

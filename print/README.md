# 打印版（`include/acm/` 的 LaTeX 排版工程）

把仓库里的赛场版头文件排成一本可双面打印的 A4 小册子：封面、目录、速查索引，每个模块一章，只讲**接口**与**约定**，并附完整源码。

排版风格：纯黑白、正文宋体、标题黑体、楷体作副标题、细横线分栏页眉页脚、大面积留白；代码用 minted 的 `vs` 配色（注释绿、关键字蓝、类型青），并套一层圆角矩形外框。

源码不是誊抄的：每章的「完整源码」由 `minted` 在编译时直接读取 `../include/acm/`，修改头文件后重新构建即可自动同步。

## 依赖

| 依赖 | 说明 |
|------|------|
| XeLaTeX（TeX Live 2024 及以上） | 需要 `ctexbook`、`fancyhdr`、`longtable`、`lastpage` 等宏包 |
| Pygments（`pygmentize`） | `minted` 的高亮后端，使用内置的 `vs` 样式 |
| `-shell-escape` | `minted` 需要调用外部程序；已在 `.latexmkrc` 中固定 |

中文使用 TeX Live 自带的 Fandol 字库（`fontset=fandol`），不依赖系统字体；西文与数学使用 Latin Modern，等宽使用 Latin Modern Mono。

## 构建

构建目标在仓库根的 `Makefile` 里，在仓库根执行：

```bash
make print        # 生成 build/print/main.pdf
make print-once   # 单遍 XeLaTeX，快速预览（目录与交叉引用可能未解析）
make print-watch  # 持续编译
make print-check  # 校验源码引用、头文件覆盖与排版前置条件
make print-clean  # 删除 build/print/
```

`make clean` 会同时清理测试用的 `build/make/` 与打印版的 `build/print/`。

输出固定写到仓库根的 `build/print/`（该目录已被 `.gitignore` 忽略），不污染源码树。

## 目录结构

```
print/
├── main.tex            装配入口：文档类、导言区、封面/目录、\include 各章
├── preamble/           导言区：宏包、字体、版式、minted 配置、排版宏
├── frontmatter/        封面、目录、速查索引
├── chapters/           每章一个 .tex，按模块编号
├── tools/check_sources.py
└── .latexmkrc
```

## 一章的固定结构

每章只有接口与约定，外加完整源码；不写算法讲解、示例与「注意事项」清单：

```latex
\chapter{模运算}\label{ch:modint}

\chaptermeta{modint.hpp}{def.hpp}{乘除 $O(1)$，求逆 $O(\log p)$}

一句话说明这个模块是什么。

\section{接口}

\apitable{
    \api{SModint<P>}{编译期模数，要求 $0<P<2^{31}$}
    ...
}

\section{约定}

接口的类型约束、下标范围、哨兵值、前置条件与错误处理，一段话讲完。

\fullsource{modint.hpp}
```

`\fullsource{...}` 的参数是相对 `include/acm/` 的路径（如 `geometry/circle.hpp`），只用于定位文件；标题统一为「完整源码」，不会在正文里重复路径。不要改成 `firstline/lastline` 行号区间，头文件增删行会让它失效。

## 源码书写约定

- **一个段落写成一行**，不要在段落内部换行；段与段之间空一行。
- 环境、表格行、列表项统一缩进 **4 个空格**；章节命令顶格。
- 行内代码用 `\code{...}`（内部会自动转义，下划线无需写成 `\_`），长标识符会自动在 `_ . : / , ( -` 处断行。
- 表格用 `\apitable{...}`（内部是 `longtable`，长表跨页并重复表头），不要直接写 `tabularx` 环境。

## 版式调整

| 想改什么 | 改哪里 |
|----------|--------|
| 标题、副标题、作者、日期 | `preamble/meta.tex` |
| 页边距、行距、标题样式、页眉页脚、目录样式 | `preamble/style.tex` |
| 字体（含改用系统字体） | `preamble/fonts.tex` 与 `main.tex` 的 `fontset` |
| 代码字号、行距、行号、配色、圆角外框 | `preamble/minted.tex` 的 `acmcode` 样式 |
| 表格、元信息行、源码引入宏 | `preamble/macros.tex` |

## 常见问题

- **`minted` 报 "You must invoke LaTeX with the -shell-escape flag"**：用 `make` 构建，或手动加 `-shell-escape`；`.latexmkrc` 已固定该参数。
- **改了代码但 PDF 没变**：`minted` 有缓存，先 `make print-clean` 再构建。
- **中文变成方框**：Fandol 字库缺失（精简版 TeX Live），安装 `texlive-lang-chinese`；不建议改用 Noto CJK 的 `.ttc`，XeTeX 会取到日文字面。
- **路径里有空格**：根 Makefile 已给路径加引号；自行调用 `latexmk` 时请给路径加引号。
- **代码外框跨页**：长源码的圆角框会在页边界自然断开（`breakable`），下一页接着画，属于预期行为；若想改成每页独立的完整框，可去掉 `breakable` 并拆分源码。
- **有一段代码缺字**：源码注释里的 `⌊ ⌋ ≡` 等符号不在 Latin Modern Mono 中，已在 `fonts.tex` 用 `newunicodechar` 映射到数学字体；新增符号时照此添加。

## 维护约定

- 代码只以 `\fullsource` 的形式出现，不要把 C++ 代码复制进 `.tex`。
- 唯一的布尔前提：`include/acm/` 下的源码不超过 80 列、不含制表符（`make print-check` 会检查）。
- 修改头文件后，除重新构建外，请复核对应章节的「接口」与「约定」是否仍然准确。

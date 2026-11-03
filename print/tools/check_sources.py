#!/usr/bin/env python3
"""校验打印版的源码引用与排版前置条件。

检查项：
  1. 每个 \\fullsource{...} 指向的 include/acm 文件存在；
  2. include/acm 下每个模块头文件恰好被引用一次（all.hpp 是汇总头，不要求成章）；
  3. 被引入的源码不含制表符，且最长行不超过 80 列（minted 版心的假设）。

用法：python3 tools/check_sources.py [--root .]
退出码：0 全部通过；1 有错误。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FULLSOURCE_RE = re.compile(r"\\fullsource\{([^}]+)\}")
INPUTMINTED_RE = re.compile(
    r"\\inputminted(?:\[[^\]]*\])?\{[A-Za-z]+\}\{([^}]+)\}"
)

MAX_COLUMNS = 80

# 汇总头不对应具体算法，不要求单独成章；若某章收录了它也不报错。
OPTIONAL = {"all.hpp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="print/ 目录（默认当前目录）")
    parser.add_argument(
        "--max-columns", type=int, default=MAX_COLUMNS, help="源码最大列数"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    main_tex = root / "main.tex"
    if not main_tex.is_file():
        print(f"error: {main_tex} 不存在，请在 print/ 目录下运行", file=sys.stderr)
        return 1

    acm_dir = root.parent / "include" / "acm"
    if not acm_dir.is_dir():
        print(f"error: {acm_dir} 不存在", file=sys.stderr)
        return 1

    headers = sorted(p.relative_to(acm_dir).as_posix() for p in acm_dir.rglob("*.hpp"))
    required = [h for h in headers if h not in OPTIONAL]

    tex_files = sorted(root.glob("chapters/*.tex")) + sorted(
        root.glob("frontmatter/*.tex")
    )
    fullsources: list[tuple[str, str]] = []
    minted: list[tuple[str, str]] = []
    for tex in tex_files:
        text = tex.read_text(encoding="utf-8")
        fullsources += [(tex.name, m) for m in FULLSOURCE_RE.findall(text)]
        minted += [(tex.name, m) for m in INPUTMINTED_RE.findall(text)]

    errors: list[str] = []
    warnings: list[str] = []

    # 1. \fullsource 目标存在
    for tex, rel in fullsources:
        if not (acm_dir / rel).is_file():
            errors.append(f"{tex}: \\fullsource 目标不存在 include/acm/{rel}")

    # 2. 覆盖情况
    referenced = [rel for _, rel in fullsources]
    missing = [h for h in required if h not in referenced]
    duplicated = sorted({h for h in referenced if referenced.count(h) > 1})
    unknown = [r for r in referenced if r not in headers]
    for h in missing:
        errors.append(f"未被任何章节收录：include/acm/{h}")
    for h in duplicated:
        errors.append(f"被多个章节重复收录：include/acm/{h}")
    for h in unknown:
        errors.append(f"\\fullsource 指向不存在的头文件：{h}")

    # 3. 被引入源码的排版前置条件
    for tex, rel in fullsources:
        target = acm_dir / rel
        if not target.is_file():
            continue
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            errors.append(f"{tex}: include/acm/{rel} 不是 UTF-8 文本")
            continue
        tabs = [i for i, line in enumerate(lines, 1) if "\t" in line]
        if tabs:
            errors.append(f"include/acm/{rel}: 含制表符（第 {tabs[:3]} 行）")
        too_wide = [i for i, line in enumerate(lines, 1) if len(line) > args.max_columns]
        if too_wide:
            errors.append(
                f"include/acm/{rel}: 有 {len(too_wide)} 行超过 {args.max_columns} 列"
                f"（第 {too_wide[:3]} 行），minted 版心会断行"
            )

    # 汇总
    print(f"print/ 目录：{root}")
    print(f"模块头文件：{len(required)} 个（另有汇总头 {len(headers) - len(required)} 个）")
    print(f"已收录：{len(set(referenced) & set(headers))} 个")
    print(f"源码引入（\\fullsource）：{len(fullsources)} 处")
    if warnings:
        print("\n提醒：")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("\n错误：")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\n全部检查通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

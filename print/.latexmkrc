# latexmk 配置：XeLaTeX + minted
#
# minted 需要 -shell-escape 调用 pygmentize；这里固定写进 $xelatex，
# 避免每次手工加参数。输出目录由 Makefile 的 -outdir 指定。
$pdf_mode = 5;                                      # 5 = xelatex
$xelatex  = 'xelatex -shell-escape -synctex=1 %O %S';
$max_repeat = 6;                                    # 目录 / 页码 / \pageref 需要多遍
$pdf_update_method = 0;

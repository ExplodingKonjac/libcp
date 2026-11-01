# ACM / cp-lib - test driver and print build
#
# The default CXX is clang++ because the GCC 15.2 libstdc++ installation in this
# environment fails to compile <mutex> (and therefore several cp tests).
# Override with: make CXX=g++ test-acm

ifeq ($(origin CXX), default)
CXX = clang++
endif
CPPFLAGS ?= -Iinclude
WARN ?= -Wall -Wextra -Wno-unused-result
OPT ?= -O2

ACSTD ?= -std=c++20
CPSTD ?= -std=c++23 -mavx2 -Wno-unknown-pragmas

OUT ?= build/make

# 打印版 LaTeX 工程（print/），需要 XeLaTeX + Pygments，并允许 -shell-escape
PRINT_DIR  ?= print
PRINT_OUT  ?= build/print
PRINT_MAIN ?= main.tex
PRINT_PDF   = $(PRINT_OUT)/main.pdf
LATEXMK    ?= latexmk
PYTHON     ?= python3
LATEXMKFLAGS = -xelatex -shell-escape -interaction=nonstopmode \
               -halt-on-error -file-line-error

.PHONY: all test test-acm test-cp test-acm-san headers clean \
        print print-once print-watch print-check print-clean

all: test

test: test-acm test-cp

test-acm: headers
	@mkdir -p $(OUT)/acm
	@set -e; for src in tests/acm/test_*.cpp; do \
		name=$$(basename $$src .cpp); \
		echo "==> acm/$$name"; \
		$(CXX) $(ACSTD) $(CPPFLAGS) $(WARN) $(OPT) $$src -o $(OUT)/acm/$$name; \
		./$(OUT)/acm/$$name; \
	done

test-cp:
	@mkdir -p $(OUT)/cp
	@set -e; for src in tests/cp/test_*.cpp; do \
		name=$$(basename $$src .cpp); \
		case $$name in \
			test_luogu|test_dft_new|test_io) echo "==> cp/$$name (skipped: needs stdin)"; continue;; \
		esac; \
		echo "==> cp/$$name"; \
		$(CXX) $(CPSTD) $(CPPFLAGS) $(WARN) $(OPT) $$src -o $(OUT)/cp/$$name; \
		./$(OUT)/cp/$$name; \
	done

test-acm-san:
	@mkdir -p $(OUT)/acm-san
	@set -e; for src in tests/acm/test_*.cpp; do \
		name=$$(basename $$src .cpp); \
		echo "==> acm-san/$$name"; \
		$(CXX) $(ACSTD) $(CPPFLAGS) $(WARN) -g -O1 \
			-fsanitize=address,undefined $$src -o $(OUT)/acm-san/$$name; \
		./$(OUT)/acm-san/$$name; \
	done

headers:
	@mkdir -p $(OUT)
	@set -e; for h in $$(find include/acm -name '*.hpp'); do \
		rel=$${h#include/}; \
		printf '#include "%s"\nint main() {}\n' "$$rel" > $(OUT)/header.cpp; \
		$(CXX) $(ACSTD) $(CPPFLAGS) $(WARN) -fsyntax-only $(OUT)/header.cpp; \
	done

# ---------------------------------------------------------------- 打印版

print:
	@mkdir -p "$(PRINT_OUT)"
	cd "$(PRINT_DIR)" && $(LATEXMK) $(LATEXMKFLAGS) "-outdir=../$(PRINT_OUT)" "$(PRINT_MAIN)"
	@printf 'PDF: %s\n' "$(PRINT_PDF)"
	@pdfinfo "$(PRINT_PDF)" | sed -n 's/^Pages:/页数:/p'

print-once:
	@mkdir -p "$(PRINT_OUT)"
	cd "$(PRINT_DIR)" && xelatex -shell-escape -interaction=nonstopmode -file-line-error \
		"-output-directory=../$(PRINT_OUT)" "$(PRINT_MAIN)" || true
	@printf 'PDF: %s（单遍结果，目录与交叉引用可能未解析）\n' "$(PRINT_PDF)"

print-watch:
	@mkdir -p "$(PRINT_OUT)"
	cd "$(PRINT_DIR)" && $(LATEXMK) $(LATEXMKFLAGS) -pvc "-outdir=../$(PRINT_OUT)" "$(PRINT_MAIN)"

print-check:
	cd "$(PRINT_DIR)" && $(PYTHON) tools/check_sources.py

print-clean:
	rm -rf "$(PRINT_OUT)"

# ---------------------------------------------------------------- 清理

clean:
	rm -rf $(OUT) $(PRINT_OUT)

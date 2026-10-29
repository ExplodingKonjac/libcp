# ACM / cp-lib test driver
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

.PHONY: all test test-acm test-cp test-acm-san headers clean

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

clean:
	rm -rf $(OUT)

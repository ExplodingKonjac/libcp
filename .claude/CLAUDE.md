# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

This is a header-only C++23 library with no build system. Compile and run directly:

```bash
# Release build
g++ -std=c++23 -O2 -Wall -o prog prog.cpp

# Debug build with sanitizers
g++ -std=c++23 -g -fsanitize=address,undefined,leak -DDADALZY -Wall -o prog prog.cpp
```

Tests are standalone `.cpp` files in `tests/`, each with its own `main()`. No test framework — just `assert()` / hand-rolled `assert_eq()`. Run any test:

```bash
g++ tests/test_graph.cpp -std=c++23 -O2 -Wall -o test_graph && ./test_graph
```

Formatting: `clang-format -i <file>` (config at `.clang-format` — Google-based, 4-space indent).

## Architecture

All library code lives under `cp/` as independent, self-contained headers. Every header includes only `"def.hpp"` (no internal dependency chains). There is no layering — headers can be included individually in any order.

Headers within `cp/utils/` are smaller utility components (`lambda.hpp`, `hash_value.hpp`, `compile_format.hpp`). The rest at `cp/` root are major data-structure or algorithm modules.

Everything sits in `namespace cp`. Type aliases and UDL literals live in `namespace cp::inline defs` (auto-imported via `using namespace cp`).

## Coding Conventions

- **`#pragma once`** on every header
- **Relative includes**: `#include "def.hpp"`, not `#include "cp/def.hpp"`
- **PascalCase** for types/classes, **snake_case** for methods/variables
- **Trailing underscore** for private members: `n_`, `m_`, `alloc_`, `t_`
- **`requires` clauses** on all templated containers — the convention is to constrain template parameters with `requires requires(...) { ... }` blocks
- **`noexcept`** on methods that shouldn't throw (modify, update, etc.)
- **`[[gnu::always_inline]]`** on hot-path helpers and expression template call operators
- **RAII everywhere** — `std::allocator_traits` for custom allocation, destructors handle cleanup, no manual `new`/`delete`

## Notable Patterns & Quirks

- `cp/utils/lambda.hpp` overloads operators on placeholder types (`_1` through `_9`) to build expression templates. These are evaluated lazily when called with arguments. Use `cp::inline placeholders` to bring `_1`..`_9` into scope.
- `cp/utils/compile_format.hpp` and `cp/fast_io.hpp` each define their own `FormatString`/`FixedString`/`""_fmt`. They **conflict**. If using both, define `CP_FORMAT_STRING` before including `fast_io.hpp` to suppress its version.
- `cp/fast_io.hpp` provides global instances `cp::qin` / `cp::qout` (not `cin`/`cout`). `scan<T>()` returns `std::optional<T>`.
- `cp/modint.hpp` stores values in Montgomery form internally. Conversion to/from plain integers happens only on construction and extraction — arithmetic operations stay in Montgomery space.
- `cp/fpoly.hpp` uses an internal `AlignedPool` bump allocator with 64-byte alignment for AVX2 vectors. NTT twiddle tables are `constexpr`-precomputed.
- Tests use `using namespace cp;` liberally. The library is designed for this usage pattern (competitive programming).

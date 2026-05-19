# PROJECT: cp-lib

C++23 header-only high-performance library for competitive programming (OI/ACM), providing fast I/O, modular arithmetic, polynomial operations, data structures, and SIMD-accelerated (SSE2/AVX2) components.

## Table of Contents

- [Stack](stack.md) — Technology choices and dependencies
- [Toolchain](toolchain.md) — Build, test, dev setup
- [Architecture](architecture.md) — Module layout and data flow
- [Progress](progress.md) — Current status and roadmap
- [Implementation](implementation.md) — Key implementation details
- [Data Model](data-model.md) — N/A (no persistent data layer)
- [API](api.md) — N/A (library, not a network service)
- [Deployment](deployment.md) — N/A (distributed as headers, not deployed)

## Tech Stack Summary

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | C++ | C++23 |
| Compiler | g++ or clang | g++-13+ / clang-16+ |
| SIMD | SSE2, AVX2 | x86-64 intrinsics |
| Platform | Linux x86-64 | — |
| Formatting | clang-format | Google-based |

## Quick Reference Commands

```bash
# Compile a program using the library
g++ -std=c++23 -O2 -Wall -o solve solve.cpp

# Compile and run a test
g++ tests/test_graph.cpp -std=c++23 -O2 -Wall -o test_graph && ./test_graph

# Debug build with sanitizers
g++ tests/test_graph.cpp -std=c++23 -g -fsanitize=address,undefined,leak \
  -o test_graph -DDADALZY -Wall

# Format all source files
clang-format -i cp/**/*.hpp tests/*.cpp

# Install headers to a prefix
./install.sh --prefix /path/to/target
```

## Project Map

- `cp/` — All library headers (core types, data structures, algorithms, utils)
- `cp/utils/` — Utility headers (compile-time format, lambda expressions, hash values)
- `tests/` — Standalone test programs (correctness + benchmarks)
- `install.sh` — Header installation script
- `.clang-format` — Code style configuration

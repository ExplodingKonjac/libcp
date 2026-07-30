---
sources:
  - "README.md"
  - "cp/**/*.hpp"
  - "tests/*.cpp"
  - "install.sh"
  - ".clangd"
  - ".clang-format"
---

# cp-lib

A C++23 header-only library of performance-oriented data structures, algorithms, and I/O utilities for OI/ACM competitive programming.

## Table of Contents

- [Stack](stack.md) — Technology choices and dependencies
- [Toolchain](toolchain.md) — Build, test, formatting, and development setup
- [Architecture](architecture.md) — Header/module layout and compile-time data flow
- [Conventions](conventions.md) — Coding standards and architectural rules
- [Progress](progress.md) — Current status, debt, and roadmap
- [Implementation](implementation.md) — Algorithms and implementation patterns
- [Data Model](data-model.md) — Persistence applicability and in-memory structures
- [API](api.md) — Network API applicability
- [Deployment](deployment.md) — Header installation and distribution

## Tech Stack Summary

| Layer | Technology | Version |
|-------|------------|---------|
| Language | C++ | C++23 |
| Platform | Linux x86-64 | Required for the documented mmap/SIMD paths |
| Toolchain | GCC or Clang | g++ 13+ or clang 16+ |
| Testing | Standalone executable tests | Standard assertions and reference comparisons |

- Header-only templates and inline facilities; no package manager or linked library target.
- Standard-library-only dependencies plus Linux/POSIX and x86 intrinsic headers.
- Performance paths use mmap, SSE2, AVX2, Montgomery arithmetic, expression templates, and widened integral geometry predicates.
- GPLv3 licensed and installable as a copied `include/cp` header tree.

## Quick Reference Commands

```bash
# Build a consumer or one test from the repository root
g++ -I. -std=c++23 -O2 -Wall tests/test_graph.cpp -o /tmp/test_graph

# Run the compiled test
/tmp/test_graph

# Build and run one part of the split 2D geometry suite
g++ -I. -std=c++23 -O2 -Wall -Wextra tests/test_circle.cpp -o /tmp/test_circle
/tmp/test_circle

# Debug with sanitizers
g++ -I. -std=c++23 -g -Wall -Wno-unused-result \
  -fsanitize=address,undefined,leak -DDADALZY \
  tests/test_hash_map.cpp -o /tmp/test_hash_map

# Format a file
clang-format -i cp/graph.hpp

# Install headers under a prefix
./install.sh --prefix <TARGET_DIRECTORY>
```

## Project Map

- `cp/` — public headers for core algorithms and data structures.
- `cp/geometry/` — independently includable circle and polygon extensions built on `cp/geometry.hpp`.
- `cp/utils/` — supporting concepts, compile-time formatting, hashing, and lambda-expression helpers.
- `tests/` — independent component tests, stress tests, benchmarks, and algorithm programs; `tests/cp` links back to `cp/`.
- `.vscode/` — per-file compile/debug integration for VS Code.
- `install.sh` — copies the public header tree into a chosen prefix.

## Tracking Exclusions

- `.claude/project-tracker/**` — legacy tracker documentation is not an input to this tracker.
- `tests/test_*` without a source suffix — generated local test executables are not source documentation inputs.

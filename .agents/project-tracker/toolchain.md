---
sources:
  - "README.md"
  - ".clangd"
  - ".clang-format"
  - ".vscode/*.json"
  - ".vscode/*.py"
  - "tests/*.cpp"
  - "install.sh"
---

# Toolchain & Dev Setup

## Build System

| Tool | Command | Output |
|------|---------|--------|
| g++ | `g++ -I. -std=c++23 -O2 -Wall <SOURCE>.cpp -o <OUTPUT>` | Native executable containing instantiated header code |
| install script | `./install.sh --prefix <TARGET_DIRECTORY>` | `<TARGET_DIRECTORY>/include/cp/` header tree |

No CMake, Make, Meson, package manifest, or compiled library target is present. Tests and consumers are built independently.

## Linting & Formatting

| Tool | Config file | Run command |
|------|-------------|-------------|
| clang-format | `.clang-format` | `clang-format -i <FILES>` |
| clangd | `.clangd` | Run through editor integration |
| Compiler warnings | `.clangd` and command line | `g++ -std=c++23 -Wall ...` |
| Sanitizers | `.vscode/tasks.json` / README examples | Add `-fsanitize=address,undefined,leak -g` |

The format is Google-derived with four-space indentation and custom brace and line-breaking rules. `.clang-format` declares `Standard: c++20`, while actual compilation targets C++23.

## Testing

| Aspect | Detail |
|--------|--------|
| Framework | Standalone C++ executables using `assert`, custom checks, randomized comparisons, and benchmarks |
| Coverage target | Not specified |
| Coverage tool | None configured |
| E2E / integration | Per-header integration programs under `tests/*.cpp`; no unified runner |

Example:

```bash
g++ -I. -std=c++23 -O2 -Wall tests/test_bitset.cpp -o /tmp/test_bitset
/tmp/test_bitset
```

The geometry suite is additionally verified with `-Wextra -Werror` under both GCC and Clang, with an ASan/UBSan/LSan build. Its measured header coverage is 91.69% of lines and 83.79% of branches.

Some programs are benchmarks or judge-style programs and may require input or substantial runtime. Treat each test source according to its `main()` behavior rather than assuming a uniform suite.

## CI/CD Pipeline

N/A — no CI configuration or automated release/deployment pipeline is present. A future pipeline should at minimum format-check, compile representative headers with supported GCC and Clang versions, run deterministic component tests, and exercise sanitizer builds.

## Development Environment

| Requirement | Value |
|-------------|-------|
| Required tools | Linux x86-64, g++ 13+ or clang 16+, a C++23 standard library |
| Optional tools | clang-format, clangd, VS Code, LLDB |
| Environment variables | None |
| Dev server / watcher | N/A |
| Recommended flags | `-std=c++23 -Wall -O2`; sanitizers and `-DDADALZY` for debug builds |

The VS Code task compiles the current file in its own directory. The tracked `tests/cp -> ../cp/` symlink makes `#include "cp/..."` work from that location.

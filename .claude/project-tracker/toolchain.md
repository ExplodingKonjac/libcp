# Toolchain & Dev Setup

## Build System

There is no build system — this is a header-only library. Users `#include` headers directly and compile with `g++`.

| Tool | Command | Output |
|------|---------|--------|
| g++ (release) | `g++ -std=c++23 -O2 -Wall -o prog prog.cpp` | Native binary |
| g++ (debug) | `g++ -std=c++23 -g -fsanitize=address,undefined,leak -DDADALZY -Wall -o prog prog.cpp` | Debug binary with sanitizers |

Compiler flags:
- `-std=c++23` — Language standard
- `-O2` — Optimization level (release)
- `-Wall` — Common warnings
- `-DDADALZY` — Enables extra debug assertions in some components (debug builds)

## Linting & Formatting

| Tool | Config file | Run command |
|------|-----------|-------------|
| clang-format | `.clang-format` (Google-based, 4-space indent) | `clang-format -i cp/**/*.hpp tests/*.cpp` |

The clang-format style extends Google with:
- 4-space indent (not 2)
- Brace wrapping after namespaces
- Short if/loop/lambda on single line allowed
- Break before binary operators (non-assignment)

## Testing

| Aspect | Detail |
|--------|--------|
| Framework | None — manual assertions with `assert_eq()` |
| Coverage target | Not formally tracked |
| Coverage tool | Not configured |
| Test structure | Standalone `.cpp` files in `tests/`, each with its own `main()` |

Tests are simple correctness checks: each test file compiles and runs independently, printing "All tests passed!" on success or exiting with code 1 on failure.

Benchmark tests (e.g., `test_fastio.cpp`, `test_radix4.cpp`) measure throughput against alternatives and print timing results.

## CI/CD Pipeline

No CI/CD is configured. The project has no `.github/workflows/` directory.

## Development Environment

| Requirement | Value |
|-----------|-------|
| Required tools | g++-13+ or clang-16+ |
| Environment variables | None required |
| IDE configuration | `.vscode/` with launch.json, tasks.json, settings.json |
| Debugger | lldb (configured in `.vscode/launch.json`) |

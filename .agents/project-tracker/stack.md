---
sources:
  - "README.md"
  - "cp/**/*.hpp"
  - "install.sh"
---

# Technology Stack

## Language & Runtime

| Property | Value |
|----------|-------|
| Language | C++23 |
| Runtime / VM | Native Linux x86-64 process; no VM |
| Package manager | None; the project is header-only |
| Supported compilers | g++ 13+ or clang 16+ |
| License | GPLv3 |

The library uses C++23 for concepts, ranges, formatting, and template-heavy compile-time composition. Native code and direct header inclusion keep integration overhead low for competitive-programming solutions.

## Frameworks & Libraries

| Dependency | Version | Purpose |
|------------|---------|---------|
| C++ standard library | Compiler-provided C++23 implementation | Containers, concepts, ranges, formatting, allocation, and numeric utilities |
| Linux/POSIX APIs | Host-provided | `mmap`, `fstat`, and file descriptors for the fast-input regular-file path |
| x86 intrinsics | Compiler-provided | SSE2 control-byte probing and AVX2 bitset/NTT acceleration |

The foundational, circle, and polygon geometry headers use only portable C++23 standard-library facilities and the project-defined fixed-width aliases; they do not add a platform or third-party dependency.

There are no third-party library dependencies. This choice makes individual headers easy to copy into contest solutions, at the cost of platform-specific optimized modules and no centralized dependency/version manifest.

## Database & Storage

| Component | Technology | Purpose |
|-----------|------------|---------|
| Primary DB | N/A | The library has no persistence layer |
| ORM / Client | N/A | No database access exists |
| Cache | N/A | `FPoly` reuses an in-process aligned memory pool, not a persistent cache |
| File storage | N/A | Fast input may map stdin when it is a regular file, but stores no data |

## Infrastructure & Services

- N/A — there are no cloud services, external APIs, containers, servers, or managed infrastructure.
- Distribution is a source-header copy performed by `install.sh` or direct repository inclusion.

## Platform Constraints

- Linux is assumed by `cp/fast_io.hpp` because it directly includes `sys/mman.h` and `sys/stat.h`.
- `cp/hash_map.hpp` uses SSE2 intrinsics; `cp/bitset.hpp` and `cp/fpoly.hpp` contain AVX2-oriented implementations.
- Consumers should validate CPU/compiler compatibility before using SIMD-heavy headers on another architecture.

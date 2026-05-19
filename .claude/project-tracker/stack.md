# Technology Stack

## Language & Runtime

| Property | Value |
|----------|-------|
| Language | C++23 |
| Compiler | g++-13+ or clang-16+ |
| Platform | Linux x86-64 |
| Package management | None — header-only, no dependencies |

## Frameworks & Libraries

The library has zero external dependencies — it uses only the C++ standard library and platform intrinsics.

| Dependency | Purpose | Rationale |
|-----------|---------|-----------|
| C++23 STL | Core data structures, concepts, ranges | Required language standard; `<concepts>`, `<bit>`, `<optional>` used heavily |
| SSE2 intrinsics (`<emmintrin.h>`) | 16-byte SIMD for hash map probing | Fast open-addressing slot scanning, checking 16 slots per iteration |
| AVX2 intrinsics (`<immintrin.h>`) | 256-bit SIMD for NTT polynomial multiplication | Radix-4 DIF/DIT butterfly — 8x int32 per cycle |
| `sys/mman.h` (Linux) | Memory-mapped file I/O | Zero-copy input for regular files; `mmap` + `madvise` |
| `cp/def.hpp` | Fixed-width type aliases and literals | Foundation used by all other headers; `cp::i32`, `cp::u64`, `1_i64`, etc. |

### Why no external dependencies?

Competitive programming environments restrict what libraries are available. A header-only library with zero external deps can be dropped into any contest setup by copying headers. The library targets the subset of C++23 supported by g++-13/clang-16, avoiding bleeding-edge features that lack compiler support.

## Database & Storage

N/A — This is a computation library with no persistent storage. All data structures operate entirely in memory.

## Infrastructure & Services

N/A — No cloud services, third-party APIs, or external infrastructure. The library runs locally during competition.

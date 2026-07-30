# Progress & Roadmap

## Current Phase

Active header-library development with a newly completed generic 2D geometry component. The repository is usable component-by-component but lacks a unified build, test, and CI harness.

## Completed

- [x] Core fixed-width aliases and callable concepts.
- [x] Fast input/output with regular-file mmap and buffered fallbacks.
- [x] Static/dynamic Montgomery modular integers and formal polynomial operations.
- [x] Fenwick, segment, and lazy segment trees.
- [x] Weighted and unweighted directed graph containers.
- [x] SIMD-probed flat hash map and allocator-aware pairing heap.
- [x] AVX2 fixed-size bitset with expression templates and set-order comparisons.
- [x] Supporting compile-time formatting, multi-mod hash values, and placeholder lambda expressions.
- [x] Standalone component tests and several stress/benchmark programs.
- [x] Prefix-based header installation script.
- [x] Generic 2D vectors, lines, circles, polygons, predicates, metrics, allocation-free intersections, and minimum enclosing circles.

## In Progress

- [ ] `<CURRENT_MILESTONE>` — no explicit milestone file or issue tracker is present; confirm the next target with maintainers.

## Known Issues & Technical Debt

- The README links `cp/compile_format.hpp`, but the tracked header is `cp/utils/compile_format.hpp`.
- There is no unified build/test runner, test discovery, CI matrix, or configured coverage measurement.
- Platform-sensitive mmap/SSE2/AVX2 behavior is not continuously tested across documented compiler versions and CPU feature sets.
- `.clang-format` declares C++20 while the project requires C++23; this is mostly metadata but can confuse tooling.
- Some tests are benchmarks or judge programs and do not share a consistent pass/fail contract.
- The compile-time formatting helper and fast-I/O formatting facilities have overlapping names guarded by `CP_FORMAT_STRING`, requiring include-order awareness.

## Roadmap

- [ ] Add a lightweight build/test harness that compiles and runs deterministic component tests.
- [ ] Add GCC/Clang CI with release and sanitizer configurations.
- [ ] Correct README paths and clearly document per-header CPU/platform requirements.
- [ ] Separate correctness tests from benchmarks and input-driven examples.
- [?] Add portable scalar fallbacks or explicit compile-time feature gates for SIMD-heavy modules.
- [?] Publish versioned releases or an installable package after the public API stabilizes.

## Recent Work

- Split foundational, circle, and polygon geometry into independently includable headers, replacing the former top-level `cp/polygon.hpp`.
- Removed the dedicated segment type while retaining point-on-segment predicates, and made `sgn`/`cmp` public.
- Standardized floating geometry on a macro-configurable global absolute epsilon.
- Added geometry tests covering compile-time contracts, extreme integer inputs, primitives, distances, intersections, convex geometry, and minimum enclosing circles.
- Fixed `FastOutput` so strings at least as large as its buffer are written safely.
- Fixed rvalue lifetime behavior in expression templates.
- Extended `Bitset` equality and three-way comparison to expression operands.
- Added and refined `Bitset` tests and style.
- Added element-wise initialization/inversion for `HashValue` and continued segment-tree work.

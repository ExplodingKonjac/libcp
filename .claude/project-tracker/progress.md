# Progress & Roadmap

## Current Phase

Active development — adding new components and refining existing interfaces.

## Completed

- [x] Type aliases (`cp/def.hpp`) with UDL literals
- [x] Compile-time format string parsing (`cp/utils/compile_format.hpp`)
- [x] Fast I/O with mmap acceleration (`cp/fast_io.hpp`)
- [x] Montgomery modular arithmetic — static and dynamic modulus (`cp/modint.hpp`)
- [x] Polynomial operations with AVX2 NTT (`cp/fpoly.hpp`)
- [x] Fenwick tree — generic binary indexed tree (`cp/fenwick_tree.hpp`)
- [x] Directed graph with adjacency list (`cp/graph.hpp`)
- [x] SSE2-accelerated flat hash map (`cp/hash_map.hpp`)
- [x] Pairing heap with modify/erase/join (`cp/pairing_heap.hpp`)
- [x] Segment tree with custom semigroup (`cp/segtree.hpp`)
- [x] Lambda expression templates (`cp/utils/lambda.hpp`)
- [x] Multi-modulus hash value (`cp/utils/hash_value.hpp`)
- [x] Header installation script (`install.sh`)

## In Progress

- [ ] Segment tree `update()` method (recently added, interface change from standard `modify()`)

## Known Issues & Technical Debt

- `cp/utils/compile_format.hpp` and `cp/fast_io.hpp` define conflicting `FormatString`/`FixedString`/`""_fmt` — users must define `CP_FORMAT_STRING` before including `fast_io.hpp` when using both modules
- No unified test runner — each test file has its own `main()` with ad-hoc assertions
- No formal benchmark suite — performance tests print results but don't track regressions
- No CI/CD pipeline — tests are run manually
- `graph.hpp`'s `add_edge` does not use `requires` constraints despite the rest of the API being heavily constrained

## Roadmap

- [?] Unify `FormatString` between `fast_io.hpp` and `compile_format.hpp`
- [?] Add CI/CD with automated test compilation and execution
- [?] Add more graph algorithms (shortest path, MST, flow, SCC)
- [?] Add string algorithms (KMP, Z-algorithm, SA, SAM)
- [?] Add geometry primitives
- [?] Consider moving from `cp/` flat structure to categorized subdirectories as component count grows

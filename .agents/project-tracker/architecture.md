---
sources:
  - "cp/**/*.hpp"
  - "tests/*.cpp"
  - "README.md"
---

# Architecture

## Overview

The project is a collection of independently includable headers. There is no runtime service entry point; consumer translation units select modules and instantiate templates at compile time.

```text
+---------------------------+
| Consumer / test .cpp file |
+-------------+-------------+
              | #include and template instantiation
              v
+-------------+-----------------------------------------+
| Public cp headers                                   |
| I/O | math/poly | geometry | trees | graph | map/bitset |
+-------------+-----------------------------------------+
              | shared aliases/concepts or math base
              v
+-------------+-----------------------------------------+
| def.hpp | utils/concepts.hpp | modint.hpp | utilities|
+-------------+-----------------------------------------+
              |
              v
+-------------------------------------------------------+
| Standard library | Linux/POSIX APIs | x86 intrinsics  |
+-------------------------------------------------------+
```

## Module Breakdown

| Module | Responsibility | Key types / exports |
|--------|----------------|---------------------|
| `cp/def.hpp` | Fixed-width and size type aliases | `i32`, `u64`, `usize`, `i128`, `f64` |
| `cp/fast_io.hpp` | Buffered/mmap input and buffered output | `FastInput`, `FastOutput`, `qin`, `qout` |
| `cp/modint.hpp` | Montgomery modular arithmetic | `SModint<P>`, `DModint`, `pow`, `sqrt` |
| `cp/fpoly.hpp` | Formal polynomial operations and SIMD NTT | `FPoly<P>`, `PolyUtils<P>` |
| `cp/fenwick_tree.hpp` | Generic Fenwick tree aggregation | `FenwickTree` |
| `cp/segtree.hpp` | Iterative semigroup segment tree | `SegTree` |
| `cp/lazy_segtree.hpp` | Range-action segment tree | `LazySegTree` |
| `cp/graph.hpp` | Weighted/unweighted directed adjacency lists | `Graph<E>`, `Graph<void>` |
| `cp/geometry.hpp` | Generic 2D vectors, primitives, predicates, metrics, and intersections | `Vec2<T>`, `Line2<T>`, `Segment2<T>`, `Circle2<T>` |
| `cp/hash_map.hpp` | SIMD-probed open-addressing map | `FlatHashMap` |
| `cp/pairing_heap.hpp` | Meldable heap with point handles | `PairingHeap` |
| `cp/bitset.hpp` | Fixed-size SIMD bitset and fused expressions | `Bitset<SIZE>`, expression concepts/operators |
| `cp/utils/*.hpp` | Concepts, compile-time format parsing, hash tuples, placeholder lambdas | `Fn`, `HashValue`, `_1`…`_9`, format helpers |

## Data Flow

1. A solution includes only the headers it needs.
2. The compiler resolves concepts, templates, expression nodes, and inline globals.
3. Runtime input may flow through `qin`, into an in-memory algorithm/data structure, then through `qout`.
4. Polynomial operations allocate/reuse aligned coefficient buffers, transform them with NTT butterflies, apply Montgomery pointwise arithmetic, and transform back.
5. Data structures keep all state in the consumer process; nothing is persisted or sent over a network.
6. Geometry consumers perform exact widened integral predicates or tolerance-aware floating calculations and receive allocation-free intersection result objects.

## Design Patterns

- **Header-only generic programming** — algorithms are parameterized by types and operations and are available without linking.
- **Concept-constrained policies** — tree operations and helpers accept callable policies checked at compile time.
- **CRTP** — modular integer variants share arithmetic through `ModintBase`.
- **Expression templates** — bitset and placeholder-lambda expressions defer evaluation and enable loop fusion.
- **Specialization** — weighted/unweighted graphs and small NTT sizes have specialized implementations.
- **Allocator-aware ownership** — the flat map and pairing heap manage storage while accepting allocator types.
- **Typed result objects** — geometry intersections encode none/point/overlap/coincident outcomes without allocation.

## Security Boundaries

There is no authentication or privilege boundary. The relevant boundary is between trusted contest code and low-level optimized internals:

- Many APIs assume valid indices, non-empty containers, suitable moduli, and compatible CPU features.
- `mmap`, raw pointers, custom allocators, SIMD loads, and manual alignment require careful lifetime and bounds handling.
- Compile-time assertions enforce some invariants, but callers remain responsible for documented preconditions.
- The install script replaces the destination `include/cp` directory, so its prefix must be chosen deliberately.

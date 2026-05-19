# Architecture

## Overview

The library is a flat collection of standalone header files under `cp/`, each providing one major component. There is no layering beyond `cp/def.hpp` as a common foundation. Headers are designed to be included independently — there are no internal dependency chains beyond `def.hpp`.

```
+----------------+     +----------------+     +----------------+
|   Data Structs |     |   Arithmetic   |     |      I/O       |
| fenwick_tree   |     | modint         |     | fast_io        |
| graph          |     | fpoly (NTT)    |     | compile_format |
| hash_map       |     | hash_value     |     |                |
| pairing_heap   |     |                |     |                |
| segtree        |     |                |     |                |
+-------+--------+     +-------+--------+     +-------+--------+
        |                      |                      |
        +----------------------+----------------------+
                               |
                        cp/def.hpp
                     (type aliases, literals)
```

## Module Breakdown

| Module | File | Responsibility | Key types |
|--------|------|---------------|-----------|
| Type defs | `cp/def.hpp` | Fixed-width type aliases and UDL literals | `i32`, `i64`, `u64`, `f64`, `i128`, `usize` |
| Fast I/O | `cp/fast_io.hpp` | High-speed input/output with mmap/buffering | `FastInput`, `FastOutput`, `qin`, `qout` |
| Compile format | `cp/utils/compile_format.hpp` | Compile-time format string decomposition | `FormatString`, `""_fmt` |
| Lambda expr | `cp/utils/lambda.hpp` | Expression templates for inline lambdas | `_1`..`_9` placeholders, operator overloads |
| Hash value | `cp/utils/hash_value.hpp` | Multi-modulus rolling hash value | `HashValue<MOD...>` |
| Modint | `cp/modint.hpp` | Montgomery modular arithmetic | `SModint<P>`, `DModint` |
| Polynomial | `cp/fpoly.hpp` | Polynomial with NTT multiplication | `FPoly<P>`, `AlignedPool` |
| Fenwick tree | `cp/fenwick_tree.hpp` | Generic binary indexed tree | `FenwickTree<T, Plus, Minus>` |
| Graph | `cp/graph.hpp` | Directed adjacency-list graph | `Graph<void>`, `Graph<E>` |
| Hash map | `cp/hash_map.hpp` | SSE2-accelerated flat hash map | `FlatHashMap<K, V>` |
| Pairing heap | `cp/pairing_heap.hpp` | Pairing heap with modify/erase | `PairingHeap<T, Compare>` |
| Segment tree | `cp/segtree.hpp` | Segment tree with custom semigroup | `SegTree<S, Mult, Alloc>` |

## Data Flow

1. **Input**: `FastInput` reads raw bytes via `mmap` (files) or `fgets` (pipes/stdin), buffers in 64 KiB chunks, and parses integers/floats/strings via `scan<T>()`.
2. **Computation**: Data structures and arithmetic modules operate on in-memory data with SIMD acceleration where applicable.
3. **Output**: `FastOutput` buffers formatted output in 64 KiB chunks, uses `std::to_chars` for integer/float conversion, and flushes on destruction.

## Design Patterns

- **Header-only**: All code lives in headers. Users `#include` what they need. No linking step.
- **CRTP / static polymorphism**: Template parameters for comparison functions, allocators, and semigroup operations avoid virtual dispatch.
- **Expression templates**: `cp/utils/lambda.hpp` uses expression templates to build composable inline lambdas without heap allocations.
- **RAII**: All resources (memory pools, file mappings, buffers) are tied to object lifetimes.
- **SIMD intrinsics**: Direct use of `__m128i`/`__m256i` for hot paths (hash table probing, NTT butterflies) rather than relying on auto-vectorization.
- **Montgomery form**: `SModint`/`DModint` store values in Montgomery representation internally, converting only on I/O, eliminating expensive division in modular multiplication.

## Security Boundaries

As a computation library with no network or filesystem exposure beyond the fast I/O layer, security concerns are limited to:

- **Memory safety**: Allocator-aware containers (`SegTree`, `FPoly`) use `std::allocator_traits` for allocation. RAII ensures cleanup on exceptions.
- **Integer overflow**: Montgomery multiplication uses 128-bit intermediates to avoid overflow. `HashValue` uses `i64` promotion in multiplication.
- **Input validation**: `FastInput::scan<T>()` returns `std::optional<T>`, allowing callers to handle malformed input. `modify()`/`update()` bounds-check indices.

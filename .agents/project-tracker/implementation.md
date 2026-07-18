---
sources:
  - "cp/**/*.hpp"
  - "tests/*.cpp"
  - "README.md"
---

# Implementation Details

## Entry Points

| Target | File | Purpose |
|--------|------|---------|
| Consumer library | `cp/**/*.hpp` | Directly included public templates, inline functions, and inline I/O globals |
| Component tests | `tests/*.cpp` | Independent executable entry points for correctness, stress, benchmark, or judge-style scenarios |
| Installer | `install.sh` | Copies `cp/` into `<PREFIX>/include/cp` |

There is no single application `main()` or compiled library entry point. A typical solution includes selected headers, reads through `cp::qin` or standard input, runs an in-memory algorithm, and writes the result.

## Key Algorithms & Logic

### Fast I/O

`FastInput` checks whether stdin is a non-empty regular file. If so, it maps the file for zero-copy traversal; otherwise it uses a 64 KiB stdio buffer. Typed scanning returns `std::optional`, while `FastOutput` buffers writes and uses `to_chars`/formatting helpers.

### Modular Arithmetic and Polynomials

`SModint` and `DModint` store Montgomery-form values to replace division-heavy modular multiplication with reductions. `FPoly` builds on that representation with aligned pooled storage, precomputed roots/inverses, radix-4 DIF/DIT NTT, AVX2 pointwise arithmetic, and Newton-style formal-series operations such as inverse, logarithm, exponential, and square root.

### Containers

- `FlatHashMap` uses open addressing, 16-byte control groups, 7-bit hash fragments, SSE2 matching, and Empty/Deleted/Sentinel states.
- `PairingHeap` uses node links and two-pass sibling merging; stable point iterators enable modify/erase operations.
- `Bitset` stores aligned machine words and represents chained operations as expression nodes so assignment can fuse traversal loops.
- Segment trees parameterize aggregation and update actions through concepts/callables rather than hard-coding arithmetic.

### 2D Geometry

`cp/geometry.hpp` provides value-semantic vector, line, segment, and circle templates. Signed integral predicates widen to `i64` or `i128`; determinant calculation uses sign/magnitude products to preserve cancellation, while results beyond `i128` saturate rather than invoking signed-overflow UB. Floating predicates use explicit absolute-plus-relative `GeometryTolerance` values. Intersections return fixed-size typed payloads and allocate no memory.

## Error Handling Strategy

- Concepts and `static_assert` reject incompatible operations, bitset sizes, and mathematical parameters at compile time.
- `std::optional` represents scan exhaustion and queries without a value.
- Some mathematical routines throw standard exceptions for invalid runtime cases.
- Low-level container operations often assume trusted callers and valid indices/non-empty state for speed; tests must cover these preconditions and ownership transitions.
- RAII and allocator traits own heap/map resources, while the fast-I/O objects manage buffering/mapping lifetimes.

## Testing Strategy

| Test level | Location | What it covers |
|------------|----------|----------------|
| Unit/component | `tests/test_bitset.cpp`, `test_graph.cpp`, `test_hash_value.cpp`, `test_utils_lambda.cpp`, etc. | Public operations, edge cases, comparisons, and type behavior |
| Geometry | `tests/test_geometry.cpp` | Compile-time contracts, numeric limits, primitives, metrics, tolerance behavior, and all intersection families |
| Stress/reference | `tests/test_hash_map.cpp`, `test_pairing_heap.cpp`, segment-tree tests | Randomized operations and comparison with simpler/reference containers |
| Algorithm/integration | `tests/test_dft_new.cpp`, `test_radix2.cpp`, `test_radix4.cpp`, `test_luogu.cpp` | Polynomial/NTT paths and contest-style integration |
| I/O/benchmark | `tests/test_fastio.cpp`, `test_io.cpp` | Throughput and accelerated input/output modes |

## Performance Considerations

- SIMD code assumes suitable x86 features and may require compiler target support.
- Expression-template operands must not dangle; materialize results when lifetimes are uncertain.
- The polynomial aligned pool reduces repeated allocation but makes capacity/lifetime behavior important.
- Hash-map performance depends on control-byte invariants, load-factor growth, and allocator correctness.
- Sanitizers are valuable for ownership bugs but change timing and should not be used for benchmark conclusions.
- Full-range `i64` geometry may exceed `i128`; wide results saturate at the representable boundary and README documents the resulting loss of exactness.

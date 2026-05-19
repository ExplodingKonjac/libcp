# Implementation Details

## Entry Points

This is a header-only library — there is no single entry point. Each header is self-contained and can be included independently. The only common dependency is `cp/def.hpp`.

Tests are standalone `.cpp` files in `tests/`, each compiled and run separately:

| Test file | Tests |
|-----------|-------|
| `tests/test_graph.cpp` | Graph correctness (unweighted, weighted, const access) |
| `tests/test_hash_map.cpp` | Hash map correctness + benchmark vs alternatives |
| `tests/test_pairing_heap.cpp` | Pairing heap correctness + benchmark |
| `tests/test_fenwick.cpp` | Fenwick tree correctness |
| `tests/test_segtree.cpp` | Segment tree correctness |
| `tests/test_utils_lambda.cpp` | Lambda expression template tests |
| `tests/test_hash_value.cpp` | Hash value tests |
| `tests/test_fastio.cpp` | Fast I/O benchmark (50M integers) |
| `tests/test_io.cpp` | Fast I/O with `CP_FASTIO_ACCELERATE` mode |
| `tests/test_radix2.cpp` | Radix-2 NTT polynomial multiplication benchmark |
| `tests/test_radix4.cpp` | Radix-4 AVX2 NTT benchmark |
| `tests/test_dft_new.cpp` | NTT + polynomial inverse benchmark |
| `tests/test_luogu.cpp` | Polynomial sqrt (Luogu-style test) |

## Key Algorithms & Logic

### Montgomery Modular Arithmetic (`modint.hpp`)

Values are stored in Montgomery form: `aR mod P` where `R = 2^32`. Multiplication uses `__int128` for the 128-bit intermediate product, then applies Montgomery reduction to avoid division:

1. Compute 128-bit product `T = a * b`
2. `m = T * P' mod R` (where `P'` is the modular inverse of `-P mod R`)
3. `t = (T + m * P) / R`
4. If `t >= P`, subtract `P`

### AVX2 Radix-4 NTT (`fpoly.hpp`)

Polynomial multiplication uses Number Theoretic Transform with radix-4 DIF (forward) and DIT (inverse) butterflies. Each iteration processes 8 int32 values simultaneously in `__m256i` vectors:

- Precomputed W4 and W8 twiddle factor tables stored in aligned static arrays
- Hand-written specializations for 2/4/8-point transforms to avoid overhead
- Memory managed via `AlignedPool` — a bump allocator with 64-byte alignment that reuses allocations across operations

### SSE2 Hash Table Probing (`hash_map.hpp`)

`FlatHashMap` uses open addressing with SSE2-accelerated probing. Each slot has a control byte encoding the hash's upper 7 bits plus Empty/Deleted/Sentinel markers:

1. Compute `hash(key)`, extract top 7 bits as the probe tag
2. Load 16 control bytes into `__m128i`
3. `_mm_cmpeq_epi8` to find slots matching the tag in one instruction
4. For each candidate slot, compare full keys
5. If Sentinel found, key is not present — insert at first Deleted/Empty slot seen

### Generic Fenwick Tree (`fenwick_tree.hpp`)

Parameterized on value type, addition operation, and subtraction operation, allowing use with non-numeric types (e.g., matrices, polynomials) as long as the operations form a group.

## Error Handling Strategy

- `FastInput::scan<T>()` returns `std::optional<T>` — `nullopt` on parse failure
- `modify()`/`update()` on data structures return `bool` — `false` on out-of-bounds index
- `cp::sqrt()` returns `std::optional<Mint>` — `nullopt` for quadratic non-residue
- Test assertions use `exit(1)` on failure — no exception handling in tests
- Library code generally assumes valid inputs (competitive programming context — input is guaranteed well-formed)

## Testing Strategy

| Test level | Location | What it covers |
|-----------|---------|---------------|
| Correctness | `tests/test_*.cpp` | Functional correctness with hand-written cases |
| Benchmark | `tests/test_fastio.cpp`, `tests/test_radix*.cpp`, `tests/test_dft_new.cpp` | Throughput comparison against alternatives |
| Adversarial | `tests/test_hash_map.cpp` | Collision stress testing for hash table |

Tests have no formal framework. Each test file:
1. Defines `assert_eq(a, b, msg)` macro/function
2. Contains named `test_*()` functions for each scenario
3. Has a `main()` that calls all test functions and prints "All tests passed!"

## Performance Considerations

- **mmap I/O**: `FastInput` uses `mmap` + `madvise(MADV_SEQUENTIAL)` for regular files, falling back to buffered `fgets` for pipes/stdin
- **Memory alignment**: `AlignedPool` in `fpoly.hpp` provides 64-byte alignment for AVX2 loads/stores, preventing penalties from unaligned access
- **Compile-time precomputation**: W4/W8 twiddle tables and inverse tables are computed at compile time via `constexpr`
- **Small-size specialization**: NTT has hand-written paths for 2/4/8-point transforms, avoiding the general butterfly overhead
- **Control-byte filtering**: `FlatHashMap` probes 16 slots per SSE2 instruction, dramatically reducing the number of key comparisons compared to linear probing

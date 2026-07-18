---
sources:
  - "cp/**/*.hpp"
  - "README.md"
---

# API Reference

N/A — the project exposes a C++ header/template API, not a network API. There are no HTTP/RPC endpoints, authentication rules, request schemas, rate limits, or pagination.

## Public Surface

Consumers include individual files under `cp/` and use symbols in the `cp` namespace. The component overview and examples live in `README.md`; authoritative signatures and constraints live in `cp/**/*.hpp`.

| Header area | Representative public API |
|-------------|---------------------------|
| I/O | `FastInput`, `FastOutput`, `qin`, `qout` |
| Math | `SModint`, `DModint`, `FPoly` |
| Geometry | `Vec2`, `Point2`, `Line2`, `Segment2`, `Circle2`, predicate/metric/intersection overloads |
| Data structures | `FenwickTree`, `SegTree`, `LazySegTree`, `Graph`, `FlatHashMap`, `PairingHeap`, `Bitset` |
| Utilities | fixed-width aliases, concepts, `HashValue`, placeholder expressions, compile-time formatting |

## Network Concerns

| Concern | Status |
|---------|--------|
| Endpoints | N/A |
| Authentication / authorization | N/A |
| Request / response schema | N/A |
| Rate limiting | N/A |
| Pagination | N/A |

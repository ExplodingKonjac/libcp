---
sources:
  - "cp/**/*.hpp"
  - "tests/*.cpp"
---

# Data Model

N/A for persistent data: this header-only algorithm library has no database, schema, ORM, migrations, or durable cache.

## In-Memory Structures

| Structure | Key fields / representation | Notes |
|-----------|-----------------------------|-------|
| `Graph<E>` | Vertex count and adjacency vectors of destinations/edge values | Directed, process-local graph storage |
| `FlatHashMap` | Control-byte array plus aligned key/value slots | Open addressing with Empty/Deleted/Sentinel states |
| `FenwickTree` | One-dimensional aggregate array | Prefix-based point-update structure |
| `SegTree` | Iterative `2n` aggregate array | Half-open range queries |
| `LazySegTree` | `4n` aggregate tree plus optional lazy action tags | Range updates and queries |
| `PairingHeap` | Linked heap nodes with child/sibling relationships | Allocator-owned nodes and stable point iterators |
| `Bitset<SIZE>` | Fixed aligned array of machine words | AVX2 operations and expression evaluation |
| `FPoly<P>` | Coefficient range backed by aligned pooled allocation | Transient formal polynomial state |
| Geometry primitives | Two coordinates, point/direction or endpoints, and center/radius | Value-semantic, allocation-free 2D objects |
| Geometry intersections | Kind enum plus zero, one, or two fixed point payloads | Represents point, overlap, and coincidence outcomes without persistence |

## Relationships

| From | To | Cardinality | Description |
|------|----|-------------|-------------|
| `Graph` | adjacency entry | 1:N | Each vertex owns zero or more outgoing edges |
| `PairingHeap` | node | 1:N | A heap owns a linked forest rooted at its top node |
| `LazySegTree` | lazy tag | 1:0..1 per tree node | Pending actions are composed until pushed |
| `FPoly` | coefficient | 1:N | A polynomial owns a logical coefficient sequence |
| `FlatHashMap` | slot/control byte | 1:N | Each capacity position has metadata and optional storage |

## Schema Migrations

| Aspect | Detail |
|--------|--------|
| Tool | N/A |
| Location | N/A |
| Strategy | Public C++ API and representation changes are source/version migrations, not database migrations |

## Caching

| Cache | Strategy | TTL | Invalidation |
|-------|----------|-----|--------------|
| N/A | No application cache exists | N/A | N/A |
| `FPoly` aligned pool | Reuse transient allocation blocks | Process lifetime / implementation-defined | Managed internally; not a data cache |

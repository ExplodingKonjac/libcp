---
sources:
  - ".clang-format"
  - ".clangd"
  - "cp/**/*.hpp"
  - "tests/*.cpp"
  - "README.md"
---

# Project Conventions

> Agents MUST read and follow these conventions.

## Coding Conventions

| Aspect | Rule | Config |
|--------|------|--------|
| Formatter | clang-format, Google-derived with project overrides | `.clang-format` |
| Linter | Compiler warnings and clangd diagnostics; no separate linter | `.clangd` |
| Language mode | Compile as C++23 | `.clangd`, README |
| Indentation | 4 spaces; case labels are not additionally indented | `.clang-format` |
| Braces | Custom wrapping; namespaces open on a new line | `.clang-format` |
| Parameters | Prefer one per line when wrapping | `.clang-format` |
| Documentation | clangd is configured for Doxygen comment format | `.clangd` |

## Naming Conventions

| Category | Convention | Example |
|----------|------------|---------|
| Files / modules | lowercase `snake_case.hpp` | `lazy_segtree.hpp` |
| Variables | lowercase or `snake_case` | `bit_size`, `qin` |
| Constants | `UPPER_SNAKE_CASE` for macros; descriptive compile-time members otherwise | `CP_FORMAT_STRING`, `LG_MAXN` |
| Functions / methods | lowercase `snake_case` | `add_edge`, `find_first_set` |
| Types / classes / concepts | PascalCase | `FlatHashMap`, `BitsetExpr` |
| Namespaces | lowercase; public names live under `cp` | `cp::detail`, `cp::literals` |

## Architectural Rules

- Keep the library header-only: definitions exposed by headers must be templates, class members, `constexpr`, or safely `inline` where the ODR requires it.
- Put public APIs in `namespace cp`; hide helpers in `cp::detail` or `cp::details` unless they are intentionally exported.
- Preserve compile-time constraints and explicit precondition checks around modulus sizes, expression sizes, and callable policies.
- Geometry predicates must widen before subtraction/multiplication, keep floating tolerances dimensionally consistent, and avoid signed-overflow UB.
- Avoid unnecessary allocations and abstraction overhead in hot paths; performance is a primary design goal.
- Keep component headers independently includable and include their direct dependencies.
- **Forbidden**: silently adding third-party dependencies, platform assumptions beyond those documented, or a linked runtime requirement without an explicit architecture decision.
- **Forbidden**: storing a `Bitset` expression-template result in `auto` when it may outlive referenced operands; materialize a concrete `Bitset` instead.

## File Organization

| What | Where | Notes |
|------|-------|-------|
| Public source | `cp/*.hpp` | Major reusable modules |
| Supporting source | `cp/utils/*.hpp` | Cross-module concepts and expression/format helpers |
| Tests and benchmarks | `tests/*.cpp` | One standalone executable per scenario/component |
| Tool configuration | Repository root and `.vscode/` | clangd, clang-format, compile/debug tasks |
| Documentation | `README.md`, `.agents/project-tracker/` | User guide and maintained project context |

## Import / Module Conventions

- Prefer quoted includes for project headers and angle brackets for standard/system headers.
- Within `cp/`, sibling headers often use relative includes such as `"def.hpp"`; utility headers used from the repository root may use `"cp/def.hpp"`.
- Public visibility is explicit through namespace placement; there are no C++20 module units or umbrella header.
- Avoid circular header dependencies. Keep the observed dependency direction from base aliases/concepts toward higher-level structures.

## Error Handling

- **Expected absence/input exhaustion**: use `std::optional`, as in scanning and optional range-query results.
- **Invalid runtime mathematical state**: use standard exceptions where implemented, such as domain-related polynomial/modular failures.
- **Static invariants**: use concepts and `static_assert`.
- **Programmer preconditions**: assertions or trusted-call assumptions are acceptable in contest-oriented hot paths, but must be documented and tested.
- **Resource ownership**: use RAII/allocators; manual memory and mappings must have clear cleanup paths.

## Testing Conventions

- Tests live under `tests/` and contain their own `main()`.
- Use descriptive component filenames such as `test_graph.cpp`.
- Prefer deterministic assertions plus randomized comparison against a simple standard-library/reference implementation for complex containers.
- Run sanitizer builds for memory-owning and low-level SIMD code.
- No coverage target is currently configured; new behavior should still receive focused regression coverage.

## Documentation Conventions

- Public-facing usage belongs in `README.md`; public or non-obvious APIs should use concise comments compatible with Doxygen tooling.
- Keep paths and examples synchronized with the repository. The current README link `cp/compile_format.hpp` should be updated to `cp/utils/compile_format.hpp`.
- Record important architecture changes in tracker documents rather than relying on implementation knowledge alone.

## Agent Instructions

- Preserve unrelated user changes and never overwrite existing tracker documents during initialization.
- Search with `rg`/`rg --files`, edit with patch-based changes, and verify relevant compile/test commands after code modifications.
- Follow test-driven development for new behavior and keep tests independently runnable.
- Review modified code for correctness, portability, memory safety, hardcoded secrets, and performance regressions.
- Do not treat legacy `.claude/project-tracker/` documentation as source evidence for this tracker.

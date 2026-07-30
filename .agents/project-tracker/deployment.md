---
sources:
  - "install.sh"
  - "README.md"
  - ".gitignore"
  - "cp/**/*.hpp"
---

# Deployment

There is no runtime service deployment. Distribution consists of making the header tree available to C++ consumers.

## Build Artifacts

| Artifact | Format | How to build |
|----------|--------|--------------|
| Public library | C++ header tree under `cp/` | No build required |
| Consumer/test executable | Native executable | `g++ -I. -std=c++23 -O2 -Wall <SOURCE>.cpp -o <OUTPUT>` |
| Prefix installation | `<PREFIX>/include/cp/` | `./install.sh --prefix <PREFIX>` |

## Packaging

`install.sh` requires an explicit prefix, removes the existing `<PREFIX>/include/cp` directory, recreates it, and recursively copies the repository's `cp/` contents. Nested modules such as `cp/geometry/circle.hpp` and `cp/geometry/polygon.hpp` therefore retain their include paths. The project does not produce archives, system packages, containers, or versioned binary releases.

## Environments

| Environment | Target | Promotion from | Notes |
|-------------|--------|----------------|-------|
| Development | Repository checkout | N/A | Include headers directly; compile standalone tests |
| Local install | `<PREFIX>/include/cp` | Development revision | Consumers add `<PREFIX>/include` to include paths |
| Production service | N/A | N/A | No hosted runtime exists |

## Health Checks

| Check | Command | Expected |
|-------|---------|----------|
| Header integration | Compile representative root and nested-header tests with a supported compiler | Successful C++23 compilation |
| Component behavior | Run the resulting deterministic test executable | Exit status 0 and no assertion failure |
| Memory safety | Run a sanitizer build of low-level containers | No ASan/UBSan/LSan findings |

## Monitoring & Alerts

N/A — the library has no long-running process. Consumers are responsible for application-level monitoring. Repository CI, when added, should monitor compiler compatibility and test regressions.

## Rollback Procedure

1. Select a previously known-good repository revision or release.
2. Re-run `./install.sh --prefix <PREFIX>` from that revision.
3. Rebuild dependent programs because this is a source-level library.
4. Run their relevant regression tests before redeployment.

The installer replaces the destination header directory, so preserve local modifications elsewhere or use a dedicated prefix.

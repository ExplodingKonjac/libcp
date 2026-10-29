# tests

```
tests/
  cp/    # 主库 include/cp 的测试（C++23 / SIMD）
  acm/   # 赛场版 include/acm 的测试（C++20）
```

所有测试都通过 `-Iinclude` 查找头文件，不再使用 `tests/cp` symlink。

## Makefile 目标

```bash
make test-acm      # C++20，编译并运行 tests/acm
make test-cp       # C++23，编译并运行 tests/cp（跳过 stdin 型测试）
make test          # 两者都跑
make test-acm-san  # acm + ASan/UBSan
make headers       # 逐个检查 include/acm 头文件可独立编译
make clean         # 清理 build/make
```

默认 `CXX=clang++`；可覆盖：

```bash
make CXX=g++ test-acm
```

需要 stdin 的测试（`test_luogu`、`test_dft_new`、`test_io`）默认跳过，可单独编译后手工传入输入运行。

## 单独编译

```bash
g++ -std=c++20 -O2 -Wall -Iinclude tests/acm/test_headers.cpp -o /tmp/test_headers
g++ -std=c++23 -O2 -Wall -Iinclude tests/cp/test_hash_map.cpp -o /tmp/test_hash_map
```

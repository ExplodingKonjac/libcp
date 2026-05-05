# cp-lib — 竞赛编程 C++23 高性能头文件库

用于算法竞赛（OI/ACM）的 C++23 header-only 高性能库，提供快速 IO、模运算、多项式运算、数据结构等组件，充分利用 x86 SIMD（SSE2/AVX2）指令集进行极致优化。

## 环境要求

- C++23 编译器（`g++-13+` 或 `clang-16+`）
- Linux x86-64（mmap 加速 IO 需要 Linux，部分组件使用 SSE2/AVX2 指令集）
- 编译选项：`-std=c++23 -Wall -O2`

## 使用方式

直接 `#include` 对应头文件即可。编译示例：

```bash
g++ -std=c++23 -O2 -Wall -o solve solve.cpp
```

许可：GPLv3

## 组件

| 头文件 | 说明 |
|--------|------|
| [`cp/def.hpp`](cp/def.hpp) | 基本类型别名 |
| [`cp/compile_format.hpp`](cp/compile_format.hpp) | 编译期格式串解析 |
| [`cp/fast_io.hpp`](cp/fast_io.hpp) | 快速 IO 库 |
| [`cp/modint.hpp`](cp/modint.hpp) | 模运算（Montgomery 形式） |
| [`cp/fpoly.hpp`](cp/fpoly.hpp) | 多项式运算（AVX2 加速 NTT） |
| [`cp/fenwick_tree.hpp`](cp/fenwick_tree.hpp) | 泛化树状数组 |
| [`cp/graph.hpp`](cp/graph.hpp) | 有向图容器 |
| [`cp/hash_map.hpp`](cp/hash_map.hpp) | 高性能扁平哈希表 |
| [`cp/pairing_heap.hpp`](cp/pairing_heap.hpp) | 配对堆 |

---

### `cp/def.hpp` — 类型别名

位于 `cp::inline defs` 命名空间，提供简洁的定长类型名：

```cpp
using cp::i32, cp::i64, cp::u64, cp::f64, cp::i128;
```

包含：`i8` / `u8` / `i16` / `u16` / `i32` / `u32` / `i64` / `u64` / `i128` / `u128` / `isize` / `usize` / `f32` / `f64`

---

### `cp/compile_format.hpp` — 编译期格式串

通过 `""_fmt` 字面量和 `FormatString` 在编译期解析格式字符串，分解为 `Literal` 段和 `Replacement` 段的编译期元组，**零运行时开销**。底层调用 `std::formatter` 完成格式化。

```cpp
#include "cp/compile_format.hpp"
using namespace cp::literals;

auto s = cp::format("hello {}!"_fmt, "world");  // "hello world!"
```

---

### `cp/fast_io.hpp` — 快速 IO

提供 `FastInput` 和 `FastOutput` 两个类以及全局实例 `qin` / `qout`。

- **读取**：在 Linux 下对常规文件使用 `mmap` 实现零拷贝读取；否则使用 64 KiB 缓冲区 + `fgets`
- **写入**：64 KiB 缓冲区，支持 `to_chars` 整数/浮点数快速转换
- **扫描**：定长整数、浮点数、字符串，自动跳过空白；`scan<T>()` 返回 `optional<T>`，支持多变量同时读取

```cpp
#include "cp/fast_io.hpp"
using cp::qin, cp::qout;

int n = qin.scan<int>().value();      // 读一个整数
auto [a, b] = qin.scan<int, int>().value(); // 读两个整数
qout.println("ans =", 42);            // 输出，自动空格分隔
```

> **注意**：`compile_format.hpp` 和 `fast_io.hpp` 各自定义了自己的 `FixedString` / `FormatString` / `""_fmt`，二者不兼容。如果同时使用两个模块，**在包含 `fast_io.hpp` 之前定义 `CP_FORMAT_STRING`** 以避免重定义。

---

### `cp/modint.hpp` — 模运算

基于 **Montgomery 约减** 的模整数，支持静态模数和动态模数。

- `SModint<P>` — 模数在编译期指定，要求 `P < 2^30` 且为奇数（素数）
- `DModint` — 模数可在运行时通过 `DModint::set_mod(p)` 修改，默认 `998244353`
- 使用 Montgomery 形式存储内部值，避免除法运算
- 提供 `pow`、`legendre`、`sqrt` (Tonelli-Shanks 二次剩余) 等辅助函数

```cpp
using Mint = cp::SModint<998244353>;
Mint a = 114514, b = 1919810;
Mint c = a * b + a / b;
Mint d = cp::pow(a, 1000);           // 快速幂
auto r = cp::sqrt(Mint(4));           // 二次剩余，返回 optional<Mint>
```

---

### `cp/fpoly.hpp` — 多项式运算

以 `SModint<P>` 为系数的多项式 `FPoly<P>`，支持以下运算：

- 加、减、乘（NTT 加速）、除（带余除法）
- 求逆 `inv`、对数 `ln`、指数 `exp`、平方根 `sqrt`
- 求导 `derivative`、积分 `integrate`

**优化特性**：

- 使用 **radix-4 DIF/DIT AVX2 加速 NTT**，每次处理 8 个 int32
- 编译期预计算所有旋转因子（W4、W8 表）
- 对 2/4/8 点小规模变换有手写特化
- 对齐内存池 (`AlignedPool`)，避免重复分配，支持内存复用
- 自动前置求逆元表

```cpp
using Poly = cp::FPoly<998244353>;
Poly f = {1, 2, 3};        // 1 + 2x + 3x^2
Poly g = {4, 5, 6};
Poly h = f * g;             // NTT 加速乘法
Poly i = cp::inv(f, 3);    // 模 x^3 逆
Poly l = cp::ln(f, 3);     // 模 x^3 对数
Poly e = cp::exp(f, 3);    // 模 x^3 指数
```

---

### `cp/fenwick_tree.hpp` — 泛化树状数组

支持自定义类型和运算的树状数组（Binary Indexed Tree / Fenwick Tree）。

```cpp
cp::FenwickTree<long long> bit(n, std::plus<>{}, std::minus<>{});
bit.add(p, x);                // 单点加
auto s = bit.sum(l, r);       // 区间和
auto p = bit.pre_sum(r);      // 前缀和
auto s = bit.suf_sum(l);      // 后缀和
```

---

### `cp/graph.hpp` — 有向图容器

基于邻接表的有向图，提供 range-based 遍历。

- `Graph<void>` — 无权图
- `Graph<E>` — 边权类型为 `E` 的有权图

```cpp
cp::Graph<void> g(n);
g.add_edge(u, v);
for (auto w : g.out(u)) { ... }           // 遍历出边
for (auto [u, v] : g.edges()) { ... }     // 遍历所有边

cp::Graph<int> gw(n);
gw.add_edge(u, v, w);
for (auto [to, weight] : gw.out(u)) { ... }
```

---

### `cp/hash_map.hpp` — 扁平哈希表

高性能开放寻址哈希表 `FlatHashMap`，类似 Swiss Table / Abseil `flat_hash_map` 设计。

- **SSE2 加速探测**：每次检查 16 个 slot（一个 `__m128i`）
- **控制字节**：每个 slot 对应一个 `u8`，编码 7 位 hash + Empty/Deleted/Sentinel 标记
- **查找快速路径**：先匹配 hash 再比较 key，减少不必要比较
- **删除标记**：使用 Deleted 标记而非 tombstone 压缩
- 支持自定义 hash、相等比较器、分配器

```cpp
cp::FlatHashMap<int, int> map;
map.try_emplace(1, 100);                // 插入
if (auto* v = map.get(1)) { ... }       // 查找
map.erase(1);                           // 删除
for (auto& [k, v] : map) { ... }        // 遍历
```

---

### `cp/pairing_heap.hpp` — 配对堆

实现配对堆（Pairing Heap），支持高效的合并与修改操作。

- `push` / `pop` / `top`
- `modify(iter, val)` — 修改任意元素的值（需要堆的稳定性更好）
- `erase(iter)` — 删除任意元素
- `join(other)` — 合并两个堆
- `point_iterator` — 操作后保持有效的迭代器（指向之前插入结点的迭代器）

```cpp
cp::PairingHeap<int> heap;
auto it = heap.push(42);
heap.modify(it, 100);        // 修改值
int mx = heap.pop();         // 弹出最大值（默认 max-heap）
```

对应 Compare `cp::PairingHeap<int, std::greater<>>` 则为小根堆。

## 构建测试

使用 `g++` 直接编译测试文件，推荐开启 sanitizer：

```bash
g++ tests/test_hash_map.cpp -std=c++23 -O2 -Wall -o test_hash_map
./test_hash_map
```

调试构建（开启 sanitizer）：

```bash
g++ tests/test_hash_map.cpp -std=c++23 -g -fsanitize=address,undefined,leak \
  -o test_hash_map -DDADALZY -Wall -Wno-unused-result
```

VSCode 用户可使用 `.vscode/tasks.json` 的 "C++ Compile" 任务一键编译当前文件。

测试文件列表：

| 文件 | 测试目标 |
|------|----------|
| `tests/test_fastio.cpp` | 快速 IO 基准（5000 万整数读取） |
| `tests/test_io.cpp` | 快速 IO（CP_FASTIO_ACCELERATE 模式） |
| `tests/test_hash_map.cpp` | 哈希表正确性 + 对比 benchmark |
| `tests/test_pairing_heap.cpp` | 配对堆正确性 + 对比 benchmark |
| `tests/test_fenwick.cpp` | 树状数组正确性 |
| `tests/test_graph.cpp` | 图容器正确性 |
| `tests/test_radix2.cpp` | 多项式乘法基准（radix-2 NTT） |
| `tests/test_radix4.cpp` | 多项式乘法基准（radix-4 AVX2 NTT） |
| `tests/test_dft_new.cpp` | 多项式 NTT + inv 基准 |
| `tests/test_luogu.cpp` | 多项式 sqrt（Luogu 风格测试） |

## 设计原则

- **Header-Only**：所有代码在头文件中，无需编译库文件
- **显式优化**：关键路径（IO、NTT、哈希探测）直接使用平台 SIMD 指令
- **泛型设计**：数据结构通过模板支持自定义类型、运算和分配器
- **安全优先**：Montgomery 形式避免溢出，`optional` 返回值处理异常输入
- **RAII 一致**：所有组件正确管理资源，支持移动语义和复制

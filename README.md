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
| [`cp/interpolation.hpp`](cp/interpolation.hpp) | 多项式插值与系数求解 |
| [`cp/fenwick_tree.hpp`](cp/fenwick_tree.hpp) | 泛化树状数组 |
| [`cp/graph.hpp`](cp/graph.hpp) | 有向图容器 |
| [`cp/suffix_automaton.hpp`](cp/suffix_automaton.hpp) | 后缀自动机 |
| [`cp/suffix_array.hpp`](cp/suffix_array.hpp) | 后缀数组，支持后缀排序、排名和最长公共前缀 |
| [`cp/max_flow.hpp`](cp/max_flow.hpp) | 最大流 |
| [`cp/min_cost_flow.hpp`](cp/min_cost_flow.hpp) | 最小费用流 |
| [`cp/min_cost_circulation.hpp`](cp/min_cost_circulation.hpp) | 最小费用可行环流 |
| [`cp/number_theory.hpp`](cp/number_theory.hpp) | 数论算法 |
| [`cp/geometry.hpp`](cp/geometry.hpp) | 二维向量与基础计算几何 |
| [`cp/geometry/circle.hpp`](cp/geometry/circle.hpp) | 圆、圆交点与最小覆盖圆 |
| [`cp/geometry/polygon.hpp`](cp/geometry/polygon.hpp) | 多边形与凸几何算法 |
| [`cp/hash_map.hpp`](cp/hash_map.hpp) | 高性能扁平哈希表 |
| [`cp/pairing_heap.hpp`](cp/pairing_heap.hpp) | 配对堆 |
| [`cp/bitset.hpp`](cp/bitset.hpp) | 高性能位集（AVX2 优化 bitset） |

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

### `cp/interpolation.hpp` — 多项式插值

提供拉格朗日插值求值和根据点值恢复多项式系数：

- `polynomial_interpolation(points, x)` — 计算插值多项式在 `x` 处的值
- `polynomial_coefficients<T>(points, eq)` — 返回升幂排列的系数

节点横坐标必须互异。系数计算包含除法，`T` 应支持适合问题的精确除法（通常使用浮点类型或域类型）。

```cpp
#include <functional>
#include <vector>
#include "cp/interpolation.hpp"

std::vector<std::pair<double, double>> points{{0, 5}, {1, 4}, {2, 7}};
double value = cp::polynomial_interpolation(points, 3);  // 14
auto coefficients = cp::polynomial_coefficients<double>(
    points, std::equal_to<>{}
);  // {5, -3, 2}
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

### `cp/suffix_automaton.hpp` — 后缀自动机

`SuffixAutomaton<SymbolT, MapT>` 在线构造字符串的后缀自动机，支持判断子串和遍历状态转移。
`MapT` 需要满足 `sam_symbol_map`，库内提供适合小整数符号集的 `DenseMap<Symbol, N>`。

```cpp
using SAM = cp::SuffixAutomaton<char, cp::DenseMap<char, 128>>;
SAM sam;
for (char c : std::string{"ababa"}) sam.extend(c);

cp::usize state = 0;
for (char c : std::string{"bab"}) {
    auto next = sam.transition(state, c);
    if (!next) break;                 // 不是原串的子串
    state = *next;
}
```

`size()` 返回状态数，`last()` 返回当前最后状态；`link(i)`、`max_len(i)` 和
`transitions(i)` 分别访问状态的后缀链接、最长字符串长度和转移边。

---

### `cp/suffix_array.hpp` — 后缀数组

`SuffixArray<Symbol>` 从随机访问范围构造后缀数组，提供按字典序排列的后缀位置、
后缀排名以及相邻后缀的最长公共前缀长度。支持字符串、整数序列等可比较的符号类型，
也支持通过类模板参数推导直接构造。

```cpp
#include "cp/suffix_array.hpp"

cp::SuffixArray suffix_array(std::string{"banana"});
auto position = suffix_array.sa(0);   // 5，最小后缀 "a"
auto rank = suffix_array.rk(1);       // 2，"anana" 的排名
auto common = suffix_array.height(2); // 3，与前一后缀的 LCP
```

`sa(i)` 返回第 `i` 个后缀的起始位置，`rk(i)` 返回从位置 `i` 开始的后缀排名，
`height(i)` 返回第 `i` 个后缀与前一个后缀的最长公共前缀长度；`height(0)` 始终为零。

---

### `cp/max_flow.hpp` — 最大流

`MaxFlow<FlowT>` 使用 Dinic 算法计算整数容量网络的最大流，`max_flow(s, t)` 返回最大流值。
边编号从 0 开始，`edge(i)` 可读取对应的边信息。

```cpp
#include "cp/max_flow.hpp"

cp::MaxFlow<int> flow(n);
flow.add_edge(u, v, capacity);
int value = flow.max_flow(source, sink);
```

---

### `cp/min_cost_flow.hpp` — 最小费用流

`MinCostFlow<FlowT, CostT>` 支持整数容量和有符号费用。
`min_cost_flow(s, t)` 返回费用最小的流，`min_cost_max_flow(s, t)` 返回最大流量下费用最小的流，
二者均返回 `{flow, cost}`。

```cpp
#include "cp/min_cost_flow.hpp"

cp::MinCostFlow<int, int> costs(n);
costs.add_edge(u, v, capacity, cost);
auto [sent, total_cost] = costs.min_cost_max_flow(source, sink);
```

`negative_cost` 参数在残量网络可能含负费用时设为 `true`；默认值适用于所有相关费用非负的初始网络。

---

### `cp/min_cost_circulation.hpp` — 最小费用可行环流

`MinCostCirculation<FlowT, CostT>` 支持带点供给和边费用的最小费用可行环流。
通过 `supply(u)` 设置点供给，`circulation()` 返回最小费用，若无可行流则返回 `std::nullopt`。

```cpp
#include "cp/min_cost_circulation.hpp"

cp::MinCostCirculation<int, int> circulation(n);
circulation.supply(source) = amount;
circulation.supply(sink) = -amount;
circulation.add_edge(u, v, capacity, cost);
auto total_cost = circulation.circulation();
```

所有点供给之和必须为零，`FlowT` 和 `CostT` 必须为有符号类型。

---

### `cp/number_theory.hpp` — 数论算法

提供模幂、扩展欧几里得、一次二元不定方程、Miller–Rabin 素性测试、Pollard–Rho 分解、Legendre
符号、Cipolla 开平方，以及可组合的 Universal-Uclidean Algorithm（`uniclidean`）。

```cpp
#include "cp/number_theory.hpp"

using i64 = cp::i64;
std::mt19937_64 gen(123456789);

i64 r = cp::qpow<i64>(2, 30, 1'000'000'007);
auto factors = cp::factorize<i64>(8051, gen); // {83, 97}
auto root = cp::cipolla<i64>(10, 13, gen);    // optional<i64>
```

这些接口目前面向有符号整数；`qpow`、`miller_rabin`、`legendre` 和 `cipolla` 的模数需满足各自注释中的
数学前提（例如素数或奇素数）。随机算法通过调用者传入的生成器取样；`factorize` 返回的质因子未保证排序。

---

### `cp/geometry.hpp` — 二维计算几何

提供泛型二维向量/点 `Vec2<T>`（`Point2<T>` 为同类型别名）与直线 `Line2<T>`。
支持向量运算、方向判断、投影/反射、距离和直线交点计算。圆与多边形算法分别位于
`cp/geometry/circle.hpp` 和 `cp/geometry/polygon.hpp`。

```cpp
#include "cp/geometry.hpp"

cp::Point2<cp::i64> a{0, 0}, b{4, 0}, p{2, 3};
auto line = cp::Line2<cp::i64>::through(a, b);
auto projected = cp::project(p, line);         // Point2<long double>{2, 0}
bool between = cp::on_segment(a, b, {2, 0});   // true
int sign = cp::sgn(-3.0);                      // -1
int order = cp::cmp(1.0, 1.0 + 5e-10);        // 0
```

数值规则：

- 坐标类型必须是有符号整数或浮点数；混合坐标类型需显式 `cast<U>()`。
- `i8`/`i16` 中间结果提升到 `i64`，`i32`/`i64` 提升到 `i128`；整数交点坐标和距离提升到 `long double`。
- 整数运算会提升到 `geometry_wide_t<T>`；若数学结果超出该类型的表示范围，则行为不受支持。
- 所有浮点几何判断统一使用绝对容差 `cp::geometry_eps`，默认值为 `1e-9L`。
- `Line2<T>::through(a, b)` 要求整数端点差可由 `T` 表示；调试构建会检查此前置条件。直接使用点和方向构造可避免该转换。
- 零方向直线无效。对无效元素调用几何算法属于前置条件错误。

可在包含任何几何头文件前通过宏修改整个程序的容差；所有翻译单元必须使用相同配置：

```cpp
#define CP_GEOMETRY_EPS 1e-7L
#include "cp/geometry.hpp"
```

直线交点结果使用无动态分配的 `LineIntersection2`，并通过
`LineIntersectionKind` 区分无交点、单点或重合。

---

### `cp/geometry/circle.hpp` — 圆与最小覆盖圆

提供圆 `Circle2<T>`、点与圆的包含关系、点到圆/圆盘的距离、直线与圆及两个圆的交点，
三点构造圆 `circle_from()`，以及期望线性时间的随机增量最小覆盖圆算法。空点集没有
最小覆盖圆，返回 `std::nullopt`；整数坐标的构造结果使用 `long double`。

```cpp
#include "cp/geometry/circle.hpp"

cp::Circle2<double> circle{{0, 0}, 5};
auto hits = cp::intersection(
    cp::Line2<double>{{0, 0}, {1, 0}}, circle
);  // (-5, 0), (5, 0)

auto circumcircle = cp::circle_from(
    cp::Point2<int>{0, 0}, cp::Point2<int>{4, 0}, cp::Point2<int>{0, 3}
);  // Circle2<long double>{{2, 1.5}, 2.5}

auto cover = cp::minimum_enclosing_circle(
    std::vector<cp::Point2<int>>{{0, 0}, {4, 0}, {0, 3}}
);  // optional<Circle2<long double>>
```

三点不共线时，`circle_from()` 返回外接圆；三点共线时，返回由最远点对作为直径的圆。
零半径圆有效，负半径圆无效。圆交点通过 `PointIntersectionKind` 区分无交点、单点、
双交点或重合。

---

### `cp/geometry/polygon.hpp` — 多边形与凸几何

提供逆时针存储的 `Polygon<T>`、面积、凸性和点与多边形的位置关系，以及上下凸壳、
完整凸包、凸多边形 Minkowski 和与半平面交。点的位置由
`PointPolygonRelation::{outside,boundary,inside}` 区分。

```cpp
#include "cp/geometry/polygon.hpp"

cp::Polygon<int> polygon{
    std::vector<cp::Point2<int>>{{0, 0}, {4, 0}, {4, 3}, {0, 3}}
};
auto area = polygon.area();                   // 12.0L
auto where = polygon.relation({2, 1});        // inside
auto hull = cp::convex_hull(polygon.vertices());
```

半平面由有向直线表示，直线左侧为可行区域。
`half_plane_intersection()` 仅在交集为非退化有界多边形时返回结果；空集、无界区域、
点或其他线性退化区域均返回 `std::nullopt`。`minkowski_sum()` 要求输入多边形为凸多边形。

---

### `cp/hash_map.hpp` — 扁平哈希表

高性能开放寻址哈希表 `FlatHashMap`，类似 Swiss Table / Abseil `flat_hash_map` 设计。

- **SSE2 加速探测**：每次检查 16 个 slot（一个 `__m128i`）
- **控制字节**：每个 slot 对应一个 `u8`，编码 7 位 hash + Empty/Deleted/Sentinel 标记
- **查找快速路径**：先匹配 hash 再比较 key，减少不必要比较
- **删除标记**：使用 Deleted 标记而非 tombstone 压缩
- 支持自定义 hash、相等比较器、分配器
- `insert` / `emplace`：重复 key 时返回 `std::nullopt`，否则返回新元素迭代器
- `try_insert` / `try_emplace`：重复 key 时保留原值并返回其迭代器
- `try_insert_with`：仅在 key 不存在时调用工厂函数构造 value

```cpp
cp::FlatHashMap<int, int> map;
map.try_emplace(1, 100);                // 原地插入
map.try_insert_with(3, [] { return 300; }); // 延迟构造 value
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

---

### `cp/bitset.hpp` - 位集

实现位集（Bitset），以 AVX2 指令集进行优化，支持一组高效的位运算操作：

- 按位与 / 或 / 异或 / 集合差（反向与非）/ 左移 / 右移
- 单点 / 区间 / 整体设为 1 / 设为 0 / 取反
- Find first set / unset
- 集合关系比较，子集 / 真子集 / 包含 / 真包含
- popcount

**优化特性**：

- 使用 AVX2 指令集加速大部分操作，每次处理 $4$ 个字，即 $256$-bits。
- 利用表达式模板（Expression Template）优化运算嵌套，优雅而高效地实现循环融合的功能。
- 速度可达 `std::bitset` 的两倍（popcnt，shift）至四倍（运算嵌套）。

> **注意**：由于表达式模板的特性，请不要在包含 `cp::Bitset` 的表达式中使用 `auto` 关键字，除非你明确知道你在干什么。错误地使用 `auto` 关键字可能带来悬垂引用问题。

```cpp
cp::Bitset<50> a, b, c;
a.set_all();                       // 整体设为 1
a.unset_bit(4);                    // 单点设为 0
b.flip_range(1, 3);                // 区间 [1, 1+3) 取反
c = a ^ b;                         // 按位异或
c -= a;                            // 集合差，即“在 c 中且不在 a 中的元素”
size_t cnt = c.popcnt();           // 求 1 的个数
size_t pos = c.find_first_set(2);  // 求 [2, 50) 中第一个 1 的位置
bool supset = a > b;               // 判断 a 真包含 b
bool subseteq = c <= b;            // 判断 c 是 b 的子集
```

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
| `tests/test_suffix_automaton.cpp` | 后缀自动机子串判断、转移与状态结构 |
| `tests/test_geometry.cpp` | 二维向量、直线、距离与直线交点正确性 |
| `tests/test_geometry_epsilon.cpp` | 自定义全局几何容差 |
| `tests/test_circle.cpp` | 圆、圆交点与最小覆盖圆 |
| `tests/test_polygon.cpp` | 多边形、凸包、Minkowski 和与半平面交 |
| `tests/test_interpolation.cpp` | 多项式插值与系数求解 |
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

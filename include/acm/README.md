# acm — 赛场抄写版

`include/acm/` 是与主库 `include/cp/` 并存的 ACM 赛场版：

- 命名空间 `acm`
- 目标标准 **C++20**
- 无 SIMD、无内存池、无 allocator、无表达式模板
- 内部变量尽量短（`n, m, g, t, e, d, q, vis, dis, f, h`）
- 每个头文件可单独复制/编译

## 包含的组件

| 头文件 | 内容 | 复杂度 |
|--------|------|--------|
| `def.hpp` | `i8…u64, i128, u128, isize, usize, f32, f64` | — |
| `fast_io.hpp` | `FastIn`/`FastOut`、`qin`/`qout` | O(字符数) |
| `modint.hpp` | `SModint<P>`、`DModint`、`pow/legendre/sqrt` | 单次运算 O(1)，幂 O(log e) |
| `number_theory.hpp` | `qpow/exgcd/biequation/is_prime/factorize/legendre/cipolla/uniclidean` | 见函数注释 |
| `suffix_array.hpp` | 后缀数组、排名、LCP | O(n log n) |
| `suffix_automaton.hpp` | 固定字符集后缀自动机 | O(n) 摊还 |
| `bipartite_matching.hpp` | Hopcroft–Karp | O(E√V) |
| `bipartite_weighted_matching.hpp` | Hungarian | O(n²m) |
| `general_matching.hpp` | Edmonds Blossom | O(n³) |
| `general_weighted_matching.hpp` | Weighted Blossom | O(n³) |
| `max_flow.hpp` | Dinic 最大流 | O(V²E) |
| `min_cost_flow.hpp` | 势能 Dijkstra + Dinic | O(FE log V) |
| `min_cost_circulation.hpp` | 网络单纯形 | 实际运行高效 |
| `fpoly.hpp` | 多项式 NTT、求逆/对数/指数/开方/带余除法 | O(n log n) |
| `interpolation.hpp` | 拉格朗日插值、系数恢复 | O(n²) |
| `geometry.hpp` | 向量、直线、线段、精确谓词、浮点交点 | 谓词 O(1) |
| `geometry/circle.hpp` | 圆、圆交点、三点圆、最小覆盖圆 | 最小覆盖圆期望 O(n) |
| `geometry/polygon.hpp` | 多边形、凸包、Minkowski 和、半平面交 | 凸包/HPI O(n log n) |
| `all.hpp` | 包含以上全部头文件 | — |

以下主库组件**不在** acm 版本中：`hash_value`、`fenwick_tree`、`segtree`、`lazy_segtree`、`bitset`、`graph`，以及 `utils/` 下的工程工具。

## 整数/浮点几何的分工

- 精确整数部分接受整数 `T`：`dot/cross/norm_sq/distance_sq`、`orientation`、`line_side`、`on_segment`、`parallel`、`perpendicular`、`on_line`、`Polygon` 的构造/`area2`/凸性/点关系/凸包/Minkowski 和、`Circle2` 的 `on_circle/contains`。
- 可能产生有理数或无厘数结果的部分用 `requires std::floating_point<T>` 拒绝整数：`Vec2::operator/`、`norm/distance/angle/rotate/normalized`、`project/reflect`、线/线段交点、线圆/圆圆交点、`circle_from`、`minimum_enclosing_circle`、`half_plane_intersection`、`distance_to_circle/disk`。
- `Polygon<int>::area()` 返回 `long double`，由精确整数 `area2()` 转换而来，不做整数除法截断。

`SModint<P>` / `DModint` 用 `x()` 取模意义下的值（不再提供 `val()`）。

## FPoly

`FPoly<P>` 继承 `std::vector<SModint<P>>`，因此可以直接使用 `size/resize/data/begin/end/operator[]`，并额外提供多项式运算符和 NTT。

```cpp
using Poly = acm::FPoly<998244353>;
Poly f{1, 2, 3}, g{4, 5};
Poly h = f * g;
Poly fi = acm::inv(f);
Poly fl = acm::ln(f);
Poly fe = acm::exp(f);
```

- `FPoly<P>::Mint` 是系数类型，`FPoly<P>::G` 是原根，`FPoly<P>::LG = ctz(P-1)`。
- 原根信息 `w[k]`/`iw[k]` 在编译期预处理为 `2^k` 次单位根及其逆。
- NTT 使用转置形式：正向 DIF、逆向 DIT，不做位翻转；`dft/idft/convolution` 等均为 `FPoly<P>` 的 static 函数。
- 要求 `P` 为奇素数，且变换长度不超过 `2^LG`；小规模卷积自动走朴素实现。

## 最短使用示例

```cpp
#include "acm/fast_io.hpp"
using namespace acm;

int main() {
    i32 n;
    qin.read(n);
    qout.println(n);
}
```

```cpp
#include "acm/max_flow.hpp"
#include "acm/modint.hpp"
#include "acm/fpoly.hpp"

acm::MaxFlow<i64> mf(n);
mf.add_edge(u, v, cap);
auto flow = mf.max_flow(s, t);

using Mint = acm::SModint<998244353>;
using Poly = acm::FPoly<998244353>;
Poly f{1, 2, 3};
auto g = f * acm::inv(f);
```

## 编译

```bash
g++ -std=c++20 -O2 -Iinclude solve.cpp -o solve
```

`include/acm` 不依赖 `include/cp`。所有头文件也可以按需单独复制到赛场代码中。

## 测试

```bash
make test-acm      # 编译并运行 tests/acm
make test-acm-san  # ASan/UBSan
make headers       # 逐头文件独立编译检查
```

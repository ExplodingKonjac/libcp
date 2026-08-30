#include <algorithm>
#include <cassert>
#include <vector>
#include "acm/segtree.hpp"

int main() {
    std::vector<int> a{5, 1, 4, 2, 3};
    acm::SegTree<int> sum(a);
    assert(sum.prod(1, 4) == 7);
    sum.set(2, 9);
    assert(sum.sum(0, 5) == 20);
    acm::SegTree<int, decltype([](int x, int y) { return std::max(x, y); })>
        mx(a, -1, {});
    assert(mx.prod(0, 5) == 5);
}

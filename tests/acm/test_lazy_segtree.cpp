#include <cassert>
#include "acm/lazy_segtree.hpp"

int main() {
    auto merge = [](long long x, long long y) { return x + y; };
    auto apply = [](long long& x, const long long& d, acm::usize n) { x += d * n; };
    auto compose = [](long long& x, const long long& y) { x += y; };
    acm::LazySegTree<long long> st(6, merge, apply, compose);
    assert(st.apply(1, 5, 3));
    assert(*st.query(0, 6) == 12);
    st.modify(2, 10);
    assert(*st.query(1, 4) == 16);
}

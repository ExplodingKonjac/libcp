#include <cassert>
#include "acm/fenwick_tree.hpp"

int main() {
    acm::FenwickTree<long long> bit(8);
    for (int i = 0; i < 8; i++) bit.add(i, i + 1);
    assert(bit.pre_sum(8) == 36);
    assert(bit.sum(2, 6) == 18);
    assert(bit.suf_sum(6) == 15);
}

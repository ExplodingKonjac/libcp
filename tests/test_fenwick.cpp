#include "cp/fast_io.hpp"
#include "cp/fenwick_tree.hpp"

using cp::qin, cp::qout, cp::FenwickTree;

int main() {
    FenwickTree<int> t(4);
    t.add(0, 114);
    t.add(1, 514);
    t.add(2, 1919);
    t.add(3, 810);

    qout.println("sum of [1, 2):", t.sum(1, 2));
    qout.println("sum of [0, 3):", t.sum(0, 3));
    qout.println("prefix sum of 2: ", t.pre_sum(2));
    qout.println("suffix sum of 2: ", t.suf_sum(2));
    return 0;
}

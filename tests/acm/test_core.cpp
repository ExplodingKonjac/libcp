#include <bitset>
#include <cassert>
#include <random>
#include <unordered_map>
#include <vector>

#include "acm/bitset.hpp"
#include "acm/fast_io.hpp"
#include "acm/fenwick_tree.hpp"
#include "acm/fpoly.hpp"
#include "acm/graph.hpp"
#include "acm/hash_map.hpp"
#include "acm/lazy_segtree.hpp"
#include "acm/pairing_heap.hpp"
#include "acm/segtree.hpp"

int main() {
    acm::FenwickTree<long long> bit(8);
    for (int i = 0; i < 8; i++) bit.add(i, i + 1);
    assert(bit.sum(2, 6) == 18 && bit.suf_sum(6) == 15);
    std::vector<int> v{5, 1, 4, 2, 3};
    acm::SegTree<int> seg(v);
    assert(seg.prod(1, 4) == 7);
    auto merge = [](long long x, long long y) { return x + y; };
    auto apply = [](long long& x, const long long& d, acm::usize n) {
        x += d * n;
    };
    auto compose = [](long long& x, const long long& y) { x += y; };
    acm::LazySegTree<long long> lazy(6, merge, apply, compose);
    lazy.apply(1, 5, 3);
    lazy.modify(2, 10);
    assert(*lazy.query(1, 4) == 16);
    acm::PairingHeap<int> heap;
    auto p = heap.push(2);
    heap.push(7);
    heap.modify(p, 9);
    assert(heap.pop() == 9 && heap.pop() == 7);
    acm::Bitset<130> bits;
    std::bitset<130> expected;
    std::mt19937 rng(1);
    for (int i = 0; i < 130; i++)
        if (rng() & 1) bits.set_bit(i), expected.set(i);
    bits.flip_range(17, 80);
    for (int i = 17; i < 97; i++) expected.flip(i);
    bits <<= 19;
    expected <<= 19;
    for (int i = 0; i < 130; i++) assert(bits[i] == expected[i]);
    acm::FlatHashMap<int, int> map;
    std::unordered_map<int, int> oracle;
    for (int i = 0; i < 3000; i++) {
        int k = rng() % 100;
        if (rng() & 1) map[k]++, oracle[k]++;
        else assert(map.erase(k) == (oracle.erase(k) != 0));
        assert(map.size() == oracle.size());
        for (auto [x, y]: oracle) assert(map.contains(x) && *map.get(x) == y);
    }
    acm::Graph<void> graph(3);
    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    assert(graph.e_size() == 2 && graph.out(0)[1] == 2);
    using Poly = acm::FPoly<998244353>;
    using Mint = Poly::Mint;
    std::vector<Mint> x(40), y(35), want(74);
    for (auto& z: x) z = rng() % 1000;
    for (auto& z: y) z = rng() % 1000;
    for (int i = 0; i < 40; i++)
        for (int j = 0; j < 35; j++) want[i + j] += x[i] * y[j];
    Poly product = Poly(x.begin(), x.end()) * Poly(y.begin(), y.end());
    for (int i = 0; i < 74; i++) assert(product[i] == want[i]);
    Poly f{1, 2, 3, 4};
    auto one = f * acm::inv(f);
    one.resize(4);
    assert(one[0] == Mint(1) && !one[1] && !one[2] && !one[3]);
}

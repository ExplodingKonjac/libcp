#include <algorithm>
#include <cassert>
#include <functional>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "acm/pairing_heap.hpp"

using acm::usize;

void test_basic() {
    acm::PairingHeap<int> h;
    assert(h.empty());
    assert(h.size() == 0);

    auto a = h.push(3);
    auto b = h.push(1);
    auto c = h.push(4);
    assert(h.size() == 3);
    assert(h.top() == 4);
    assert(*c == 4);
    assert(*a == 3 && *b == 1);

    assert(h.pop() == 4);
    h.modify(b, 9);
    assert(h.top() == 9);
    h.modify(b, 2);
    assert(h.top() == 3);
    assert(h.erase(a) == 3);
    assert(h.size() == 1);
    assert(h.top() == 2);
    assert(h.pop() == 2);
    assert(h.empty());
}

void test_single() {
    acm::PairingHeap<int> h;
    auto it = h.push(42);
    assert(h.size() == 1);
    assert(h.top() == 42 && *it == 42);

    h.modify(it, 99);
    assert(h.top() == 99);
    h.modify(it, 1);
    assert(h.top() == 1);
    assert(h.erase(it) == 1);
    assert(h.empty());

    h.push(7);
    assert(h.pop() == 7);
    assert(h.empty());
}

void test_copy_move() {
    acm::PairingHeap<int> a;
    for (int i = 0; i < 100; i++) a.push(i);

    acm::PairingHeap<int> b(a);
    assert(b.size() == 100);
    assert(b.top() == 99);
    a.pop();
    assert(a.top() == 98);
    assert(b.top() == 99);

    acm::PairingHeap<int> c;
    c.push(1000);
    c = b;
    assert(c.size() == 100 && c.top() == 99);
    c.pop();
    assert(b.top() == 99 && c.top() == 98);

    acm::PairingHeap<int>& same = c;
    auto& self = (c = same);
    assert(&self == &c);
    assert(c.size() == 99);

    acm::PairingHeap<int> d(std::move(c));
    assert(d.size() == 99 && c.empty());

    acm::PairingHeap<int> e;
    e.push(7);
    e = std::move(d);
    assert(e.top() == 98 && d.empty());
}

void test_min_heap() {
    acm::PairingHeap<int, std::greater<int>> h;
    h.push(30);
    h.push(10);
    h.push(20);
    assert(h.top() == 10);
    assert(h.pop() == 10);
    assert(h.top() == 20);
    assert(h.pop() == 20);
    assert(h.pop() == 30);
    assert(h.empty());
}

void test_emplace() {
    acm::PairingHeap<std::pair<int, int>> h;
    auto it = h.emplace(3, 5);
    h.emplace(1, 2);
    h.emplace(3, 7);
    assert(h.top() == std::make_pair(3, 7));
    assert(*it == std::make_pair(3, 5));
    assert(h.erase(it) == std::make_pair(3, 5));
    assert(h.size() == 2);
}

void test_join() {
    acm::PairingHeap<int> a, b;
    for (int i = 0; i < 500; i++) a.push(i);
    for (int i = 500; i < 1000; i++) b.push(i);
    a.join(b);
    assert(a.size() == 1000);
    assert(b.empty());
    for (int i = 999; i >= 0; i--) assert(a.pop() == i);
    assert(a.empty());
}

void test_clear() {
    acm::PairingHeap<int> h;
    for (int i = 0; i < 1000; i++) h.push(i);
    h.clear();
    assert(h.empty());
    assert(h.size() == 0);
    h.push(5);
    h.push(10);
    assert(h.top() == 10 && h.size() == 2);
}

void test_sorted_output() {
    const int n = 10000;
    std::mt19937 rng(42);
    std::vector<int> data(n);
    acm::PairingHeap<int> h;
    for (int i = 0; i < n; i++) {
        data[i] = int(rng());
        h.push(data[i]);
    }
    std::sort(data.begin(), data.end(), std::greater<int>());
    for (int i = 0; i < n; i++) {
        assert(h.top() == data[i]);
        assert(h.pop() == data[i]);
    }
    assert(h.empty());
}

void test_random() {
    const int n = 2000;
    std::mt19937 rng(20261103);
    acm::PairingHeap<int> h;
    std::multiset<int> ref;
    std::vector<acm::PairingHeap<int>::point_iterator> its;

    for (int i = 0; i < n; i++) {
        int v = int(rng() % 1000000);
        its.push_back(h.push(v));
        ref.insert(v);
    }
    assert(h.size() == usize(n));
    assert(h.top() == *ref.rbegin());

    for (int i = 0; i < n; i++) {
        int v = int(rng() % 1000000);
        ref.erase(ref.find(*its[i]));
        ref.insert(v);
        h.modify(its[i], v);
        assert(*its[i] == v);
    }
    assert(h.size() == usize(n));
    assert(h.top() == *ref.rbegin());

    std::vector<char> alive(n, true);
    for (int i = 0; i < n / 2; i++) {
        int k = int(rng() % n);
        if (!alive[k]) continue;
        int old = *its[k];
        ref.erase(ref.find(old));
        assert(h.erase(its[k]) == old);
        alive[k] = false;
    }
    assert(h.size() == ref.size());

    for (auto it = ref.rbegin(); it != ref.rend(); ++it) {
        assert(h.top() == *it);
        assert(h.pop() == *it);
    }
    assert(h.empty());
}

int main() {
    test_basic();
    test_single();
    test_copy_move();
    test_min_heap();
    test_emplace();
    test_join();
    test_clear();
    test_sorted_output();
    test_random();
    return 0;
}

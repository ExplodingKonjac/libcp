#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cp/link_cut_tree.hpp"

using namespace cp;

struct MaxOp {
    int operator()(int a, int b) const { return std::max(a, b); }
};

template <typename T, typename Op>
struct Naive {
    usize n;
    std::vector<std::vector<usize>> adj;
    std::vector<T> val;
    Op op;

    Naive(usize n, std::vector<T> values, Op op = {}):
        n{n}, adj(n), val{std::move(values)}, op{std::move(op)} {}

    bool connected(usize s, usize t) const {
        if (s == t) return true;
        std::vector<char> vis(n);
        std::queue<usize> q;
        q.push(s);
        vis[s] = true;
        while (!q.empty()) {
            usize u = q.front();
            q.pop();
            for (usize v: adj[u]) {
                if (vis[v]) continue;
                vis[v] = true;
                q.push(v);
            }
        }
        return vis[t];
    }
    std::optional<std::vector<usize>> path_nodes(usize s, usize t) const {
        std::vector<usize> par(n, usize(-1));
        std::queue<usize> q;
        q.push(s);
        par[s] = s;
        while (!q.empty()) {
            usize u = q.front();
            q.pop();
            if (u == t) break;
            for (usize v: adj[u]) {
                if (par[v] != usize(-1)) continue;
                par[v] = u;
                q.push(v);
            }
        }
        if (par[t] == usize(-1)) return std::nullopt;
        std::vector<usize> path;
        for (usize x = t;; x = par[x]) {
            path.push_back(x);
            if (x == s) break;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }
    std::optional<T> path(usize s, usize t) const {
        auto nodes = path_nodes(s, t);
        if (!nodes) return std::nullopt;
        T res = val[(*nodes)[0]];
        for (usize i = 1; i < nodes->size(); i++)
            res = op(res, val[(*nodes)[i]]);
        return res;
    }
    void add(usize u, usize v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    void del(usize u, usize v) {
        auto& a = adj[u];
        auto& b = adj[v];
        a.erase(std::find(a.begin(), a.end(), v));
        b.erase(std::find(b.begin(), b.end(), u));
    }
    bool has(usize u, usize v) const {
        return std::find(adj[u].begin(), adj[u].end(), v) != adj[u].end();
    }
};

void test_empty_and_single() {
    LinkCutTree<int> empty;
    assert(empty.size() == 0);
    assert(empty.query(0, 0) == std::nullopt);
    assert(empty.find_root(0) == LinkCutTree<int>::npos);
    assert(!empty.connected(0, 0));
    assert(!empty.make_root(0));
    assert(!empty.link(0, 0));
    assert(!empty.cut(0, 0));
    assert(!empty.modify(0, 0));
    assert(!empty.update(0, [](int x) { return x; }));

    LinkCutTree<int> zero(0, 0);
    assert(zero.size() == 0);
    assert(zero.query(0, 0) == std::nullopt);

    LinkCutTree<int> one(1, 42);
    assert(one.size() == 1);
    assert(one.query(0, 0) == 42);
    assert(one.find_root(0) == 0);
    assert(one.connected(0, 0));
    assert(one.make_root(0));
    assert(!one.link(0, 0));
    assert(!one.cut(0, 0));
    assert(!one.query(0, 1));
    assert(one.modify(0, 7));
    assert(one.query(0, 0) == 7);
    assert(one.update(0, [](int x) { return x + 1; }));
    assert(one.query(0, 0) == 8);
}

void test_basic_sum() {
    LinkCutTree<int> lct(5, [](usize i) { return int(i + 1); });
    assert(lct.link(0, 1));
    assert(lct.link(1, 2));
    assert(lct.link(2, 3));
    assert(lct.link(3, 4));
    assert(!lct.link(0, 4));
    assert(lct.query(0, 4) == 15);
    assert(lct.query(4, 0) == 15);
    assert(lct.query(2, 2) == 3);

    assert(lct.make_root(2));
    assert(lct.find_root(0) == 2);
    assert(lct.find_root(1) == 2);
    assert(lct.find_root(2) == 2);
    assert(lct.find_root(3) == 2);
    assert(lct.find_root(4) == 2);
    assert(lct.query(0, 4) == 15);

    assert(lct.modify(2, 10));
    assert(lct.query(0, 4) == 22);
    assert(lct.update(4, [](int x) { return x * 2; }));
    assert(lct.query(0, 4) == 27);

    assert(lct.cut(2, 3));
    assert(!lct.connected(0, 4));
    assert(lct.connected(0, 2));
    assert(lct.query(0, 4) == std::nullopt);
    assert(!lct.cut(2, 3));

    assert(lct.link(3, 2));
    assert(lct.connected(0, 4));
    assert(lct.query(0, 4) == 27);
}

void test_max_without_identity() {
    std::vector<int> values{-5, -2, -9, -1, -7};
    LinkCutTree<int, MaxOp> lct(5, [&](usize i) { return values[i]; });
    assert(lct.link(0, 1));
    assert(lct.link(1, 2));
    assert(lct.link(2, 3));
    assert(lct.link(3, 4));
    assert(lct.query(0, 4) == -1);
    assert(lct.query(0, 2) == -2);
    assert(lct.make_root(4));
    assert(lct.query(0, 4) == -1);
    assert(lct.modify(2, -10));
    assert(lct.query(0, 4) == -1);
    assert(lct.cut(1, 2));
    assert(lct.query(0, 4) == std::nullopt);
    assert(lct.query(0, 1) == -2);
    assert(lct.query(2, 4) == -1);
}

void test_xor_p3690() {
    usize n = 8;
    std::vector<int> values{1, 2, 3, 4, 5, 6, 7, 8};
    Naive<int, std::bit_xor<int>> naive(n, values);
    LinkCutTree<int, std::bit_xor<int>> lct(n, [&](usize i) {
        return values[i];
    });

    auto link = [&](usize u, usize v) {
        bool expected = u != v && !naive.connected(u, v);
        assert(lct.link(u, v) == expected);
        if (expected) naive.add(u, v);
    };
    auto cut = [&](usize u, usize v) {
        bool expected = u != v && naive.has(u, v);
        assert(lct.cut(u, v) == expected);
        if (expected) naive.del(u, v);
    };
    auto modify = [&](usize u, int v) {
        values[u] = v;
        naive.val[u] = v;
        assert(lct.modify(u, v));
    };

    link(0, 1);
    link(1, 2);
    link(2, 3);
    link(3, 4);
    link(4, 5);
    link(5, 6);
    link(6, 7);
    for (usize u = 0; u < n; u++) {
        for (usize v = 0; v < n; v++) {
            assert(lct.query(u, v) == naive.path(u, v));
        }
    }
    modify(3, 99);
    assert(lct.query(0, 7) == naive.path(0, 7));
    assert(lct.make_root(5));
    assert(lct.find_root(0) == 5);
    assert(lct.query(0, 3) == naive.path(0, 3));
    cut(5, 6);
    assert(lct.query(0, 7) == std::nullopt);
    assert(lct.query(0, 5) == naive.path(0, 5));
    assert(lct.query(6, 7) == naive.path(6, 7));
    link(5, 6);
    assert(lct.query(0, 7) == naive.path(0, 7));
}

void test_random_forest() {
    std::mt19937 rng(20261101);
    usize n = 16;
    std::vector<int> values(n);
    for (usize i = 0; i < n; i++) values[i] = int(i) - 8;
    Naive<int, std::plus<int>> naive(n, values);
    LinkCutTree<int> lct(n, [&](usize i) { return values[i]; });

    std::uniform_int_distribution<int> op_dist(0, 6);
    std::uniform_int_distribution<usize> v_dist(0, n - 1);
    for (int step = 0; step < 5000; step++) {
        usize u = v_dist(rng), v = v_dist(rng);
        int op = op_dist(rng);
        if (op == 0) {
            int x = int(rng() % 41) - 20;
            values[u] = x;
            naive.val[u] = x;
            assert(lct.modify(u, x));
        } else if (op == 1) {
            assert(lct.connected(u, v) == naive.connected(u, v));
            assert(lct.query(u, v) == naive.path(u, v));
        } else if (op == 2) {
            bool expected = u != v && !naive.connected(u, v);
            assert(lct.link(u, v) == expected);
            if (expected) naive.add(u, v);
        } else if (op == 3) {
            bool expected = u != v && naive.has(u, v);
            assert(lct.cut(u, v) == expected);
            if (expected) naive.del(u, v);
        } else if (op == 4) {
            assert(lct.make_root(u));
            if (naive.connected(u, v)) assert(lct.find_root(v) == u);
        } else if (op == 5) {
            values[u]++;
            naive.val[u]++;
            assert(lct.update(u, [](int x) { return x + 1; }));
        } else {
            assert(lct.query(u, v) == naive.path(u, v));
        }

        if (step % 100 == 0) {
            for (usize i = 0; i < n; i++) {
                assert(lct.find_root(i) < n);
                assert(lct.find_root(lct.find_root(i)) == lct.find_root(i));
                for (usize j = 0; j < n; j++) {
                    assert(lct.connected(i, j) == naive.connected(i, j));
                    assert(lct.query(i, j) == naive.path(i, j));
                }
            }
        }
    }
}

void test_long_chain() {
    usize n = 50000;
    LinkCutTree<long long> lct(n, [](usize i) { return (long long)i; });
    for (usize i = 1; i < n; i++) assert(lct.link(i - 1, i));
    long long expected = (long long)n * (long long)(n - 1) / 2;
    assert(lct.query(0, n - 1) == expected);
    assert(lct.make_root(n - 1));
    assert(lct.query(0, n - 1) == expected);
    assert(lct.cut(n / 2 - 1, n / 2));
    assert(!lct.connected(0, n - 1));
    assert(lct.query(0, n - 1) == std::nullopt);
    assert(lct.link(n / 2 - 1, n / 2));
    assert(lct.query(0, n - 1) == expected);
}

void test_copy_move() {
    LinkCutTree<int> lct(4, [](usize i) { return int(i + 1); });
    assert(lct.link(0, 1));
    assert(lct.link(1, 2));
    assert(lct.link(2, 3));
    auto copy = lct;
    assert(copy.query(0, 3) == 10);
    assert(copy.make_root(2));
    assert(copy.find_root(0) == 2);
    auto moved = std::move(copy);
    assert(moved.query(0, 3) == 10);
}

void test_ctad() {
    LinkCutTree lct(3, [](usize i) { return int(i + 1); });
    static_assert(std::is_same_v<decltype(lct), LinkCutTree<int>>);
    assert(lct.query(0, 0) == 1);
    assert(lct.link(0, 1));
    assert(lct.query(0, 1) == 3);
}

int main() {
    test_empty_and_single();
    test_basic_sum();
    test_max_without_identity();
    test_xor_p3690();
    test_random_forest();
    test_long_chain();
    test_copy_move();
    test_ctad();
    std::cout << "All tests passed!\n";
}

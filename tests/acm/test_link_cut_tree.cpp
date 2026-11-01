#include <algorithm>
#include <cassert>
#include <functional>
#include <optional>
#include <queue>
#include <random>
#include <vector>

#include "acm/link_cut_tree.hpp"

using acm::usize;

template <typename T, typename Op>
struct Naive {
    usize n;
    std::vector<std::vector<usize>> adj;
    std::vector<T> val;
    Op op;

    Naive(usize n, std::vector<T> values, Op op = {}):
        n(n), adj(n), val(values), op(op) {}

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
    std::optional<T> path(usize s, usize t) const {
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
        std::vector<usize> nodes;
        for (usize x = t;; x = par[x]) {
            nodes.push_back(x);
            if (x == s) break;
        }
        T res = val[nodes.back()];
        for (usize i = nodes.size() - 1; i > 0; i--)
            res = op(res, val[nodes[i - 1]]);
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
    acm::LinkCutTree<int> empty;
    assert(empty.size() == 0);
    assert(!empty.query(0, 0));
    assert(empty.find_root(0) == acm::LinkCutTree<int>::npos);
    assert(!empty.connected(0, 0));
    assert(!empty.make_root(0));
    assert(!empty.link(0, 0));
    assert(!empty.cut(0, 0));
    assert(!empty.modify(0, 0));
    assert(!empty.update(0, [](int x) { return x; }));

    acm::LinkCutTree<int> one(1, 42);
    assert(one.size() == 1);
    assert(one.query(0, 0) == 42);
    assert(one.find_root(0) == 0);
    assert(one.connected(0, 0));
    assert(one.make_root(0));
    assert(!one.link(0, 0));
    assert(!one.cut(0, 0));
    assert(one.modify(0, 7));
    assert(one.update(0, [](int x) { return x + 1; }));
    assert(one.query(0, 0) == 8);
}

void test_xor_and_makeroot() {
    usize n = 7;
    std::vector<int> values{1, 2, 4, 8, 16, 32, 64};
    Naive<int, std::bit_xor<int>> naive(n, values);
    acm::LinkCutTree<int, std::bit_xor<int>> lct(values);

    for (usize i = 1; i < n; i++) {
        assert(lct.link(i - 1, i));
        naive.add(i - 1, i);
    }
    for (usize u = 0; u < n; u++)
        for (usize v = 0; v < n; v++)
            assert(lct.query(u, v) == naive.path(u, v));

    assert(lct.make_root(3));
    assert(lct.find_root(0) == 3);
    assert(lct.find_root(6) == 3);
    assert(lct.query(0, 6) == naive.path(0, 6));

    values[4] = 7;
    naive.val[4] = 7;
    assert(lct.modify(4, 7));
    assert(lct.query(0, 6) == naive.path(0, 6));

    assert(lct.cut(2, 3));
    naive.del(2, 3);
    assert(!lct.connected(0, 6));
    assert(!lct.query(0, 6));
    assert(lct.query(0, 2) == naive.path(0, 2));
    assert(lct.query(3, 6) == naive.path(3, 6));

    assert(lct.link(2, 3));
    naive.add(2, 3);
    assert(lct.query(0, 6) == naive.path(0, 6));
}

void test_random_forest() {
    std::mt19937 rng(20261101);
    usize n = 12;
    std::vector<int> values(n);
    for (usize i = 0; i < n; i++) values[i] = int(i) - 6;
    Naive<int, std::plus<int>> naive(n, values);
    acm::LinkCutTree<int> lct(n, 0);
    for (usize i = 0; i < n; i++) assert(lct.modify(i, values[i]));

    std::uniform_int_distribution<int> opd(0, 5);
    std::uniform_int_distribution<usize> vd(0, n - 1);
    for (int step = 0; step < 3000; step++) {
        usize u = vd(rng), v = vd(rng);
        int op = opd(rng);
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
        } else {
            values[u]++;
            naive.val[u]++;
            assert(lct.update(u, [](int x) { return x + 1; }));
        }
        if (step % 100 == 0) {
            for (usize i = 0; i < n; i++)
                for (usize j = 0; j < n; j++) {
                    assert(lct.connected(i, j) == naive.connected(i, j));
                    assert(lct.query(i, j) == naive.path(i, j));
                }
        }
    }
}

void test_long_chain() {
    usize n = 20000;
    acm::LinkCutTree<long long> lct(n, 0);
    for (usize i = 0; i < n; i++) assert(lct.modify(i, (long long)i));
    for (usize i = 1; i < n; i++) assert(lct.link(i - 1, i));
    long long expected = (long long)n * (long long)(n - 1) / 2;
    assert(lct.query(0, n - 1) == expected);
    assert(lct.make_root(n - 1));
    assert(lct.query(0, n - 1) == expected);
    assert(lct.cut(n / 2 - 1, n / 2));
    assert(!lct.query(0, n - 1));
}

int main() {
    test_empty_and_single();
    test_xor_and_makeroot();
    test_random_forest();
    test_long_chain();
    return 0;
}

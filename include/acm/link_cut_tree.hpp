#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "def.hpp"

namespace acm
{

template <typename T = i32, typename Op = std::plus<T>>
class LinkCutTree {
public:
    static constexpr usize npos = usize(-1);

private:
    struct Node {
        usize lc, rc, fa;
        bool rev;
        T val, sum;

        explicit Node(T v):
            lc(npos),
            rc(npos),
            fa(npos),
            rev(false),
            val(std::move(v)),
            sum(val) {}
    };

    usize n = 0;
    std::vector<Node> t;
    Op op{};

    bool islc(usize x) const {
        usize f = t[x].fa;
        return f != npos && t[f].lc == x;
    }
    bool nroot(usize x) const {
        usize f = t[x].fa;
        return f == npos || (t[f].lc != x && t[f].rc != x);
    }
    void pushup(usize x) {
        T s = t[x].val;
        if (t[x].lc != npos) s = op(t[t[x].lc].sum, s);
        if (t[x].rc != npos) s = op(s, t[t[x].rc].sum);
        t[x].sum = std::move(s);
    }
    void pushrev(usize x) {
        std::swap(t[x].lc, t[x].rc);
        t[x].rev ^= 1;
    }
    void pushdown(usize x) {
        if (!t[x].rev) return;
        if (t[x].lc != npos) pushrev(t[x].lc);
        if (t[x].rc != npos) pushrev(t[x].rc);
        t[x].rev = false;
    }
    void pushall(usize x) {
        if (!nroot(x)) pushall(t[x].fa);
        pushdown(x);
    }
    void rotate(usize x) {
        usize f = t[x].fa, g = t[f].fa;
        if (islc(x)) {
            t[f].lc = t[x].rc;
            if (t[x].rc != npos) t[t[x].rc].fa = f;
            t[x].rc = f;
        } else {
            t[f].rc = t[x].lc;
            if (t[x].lc != npos) t[t[x].lc].fa = f;
            t[x].lc = f;
        }
        if (!nroot(f)) (t[g].lc == f ? t[g].lc : t[g].rc) = x;
        t[x].fa = g;
        t[f].fa = x;
        pushup(f);
        pushup(x);
    }
    void splay(usize x) {
        pushall(x);
        for (; !nroot(x); rotate(x)) {
            usize f = t[x].fa;
            if (!nroot(f)) rotate(islc(f) == islc(x) ? f : x);
        }
    }
    void access(usize x) {
        for (usize y = npos; x != npos; x = t[y = x].fa) {
            splay(x);
            t[x].rc = y;
            if (y != npos) t[y].fa = x;
            pushup(x);
        }
    }
    void makeroot(usize x) {
        access(x);
        splay(x);
        pushrev(x);
    }
    usize findroot(usize x) {
        access(x);
        splay(x);
        while (true) {
            pushdown(x);
            if (t[x].lc == npos) break;
            x = t[x].lc;
        }
        splay(x);
        return x;
    }

public:
    LinkCutTree() = default;

    template <typename R>
    explicit LinkCutTree(const R& a, Op op = {}): n(a.size()), t(), op(op) {
        t.reserve(n);
        for (usize i = 0; i < n; i++) t.emplace_back(a[i]);
    }
    LinkCutTree(usize n, T v, Op op = {}): n(n), t(), op(op) {
        t.reserve(n);
        for (usize i = 0; i < n; i++) t.emplace_back(v);
    }

    usize size() const { return n; }

    bool modify(usize u, T v) {
        if (u >= n) return false;
        splay(u);
        t[u].val = std::move(v);
        pushup(u);
        return true;
    }
    template <typename F>
    bool update(usize u, F f) {
        if (u >= n) return false;
        splay(u);
        t[u].val = f(t[u].val);
        pushup(u);
        return true;
    }
    std::optional<T> query(usize u, usize v) {
        if (u >= n || v >= n) return std::nullopt;
        makeroot(u);
        if (findroot(v) != u) return std::nullopt;
        access(v);
        splay(v);
        return t[v].sum;
    }
    usize find_root(usize u) { return u >= n ? npos : findroot(u); }
    bool connected(usize u, usize v) {
        if (u >= n || v >= n) return false;
        if (u == v) return true;
        return findroot(u) == findroot(v);
    }
    bool make_root(usize u) {
        if (u >= n) return false;
        makeroot(u);
        return true;
    }
    bool link(usize u, usize v) {
        if (u >= n || v >= n || u == v) return false;
        makeroot(u);
        if (findroot(v) == u) return false;
        t[u].fa = v;
        return true;
    }
    bool cut(usize u, usize v) {
        if (u >= n || v >= n || u == v) return false;
        makeroot(u);
        access(v);
        splay(v);
        if (t[v].lc != u || t[u].rc != npos) return false;
        t[v].lc = npos;
        t[u].fa = npos;
        pushup(v);
        return true;
    }
};

}  // namespace acm

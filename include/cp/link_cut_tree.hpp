#pragma once

#include <concepts>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "def.hpp"
#include "utils/concepts.hpp"

namespace cp
{

template <
    typename T,
    Fn<T(T, T)> Op = std::plus<T>,
    typename Alloc = std::allocator<T>
>
    requires requires { typename std::allocator_traits<Alloc>; }
class LinkCutTree {
public:
    static constexpr usize npos = usize(-1);

private:
    struct Node {
        usize lc, rc, fa;
        bool rev;
        T val, sum;

        explicit Node(T v):
            lc{npos},
            rc{npos},
            fa{npos},
            rev{false},
            val{std::move(v)},
            sum{val} {}
    };
    using NodeAlloc =
        typename std::allocator_traits<Alloc>::template rebind_alloc<Node>;

    bool _is_left(usize x) const {
        usize f = _t[x].fa;
        return f != npos && _t[f].lc == x;
    }
    bool _is_root(usize x) const {
        usize f = _t[x].fa;
        return f == npos || (_t[f].lc != x && _t[f].rc != x);
    }
    void _pushup(usize x) {
        T s = _t[x].val;
        if (_t[x].lc != npos) s = _op(_t[_t[x].lc].sum, s);
        if (_t[x].rc != npos) s = _op(s, _t[_t[x].rc].sum);
        _t[x].sum = std::move(s);
    }
    void _pushrev(usize x) {
        std::swap(_t[x].lc, _t[x].rc);
        _t[x].rev ^= 1;
    }
    void _pushdown(usize x) {
        if (!_t[x].rev) return;
        if (_t[x].lc != npos) _pushrev(_t[x].lc);
        if (_t[x].rc != npos) _pushrev(_t[x].rc);
        _t[x].rev = false;
    }
    void _pushall(usize x) {
        if (!_is_root(x)) _pushall(_t[x].fa);
        _pushdown(x);
    }
    void _rotate(usize x) {
        usize f = _t[x].fa, g = _t[f].fa;
        if (_is_left(x)) {
            _t[f].lc = _t[x].rc;
            if (_t[x].rc != npos) _t[_t[x].rc].fa = f;
            _t[x].rc = f;
        } else {
            _t[f].rc = _t[x].lc;
            if (_t[x].lc != npos) _t[_t[x].lc].fa = f;
            _t[x].lc = f;
        }
        if (!_is_root(f)) (_t[g].lc == f ? _t[g].lc : _t[g].rc) = x;
        _t[x].fa = g;
        _t[f].fa = x;
        _pushup(f);
        _pushup(x);
    }
    void _splay(usize x) {
        _pushall(x);
        for (; !_is_root(x); _rotate(x)) {
            usize f = _t[x].fa;
            if (!_is_root(f)) _rotate(_is_left(f) == _is_left(x) ? f : x);
        }
    }
    void _access(usize x) {
        for (usize y = npos; x != npos; x = _t[y = x].fa) {
            _splay(x);
            _t[x].rc = y;
            if (y != npos) _t[y].fa = x;
            _pushup(x);
        }
    }
    void _make_root(usize x) {
        _access(x);
        _splay(x);
        _pushrev(x);
    }
    usize _find_root(usize x) {
        _access(x);
        _splay(x);
        while (true) {
            _pushdown(x);
            if (_t[x].lc == npos) break;
            x = _t[x].lc;
        }
        _splay(x);
        return x;
    }

    std::vector<Node, NodeAlloc> _t;
    [[no_unique_address]] Op _op{};

public:
    LinkCutTree() = default;

    template <FnMut<T(usize)> F>
    LinkCutTree(usize n, F gen, Op op = {}, Alloc alloc = {}):
        _t{NodeAlloc{alloc}}, _op{std::move(op)} {
        _t.reserve(n);
        for (usize i = 0; i < n; i++) _t.emplace_back(gen(i));
    }
    LinkCutTree(usize n, T v, Op op = {}, Alloc alloc = {}):
        LinkCutTree(
            n, [v](usize) { return v; }, std::move(op), std::move(alloc)
        ) {}

    usize size() const noexcept { return _t.size(); }

    bool modify(usize u, T v) {
        if (u >= _t.size()) return false;
        _splay(u);
        _t[u].val = std::move(v);
        _pushup(u);
        return true;
    }

    template <FnOnce<T(T)> F>
    bool update(usize u, F func) {
        if (u >= _t.size()) return false;
        _splay(u);
        _t[u].val = std::forward<F>(func)(_t[u].val);
        _pushup(u);
        return true;
    }

    std::optional<T> query(usize u, usize v) {
        if (u >= _t.size() || v >= _t.size()) return std::nullopt;
        _make_root(u);
        if (_find_root(v) != u) return std::nullopt;
        _access(v);
        _splay(v);
        return _t[v].sum;
    }

    usize find_root(usize u) {
        if (u >= _t.size()) return npos;
        return _find_root(u);
    }

    bool connected(usize u, usize v) {
        if (u >= _t.size() || v >= _t.size()) return false;
        if (u == v) return true;
        return _find_root(u) == _find_root(v);
    }

    bool make_root(usize u) {
        if (u >= _t.size()) return false;
        _make_root(u);
        return true;
    }

    bool link(usize u, usize v) {
        if (u >= _t.size() || v >= _t.size() || u == v) return false;
        _make_root(u);
        if (_find_root(v) == u) return false;
        _t[u].fa = v;
        return true;
    }

    bool cut(usize u, usize v) {
        if (u >= _t.size() || v >= _t.size() || u == v) return false;
        _make_root(u);
        _access(v);
        _splay(v);
        if (_t[v].lc != u || _t[u].rc != npos) return false;
        _t[v].lc = npos;
        _t[u].fa = npos;
        _pushup(v);
        return true;
    }
};

template <
    std::invocable<usize> F,
    typename T = std::remove_cvref_t<std::invoke_result_t<F, usize>>,
    typename Op = std::plus<T>,
    typename Alloc = std::allocator<T>
>
LinkCutTree(usize, F, Op = {}, Alloc = {}) -> LinkCutTree<T, Op, Alloc>;

}  // namespace cp

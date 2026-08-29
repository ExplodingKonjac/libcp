#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

#include "cp/def.hpp"

namespace cp
{

namespace detail
{

// Doubly linked lists to maintain the tree in Network Simplex.
class LinkedLists {
private:
    std::vector<usize> _pre, _nxt;

public:
    LinkedLists(usize n): _pre(n), _nxt(n) {
        std::iota(_pre.begin(), _pre.end(), 0);
        std::iota(_nxt.begin(), _nxt.end(), 0);
    }
    usize prev(usize i) { return _pre[i]; }
    usize next(usize i) { return _nxt[i]; }
    void link(usize u, usize v) { _nxt[u] = v, _pre[v] = u; }
    void erase(usize u) {
        _pre[_nxt[u]] = _pre[u];
        _nxt[_pre[u]] = _nxt[u];
        _pre[u] = _nxt[u] = u;
    }
    void insert(usize pos, usize u) { link(_pre[pos], u), link(u, pos); }
};

}  // namespace detail

template <typename FlowT, typename CostT>
class MinCostCirculation {
    static_assert(std::is_signed_v<FlowT>, "FlowT must be signed type");
    static_assert(std::is_signed_v<CostT>, "CostT must be signed type");

public:
    struct Node {
        FlowT supply;
        CostT potential;
    };
    struct Edge {
        usize from, to;
        FlowT capacity, flow;
        CostT cost;
    };

private:
    enum class ArcState { UPPER = -1, LOWER = 1, TREE = 0 };
    struct _Node: Node {
        usize par, pre;
    };
    struct _Edge: Edge {
        ArcState sta;
    };

    std::vector<_Node> _nd;
    std::vector<_Edge> _ed;

public:
    MinCostCirculation() = default;
    MinCostCirculation(usize V): _nd(V) {}

    usize size_V() const { return _nd.size(); }
    usize size_E() const { return _ed.size(); }

    FlowT& supply(usize i) { return _nd[i].supply; }
    FlowT supply(usize i) const { return _nd[i].supply; }
    const Node& node(usize i) const { return static_cast<const Node&>(_nd[i]); }
    const Edge& edge(usize i) const { return static_cast<const Edge&>(_ed[i]); }
    usize add_edge(usize u, usize v, FlowT cap, CostT cost) {
        _ed.push_back(_Edge{{u, v, cap, 0, cost}, ArcState::LOWER});
        return _ed.size() - 1;
    }

    std::optional<CostT> circulation();
};

template <typename FlowT, typename CostT>
std::optional<CostT> MinCostCirculation<FlowT, CostT>::circulation() {
    constexpr usize npos = -1;

    usize n = size_V(), m = size_E(), nxt_arc = 0, B = 0, cur_tim = 1;
    std::vector<usize> p{}, q(n + 1), vis(n + 1);
    detail::LinkedLists tree(2 * n + 2);

    auto reduced = [&](usize e) {
        auto u = _ed[e].from, v = _ed[e].to;
        return _ed[e].cost + _nd[u].potential - _nd[v].potential;
    };
    auto sreduced = [&](usize e) {
        return static_cast<int>(_ed[e].sta) * reduced(e);
    };
    auto select = [&] {
        auto res = npos, cnt1 = B, cnt2 = m;
        CostT mn = 0;
        for (auto& e = nxt_arc; cnt2--; e++, (e == m) && (e = 0)) {
            auto c = sreduced(p[e]);
            if (c < mn) mn = c, res = p[e];
            if (!--cnt1 && mn < 0) break;
            else if (cnt1 == 0) cnt1 = B;
        }
        return res;
    };
    auto pivot = [&](usize in_arc) {
        auto u_in = _ed[in_arc].from, v_in = _ed[in_arc].to;
        auto w = [&] {
            cur_tim++;
            for (auto x = u_in; x != npos; x = _nd[x].par) vis[x] = cur_tim;
            for (auto x = v_in;; x = _nd[x].par)
                if (vis[x] == cur_tim) return x;
        }();
        auto S = u_in, T = v_in, u_out = npos;
        if (_ed[in_arc].sta == ArcState::UPPER) std::swap(S, T);
        auto df = _ed[in_arc].capacity;
        enum { S_SIDE, T_SIDE, SAME } side = SAME;
        for (auto u = S; u != w && df; u = _nd[u].par) {
            auto e = _nd[u].pre;
            auto d =
                u == _ed[e].to ? _ed[e].capacity - _ed[e].flow : _ed[e].flow;
            if (d < df) df = d, u_out = u, side = S_SIDE;
        }
        for (auto u = T; u != w && (df || side != T_SIDE); u = _nd[u].par) {
            auto e = _nd[u].pre;
            auto d =
                u == _ed[e].to ? _ed[e].flow : _ed[e].capacity - _ed[e].flow;
            if (d <= df) df = d, u_out = u, side = T_SIDE;
        }
        if (df) {
            for (auto u = S; u != w; u = _nd[u].par) {
                auto e = _nd[u].pre;
                _ed[e].flow += u == _ed[e].to ? df : -df;
            }
            for (auto u = T; u != w; u = _nd[u].par) {
                auto e = _nd[u].pre;
                _ed[e].flow += u == _ed[e].to ? -df : df;
            }
            _ed[in_arc].flow += static_cast<int>(_ed[in_arc].sta) * df;
        }
        if (side == SAME) {
            auto state = static_cast<int>(_ed[in_arc].sta);
            _ed[in_arc].sta = static_cast<ArcState>(-state);
            return;
        }
        auto out_arc = _nd[u_out].pre;
        _ed[in_arc].sta = ArcState::TREE;
        _ed[out_arc].sta =
            _ed[out_arc].flow ? ArcState::UPPER : ArcState::LOWER;
        u_in = S, v_in = T;
        if (side != S_SIDE) std::swap(u_in, v_in);
        usize ql = 0, qr = 0;
        for (auto u = u_in; u != u_out; u = _nd[u].par) q[qr++] = u;
        for (usize i = qr; i > 0; i--) {
            auto u = q[i - 1], parent = _nd[u].par;
            tree.erase(parent);
            tree.insert(u + n, parent);
            _nd[parent].par = u;
            _nd[parent].pre = _nd[u].pre;
        }
        tree.erase(u_in);
        tree.insert(v_in + n, u_in);
        _nd[u_in].par = v_in;
        _nd[u_in].pre = in_arc;
        auto dp = reduced(in_arc);
        if (u_in == _ed[in_arc].from) dp = -dp;
        for (q[ql = 0] = u_in, qr = 1; ql < qr;) {
            auto u = q[ql++];
            for (auto v = tree.next(u + n); v != u + n; v = tree.next(v))
                q[qr++] = v;
            _nd[u].potential += dp;
        }
    };
    auto simplex = [&] {
        std::mt19937 rng(std::random_device{}());
        p.resize(_ed.size());
        std::iota(p.begin(), p.end(), 0);
        std::shuffle(p.begin(), p.end(), rng);
        B = std::max(
            static_cast<usize>(std::ceil(std::sqrt(_ed.size()))),
            std::min<usize>(5, _nd.size())
        );
        for (usize e; (e = select()) != npos;) pivot(e);
    };

    FlowT supply_sum = 0;
    for (usize u = 0; u < n; u++) supply_sum += _nd[u].supply;
    if (supply_sum != 0) return std::nullopt;
    CostT artificial_cost = 1;
    for (auto& e: _ed) artificial_cost += std::abs(e.cost);
    _ed.resize(m + n);
    _nd.push_back(_Node{{0, 0}, npos, npos});
    for (usize e = 0; e < m; e++) _ed[e].sta = ArcState::LOWER;
    for (usize u = 0, e = m; u < n; u++, e++) {
        _nd[u].par = n;
        _nd[u].pre = e;
        tree.insert(2 * n + 1, u);
        bool positive = _nd[u].supply >= 0;
        if (positive) {
            _ed[e] =
                _Edge{{u, n, _nd[u].supply, _nd[u].supply, artificial_cost},
                      ArcState::TREE};
        } else {
            _ed[e] =
                _Edge{{n, u, -_nd[u].supply, -_nd[u].supply, artificial_cost},
                      ArcState::TREE};
        }
        _nd[u].potential = positive ? -artificial_cost : artificial_cost;
    }
    m += n, n++, simplex(), n--, m -= n;
    _nd.resize(n);
    for (usize e = m; e < n + m; e++) {
        if (_ed[e].flow != 0) {
            _ed.resize(m);
            return std::nullopt;
        }
    }
    _ed.resize(m);
    CostT cost = 0;
    for (auto& e: _ed) cost += e.cost * e.flow;
    return cost;
}

}  // namespace cp

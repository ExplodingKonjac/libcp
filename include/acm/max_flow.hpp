#pragma once

#include <algorithm>
#include <concepts>
#include <limits>
#include <vector>

#include "def.hpp"

namespace acm
{

template <std::integral FlowT = int>
class MaxFlow {
public:
    struct Edge {
        usize from, to;
        FlowT capacity, flow;
    };

private:
    struct Arc {
        usize to;
        usize nxt;
        FlowT w;
    };

    static constexpr usize npos = -1;

    std::vector<usize> _head;
    std::vector<Arc> _e;

public:
    MaxFlow() = default;
    MaxFlow(usize V): _head(V, npos) {}

    usize size_V() const { return _head.size(); }
    usize size_E() const { return _e.size() / 2; }

    Edge edge(usize i) const {
        const auto& e = _e[2 * i];
        const auto& rev = _e[2 * i + 1];
        return {rev.to, e.to, e.w + rev.w, rev.w};
    }
    usize add_edge(usize u, usize v, FlowT capacity) {
        usize i = size_E();
        _e.push_back({v, _head[u], capacity});
        _head[u] = 2 * i;
        _e.push_back({u, _head[v], 0});
        _head[v] = 2 * i + 1;
        return i;
    }

    FlowT max_flow(usize s, usize t);
};

template <std::integral FlowT>
FlowT MaxFlow<FlowT>::max_flow(usize s, usize t) {
    if (s == t) return 0;
    constexpr FlowT inf = std::numeric_limits<FlowT>::max();
    usize n = size_V();
    std::vector<usize> dep(n), cur(n), q(n);

    auto bfs = [&] {
        dep.assign(n, 0);
        cur = _head;
        usize hd = 0, tl = 0;
        dep[s] = 1;
        q[tl++] = s;
        while (hd < tl) {
            usize u = q[hd++];
            for (usize i = _head[u]; i != npos; i = _e[i].nxt) {
                const auto& e = _e[i];
                if (!e.w || dep[e.to]) continue;
                dep[e.to] = dep[u] + 1;
                q[tl++] = e.to;
                if (e.to == t) return true;
            }
        }
        return false;
    };
    auto dfs = [&](auto&& self, usize u, FlowT flow) -> FlowT {
        if (u == t || !flow) return flow;
        FlowT rem = flow;
        for (usize& i = cur[u]; i != npos; i = _e[i].nxt) {
            if (!_e[i].w || dep[_e[i].to] != dep[u] + 1) continue;
            FlowT c = self(self, _e[i].to, std::min(rem, _e[i].w));
            _e[i].w -= c;
            _e[i ^ 1].w += c;
            if (!(rem -= c)) break;
        }
        if (rem == flow) dep[u] = 0;
        return flow - rem;
    };

    FlowT result = 0;
    while (bfs()) result += dfs(dfs, s, inf);
    return result;
}

}  // namespace acm

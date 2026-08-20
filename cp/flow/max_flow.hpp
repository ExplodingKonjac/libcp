#pragma once

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>

#include "cp/def.hpp"

namespace cp
{

template <typename FlowT = int>
class MaxFlow {
    static_assert(std::is_integral_v<FlowT>, "FlowT must be integral type");

public:
    struct Edge {
        usize from, to;
        FlowT capacity, flow;
    };

private:
    struct Arc {
        usize edge;
        bool reverse;
    };

    std::vector<std::vector<Arc>> _adj;
    std::vector<Edge> _edges;

public:
    MaxFlow() = default;
    MaxFlow(usize V): _adj(V) {}

    usize size_V() const { return _adj.size(); }
    usize size_E() const { return _edges.size(); }

    Edge& edge(usize i) { return _edges[i]; }
    Edge edge(usize i) const { return _edges[i]; }
    usize add_edge(usize u, usize v, FlowT capacity) {
        usize i = _edges.size();
        _edges.push_back({u, v, capacity, 0});
        _adj[u].push_back({i, false});
        _adj[v].push_back({i, true});
        return i;
    }

    FlowT max_flow(usize s, usize t);
};

template <typename FlowT>
FlowT MaxFlow<FlowT>::max_flow(usize s, usize t) {
    if (s == t) return 0;
    constexpr usize npos = -1;
    constexpr FlowT inf = std::numeric_limits<FlowT>::max();
    usize n = size_V();
    std::vector<usize> level(n), head(n), queue(n);
    auto residual = [&](Arc a) {
        auto& e = _edges[a.edge];
        return a.reverse ? e.flow : e.capacity - e.flow;
    };
    auto augment = [&](Arc a, FlowT flow) {
        if (a.reverse) _edges[a.edge].flow -= flow;
        else _edges[a.edge].flow += flow;
    };
    auto bfs = [&] {
        level.assign(n, npos);
        head.assign(n, 0);
        usize front = 0, back = 0;
        level[s] = 0;
        queue[back++] = s;
        while (front < back) {
            usize u = queue[front++];
            for (auto a: _adj[u]) {
                usize v = _edges[a.edge].from ^ _edges[a.edge].to ^ u;
                if (residual(a) && level[v] == npos) {
                    level[v] = level[u] + 1;
                    queue[back++] = v;
                }
            }
        }
        return level[t] != npos;
    };
    auto dfs = [&](auto&& self, usize u, FlowT flow) -> FlowT {
        if (u == t || !flow) return flow;
        for (auto& i = head[u]; i < _adj[u].size(); i++) {
            auto a = _adj[u][i];
            usize v = _edges[a.edge].from ^ _edges[a.edge].to ^ u;
            FlowT available = residual(a);
            if (!available || level[v] != level[u] + 1) continue;
            FlowT pushed = self(self, v, std::min(flow, available));
            if (pushed) {
                augment(a, pushed);
                return pushed;
            }
        }
        level[u] = npos;
        return 0;
    };

    FlowT result = 0;
    while (bfs()) {
        while (FlowT flow = dfs(dfs, s, inf)) result += flow;
    }
    return result;
}

}  // namespace cp

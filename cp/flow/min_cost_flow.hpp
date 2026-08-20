#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <type_traits>
#include <utility>
#include <vector>

#include "cp/def.hpp"

namespace cp
{

template <typename FlowT = int, typename CostT = int>
class MinCostFlow {
    static_assert(std::is_integral_v<FlowT>, "FlowT must be integral type");
    static_assert(std::is_signed_v<CostT>, "CostT must be signed type");

public:
    struct Edge {
        usize from, to;
        FlowT capacity, flow;
        CostT cost;
    };

private:
    struct Arc {
        usize edge;
        bool reverse;
    };

    std::vector<std::vector<Arc>> _adj;
    std::vector<Edge> _edges;
    bool _solved = false, _dirty = false;

    template <bool maximize>
    std::pair<FlowT, CostT> _flow(usize s, usize t, bool negative_cost);

public:
    MinCostFlow() = default;
    MinCostFlow(usize V): _adj(V) {}

    usize size_V() const { return _adj.size(); }
    usize size_E() const { return _edges.size(); }

    const Edge& edge(usize i) const { return _edges[i]; }
    usize add_edge(usize u, usize v, FlowT capacity, CostT cost) {
        usize i = _edges.size();
        _edges.push_back({u, v, capacity, 0, cost});
        _adj[u].push_back({i, false});
        _adj[v].push_back({i, true});
        _dirty = true;
        return i;
    }

    std::pair<FlowT, CostT> min_cost_flow(
        usize s, usize t, bool negative_cost = false
    ) {
        return _flow<false>(s, t, negative_cost);
    }
    std::pair<FlowT, CostT> min_cost_max_flow(
        usize s, usize t, bool negative_cost = false
    ) {
        return _flow<true>(s, t, negative_cost);
    }
};

template <typename FlowT, typename CostT>
template <bool maximize>
std::pair<FlowT, CostT> MinCostFlow<FlowT, CostT>::_flow(
    usize s, usize t, bool negative_cost
) {
    if (s == t) return {0, 0};
    constexpr usize npos = -1;
    constexpr FlowT flow_inf = std::numeric_limits<FlowT>::max();
    constexpr CostT cost_inf = std::numeric_limits<CostT>::max();
    usize n = size_V();
    FlowT old_flow = 0;
    CostT old_cost = 0;
    if (_solved && _dirty) {
        // ponytail: recompute after graph changes; add cycle cancellation
        // if repeated reoptimization becomes a bottleneck.
        FlowT out = 0, in = 0;
        for (auto& e: _edges) {
            if (e.from == s) out += e.flow;
            if (e.to == s) in += e.flow;
            old_cost += static_cast<CostT>(e.flow) * e.cost;
            e.flow = 0;
        }
        old_flow = out - in;
    }
    std::vector<usize> level(n), head(n), queue(n);
    std::vector<std::uint8_t> visited(n), in_queue(n);
    std::vector<CostT> distance(n), potential(n);
    auto residual = [&](Arc a) {
        auto& e = _edges[a.edge];
        return a.reverse ? e.flow : e.capacity - e.flow;
    };
    auto cost = [&](Arc a) {
        return a.reverse ? -_edges[a.edge].cost : _edges[a.edge].cost;
    };
    auto to = [&](usize u, Arc a) {
        auto& e = _edges[a.edge];
        return e.from ^ e.to ^ u;
    };
    auto augment = [&](Arc a, FlowT flow) {
        if (a.reverse) _edges[a.edge].flow -= flow;
        else _edges[a.edge].flow += flow;
    };
    auto spfa = [&] {
        distance.assign(n, cost_inf);
        in_queue.assign(n, false);
        std::queue<usize> q;
        distance[s] = 0;
        in_queue[s] = true;
        q.push(s);
        while (!q.empty()) {
            usize u = q.front();
            q.pop();
            in_queue[u] = false;
            for (auto a: _adj[u]) {
                if (!residual(a)) continue;
                usize v = to(u, a);
                CostT next =
                    distance[u] + cost(a) + potential[u] - potential[v];
                if (next >= distance[v]) continue;
                distance[v] = next;
                if (!in_queue[v]) {
                    in_queue[v] = true;
                    q.push(v);
                }
            }
        }
        return distance[t] != cost_inf;
    };
    auto dijkstra = [&] {
        distance.assign(n, cost_inf);
        visited.assign(n, false);
        using Item = std::pair<CostT, usize>;
        std::priority_queue<Item, std::vector<Item>, std::greater<>> q;
        q.emplace(distance[s] = 0, s);
        while (!q.empty()) {
            usize u = q.top().second;
            q.pop();
            if (visited[u]) continue;
            visited[u] = true;
            for (auto a: _adj[u]) {
                if (!residual(a)) continue;
                usize v = to(u, a);
                CostT next =
                    distance[u] + cost(a) + potential[u] - potential[v];
                if (next >= distance[v]) continue;
                distance[v] = next;
                q.emplace(next, v);
            }
        }
        return distance[t] != cost_inf;
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
                usize v = to(u, a);
                if (residual(a) &&
                    level[v] == npos &&
                    potential[v] == potential[u] + cost(a)) {
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
            usize v = to(u, a);
            FlowT available = residual(a);
            if (!available ||
                level[v] != level[u] + 1 ||
                potential[v] != potential[u] + cost(a))
                continue;
            FlowT pushed = self(self, v, std::min(flow, available));
            if (pushed) {
                augment(a, pushed);
                return pushed;
            }
        }
        level[u] = npos;
        return 0;
    };

    FlowT total_flow = 0;
    CostT total_cost = 0;
    for (auto e: _edges) {
        negative_cost |=
            (e.flow < e.capacity && e.cost < 0) || (e.flow > 0 && e.cost > 0);
    }
    bool reachable = negative_cost ? spfa() : dijkstra();
    while (reachable) {
        for (usize u = 0; u < n; u++) {
            if (distance[u] != cost_inf) potential[u] += distance[u];
        }
        if (!maximize && potential[t] >= 0) break;
        FlowT delta = 0;
        while (bfs()) {
            while (FlowT flow = dfs(dfs, s, flow_inf)) delta += flow;
        }
        total_flow += delta;
        total_cost += static_cast<CostT>(delta) * potential[t];
        reachable = dijkstra();
    }
    _solved = true;
    _dirty = false;
    return {total_flow - old_flow, total_cost - old_cost};
}

}  // namespace cp

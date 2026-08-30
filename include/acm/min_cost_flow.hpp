#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "def.hpp"

namespace acm
{
template <std::integral FlowT = int, typename CostT = int>
    requires(std::signed_integral<CostT> || std::floating_point<CostT>)
class MinCostFlow {
public:
    struct Edge {
        usize from, to;
        FlowT capacity, flow;
        CostT cost;
    };

private:
    struct Arc {
        usize to;
        usize nxt;
        FlowT w;
        CostT cost;
    };

    static constexpr usize npos = -1;

    std::vector<usize> _head;
    std::vector<Arc> _e;

    template <bool maximize>
    std::pair<FlowT, CostT> _flow(usize s, usize t);

public:
    MinCostFlow() = default;
    MinCostFlow(usize V): _head(V, npos) {}

    usize size_V() const { return _head.size(); }
    usize size_E() const { return _e.size() / 2; }

    Edge edge(usize i) const {
        const auto& e = _e[2 * i];
        const auto& rev = _e[2 * i + 1];
        return {rev.to, e.to, e.w + rev.w, rev.w, e.cost};
    }
    usize add_edge(usize u, usize v, FlowT capacity, CostT cost) {
        usize i = size_E();
        _e.push_back({v, _head[u], capacity, cost});
        _head[u] = 2 * i;
        _e.push_back({u, _head[v], 0, -cost});
        _head[v] = 2 * i + 1;
        return i;
    }

    std::pair<FlowT, CostT> min_cost_flow(usize s, usize t) {
        return _flow<false>(s, t);
    }
    std::pair<FlowT, CostT> min_cost_max_flow(usize s, usize t) {
        return _flow<true>(s, t);
    }
};

template <std::integral FlowT, typename CostT>
    requires(std::signed_integral<CostT> || std::floating_point<CostT>)
template <bool maximize>
std::pair<FlowT, CostT> MinCostFlow<FlowT, CostT>::_flow(usize s, usize t) {
    if (s == t) return {0, 0};
    constexpr FlowT flow_inf = std::numeric_limits<FlowT>::max();
    constexpr CostT cost_inf = std::numeric_limits<CostT>::max();
    usize n = size_V();
    std::vector<usize> dep(n), cur(n), q;
    q.reserve(n);
    std::vector<u8> vis(n), inq(n);
    std::vector<CostT> dis(n), p(n);
    auto spfa = [&] {
        dis.assign(n, cost_inf);
        inq.assign(n, false);
        q.clear();
        dis[s] = 0;
        inq[s] = true;
        q.push_back(s);
        for (usize hd = 0; hd < q.size(); hd++) {
            usize u = q[hd];
            inq[u] = false;
            for (usize i = _head[u]; i != npos; i = _e[i].nxt) {
                const auto& e = _e[i];
                if (!e.w) continue;
                CostT next = dis[u] + e.cost + p[u] - p[e.to];
                if (next >= dis[e.to]) continue;
                dis[e.to] = next;
                if (!inq[e.to]) {
                    inq[e.to] = true;
                    q.push_back(e.to);
                }
            }
        }
        return dis[t] != cost_inf;
    };
    auto dijkstra = [&] {
        dis.assign(n, cost_inf);
        vis.assign(n, false);
        using Item = std::pair<CostT, usize>;
        std::priority_queue<Item, std::vector<Item>, std::greater<>> q;
        q.emplace(dis[s] = 0, s);
        while (!q.empty()) {
            usize u = q.top().second;
            q.pop();
            if (vis[u]) continue;
            vis[u] = true;
            for (usize i = _head[u]; i != npos; i = _e[i].nxt) {
                const auto& e = _e[i];
                if (!e.w) continue;
                CostT next = dis[u] + e.cost + p[u] - p[e.to];
                if (next >= dis[e.to]) continue;
                dis[e.to] = next;
                q.emplace(next, e.to);
            }
        }
        return dis[t] != cost_inf;
    };
    auto bfs = [&] {
        dep.assign(n, 0);
        cur = _head;
        q.clear();
        dep[s] = 1;
        q.push_back(s);
        for (usize hd = 0; hd < q.size(); hd++) {
            usize u = q[hd];
            for (usize i = _head[u]; i != npos; i = _e[i].nxt) {
                const auto& e = _e[i];
                if (!e.w || dep[e.to] || p[e.to] != p[u] + e.cost) continue;
                dep[e.to] = dep[u] + 1;
                q.push_back(e.to);
                if (e.to == t) return true;
            }
        }
        return false;
    };
    auto dfs = [&](auto&& self, usize u, FlowT flow) -> FlowT {
        if (u == t || !flow) return flow;
        FlowT rem = flow;
        for (usize& i = cur[u]; i != npos; i = _e[i].nxt) {
            if (!_e[i].w ||
                dep[_e[i].to] != dep[u] + 1 ||
                p[_e[i].to] != p[u] + _e[i].cost)
                continue;
            FlowT c = self(self, _e[i].to, std::min(rem, _e[i].w));
            _e[i].w -= c;
            _e[i ^ 1].w += c;
            if (!(rem -= c)) break;
        }
        if (rem == flow) dep[u] = 0;
        return flow - rem;
    };

    FlowT total_flow = 0;
    CostT total_cost = 0;
    bool neg =
        std::ranges::any_of(_e, [](auto& e) { return e.w && e.cost < 0; });
    if (neg ? spfa() : dijkstra()) {
        do {
            for (usize u = 0; u < n; u++) {
                if (dis[u] != cost_inf) p[u] += dis[u];
            }
            if (!maximize && p[t] >= 0) break;
            FlowT delta = 0;
            while (bfs()) delta += dfs(dfs, s, flow_inf);
            total_flow += delta;
            total_cost += static_cast<CostT>(delta) * p[t];
        } while (dijkstra());
    }
    return {total_flow, total_cost};
}

}  // namespace acm

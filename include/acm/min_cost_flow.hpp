#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <type_traits>
#include <utility>
#include <vector>

#include "def.hpp"

namespace acm
{

template <typename FlowT = i32, typename CostT = i64>
class MinCostFlow {
    static_assert(std::is_integral_v<FlowT>);
    static_assert(std::is_integral_v<CostT> && std::is_signed_v<CostT>);

public:
    struct Edge {
        usize from, to;
        FlowT capacity, flow;
        CostT cost;
    };

private:
    struct Arc {
        usize to, nxt;
        FlowT w;
        CostT c;
    };

    static constexpr usize npos = usize(-1);

    std::vector<usize> head;
    std::vector<Arc> e;

    template <bool maximize>
    std::pair<FlowT, CostT> flow(usize s, usize t) {
        if (s == t) return {0, 0};
        constexpr FlowT flow_inf = std::numeric_limits<FlowT>::max();
        constexpr CostT cost_inf = std::numeric_limits<CostT>::max() / 4;
        usize n = size_V();
        std::vector<usize> dep(n), cur(n), q;
        std::vector<u8> vis(n);
        std::vector<CostT> dis(n), h(n, 0);
        q.reserve(n);

        auto spfa = [&] {
            dis.assign(n, cost_inf);
            vis.assign(n, false);
            q.clear();
            dis[s] = 0;
            vis[s] = true;
            q.push_back(s);
            for (usize hd = 0; hd < q.size(); hd++) {
                usize u = q[hd];
                vis[u] = false;
                for (usize i = head[u]; i != npos; i = e[i].nxt) {
                    usize v = e[i].to;
                    if (!e[i].w) continue;
                    CostT tmp = dis[u] + e[i].c + h[u] - h[v];
                    if (tmp < dis[v]) {
                        dis[v] = tmp;
                        if (!vis[v]) vis[v] = true, q.push_back(v);
                    }
                }
            }
            return dis[t] != cost_inf;
        };
        auto dijkstra = [&] {
            using Item = std::pair<CostT, usize>;
            std::priority_queue<Item, std::vector<Item>, std::greater<>> q;
            dis.assign(n, cost_inf);
            vis.assign(n, false);
            q.push({dis[s] = 0, s});
            while (!q.empty()) {
                auto [d, u] = q.top();
                q.pop();
                if (vis[u]) continue;
                vis[u] = true;
                for (usize i = head[u]; i != npos; i = e[i].nxt) {
                    if (!e[i].w) continue;
                    usize v = e[i].to;
                    CostT tmp = d + e[i].c + h[u] - h[v];
                    if (tmp < dis[v]) q.push({dis[v] = tmp, v});
                }
            }
            return dis[t] != cost_inf;
        };
        auto bfs = [&] {
            dep.assign(n, npos);
            cur = head;
            q.clear();
            dep[s] = 0;
            q.push_back(s);
            for (usize hd = 0; hd < q.size(); hd++) {
                usize u = q[hd];
                for (usize i = head[u]; i != npos; i = e[i].nxt) {
                    usize v = e[i].to;
                    if (!e[i].w || dep[v] != npos || h[v] != h[u] + e[i].c)
                        continue;
                    dep[v] = dep[u] + 1;
                    q.push_back(v);
                }
            }
            return dep[t] != npos;
        };
        auto dfs = [&](auto&& self, usize u, FlowT flow) -> FlowT {
            if (u == t || !flow) return flow;
            FlowT res = flow;
            for (usize& i = cur[u]; i != npos; i = e[i].nxt) {
                if (!e[i].w ||
                    dep[e[i].to] != dep[u] + 1 ||
                    h[e[i].to] != h[u] + e[i].c)
                    continue;
                FlowT c = self(self, e[i].to, std::min(res, e[i].w));
                e[i].w -= c;
                e[i ^ 1].w += c;
                if (!(res -= c)) break;
            }
            if (res == flow) dep[u] = npos;
            return flow - res;
        };

        bool neg = std::any_of(e.begin(), e.end(), [](auto& x) {
            return x.w > 0 && x.c < 0;
        });
        FlowT total_flow = 0;
        CostT total_cost = 0;
        while (neg ? spfa() : dijkstra()) {
            neg = false;
            for (usize u = 0; u < n; u++) {
                if (dis[u] != cost_inf) h[u] += dis[u];
            }
            if (!maximize && h[t] >= 0) break;
            FlowT delta = 0;
            while (bfs()) delta += dfs(dfs, s, flow_inf);
            total_flow += delta;
            total_cost += CostT(delta) * h[t];
        }
        return {total_flow, total_cost};
    }

public:
    MinCostFlow() = default;
    explicit MinCostFlow(usize n): head(n, npos) {}

    usize size_V() const { return head.size(); }
    usize size_E() const { return e.size() / 2; }

    Edge edge(usize i) const {
        const auto& a = e[2 * i];
        const auto& b = e[2 * i + 1];
        return {b.to, a.to, a.w + b.w, b.w, a.c};
    }

    usize add_edge(usize u, usize v, FlowT cap, CostT cost) {
        usize i = size_E();
        e.push_back({v, head[u], cap, cost});
        head[u] = 2 * i;
        e.push_back({u, head[v], 0, -cost});
        head[v] = 2 * i + 1;
        return i;
    }

    std::pair<FlowT, CostT> min_cost_flow(usize s, usize t) {
        return flow<false>(s, t);
    }
    std::pair<FlowT, CostT> min_cost_max_flow(usize s, usize t) {
        return flow<true>(s, t);
    }
};

}  // namespace acm

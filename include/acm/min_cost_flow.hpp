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
        CostT cost;
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
        q.reserve(n);
        std::vector<u8> vis(n), inq(n);
        std::vector<CostT> dis(n), pot(n, 0);

        auto spfa = [&] {
            dis.assign(n, cost_inf);
            inq.assign(n, false);
            std::queue<usize> qu;
            dis[s] = 0;
            inq[s] = true;
            qu.push(s);
            while (!qu.empty()) {
                usize u = qu.front();
                qu.pop();
                inq[u] = false;
                for (usize i = head[u]; i != npos; i = e[i].nxt) {
                    const auto& a = e[i];
                    if (!a.w) continue;
                    CostT nd = dis[u] + a.cost + pot[u] - pot[a.to];
                    if (nd >= dis[a.to]) continue;
                    dis[a.to] = nd;
                    if (!inq[a.to]) inq[a.to] = true, qu.push(a.to);
                }
            }
            return dis[t] != cost_inf;
        };

        auto dijkstra = [&] {
            dis.assign(n, cost_inf);
            vis.assign(n, false);
            using Item = std::pair<CostT, usize>;
            std::priority_queue<Item, std::vector<Item>, std::greater<>> pq;
            dis[s] = 0;
            pq.push({0, s});
            while (!pq.empty()) {
                auto [d, u] = pq.top();
                pq.pop();
                if (vis[u]) continue;
                vis[u] = true;
                for (usize i = head[u]; i != npos; i = e[i].nxt) {
                    const auto& a = e[i];
                    if (!a.w) continue;
                    CostT nd = d + a.cost + pot[u] - pot[a.to];
                    if (nd >= dis[a.to]) continue;
                    dis[a.to] = nd;
                    pq.push({nd, a.to});
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
            for (usize h = 0; h < q.size(); h++) {
                usize u = q[h];
                for (usize i = head[u]; i != npos; i = e[i].nxt) {
                    const auto& a = e[i];
                    if (!a.w || dep[a.to] != npos ||
                        pot[a.to] != pot[u] + a.cost)
                        continue;
                    dep[a.to] = dep[u] + 1;
                    q.push_back(a.to);
                }
            }
            return dep[t] != npos;
        };

        auto dfs = [&](auto&& self, usize u, FlowT f) -> FlowT {
            if (u == t || !f) return f;
            for (usize& i = cur[u]; i != npos; i = e[i].nxt) {
                auto& a = e[i];
                if (!a.w || dep[a.to] != dep[u] + 1 ||
                    pot[a.to] != pot[u] + a.cost)
                    continue;
                FlowT c = self(self, a.to, std::min(f, a.w));
                if (!c) continue;
                a.w -= c;
                e[i ^ 1].w += c;
                return c;
            }
            dep[u] = npos;
            return 0;
        };

        bool neg = false;
        for (const auto& a: e)
            if (a.w > 0 && a.cost < 0) {
                neg = true;
                break;
            }

        FlowT total_flow = 0;
        CostT total_cost = 0;
        if (!(neg ? spfa() : dijkstra())) return {0, 0};
        do {
            for (usize u = 0; u < n; u++)
                if (dis[u] != cost_inf) pot[u] += dis[u];
            if (!maximize && pot[t] >= 0) break;
            FlowT delta = 0;
            while (bfs())
                while (FlowT f = dfs(dfs, s, flow_inf)) delta += f;
            total_flow += delta;
            total_cost += CostT(delta) * pot[t];
        } while (dijkstra());
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
        return {b.to, a.to, a.w + b.w, b.w, a.cost};
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

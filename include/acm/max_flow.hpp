#pragma once

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>

#include "def.hpp"

namespace acm
{

template <typename FlowT = i32>
class MaxFlow {
    static_assert(std::is_integral_v<FlowT>);

public:
    struct Edge {
        usize from, to;
        FlowT capacity, flow;
    };

private:
    struct Arc {
        usize to, nxt;
        FlowT w;
    };

    static constexpr usize npos = usize(-1);

    std::vector<usize> head;
    std::vector<Arc> e;

public:
    MaxFlow() = default;
    explicit MaxFlow(usize n): head(n, npos) {}

    usize size_V() const { return head.size(); }
    usize size_E() const { return e.size() / 2; }

    Edge edge(usize i) const {
        const auto& a = e[2 * i];
        const auto& b = e[2 * i + 1];
        return {b.to, a.to, a.w + b.w, b.w};
    }

    usize add_edge(usize u, usize v, FlowT cap) {
        usize i = size_E();
        e.push_back({v, head[u], cap});
        head[u] = 2 * i;
        e.push_back({u, head[v], 0});
        head[v] = 2 * i + 1;
        return i;
    }

    FlowT max_flow(usize s, usize t) {
        if (s == t) return 0;
        constexpr FlowT INF = std::numeric_limits<FlowT>::max();
        usize n = size_V();
        std::vector<usize> dep(n), cur(n), q(n);

        auto bfs = [&] {
            dep.assign(n, npos);
            cur = head;
            usize h = 0, r = 0;
            dep[s] = 0;
            q[r++] = s;
            while (h < r) {
                usize u = q[h++];
                for (usize i = head[u]; i != npos; i = e[i].nxt) {
                    const auto& a = e[i];
                    if (!a.w || dep[a.to] != npos) continue;
                    dep[a.to] = dep[u] + 1;
                    q[r++] = a.to;
                }
            }
            return dep[t] != npos;
        };

        auto dfs = [&](auto&& self, usize u, FlowT f) -> FlowT {
            if (u == t || !f) return f;
            for (usize& i = cur[u]; i != npos; i = e[i].nxt) {
                auto& a = e[i];
                if (!a.w || dep[a.to] != dep[u] + 1) continue;
                FlowT c = self(self, a.to, std::min(f, a.w));
                if (!c) continue;
                a.w -= c;
                e[i ^ 1].w += c;
                return c;
            }
            dep[u] = npos;
            return 0;
        };

        FlowT ans = 0;
        while (bfs())
            while (FlowT f = dfs(dfs, s, INF)) ans += f;
        return ans;
    }
};

}  // namespace acm

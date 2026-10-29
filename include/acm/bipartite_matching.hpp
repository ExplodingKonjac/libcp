#pragma once

#include <queue>
#include <utility>
#include <vector>

#include "def.hpp"

namespace acm
{

class BipartiteMatching {
    usize n, m;
    std::vector<std::vector<usize>> g;

public:
    static constexpr usize npos = usize(-1);

    BipartiteMatching(usize n, usize m): n(n), m(m), g(n) {}

    usize size_l() const { return n; }
    usize size_r() const { return m; }
    void add_edge(usize u, usize v) { g[u].push_back(v); }

    std::pair<usize, std::vector<usize>> max_matching() {
        std::vector<usize> ml(n, npos), mr(m, npos), dis(n);
        usize ans = 0, shortest = npos;

        auto bfs = [&] {
            std::queue<usize> q;
            shortest = npos;
            for (usize u = 0; u < n; u++) {
                if (ml[u] == npos) dis[u] = 0, q.push(u);
                else dis[u] = npos;
            }
            while (!q.empty()) {
                usize u = q.front();
                q.pop();
                if (dis[u] + 1 >= shortest) continue;
                for (usize v: g[u]) {
                    usize w = mr[v];
                    if (w == npos) shortest = dis[u] + 1;
                    else if (dis[w] == npos) {
                        dis[w] = dis[u] + 1;
                        q.push(w);
                    }
                }
            }
            return shortest != npos;
        };

        auto dfs = [&](auto&& self, usize u) -> bool {
            for (usize v: g[u]) {
                usize w = mr[v];
                if ((w == npos && dis[u] + 1 == shortest) ||
                    (w != npos && dis[w] == dis[u] + 1 && self(self, w))) {
                    ml[u] = v;
                    mr[v] = u;
                    return true;
                }
            }
            dis[u] = npos;
            return false;
        };

        while (bfs())
            for (usize u = 0; u < n; u++)
                if (ml[u] == npos && dfs(dfs, u)) ans++;
        return {ans, ml};
    }
};

}  // namespace acm

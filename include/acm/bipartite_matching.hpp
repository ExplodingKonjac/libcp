#pragma once

#include <queue>
#include <vector>

#include "def.hpp"

namespace acm
{

class BipartiteMatching {
private:
    usize _n, _m;
    std::vector<std::vector<usize>> _g;

public:
    static constexpr usize npos = -1;

    BipartiteMatching(usize n, usize m): _n{n}, _m{m}, _g(n) {}

    usize size_l() const { return _n; }
    usize size_r() const { return _m; }

    void add_edge(usize u, usize v) { _g[u].push_back(v); }

    std::pair<usize, std::vector<usize>> max_matching() {
        std::vector<usize> _ml(_n, npos), _mr(_m, npos), _dis(_n);
        usize ans = 0, shortest;
        auto bfs = [&]() {
            std::queue<usize> q;
            shortest = npos;
            for (usize u = 0; u < _n; u++) {
                if (_ml[u] == npos) {
                    _dis[u] = 0;
                    q.push(u);
                } else {
                    _dis[u] = npos;
                }
            }
            while (!q.empty()) {
                usize u = q.front();
                q.pop();
                if (_dis[u] + 1 >= shortest) continue;
                for (usize v: _g[u]) {
                    usize w = _mr[v];
                    if (w == npos) {
                        shortest = _dis[u] + 1;
                    } else if (_dis[w] == npos) {
                        _dis[w] = _dis[u] + 1;
                        q.push(w);
                    }
                }
            }
            return shortest != npos;
        };
        auto dfs = [&](auto&& self, usize u) -> bool {
            for (usize v: _g[u]) {
                usize w = _mr[v];
                if ((w == npos && _dis[u] + 1 == shortest) ||
                    (w != npos && _dis[w] == _dis[u] + 1 && self(self, w))) {
                    _ml[u] = v;
                    _mr[v] = u;
                    return true;
                }
            }
            _dis[u] = npos;
            return false;
        };
        while (bfs()) {
            for (usize u = 0; u < _n; u++) {
                if (_ml[u] == npos && dfs(dfs, u)) ans++;
            }
        }
        return {ans, _ml};
    }
};

}  // namespace acm

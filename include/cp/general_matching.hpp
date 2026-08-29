#pragma once

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "cp/def.hpp"

namespace cp
{

class GeneralMatching {
private:
    usize _n;
    std::vector<std::vector<usize>> _g;

public:
    static constexpr usize npos = -1;

    explicit GeneralMatching(usize n): _n{n}, _g(n) {}

    usize size() const { return _n; }

    void add_edge(usize u, usize v) {
        _g[u].push_back(v);
        _g[v].push_back(u);
    }

    std::pair<usize, std::vector<usize>> max_matching() {
        std::vector<usize> match(_n, npos), parent(_n), base(_n), q;
        std::vector<u8> used(_n), blossom(_n);

        auto lca = [&](usize u, usize v) {
            std::vector<u8> seen(_n);
            while (true) {
                u = base[u];
                seen[u] = true;
                if (match[u] == npos) break;
                u = parent[match[u]];
            }
            while (!seen[base[v]]) v = parent[match[base[v]]];
            return base[v];
        };
        auto mark_path = [&](usize u, usize b, usize child) {
            while (base[u] != b) {
                blossom[base[u]] = blossom[base[match[u]]] = true;
                parent[u] = child;
                child = match[u];
                u = parent[match[u]];
            }
        };
        auto find_path = [&](usize root) {
            std::iota(base.begin(), base.end(), 0);
            std::fill(used.begin(), used.end(), false);
            std::fill(parent.begin(), parent.end(), npos);
            q.clear();
            q.push_back(root);
            used[root] = true;
            for (usize head = 0; head < q.size(); ++head) {
                usize u = q[head];
                for (usize v: _g[u]) {
                    if (base[u] == base[v] || match[u] == v) continue;
                    if (v == root ||
                        (match[v] != npos && parent[match[v]] != npos)) {
                        usize b = lca(u, v);
                        std::fill(blossom.begin(), blossom.end(), false);
                        mark_path(u, b, v);
                        mark_path(v, b, u);
                        for (usize x = 0; x < _n; ++x) {
                            if (!blossom[base[x]]) continue;
                            base[x] = b;
                            if (!used[x]) {
                                used[x] = true;
                                q.push_back(x);
                            }
                        }
                    } else if (parent[v] == npos) {
                        parent[v] = u;
                        if (match[v] == npos) return v;
                        v = match[v];
                        used[v] = true;
                        q.push_back(v);
                    }
                }
            }
            return npos;
        };

        usize result = 0;
        for (usize root = 0; root < _n; ++root) {
            if (match[root] != npos) continue;
            usize v = find_path(root);
            if (v == npos) continue;
            ++result;
            while (v != npos) {
                usize u = parent[v], next = match[u];
                match[v] = u;
                match[u] = v;
                v = next;
            }
        }
        return {result, std::move(match)};
    }
};

}  // namespace cp

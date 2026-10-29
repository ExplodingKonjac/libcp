#pragma once

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "def.hpp"

namespace acm
{

class GeneralMatching {
    usize n;
    std::vector<std::vector<usize>> g;

public:
    static constexpr usize npos = usize(-1);

    explicit GeneralMatching(usize n): n(n), g(n) {}
    usize size() const { return n; }

    void add_edge(usize u, usize v) {
        g[u].push_back(v);
        g[v].push_back(u);
    }

    std::pair<usize, std::vector<usize>> max_matching() {
        std::vector<usize> match(n, npos), fa(n), base(n), q;
        std::vector<u8> used(n), bl(n);

        auto lca = [&](usize u, usize v) {
            std::vector<u8> seen(n);
            while (true) {
                u = base[u];
                seen[u] = true;
                if (match[u] == npos) break;
                u = fa[match[u]];
            }
            while (!seen[base[v]]) v = fa[match[base[v]]];
            return base[v];
        };
        auto mark = [&](usize u, usize b, usize child) {
            while (base[u] != b) {
                bl[base[u]] = bl[base[match[u]]] = true;
                fa[u] = child;
                child = match[u];
                u = fa[match[u]];
            }
        };
        auto find = [&](usize root) {
            std::iota(base.begin(), base.end(), usize(0));
            std::fill(used.begin(), used.end(), false);
            std::fill(fa.begin(), fa.end(), npos);
            q.clear();
            q.push_back(root);
            used[root] = true;
            for (usize h = 0; h < q.size(); h++) {
                usize u = q[h];
                for (usize v: g[u]) {
                    if (base[u] == base[v] || match[u] == v) continue;
                    if (v == root ||
                        (match[v] != npos && fa[match[v]] != npos)) {
                        usize b = lca(u, v);
                        std::fill(bl.begin(), bl.end(), false);
                        mark(u, b, v);
                        mark(v, b, u);
                        for (usize x = 0; x < n; x++) {
                            if (!bl[base[x]]) continue;
                            base[x] = b;
                            if (!used[x]) used[x] = true, q.push_back(x);
                        }
                    } else if (fa[v] == npos) {
                        fa[v] = u;
                        if (match[v] == npos) return v;
                        v = match[v];
                        used[v] = true;
                        q.push_back(v);
                    }
                }
            }
            return npos;
        };

        usize ans = 0;
        for (usize root = 0; root < n; root++) {
            if (match[root] != npos) continue;
            usize v = find(root);
            if (v == npos) continue;
            ans++;
            while (v != npos) {
                usize u = fa[v], nxt = match[u];
                match[v] = u;
                match[u] = v;
                v = nxt;
            }
        }
        return {ans, std::move(match)};
    }
};

}  // namespace acm

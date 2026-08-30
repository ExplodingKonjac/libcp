#pragma once

#include <utility>
#include <vector>

#include "def.hpp"

namespace acm
{

template <typename E = void>
class Graph {
public:
    struct Edge {
        usize to;
        E weight;
    };

private:
    std::vector<std::vector<Edge>> g;
    usize m = 0;

public:
    explicit Graph(usize n = 0): g(n) {}
    usize v_size() const { return g.size(); }
    usize e_size() const { return m; }
    void add_edge(usize u, usize v, E w) {
        g[u].push_back({v, std::move(w)}), m++;
    }
    const std::vector<Edge>& out(usize u) const { return g[u]; }
};

template <>
class Graph<void> {
    std::vector<std::vector<usize>> g;
    usize m = 0;

public:
    explicit Graph(usize n = 0): g(n) {}
    usize v_size() const { return g.size(); }
    usize e_size() const { return m; }
    void add_edge(usize u, usize v) { g[u].push_back(v), m++; }
    const std::vector<usize>& out(usize u) const { return g[u]; }
};

}  // namespace acm

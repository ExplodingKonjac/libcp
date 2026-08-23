#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include "cp/bipartite_matching.hpp"
#include "cp/bipartite_weighted_matching.hpp"

using cp::BipartiteMatching;
using cp::BipartiteWeightedMatching;
using cp::usize;

usize brute_matching(const std::vector<std::vector<bool>>& edges) {
    const usize n = edges.size();
    const usize m = n ? edges[0].size() : 0;
    std::vector<bool> used(m);
    auto dfs = [&](auto&& self, usize u) -> usize {
        if (u == n) return 0;
        usize best = self(self, u + 1);
        for (usize v = 0; v < m; v++) {
            if (!edges[u][v] || used[v]) continue;
            used[v] = true;
            best = std::max(best, 1 + self(self, u + 1));
            used[v] = false;
        }
        return best;
    };
    return dfs(dfs, 0);
}

void check_matching(const std::vector<std::vector<bool>>& edges) {
    const usize n = edges.size();
    const usize m = n ? edges[0].size() : 0;
    BipartiteMatching graph(n, m);
    for (usize u = 0; u < n; u++)
        for (usize v = 0; v < m; v++)
            if (edges[u][v]) graph.add_edge(u, v);

    const auto expected = brute_matching(edges);
    auto [size, match] = graph.max_matching();
    assert(size == expected);
    assert(match.size() == n);
    std::vector<bool> used(m);
    usize actual = 0;
    for (usize u = 0; u < n; u++) {
        if (match[u] == BipartiteMatching::npos) continue;
        assert(match[u] < m && edges[u][match[u]] && !used[match[u]]);
        used[match[u]] = true;
        actual++;
    }
    assert(actual == size);
}

std::optional<int> brute_weighted_matching(
    const std::vector<std::vector<std::optional<int>>>& edges
) {
    const usize n = edges.size();
    const usize m = n ? edges[0].size() : 0;
    if (n > m) return std::nullopt;
    std::vector<bool> used(m);
    std::optional<int> best;
    auto dfs = [&](auto&& self, usize u, int sum) -> void {
        if (u == n) {
            if (!best || sum > *best) best = sum;
            return;
        }
        for (usize v = 0; v < m; v++) {
            if (!edges[u][v] || used[v]) continue;
            used[v] = true;
            self(self, u + 1, sum + *edges[u][v]);
            used[v] = false;
        }
    };
    dfs(dfs, 0, 0);
    return best;
}

void check_weighted_matching(
    const std::vector<std::vector<std::optional<int>>>& edges
) {
    const usize n = edges.size();
    const usize m = n ? edges[0].size() : 0;
    BipartiteWeightedMatching<int> graph(n, m);
    for (usize u = 0; u < n; u++)
        for (usize v = 0; v < m; v++)
            if (edges[u][v]) graph.add_edge(u, v, *edges[u][v]);

    const auto expected = brute_weighted_matching(edges);
    const auto actual = graph.max_weighted_matching();
    assert(actual.has_value() == expected.has_value());
    if (!actual) return;
    assert(actual->first == *expected);
    assert(actual->second.size() == n);
    std::vector<bool> used(m);
    int sum = 0;
    for (usize u = 0; u < n; u++) {
        const usize v = actual->second[u];
        assert(v < m && edges[u][v] && !used[v]);
        used[v] = true;
        sum += *edges[u][v];
    }
    assert(sum == actual->first);
}

void test_edge_operations() {
    BipartiteWeightedMatching<int> graph(1, 1);
    assert(graph.get_edge(0, 0) == decltype(graph)::no_edge);
    graph.add_edge(0, 0, 3);
    graph.add_edge(0, 0, 2);
    assert(graph.get_edge(0, 0) == 3);
    graph.set_edge(0, 0, -4);
    assert(graph.get_edge(0, 0) == -4);
    assert(graph.max_weighted_matching()->first == -4);
}

void test_edge_cases() {
    BipartiteMatching empty(0, 3);
    assert(empty.size_l() == 0 && empty.size_r() == 3);
    assert((empty.max_matching() == std::pair{usize{0}, std::vector<usize>{}}));

    BipartiteWeightedMatching<int> weighted_empty(0, 3);
    assert(
        (weighted_empty.max_weighted_matching() ==
         std::pair{0, std::vector<usize>{}})
    );

    const int limit = std::numeric_limits<int>::max();
    check_weighted_matching({{limit, -1}, {limit - 1, -limit}});
}

void test_random_cases() {
    std::mt19937 rng(0x5eed);
    for (int tc = 0; tc < 1000; tc++) {
        const usize n = rng() % 6;
        const usize m = rng() % 6;
        std::vector<std::vector<bool>> edges(n, std::vector<bool>(m));
        std::vector<std::vector<std::optional<int>>> weighted(
            n, std::vector<std::optional<int>>(m)
        );
        for (usize u = 0; u < n; u++) {
            for (usize v = 0; v < m; v++) {
                edges[u][v] = rng() % 3 != 0;
                if (edges[u][v]) weighted[u][v] = int(rng() % 21) - 10;
            }
        }
        check_matching(edges);
        check_weighted_matching(weighted);
    }
}

int main() {
    check_matching({});
    check_matching({{false, false}, {false, false}});
    check_matching({{true, true}, {true, false}});
    check_weighted_matching({});
    check_weighted_matching({{1}, {2}});
    check_weighted_matching({{1, std::nullopt}, {1, std::nullopt}});
    check_weighted_matching({{-5, -2}, {-4, -9}});
    test_edge_cases();
    test_edge_operations();
    test_random_cases();
    std::cout << "All tests passed!\n";
}

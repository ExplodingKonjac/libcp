#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

#include "acm/general_matching.hpp"
#include "acm/general_weighted_matching.hpp"

using acm::GeneralMatching;
using acm::GeneralWeightedMatching;
using acm::usize;

usize brute_matching(const std::vector<std::vector<bool>>& edges) {
    const usize n = edges.size();
    auto dfs = [&](auto&& self, usize used) -> usize {
        usize u = 0;
        while (u < n && (used >> u & 1)) u++;
        if (u == n) return 0;
        usize best = self(self, used | usize{1} << u);
        for (usize v = u + 1; v < n; v++) {
            if (edges[u][v] && !(used >> v & 1))
                best = std::max(
                    best, 1 + self(self, used | usize{1} << u | usize{1} << v)
                );
        }
        return best;
    };
    return dfs(dfs, 0);
}

void check_matching(const std::vector<std::vector<bool>>& edges) {
    const usize n = edges.size();
    GeneralMatching graph(n);
    for (usize u = 0; u < n; u++)
        for (usize v = u + 1; v < n; v++)
            if (edges[u][v]) graph.add_edge(u, v);

    auto [size, match] = graph.max_matching();
    assert(graph.size() == n && match.size() == n);
    assert(size == brute_matching(edges));
    usize actual = 0;
    for (usize u = 0; u < n; u++) {
        if (match[u] == GeneralMatching::npos) continue;
        assert(match[u] < n && match[match[u]] == u && edges[u][match[u]]);
        actual += u < match[u];
    }
    assert(actual == size);
}

template <typename T>
std::optional<T> brute_weighted_matching(
    const std::vector<std::vector<std::optional<T>>>& edges
) {
    const usize n = edges.size();
    if (n % 2) return std::nullopt;
    std::optional<T> best;
    auto dfs = [&](auto&& self, usize used, T sum) -> void {
        usize u = 0;
        while (u < n && (used >> u & 1)) u++;
        if (u == n) {
            if (!best || sum > *best) best = sum;
            return;
        }
        for (usize v = u + 1; v < n; v++) {
            if (edges[u][v] && !(used >> v & 1))
                self(
                    self, used | usize{1} << u | usize{1} << v,
                    sum + *edges[u][v]
                );
        }
    };
    dfs(dfs, 0, T{});
    return best;
}

template <typename T>
void check_weighted_matching(
    const std::vector<std::vector<std::optional<T>>>& edges
) {
    const usize n = edges.size();
    GeneralWeightedMatching<T> graph(n);
    for (usize u = 0; u < n; u++)
        for (usize v = u + 1; v < n; v++)
            if (edges[u][v]) graph.add_edge(u, v, *edges[u][v]);

    const auto expected = brute_weighted_matching(edges);
    const auto actual = graph.max_weighted_matching();
    assert(graph.size() == n && actual.has_value() == expected.has_value());
    if (!actual) return;
    if constexpr (std::floating_point<T>)
        assert(std::abs(actual->first - *expected) < T{1e-8});
    else assert(actual->first == *expected);
    assert(actual->second.size() == n);
    T sum{};
    for (usize u = 0; u < n; u++) {
        usize v = actual->second[u];
        assert(v < n && actual->second[v] == u && edges[u][v]);
        if (u < v) sum += *edges[u][v];
    }
    if constexpr (std::floating_point<T>)
        assert(std::abs(actual->first - sum) < T{1e-8});
    else assert(actual->first == sum);
}

void test_edge_operations() {
    GeneralMatching plain(2);
    plain.add_edge(0, 1);
    plain.add_edge(0, 1);
    assert(plain.max_matching().first == 1);
    assert(plain.max_matching().first == 1);

    GeneralWeightedMatching<int> graph(2);
    assert(graph.get_edge(0, 1) == decltype(graph)::no_edge);
    graph.add_edge(0, 1, -4);
    graph.add_edge(1, 0, -7);
    assert(graph.get_edge(0, 1) == -4 && graph.get_edge(1, 0) == -4);
    graph.set_edge(1, 0, -9);
    assert(graph.get_edge(0, 1) == -9);
    assert(graph.max_weighted_matching()->first == -9);
    assert(graph.max_weighted_matching()->first == -9);

    GeneralWeightedMatching<long long> wide(2);
    wide.add_edge(0, 1, std::numeric_limits<long long>::max());
    assert(
        wide.max_weighted_matching()->first ==
        std::numeric_limits<long long>::max()
    );
}

void test_fixed_cases() {
    check_matching({});
    check_matching(std::vector(3, std::vector<bool>(3)));
    check_matching(
        {
            {false, true, true, false, false},
            {true, false, true, true, false},
            {true, true, false, false, false},
            {false, true, false, false, true},
            {false, false, false, true, false},
        }
    );
    std::vector nested(9, std::vector<bool>(9));
    for (
        auto [u, v]: std::vector<std::pair<usize, usize>>{{0, 1},
                                                          {1, 2},
                                                          {2, 0},
                                                          {3, 4},
                                                          {4, 5},
                                                          {5, 3},
                                                          {6, 7},
                                                          {7, 8},
                                                          {8, 6},
                                                          {2, 3},
                                                          {5, 6},
                                                          {8, 0}}
    )
        nested[u][v] = nested[v][u] = true;
    check_matching(nested);

    check_weighted_matching<int>({});
    check_weighted_matching<int>({{std::nullopt}});
    check_weighted_matching<int>({
        {std::nullopt, -4, std::nullopt, std::nullopt},
        {-4, std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt, -5},
        {std::nullopt, std::nullopt, -5, std::nullopt},
    });
    check_weighted_matching<int>({
        {std::nullopt, 8, 7, 6},
        {8, std::nullopt, 6, 1},
        {7, 6, std::nullopt, 1},
        {6, 1, 1, std::nullopt},
    });
    check_weighted_matching<double>({
        {std::nullopt, 1.25, 0.5, std::nullopt},
        {1.25, std::nullopt, std::nullopt, -2.75},
        {0.5, std::nullopt, std::nullopt, 3.125},
        {std::nullopt, -2.75, 3.125, std::nullopt},
    });
}

void test_random_cases() {
    std::mt19937 rng(0x5eed);
    for (int tc = 0; tc < 2000; tc++) {
        usize n = rng() % 11;
        std::vector edges(n, std::vector<bool>(n));
        for (usize u = 0; u < n; u++)
            for (usize v = u + 1; v < n; v++)
                edges[u][v] = edges[v][u] = rng() % 3 != 0;
        check_matching(edges);
    }
    for (int tc = 0; tc < 1000; tc++) {
        usize n = rng() % 9;
        std::vector edges(n, std::vector<std::optional<int>>(n, std::nullopt));
        for (usize u = 0; u < n; u++) {
            for (usize v = u + 1; v < n; v++) {
                if (rng() % 3 == 0) continue;
                edges[u][v] = edges[v][u] = int(rng() % 41) - 20;
            }
        }
        check_weighted_matching(edges);
    }
    for (int tc = 0; tc < 300; tc++) {
        usize n = 2 * (rng() % 5);
        std::vector edges(
            n, std::vector<std::optional<double>>(n, std::nullopt)
        );
        for (usize u = 0; u < n; u++) {
            for (usize v = u + 1; v < n; v++) {
                if (rng() % 4 == 0) continue;
                double w = (int(rng() % 401) - 200) / 10.0;
                edges[u][v] = edges[v][u] = w;
            }
        }
        check_weighted_matching(edges);
    }
}

void test_large_weighted_smoke() {
    constexpr usize n = 200;
    GeneralWeightedMatching<int> graph(n);
    for (usize u = 0; u < n; u++) {
        for (usize v = u + 1; v < n; v++) {
            int w = static_cast<int>((u * 1009 + v * 9176) % 1000);
            graph.add_edge(u, v, w);
        }
    }
    for (usize u = 0; u < n; u += 2) graph.set_edge(u, u + 1, 1'000'000);

    auto result = graph.max_weighted_matching();
    assert(result && result->first == static_cast<int>(n / 2) * 1'000'000);
    for (usize u = 0; u < n; u++) assert(result->second[u] == (u ^ 1));
}

int main() {
    test_edge_operations();
    test_fixed_cases();
    test_random_cases();
    test_large_weighted_smoke();
    std::cout << "All tests passed!\n";
}

#include <algorithm>
#include <cassert>
#include <iostream>
#include <optional>
#include <random>
#include <vector>

#include "cp/flow/min_cost_circulation.hpp"

using cp::MinCostCirculation;
using cp::usize;

struct Edge {
    usize from, to;
    int capacity, cost;
};

std::optional<int> brute_force(
    const std::vector<int>& supply, const std::vector<Edge>& edges
) {
    std::optional<int> best;
    std::vector<int> balance(supply.size());
    auto dfs = [&](auto&& self, usize i, int cost) -> void {
        if (i == edges.size()) {
            if (balance == supply && (!best || cost < *best)) best = cost;
            return;
        }
        const auto& e = edges[i];
        for (int flow = 0; flow <= e.capacity; flow++) {
            balance[e.from] += flow;
            balance[e.to] -= flow;
            self(self, i + 1, cost + flow * e.cost);
            balance[e.from] -= flow;
            balance[e.to] += flow;
        }
    };
    dfs(dfs, 0, 0);
    return best;
}

void check_case(
    const std::vector<int>& supply, const std::vector<Edge>& edges
) {
    MinCostCirculation<int, int> g(supply.size());
    for (usize u = 0; u < supply.size(); u++) g.node(u).supply = supply[u];
    for (auto e: edges) g.add_edge(e.from, e.to, e.capacity, e.cost);

    auto expected = brute_force(supply, edges);
    auto actual = g.circulation();
    assert(actual == expected);
    if (!actual) return;

    int cost = 0;
    std::vector<int> balance(supply.size());
    for (usize i = 0; i < edges.size(); i++) {
        auto e = g.edge(i);
        assert(0 <= e.flow && e.flow <= e.capacity);
        balance[e.from] += e.flow;
        balance[e.to] -= e.flow;
        cost += e.flow * e.cost;
    }
    assert(balance == supply);
    assert(cost == *actual);
}

void test_edge_cases() {
    check_case({}, {});
    check_case({1}, {});
    check_case({2, -2}, {{0, 1, 1, 0}});
    check_case({0}, {{0, 0, 3, -2}});
    check_case({0, 0}, {{0, 1, 2, -3}, {1, 0, 2, 1}});
}

void test_parallel_edges_and_flow() {
    MinCostCirculation<int, int> g(2);
    g.node(0).supply = 3;
    g.node(1).supply = -3;
    auto cheap = g.add_edge(0, 1, 2, 1);
    auto expensive = g.add_edge(0, 1, 2, 5);

    assert(g.circulation() == 7);
    assert(g.edge(cheap).flow == 2);
    assert(g.edge(expensive).flow == 1);
}

void test_against_brute_force() {
    std::mt19937 rng(0x5eed);
    for (int tc = 0; tc < 2000; tc++) {
        usize n = rng() % 4 + 1;
        usize m = rng() % 7;
        std::vector<int> supply(n);
        int sum = 0;
        for (usize u = 0; u + 1 < n; u++) {
            supply[u] = static_cast<int>(rng() % 5) - 2;
            sum += supply[u];
        }
        supply.back() = -sum;

        std::vector<Edge> edges;
        for (usize i = 0; i < m; i++) {
            edges.push_back(
                {
                    rng() % n,
                    rng() % n,
                    static_cast<int>(rng() % 3),
                    static_cast<int>(rng() % 9) - 4,
                }
            );
        }
        check_case(supply, edges);
    }
}

int main() {
    test_edge_cases();
    test_parallel_edges_and_flow();
    test_against_brute_force();
    std::cout << "All tests passed!\n";
}

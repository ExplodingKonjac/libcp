#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include "cp/max_flow.hpp"
#include "cp/min_cost_flow.hpp"

using cp::MaxFlow;
using cp::MinCostFlow;
using cp::usize;

struct Edge {
    usize from, to;
    int capacity, cost;
};

int min_cut(usize n, usize s, usize t, const std::vector<Edge>& edges) {
    int best = std::numeric_limits<int>::max();
    for (usize mask = 0; mask < (usize{1} << n); mask++) {
        if (!(mask >> s & 1) || (mask >> t & 1)) continue;
        int capacity = 0;
        for (auto e: edges)
            if ((mask >> e.from & 1) && !(mask >> e.to & 1))
                capacity += e.capacity;
        best = std::min(best, capacity);
    }
    return best;
}

void check_max_flow(usize n, const std::vector<Edge>& edges) {
    MaxFlow<int> g(n);
    for (auto e: edges) g.add_edge(e.from, e.to, e.capacity);

    int flow = g.max_flow(0, n - 1);
    assert(flow == min_cut(n, 0, n - 1, edges));

    std::vector<int> balance(n);
    for (usize i = 0; i < edges.size(); i++) {
        auto e = g.edge(i);
        assert(0 <= e.flow && e.flow <= e.capacity);
        balance[e.from] += e.flow;
        balance[e.to] -= e.flow;
    }
    assert(balance[0] == flow);
    assert(balance[n - 1] == -flow);
    for (usize u = 1; u + 1 < n; u++) assert(balance[u] == 0);
}

std::vector<std::optional<int>> brute_costs(
    usize n, const std::vector<Edge>& edges
) {
    usize max_flow = 0;
    for (auto e: edges) max_flow += e.capacity;
    std::vector<std::optional<int>> best(max_flow + 1);
    std::vector<int> balance(n);
    auto dfs = [&](auto&& self, usize i, int cost) -> void {
        if (i == edges.size()) {
            for (usize u = 1; u + 1 < n; u++)
                if (balance[u] != 0) return;
            int flow = balance[0];
            if (flow < 0 || balance[n - 1] != -flow) return;
            auto& result = best[flow];
            if (!result || cost < *result) result = cost;
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

void check_flow_state(
    const MinCostFlow<int, int>& g, usize n, int flow, int cost
) {
    std::vector<int> balance(n);
    int actual_cost = 0;
    for (usize i = 0; i < g.size_E(); i++) {
        auto e = g.edge(i);
        assert(0 <= e.flow && e.flow <= e.capacity);
        balance[e.from] += e.flow;
        balance[e.to] -= e.flow;
        actual_cost += e.flow * e.cost;
    }
    assert(balance[0] == flow);
    assert(balance[n - 1] == -flow);
    for (usize u = 1; u + 1 < n; u++) assert(balance[u] == 0);
    assert(actual_cost == cost);
}

void check_min_cost_flow(
    usize n, const std::vector<Edge>& edges, bool negative_cost
) {
    auto expected = brute_costs(n, edges);
    usize max_flow = expected.size() - 1;
    while (!expected[max_flow]) max_flow--;

    MinCostFlow<int, int> g(n);
    for (auto e: edges) g.add_edge(e.from, e.to, e.capacity, e.cost);
    auto [flow, cost] = g.min_cost_max_flow(0, n - 1, negative_cost);
    assert(flow == static_cast<int>(max_flow));
    assert(cost == expected[max_flow]);
    check_flow_state(g, n, flow, cost);

    MinCostFlow<int, int> h(n);
    for (auto e: edges) h.add_edge(e.from, e.to, e.capacity, e.cost);
    auto [min_flow, min_cost] = h.min_cost_flow(0, n - 1, negative_cost);
    auto best =
        *std::min_element(expected.begin(), expected.end(), [](auto a, auto b) {
            if (!a) return false;
            if (!b) return true;
            return *a < *b;
        });
    assert(expected[min_flow] == min_cost);
    assert(min_cost == best);
    check_flow_state(h, n, min_flow, min_cost);
}

void test_edge_cases() {
    MaxFlow<int> g(2);
    assert(g.max_flow(0, 0) == 0);
    assert(g.max_flow(0, 1) == 0);

    check_max_flow(2, {{0, 1, 3, 0}, {0, 1, 4, 0}, {0, 0, 9, 0}});
    check_min_cost_flow(3, {{0, 1, 2, -3}, {1, 2, 2, 1}, {0, 2, 1, 4}}, true);
    check_min_cost_flow(3, {{0, 1, 2, 1}, {1, 2, 1, 2}}, false);

    MaxFlow<unsigned> unsigned_max_flow(2);
    unsigned_max_flow.add_edge(0, 1, 3);
    assert(unsigned_max_flow.max_flow(0, 1) == 3);
    MinCostFlow<unsigned, int> unsigned_min_cost_flow(2);
    unsigned_min_cost_flow.add_edge(0, 1, 3, 2);
    assert(
        (unsigned_min_cost_flow.min_cost_max_flow(0, 1) == std::pair{3U, 6})
    );
}

void test_repeated_min_cost_flow() {
    MinCostFlow<int, int> g(4);
    g.add_edge(0, 1, 1, 0);
    g.add_edge(1, 2, 1, 5);
    g.add_edge(2, 3, 1, 0);
    g.add_edge(0, 2, 1, 10);
    assert((g.min_cost_max_flow(0, 3) == std::pair{1, 5}));

    g.add_edge(1, 3, 1, 0);
    assert((g.min_cost_max_flow(0, 3) == std::pair{1, 5}));
}

void test_max_flow_against_min_cut() {
    std::mt19937 rng(0x12345);
    for (int tc = 0; tc < 2000; tc++) {
        usize n = rng() % 5 + 2;
        usize m = rng() % 15;
        std::vector<Edge> edges;
        for (usize i = 0; i < m; i++)
            edges.push_back({rng() % n, rng() % n, int(rng() % 8), 0});
        check_max_flow(n, edges);
    }
}

void test_min_cost_flow_against_brute_force() {
    std::mt19937 rng(0x6789a);
    for (int tc = 0; tc < 2000; tc++) {
        usize n = rng() % 3 + 2;
        usize m = rng() % 7;
        std::vector<Edge> edges;
        bool negative_cost = tc % 2 == 0;
        for (usize i = 0; i < m; i++) {
            usize from = negative_cost ? rng() % (n - 1) : rng() % n;
            usize to =
                negative_cost ? from + 1 + rng() % (n - from - 1) : rng() % n;
            int cost = negative_cost ? int(rng() % 9) - 4 : int(rng() % 5);
            edges.push_back({from, to, int(rng() % 3), cost});
        }
        check_min_cost_flow(n, edges, negative_cost);
    }
}

int main() {
    test_edge_cases();
    test_repeated_min_cost_flow();
    test_max_flow_against_min_cut();
    test_min_cost_flow_against_brute_force();
    std::cout << "All tests passed!\n";
}

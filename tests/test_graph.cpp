#include <iostream>
#include <set>
#include <tuple>
#include <vector>

#include "cp/graph.hpp"

using namespace cp;

template <typename T, typename U>
void assert_eq(const T& a, const U& b, const char* msg) {
    if (static_cast<std::common_type_t<T, U>>(a)
        != static_cast<std::common_type_t<T, U>>(b)) {
        std::cerr << "Assertion failed: " << msg << "\n";
        exit(1);
    }
}

void test_unweighted() {
    Graph<void> g(4);
    assert_eq(g.v_size(), 4, "v_size unweighted initialized");
    assert_eq(g.e_size(), 0, "e_size unweighted initialized");

    g.add_edge(0, 1);
    g.add_edge(0, 2);
    g.add_edge(1, 3);
    g.add_edge(2, 3);

    assert_eq(g.e_size(), 4, "e_size unweighted after add_edge");

    std::vector<usize> out0;
    for (auto v: g.out(0)) out0.push_back(v);
    assert_eq(out0.size(), 2, "out size unweighted");
    assert_eq(out0[0], 1, "out unweighted [0]");
    assert_eq(out0[1], 2, "out unweighted [1]");

    std::set<std::pair<usize, usize>> edges;
    for (auto e: g.edges()) {
        edges.emplace(std::get<0>(e), std::get<1>(e));
    }
    assert_eq(edges.size(), 4, "edges size unweighted");
    assert_eq(edges.count({0, 1}), 1, "edges contains 0->1");
    assert_eq(edges.count({0, 2}), 1, "edges contains 0->2");
    assert_eq(edges.count({1, 3}), 1, "edges contains 1->3");
    assert_eq(edges.count({2, 3}), 1, "edges contains 2->3");
}

void test_weighted() {
    Graph<int> g(3);
    assert_eq(g.v_size(), 3, "v_size weighted");
    assert_eq(g.e_size(), 0, "e_size weighted");

    g.add_edge(0, 1, 10);
    g.add_edge(1, 2, 20);
    assert_eq(g.e_size(), 2, "e_size weighted after add_edge");

    std::vector<std::pair<usize, int>> out1;
    for (auto e: g.out(1)) {
        out1.push_back(e);
    }
    assert_eq(out1.size(), 1, "out size weighted");
    assert_eq(out1[0].first, 2, "out weighted dest");
    assert_eq(out1[0].second, 20, "out weighted weight");

    std::set<std::tuple<usize, usize, int>> edges;
    for (auto e: g.edges()) {
        edges.emplace(std::get<0>(e), std::get<1>(e), std::get<2>(e));
    }
    assert_eq(edges.size(), 2, "edges size weighted");
    assert_eq(edges.count({0, 1, 10}), 1, "edges count 0->1(10)");
    assert_eq(edges.count({1, 2, 20}), 1, "edges count 1->2(20)");
}

void test_const_graph() {
    Graph<int> g(2);
    g.add_edge(0, 1, 5);
    const auto& cg = g;

    usize sum_out = 0;
    for (auto e: cg.out(0)) {
        sum_out += e.second;
    }
    assert_eq(sum_out, 5, "const out weighted");

    usize sum_edge = 0;
    for (auto e: cg.edges()) {
        sum_edge += std::get<2>(e);
    }
    assert_eq(sum_edge, 5, "const edges weighted");
}

void test_const_graph_unweighted() {
    Graph<void> g(2);
    g.add_edge(0, 1);
    const auto& cg = g;

    usize sum_out = 0;
    for (auto v: cg.out(0)) {
        sum_out += v;
    }
    assert_eq(sum_out, 1, "const out unweighted");

    usize sum_edge = 0;
    for (auto e: cg.edges()) {
        sum_edge += std::get<1>(e);
    }
    assert_eq(sum_edge, 1, "const edges unweighted");
}

int main() {
    test_unweighted();
    test_weighted();
    test_const_graph();
    test_const_graph_unweighted();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}

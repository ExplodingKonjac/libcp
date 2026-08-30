#include <cassert>
#include "acm/graph.hpp"

int main() {
    acm::Graph<void> g(3); g.add_edge(0, 1); g.add_edge(0, 2);
    assert(g.v_size() == 3 && g.e_size() == 2 && g.out(0)[1] == 2);
    acm::Graph<int> w(2); w.add_edge(0, 1, 7);
    assert(w.out(0)[0].to == 1 && w.out(0)[0].weight == 7);
}

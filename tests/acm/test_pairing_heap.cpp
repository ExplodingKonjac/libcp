#include <cassert>
#include "acm/pairing_heap.hpp"

int main() {
    acm::PairingHeap<int> a, b;
    auto p = a.push(2);
    auto q = a.push(7); b.push(5); a.join(b);
    assert(a.top() == 7 && b.empty());
    a.modify(p, 9);
    assert(a.pop() == 9);
    assert(a.erase(q) == 7);
    a.push(11);
    a.push(4);
    auto c = a;
    assert(c.size() == a.size() && c.top() == 11);
    a.pop();
    assert(c.top() == 11);
    c.clear();
    assert(c.empty());
}

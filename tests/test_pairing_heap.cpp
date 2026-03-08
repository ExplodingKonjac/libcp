#include <cassert>
#include <iostream>
#include <vector>

#include "cp/pairing_heap.hpp"

using cp::PairingHeap;

int main() {
    PairingHeap<int> heap;
    assert(heap.empty());
    assert(heap.size() == 0);

    auto it1 = heap.push(10);
    heap.push(20);
    heap.push(5);

    assert(heap.size() == 3);
    assert(!heap.empty());
    assert(heap.top() == 20);

    heap.pop();
    assert(heap.size() == 2);
    assert(heap.top() == 10);

    heap.push(30);
    assert(heap.top() == 30);
    assert(heap.size() == 3);

    // Test erase
    auto it4 = heap.push(25);
    assert(heap.top() == 30);
    int erased = heap.erase(it4);
    assert(erased == 25);
    assert(heap.size() == 3);

    // Test modify (decrease key)
    heap.modify(it1, 2);  // 10 -> 2
    // heap should have 30, 5, 2
    assert(heap.top() == 30);
    heap.pop();
    assert(heap.top() == 5);
    heap.pop();
    assert(heap.top() == 2);
    heap.pop();
    assert(heap.empty());

    // Test modify (increase key)
    auto it5 = heap.push(50);
    heap.push(60);
    heap.modify(it5, 70);
    assert(heap.top() == 70);
    heap.pop();
    assert(heap.top() == 60);
    heap.pop();

    // Test join
    PairingHeap<int> h1;
    h1.push(100);
    h1.push(50);

    PairingHeap<int> h2;
    h2.push(200);
    h2.push(70);

    h1.join(h2);
    assert(h1.size() == 4);
    assert(h2.size() == 0);
    assert(h1.top() == 200);

    std::cout << "All tests passed.\n";
    return 0;
}
#include <cassert>
#include <chrono>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

#include "cp/pairing_heap.hpp"

using cp::PairingHeap;

void test_basic() {
    std::cout << "test_basic... ";

    PairingHeap<int> heap;
    assert(heap.empty());
    assert(heap.size() == 0);

    heap.push(10);
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

    // Test erase (non-root)
    auto it4 = heap.push(25);
    assert(heap.top() == 30);
    int erased = heap.erase(it4);
    assert(erased == 25);
    assert(heap.size() == 3);

    // Test modify (decrease key, non-root: 10 -> 2)
    auto it1 = heap.push(10);
    heap.modify(it1, 2);
    // heap has {30, 10, 5, 2}
    assert(heap.top() == 30);
    heap.pop();
    assert(heap.top() == 10);
    heap.pop();
    assert(heap.top() == 5);
    heap.pop();
    assert(heap.top() == 2);
    heap.pop();
    assert(heap.empty());

    // Test modify (increase key, non-root)
    heap.push(60);
    auto it5 = heap.push(50);
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

    std::cout << "OK\n";
}

void test_single_element() {
    std::cout << "test_single_element... ";

    PairingHeap<int> heap;
    auto it = heap.push(42);
    assert(heap.size() == 1);
    assert(heap.top() == 42);
    assert(*it == 42);

    // Modify root (increase)
    heap.modify(it, 99);
    assert(heap.top() == 99);

    // Modify root (decrease)
    heap.modify(it, 1);
    assert(heap.top() == 1);

    // Erase root (only element)
    int val = heap.erase(it);
    assert(val == 1);
    assert(heap.empty());

    // Pop single element
    heap.push(7);
    assert(heap.pop() == 7);
    assert(heap.empty());

    // Erase non-root in two-element heap
    heap.push(100);
    auto it2 = heap.push(50);
    val = heap.erase(it2);
    assert(val == 50);
    assert(heap.size() == 1);
    assert(heap.top() == 100);

    std::cout << "OK\n";
}

void test_clear() {
    std::cout << "test_clear... ";

    PairingHeap<int> heap;
    for (int i = 0; i < 1000; i++) heap.push(i);
    assert(heap.size() == 1000);
    heap.clear();
    assert(heap.empty());
    assert(heap.size() == 0);

    // Reuse after clear
    heap.push(5);
    heap.push(10);
    assert(heap.top() == 10);
    assert(heap.size() == 2);

    std::cout << "OK\n";
}

void test_copy() {
    std::cout << "test_copy... ";

    PairingHeap<int> h1;
    h1.push(10);
    h1.push(20);
    h1.push(30);

    // Copy constructor
    PairingHeap<int> h2(h1);
    assert(h2.size() == 3);
    assert(h2.top() == 30);
    // Mutate original — copy should be independent
    h1.pop();
    assert(h1.top() == 20);
    assert(h2.top() == 30);

    // Copy assignment
    PairingHeap<int> h3;
    h3.push(999);
    h3 = h2;
    assert(h3.size() == 3);
    assert(h3.top() == 30);
    // Mutate copy — original should be independent
    h3.pop();
    assert(h2.top() == 30);
    assert(h3.top() == 20);

    // Self-assignment
    auto& ref = (h3 = h3);
    assert(&ref == &h3);
    assert(h3.size() == 2);

    // Drain all copies to verify no double-free
    while (!h1.empty()) h1.pop();
    while (!h2.empty()) h2.pop();
    while (!h3.empty()) h3.pop();

    std::cout << "OK\n";
}

void test_move() {
    std::cout << "test_move... ";

    PairingHeap<int> h1;
    h1.push(10);
    h1.push(20);
    h1.push(30);

    // Move constructor
    PairingHeap<int> h2(std::move(h1));
    assert(h2.size() == 3);
    assert(h2.top() == 30);
    assert(h1.empty());

    // Move assignment
    PairingHeap<int> h3;
    h3.push(999);
    h3 = std::move(h2);
    assert(h3.size() == 3);
    assert(h3.top() == 30);
    assert(h2.empty());

    // Self move-assignment
    auto& ref = (h3 = std::move(h3));
    assert(&ref == &h3);
    assert(h3.size() == 3);

    std::cout << "OK\n";
}

void test_min_heap() {
    std::cout << "test_min_heap... ";

    PairingHeap<int, std::greater<int>> heap;
    heap.push(30);
    heap.push(10);
    heap.push(20);
    assert(heap.top() == 10);
    heap.pop();
    assert(heap.top() == 20);
    heap.pop();
    assert(heap.top() == 30);
    heap.pop();
    assert(heap.empty());

    std::cout << "OK\n";
}

void test_emplace() {
    std::cout << "test_emplace... ";

    PairingHeap<std::pair<int, int>> heap;
    auto it = heap.push({3, 5});
    heap.push({1, 2});
    heap.push({3, 7});

    assert(heap.top() == std::make_pair(3, 7));
    heap.pop();
    assert(heap.top() == std::make_pair(3, 5));
    assert(*it == std::make_pair(3, 5));
    heap.pop();
    assert(heap.top() == std::make_pair(1, 2));
    heap.pop();
    assert(heap.empty());

    std::cout << "OK\n";
}

void test_sorted_output() {
    std::cout << "test_sorted_output... ";

    const int N = 10000;
    std::mt19937 rng(42);
    PairingHeap<int> heap;
    std::vector<int> data(N);
    for (int i = 0; i < N; i++) {
        data[i] = rng();
        heap.push(data[i]);
    }
    std::sort(data.begin(), data.end(), std::greater<int>());
    for (int i = 0; i < N; i++) {
        assert(heap.top() == data[i]);
        heap.pop();
    }
    assert(heap.empty());

    std::cout << "OK\n";
}

void test_stress_modify_erase() {
    std::cout << "test_stress_modify_erase... ";

    const int N = 10000;
    std::mt19937 rng(12345);
    PairingHeap<int> heap;
    std::vector<PairingHeap<int>::point_iterator> iters;

    for (int i = 0; i < N; i++) {
        iters.push_back(heap.push(rng() % 1000000));
    }

    // Random modify
    for (int i = 0; i < N; i++) {
        int new_val = rng() % 1000000;
        heap.modify(iters[i], new_val);
    }

    // Erase half
    for (int i = 0; i < N / 2; i++) {
        heap.erase(iters[i]);
    }
    assert(heap.size() == N - N / 2);

    // Pop the rest and verify sorted order
    int prev = heap.pop();
    while (!heap.empty()) {
        int cur = heap.pop();
        assert(cur <= prev);
        prev = cur;
    }

    std::cout << "OK\n";
}

void test_join_multiple() {
    std::cout << "test_join_multiple... ";

    const int K = 10;
    const int N = 500;
    std::mt19937 rng(777);
    PairingHeap<int> combined;

    for (int k = 0; k < K; k++) {
        PairingHeap<int> tmp;
        for (int i = 0; i < N; i++) tmp.push(rng() % 1000000);
        combined.join(tmp);
        assert(tmp.empty());
    }
    assert(combined.size() == K * N);

    int prev = combined.pop();
    while (!combined.empty()) {
        int cur = combined.pop();
        assert(cur <= prev);
        prev = cur;
    }

    std::cout << "OK\n";
}

void test_all_correctness() {
    std::cout << "Running correctness tests...\n";
    test_basic();
    test_single_element();
    test_clear();
    test_copy();
    test_move();
    test_min_heap();
    test_emplace();
    test_sorted_output();
    test_stress_modify_erase();
    test_join_multiple();
    std::cout << "All correctness tests passed.\n";
}

void test_performance() {
    std::cout << "Running performance test...\n";
    const int N = 2000000;
    std::mt19937 rng(1337);
    std::vector<int> data(N);
    for (int i = 0; i < N; ++i) {
        data[i] = rng();
    }

    auto start = std::chrono::high_resolution_clock::now();
    PairingHeap<int> heap;
    for (int i = 0; i < N; ++i) {
        heap.push(data[i]);
    }
    for (int i = 0; i < N; ++i) {
        heap.pop();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "PairingHeap time: " << diff.count() << " s\n";

    start = std::chrono::high_resolution_clock::now();
    std::priority_queue<int> pq;
    for (int i = 0; i < N; ++i) {
        pq.push(data[i]);
    }
    for (int i = 0; i < N; ++i) {
        pq.pop();
    }
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "std::priority_queue time: " << diff.count() << " s\n";
}

int main() {
    test_all_correctness();
    test_performance();
    return 0;
}
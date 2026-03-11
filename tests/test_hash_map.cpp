#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "cp/hash_map.hpp"

using cp::FlatHashMap;

// ============================================================
// Correctness tests
// ============================================================

void test_default_construct() {
    std::cout << "test_default_construct... ";
    FlatHashMap<int, int> map;
    assert(map.empty());
    assert(map.size() == 0);
    assert(map.begin() == map.end());
    assert(map.find(0) == map.end());
    assert(map.get(42) == nullptr);
    std::cout << "OK\n";
}

void test_try_emplace_basic() {
    std::cout << "test_try_emplace_basic... ";
    FlatHashMap<int, int> map;

    // Insert new elements
    auto it1 = map.try_emplace(1, 100);
    assert(it1->first == 1);
    assert(it1->second == 100);
    assert(map.size() == 1);

    auto it2 = map.try_emplace(2, 200);
    assert(it2->first == 2);
    assert(it2->second == 200);
    assert(map.size() == 2);

    // try_emplace on existing key should not overwrite
    auto it1b = map.try_emplace(1, 999);
    assert(it1b->first == 1);
    assert(it1b->second == 100);  // unchanged
    assert(map.size() == 2);

    std::cout << "OK\n";
}

void test_emplace() {
    std::cout << "test_emplace... ";
    FlatHashMap<int, int> map;

    auto res1 = map.emplace(std::pair{1, 10});
    assert(res1.has_value());
    assert((*res1)->first == 1);
    assert((*res1)->second == 10);

    // Duplicate insert should return nullopt
    auto res2 = map.emplace(std::pair{1, 20});
    assert(!res2.has_value());
    assert(map.size() == 1);

    std::cout << "OK\n";
}

void test_insert() {
    std::cout << "test_insert... ";
    FlatHashMap<int, int> map;

    auto res1 = map.insert(std::pair{5, 50});
    assert(res1.has_value());
    assert((*res1)->first == 5);

    // Duplicate
    auto res2 = map.insert(std::pair{5, 99});
    assert(!res2.has_value());
    assert(map.size() == 1);

    std::cout << "OK\n";
}

void test_get() {
    std::cout << "test_get... ";
    FlatHashMap<int, int> map;
    map.try_emplace(1, 10);
    map.try_emplace(2, 20);

    int* p = map.get(1);
    assert(p != nullptr);
    assert(*p == 10);

    // Modify through pointer
    *p = 42;
    assert(*map.get(1) == 42);

    // Non-existent key
    assert(map.get(999) == nullptr);

    // Const version
    const auto& cmap = map;
    const int* cp = cmap.get(2);
    assert(cp != nullptr);
    assert(*cp == 20);
    assert(cmap.get(999) == nullptr);

    std::cout << "OK\n";
}

void test_find() {
    std::cout << "test_find... ";
    FlatHashMap<int, int> map;
    map.try_emplace(10, 100);
    map.try_emplace(20, 200);

    auto it = map.find(10);
    assert(it != map.end());
    assert(it->first == 10);
    assert(it->second == 100);

    auto it2 = map.find(999);
    assert(it2 == map.end());

    // Const find
    const auto& cmap = map;
    auto cit = cmap.find(20);
    assert(cit != cmap.end());
    assert(cit->second == 200);

    std::cout << "OK\n";
}

void test_erase_by_key() {
    std::cout << "test_erase_by_key... ";
    FlatHashMap<int, int> map;
    map.try_emplace(1, 10);
    map.try_emplace(2, 20);
    map.try_emplace(3, 30);

    assert(map.erase(2));
    assert(map.size() == 2);
    assert(map.get(2) == nullptr);

    // Erase non-existent key
    assert(!map.erase(999));
    assert(map.size() == 2);

    // Remaining keys still accessible
    assert(*map.get(1) == 10);
    assert(*map.get(3) == 30);

    std::cout << "OK\n";
}

void test_erase_by_iterator() {
    std::cout << "test_erase_by_iterator... ";
    FlatHashMap<int, int> map;
    map.try_emplace(1, 10);
    map.try_emplace(2, 20);

    auto it = map.find(1);
    assert(it != map.end());
    map.erase(it);
    assert(map.size() == 1);
    assert(map.get(1) == nullptr);
    assert(*map.get(2) == 20);

    std::cout << "OK\n";
}

void test_insert_after_erase() {
    std::cout << "test_insert_after_erase... ";
    FlatHashMap<int, int> map;

    // Insert and erase, then re-insert same key
    map.try_emplace(42, 1);
    assert(map.erase(42));
    assert(map.get(42) == nullptr);

    map.try_emplace(42, 2);
    assert(*map.get(42) == 2);
    assert(map.size() == 1);

    std::cout << "OK\n";
}

void test_try_insert_with() {
    std::cout << "test_try_insert_with... ";
    FlatHashMap<int, int> map;

    int call_count = 0;
    auto factory = [&]() {
        call_count++;
        return 100;
    };

    auto it1 = map.try_insert_with(1, factory);
    assert(it1->first == 1);
    assert(it1->second == 100);
    assert(call_count == 1);

    // Existing key — factory should NOT be called
    auto it2 = map.try_insert_with(1, factory);
    assert(it2->second == 100);
    assert(call_count == 1);  // not called again

    std::cout << "OK\n";
}

void test_iteration() {
    std::cout << "test_iteration... ";
    FlatHashMap<int, int> map;

    const int N = 50;
    for (int i = 0; i < N; i++) {
        map.try_emplace(i, i * 10);
    }

    // Count via iteration
    int count = 0;
    std::vector<bool> seen(N, false);
    for (auto it = map.begin(); it != map.end(); ++it) {
        assert(it->first >= 0 && it->first < N);
        assert(it->second == it->first * 10);
        assert(!seen[it->first]);
        seen[it->first] = true;
        count++;
    }
    assert(count == N);

    // Range-for
    count = 0;
    for (auto& [k, v]: map) {
        assert(v == k * 10);
        count++;
    }
    assert(count == N);

    std::cout << "OK\n";
}

void test_empty_iteration() {
    std::cout << "test_empty_iteration... ";
    FlatHashMap<int, int> map;
    int count = 0;
    for (auto& [k, v]: map) {
        (void)k;
        (void)v;
        count++;
    }
    assert(count == 0);
    std::cout << "OK\n";
}

void test_move_construct() {
    std::cout << "test_move_construct... ";
    FlatHashMap<int, int> map;
    map.try_emplace(1, 10);
    map.try_emplace(2, 20);

    FlatHashMap<int, int> map2(std::move(map));
    assert(map2.size() == 2);
    assert(*map2.get(1) == 10);
    assert(*map2.get(2) == 20);

    // Moved-from should be empty
    assert(map.empty());
    assert(map.begin() == map.end());

    std::cout << "OK\n";
}

void test_move_assign() {
    std::cout << "test_move_assign... ";
    FlatHashMap<int, int> map;
    map.try_emplace(1, 10);

    FlatHashMap<int, int> map2;
    map2.try_emplace(99, 99);

    map2 = std::move(map);
    assert(map2.size() == 1);
    assert(*map2.get(1) == 10);
    assert(map2.get(99) == nullptr);

    std::cout << "OK\n";
}

void test_rehash_preserves_data() {
    std::cout << "test_rehash_preserves_data... ";
    FlatHashMap<int, int> map;

    // Insert enough elements to trigger multiple rehashes
    // capacity starts at 15, growth_left = 15*7/8 = 13
    const int N = 200;
    for (int i = 0; i < N; i++) {
        map.try_emplace(i, i * 7);
    }
    assert(map.size() == (size_t)N);

    // Verify all data survived rehashes
    for (int i = 0; i < N; i++) {
        int* p = map.get(i);
        assert(p != nullptr);
        assert(*p == i * 7);
    }

    std::cout << "OK\n";
}

void test_erase_then_heavy_insert() {
    std::cout << "test_erase_then_heavy_insert... ";
    FlatHashMap<int, int> map;

    // Fill, erase half, insert more — stresses deleted slot handling
    for (int i = 0; i < 100; i++) {
        map.try_emplace(i, i);
    }
    for (int i = 0; i < 50; i++) {
        assert(map.erase(i));
    }
    assert(map.size() == 50);

    // Insert new keys
    for (int i = 100; i < 200; i++) {
        map.try_emplace(i, i);
    }
    assert(map.size() == 150);

    // Verify
    for (int i = 0; i < 50; i++) {
        assert(map.get(i) == nullptr);
    }
    for (int i = 50; i < 200; i++) {
        assert(map.get(i) != nullptr);
        assert(*map.get(i) == i);
    }

    std::cout << "OK\n";
}

void test_hash_collision_handling() {
    std::cout << "test_hash_collision_handling... ";

    // Custom hash that always returns the same value — maximizes collisions
    struct BadHash {
        size_t operator()(int) const { return 42; }
    };

    FlatHashMap<int, int, BadHash> map;
    const int N = 50;
    for (int i = 0; i < N; i++) {
        map.try_emplace(i, i * 3);
    }
    assert(map.size() == (size_t)N);

    for (int i = 0; i < N; i++) {
        int* p = map.get(i);
        assert(p != nullptr);
        assert(*p == i * 3);
    }

    // Erase some and verify others survive
    for (int i = 0; i < N; i += 2) {
        assert(map.erase(i));
    }
    for (int i = 0; i < N; i++) {
        if (i % 2 == 0) {
            assert(map.get(i) == nullptr);
        } else {
            assert(*map.get(i) == i * 3);
        }
    }

    std::cout << "OK\n";
}

void test_modify_through_iterator() {
    std::cout << "test_modify_through_iterator... ";
    FlatHashMap<int, int> map;
    map.try_emplace(1, 10);

    auto it = map.find(1);
    assert(it != map.end());
    it->second = 999;
    assert(*map.get(1) == 999);

    std::cout << "OK\n";
}

void test_swap() {
    std::cout << "test_swap... ";
    FlatHashMap<int, int> a, b;
    a.try_emplace(1, 10);
    b.try_emplace(2, 20);
    b.try_emplace(3, 30);

    a.swap(b);
    assert(a.size() == 2);
    assert(b.size() == 1);
    assert(*a.get(2) == 20);
    assert(*b.get(1) == 10);

    std::cout << "OK\n";
}

// ============================================================
// Stress tests
// ============================================================

void test_stress_correctness() {
    std::cout << "test_stress_correctness... ";
    const int N = 100000;
    std::mt19937 rng(2026);

    FlatHashMap<int, int> fmap;
    std::unordered_map<int, int> ref;

    for (int i = 0; i < N; i++) {
        int op = rng() % 3;
        int key = rng() % (N / 5);
        int val = rng();

        if (op == 0) {
            // Insert
            fmap.try_emplace(key, val);
            ref.emplace(key, val);
        } else if (op == 1) {
            // Erase
            bool fe = fmap.erase(key);
            bool re = ref.erase(key) > 0;
            assert(fe == re);
        } else {
            // Lookup
            int* fp = fmap.get(key);
            auto rit = ref.find(key);
            if (rit == ref.end()) {
                assert(fp == nullptr);
            } else {
                assert(fp != nullptr);
                assert(*fp == rit->second);
            }
        }
    }

    // Final consistency check
    assert(fmap.size() == ref.size());
    for (auto& [k, v]: ref) {
        int* fp = fmap.get(k);
        assert(fp != nullptr);
        assert(*fp == v);
    }

    std::cout << "OK\n";
}

void test_stress_iteration_consistency() {
    std::cout << "test_stress_iteration_consistency... ";
    const int N = 5000;
    std::mt19937 rng(9999);

    FlatHashMap<int, int> map;

    for (int i = 0; i < N; i++) {
        map.try_emplace(rng() % 2000, i);
    }

    // Count via iteration should match size()
    size_t counted = 0;
    for (auto it = map.begin(); it != map.end(); ++it) {
        counted++;
    }
    assert(counted == map.size());

    std::cout << "OK\n";
}

// ============================================================
// Performance test
// ============================================================

void test_performance() {
    std::cout << "Running performance test...\n";
    const int N = 2000000;
    std::mt19937 rng(1337);
    std::vector<int> keys(N);
    for (int i = 0; i < N; i++) keys[i] = rng();

    {
        auto start = std::chrono::high_resolution_clock::now();
        FlatHashMap<int, int> map;
        for (int i = 0; i < N; i++) map.try_emplace(keys[i], i);
        for (int i = 0; i < N; i++) map.erase(keys[i]);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "FlatHashMap time: " << diff.count() << " s\n";
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        std::unordered_map<int, int> map;
        for (int i = 0; i < N; i++) map.emplace(keys[i], i);
        for (int i = 0; i < N; i++) map.erase(keys[i]);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "std::unordered_map time: " << diff.count() << " s\n";
    }
}

// ============================================================
// Main
// ============================================================

void test_all_correctness() {
    std::cout << "Running correctness tests...\n";
    test_default_construct();
    test_try_emplace_basic();
    test_emplace();
    test_insert();
    test_get();
    test_find();
    test_erase_by_key();
    test_erase_by_iterator();
    test_insert_after_erase();
    test_try_insert_with();
    test_iteration();
    test_empty_iteration();
    test_move_construct();
    test_move_assign();
    test_rehash_preserves_data();
    test_erase_then_heavy_insert();
    test_hash_collision_handling();
    test_modify_through_iterator();
    test_swap();
    test_stress_correctness();
    test_stress_iteration_consistency();
    std::cout << "All correctness tests passed.\n";
}

int main() {
    test_all_correctness();
    test_performance();
    return 0;
}
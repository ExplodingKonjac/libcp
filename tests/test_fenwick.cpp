#include "cp/fenwick_tree.hpp"
#include <iostream>
#include <vector>
#include <numeric>

using namespace cp;

struct MultZero {
    auto operator()() const { return 1; }
};

void test_basic() {
    usize n = 10;
    FenwickTree<int> ft(n);
    std::vector<int> naive(n, 0);

    for (int i = 0; i < n; ++i) {
        ft.add(i, i + 1);
        naive[i] += i + 1;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i; j <= n; ++j) {
            int expected = 0;
            for (int k = i; k < j; ++k) expected += naive[k];
            int actual = ft.sum(i, j);
            if (expected != actual) {
                std::cerr << "test_basic failed at sum(" << i << ", " << j << "): expected " << expected << ", got " << actual << "\n";
                exit(1);
            }
        }
    }
}

void test_zero_tree() {
    FenwickTree<int> ft(0);
    int res = ft.sum(0, 0);
    if (res != 0) {
        std::cerr << "test_zero_tree failed\n";
        exit(1);
    }
}

void test_add_zero() {
    FenwickTree<int> ft(5);
    ft.add(0, 10);
    if (ft.sum(0, 1) != 10) {
        std::cerr << "test_add_zero failed\n";
        exit(1);
    }
}

void test_single_element() {
    FenwickTree<int> ft(1);
    ft.add(0, 42);
    if (ft.sum(0, 1) != 42) {
        std::cerr << "test_single_element failed\n";
        exit(1);
    }
}

void test_pre_suf_sum() {
    usize n = 5;
    FenwickTree<int> ft(n);
    for (int i = 0; i < n; ++i) ft.add(i, i + 1);
    
    for (int i = 0; i <= n; ++i) {
        int expected_pre = 0;
        for (int k = 0; k < i; ++k) expected_pre += (k + 1);
        int expected_suf = 0;
        for (int k = i; k < n; ++k) expected_suf += (k + 1);
        
        if (ft.pre_sum(i) != expected_pre) {
            std::cerr << "test_pre_suf_sum pre_sum(" << i << ") failed\n";
            exit(1);
        }
        if (ft.suf_sum(i) != expected_suf) {
            std::cerr << "test_pre_suf_sum suf_sum(" << i << ") failed\n";
            exit(1);
        }
    }
}

void test_mult() {
    FenwickTree<int, std::multiplies<int>, std::divides<int>, MultZero> ft(5);
    for (int i = 0; i < 5; ++i) ft.add(i, i + 1); // prod = 120
    if (ft.sum(0, 5) != 120) {
        std::cerr << "test_mult failed: expected 120, got " << ft.sum(0, 5) << "\n";
        exit(1);
    }
    if (ft.sum(1, 4) != 24) { // 2*3*4 = 24
        std::cerr << "test_mult failed on sum(1, 4): expected 24, got " << ft.sum(1, 4) << "\n";
        exit(1);
    }
}

int main() {
    test_basic();
    test_zero_tree();
    test_add_zero();
    test_single_element();
    test_pre_suf_sum();
    test_mult();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}

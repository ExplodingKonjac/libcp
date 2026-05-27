#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include "cp/lazy_segtree.hpp"

using namespace cp;

namespace
{

struct SumValue {
    long long sum = 0;
    usize len = 0;

    friend bool operator==(const SumValue&, const SumValue&) = default;
};

struct SumPlus {
    SumValue operator()(const SumValue& lhs, const SumValue& rhs) const {
        return {lhs.sum + rhs.sum, lhs.len + rhs.len};
    }
};

struct RangeAdd {
    SumValue operator()(long long add, const SumValue& value) const {
        return {value.sum + add * static_cast<long long>(value.len), value.len};
    }
    long long operator()(long long lhs, long long rhs) const {
        return lhs + rhs;
    }
};

long long expect_sum(const std::vector<long long>& values, usize l, usize r) {
    return std::accumulate(
        values.begin() + static_cast<std::ptrdiff_t>(l),
        values.begin() + static_cast<std::ptrdiff_t>(r), 0LL
    );
}

void require(bool cond, const char* message) {
    if (!cond) {
        std::cerr << message << "\n";
        std::exit(1);
    }
}

void test_basic_range_add_and_query() {
    LazySegTree<SumValue, long long, SumPlus, RangeAdd> seg(6, [](usize i) {
        return SumValue{static_cast<long long>(i + 1), 1};
    });

    require(seg.apply(1, 5, 3), "apply(1, 5, 3) should succeed");
    auto total = seg.query(0, 6);
    require(
        total && total->sum == 33, "total sum after first range add is wrong"
    );

    auto mid = seg.query(1, 5);
    require(mid && mid->sum == 26, "range sum after first range add is wrong");

    require(seg.apply(0, 6, -2), "apply(0, 6, -2) should succeed");
    total = seg.query(0, 6);
    require(
        total && total->sum == 21, "total sum after second range add is wrong"
    );
    require(seg.all().sum == 21, "all() should match the full-range sum");
}

void test_point_modify_and_update() {
    LazySegTree<SumValue, long long, SumPlus, RangeAdd> seg(4, [](usize i) {
        return SumValue{static_cast<long long>(10 * (i + 1)), 1};
    });

    require(seg.apply(1, 4, 5), "apply before point operations should succeed");
    require(seg.modify(2, {7, 1}), "modify(2, {7, 1}) should succeed");
    auto total = seg.query(0, 4);
    require(total && total->sum == 87, "sum after modify is wrong");

    require(
        seg.update(
            1,
            [](const SumValue& value) {
                return SumValue{value.sum * 2, value.len};
            }
        ),
        "update(1, ...) should succeed"
    );
    total = seg.query(0, 4);
    require(total && total->sum == 112, "sum after update is wrong");
}

void test_invalid_ranges() {
    LazySegTree<SumValue, long long, SumPlus, RangeAdd> seg(3, [](usize i) {
        return SumValue{static_cast<long long>(i), 1};
    });

    require(!seg.apply(2, 2, 1), "empty range apply should fail");
    require(!seg.apply(0, 4, 1), "out-of-range apply should fail");
    require(!seg.modify(3, {0, 1}), "out-of-range modify should fail");
    require(
        !seg.update(5, [](const SumValue& value) { return value; }),
        "out-of-range update should fail"
    );
    require(
        !seg.query(1, 1).has_value(), "empty range query should return nullopt"
    );
    require(
        !seg.query(0, 4).has_value(), "out-of-range query should return nullopt"
    );
}

void test_randomized_range_add() {
    constexpr usize n = 512;
    constexpr int operations = 8000;
    std::mt19937_64 rng(20260527);
    std::uniform_int_distribution<long long> value_dist(-500, 500);
    std::uniform_int_distribution<long long> add_dist(-200, 200);
    std::uniform_int_distribution<usize> index_dist(0, n - 1);
    std::bernoulli_distribution update_dist(0.55);
    std::bernoulli_distribution point_assign_dist(0.15);

    std::vector<long long> values(n);
    for (auto& value: values) value = value_dist(rng);

    LazySegTree<SumValue, long long, SumPlus, RangeAdd> seg(n, [&](usize i) {
        return SumValue{values[i], 1};
    });

    for (int step = 0; step < operations; ++step) {
        if (update_dist(rng)) {
            usize l = index_dist(rng);
            usize r = index_dist(rng);
            if (l > r) std::swap(l, r);
            ++r;
            long long add = add_dist(rng);

            for (usize i = l; i < r; ++i) values[i] += add;
            require(seg.apply(l, r, add), "random apply should succeed");
            continue;
        }

        if (point_assign_dist(rng)) {
            usize p = index_dist(rng);
            values[p] = value_dist(rng);
            require(
                seg.modify(p, {values[p], 1}), "random modify should succeed"
            );
            continue;
        }

        usize l = index_dist(rng);
        usize r = index_dist(rng);
        if (l > r) std::swap(l, r);
        ++r;

        auto actual = seg.query(l, r);
        long long expected = expect_sum(values, l, r);
        if (!actual || actual->sum != expected) {
            std::cerr
                << "test_randomized_range_add failed at step "
                << step
                << " for query("
                << l
                << ", "
                << r
                << "): expected "
                << expected
                << ", got "
                << (actual ? std::to_string(actual->sum) : "nullopt")
                << "\n";
            std::exit(1);
        }
    }

    auto total = seg.query(0, n);
    long long expected_total = expect_sum(values, 0, n);
    require(total && total->sum == expected_total, "final total sum is wrong");
    require(
        seg.all().sum == expected_total,
        "all() should match the final total sum"
    );
}

}  // namespace

int main() {
    test_basic_range_add_and_query();
    test_point_modify_and_update();
    test_invalid_ranges();
    test_randomized_range_add();
    std::cout << "All lazy segtree tests passed!" << std::endl;
    return 0;
}

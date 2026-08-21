#pragma once

#include <concepts>
#include <functional>
#include <ranges>
#include <tuple>
#include <utility>
#include <vector>

#include "cp/def.hpp"

namespace cp
{

namespace detail
{

template <typename T, typename R>
std::vector<std::pair<T, T>> __to_vec(R&& r) {
    std::vector<std::pair<T, T>> p;
    for (auto&& x: r) {
        if constexpr (std::is_lvalue_reference_v<R>) {
            p.emplace_back(T(get<0>(x)), T(get<1>(x)));
        } else {
            p.emplace_back(T(get<0>(std::move(x))), T(get<1>(std::move(x))));
        }
    }
    return p;
}

}  // namespace detail

template <typename T, typename F, typename S>
concept pair_like = (std::tuple_size_v<T> == 2) && requires(T p) {
    { get<0>(p) } -> std::convertible_to<F>;
    { get<1>(p) } -> std::convertible_to<S>;
};

template <typename T>
concept arithmetic_like = requires(T x) {
    T{0};
    { x + x } -> std::convertible_to<T>;
    { x - x } -> std::convertible_to<T>;
    { x * x } -> std::convertible_to<T>;
    { x / x } -> std::convertible_to<T>;
    { -x } -> std::convertible_to<T>;
};

// Calculate f(x0) where f(x) is the polynomial decided by (x, y) pairs in `r`.
// T should not be native integers since division is used.
template <arithmetic_like T, std::ranges::input_range R>
    requires pair_like<std::ranges::range_value_t<R>, T, T>
T polynomial_interpolation(R&& r, T x0) {
    auto p = detail::__to_vec<T>(std::forward<R>(r));
    if (p.empty()) return T{0};
    usize n = p.size();
    T res{0};
    for (usize i = 0; i < n; i++) {
        T num{p[i].second}, den{1};
        for (usize j = 0; j < n; j++) {
            if (i == j) continue;
            num = num * (x0 - p[j].first);
            den = den * (p[i].first - p[j].first);
        }
        res += num / den;
    }
    return res;
}

// Resolves the polynomial coefficients decided by (x, y) pairs in `r`.
// T should not be native integers since division is used.
template <arithmetic_like T, std::ranges::input_range R, typename F>
    requires pair_like<std::ranges::range_value_t<R>, T, T>
std::vector<T> polynomial_coefficients(R&& r, F eq = std::equal_to<>{}) {
    auto p = detail::__to_vec<T>(std::forward<R>(r));
    if (p.empty()) return {T{0}};
    usize n = p.size();
    std::vector<T> A(n + 1, T{0}), B(n, T{0}), res(n, T{0});
    A[0] = T{1};
    for (usize i = 0; i < n; i++) {
        for (usize j = n; j; j--) A[j] = A[j - 1] - p[i].first * A[j];
        A[0] *= (-p[i].first);
    }
    for (usize i = 0; i < n; i++) {
        auto [x, y] = p[i];
        if (eq(x, T{0})) {
            copy(A.begin() + 1, A.end(), B.begin());
        } else {
            T tmp = -T{1} / x;
            B[0] = A[0] * tmp;
            for (usize j = 1; j < n; j++) B[j] = (A[j] - B[j - 1]) * tmp;
        }
        T tmp{1};
        for (usize j = 0; j < n; j++) {
            if (i != j) tmp *= (x - p[j].first);
        }
        tmp = y / tmp;
        for (usize j = 0; j < n; j++) res[j] += B[j] * tmp;
    }
    while (res.size() > 1 && eq(res.back(), T{0})) res.pop_back();
    return res;
}

}  // namespace cp

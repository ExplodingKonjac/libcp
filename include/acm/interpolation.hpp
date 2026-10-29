#pragma once

#include <algorithm>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "def.hpp"

namespace acm
{

namespace detail
{

template <typename T, typename R>
std::vector<std::pair<T, T>> to_pairs(R&& r) {
    std::vector<std::pair<T, T>> p;
    for (auto&& x: r)
        p.emplace_back(T(std::get<0>(x)), T(std::get<1>(x)));
    return p;
}

}  // namespace detail

template <typename T, typename R>
T polynomial_interpolation(const R& r, T x0) {
    auto p = detail::to_pairs<T>(r);
    usize n = p.size();
    if (!n) return T{0};
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

template <typename T, typename R, typename F = std::equal_to<>>
std::vector<T> polynomial_coefficients(const R& r, F eq = F{}) {
    auto p = detail::to_pairs<T>(r);
    usize n = p.size();
    if (!n) return {T{0}};
    std::vector<T> A(n + 1, T{0}), B(n, T{0}), res(n, T{0});
    A[0] = T{1};
    for (usize i = 0; i < n; i++) {
        for (usize j = n; j; j--) A[j] = A[j - 1] - p[i].first * A[j];
        A[0] *= (-p[i].first);
    }
    for (usize i = 0; i < n; i++) {
        auto [x, y] = p[i];
        if (eq(x, T{0})) {
            std::copy(A.begin() + 1, A.end(), B.begin());
        } else {
            T tmp = -T{1} / x;
            B[0] = A[0] * tmp;
            for (usize j = 1; j < n; j++) B[j] = (A[j] - B[j - 1]) * tmp;
        }
        T tmp{1};
        for (usize j = 0; j < n; j++)
            if (i != j) tmp *= (x - p[j].first);
        tmp = y / tmp;
        for (usize j = 0; j < n; j++) res[j] += B[j] * tmp;
    }
    while (res.size() > 1 && eq(res.back(), T{0})) res.pop_back();
    return res;
}

}  // namespace acm

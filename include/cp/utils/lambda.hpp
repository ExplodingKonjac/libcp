#pragma once

#include <concepts>
#include <tuple>
#include <utility>

#include "cp/def.hpp"

namespace cp
{

namespace detail
{

struct LambdaExpr {};

template <usize I>
struct Placeholder: LambdaExpr {
    template <typename... Args>
    [[gnu::always_inline]] auto operator()(Args&&... args) const {
        return get<I - 1>(std::forward_as_tuple(std::forward<Args>(args)...));
    }
};

template <typename T>
struct ConstExpr: LambdaExpr {
    T value;
    [[gnu::always_inline]] constexpr T operator()(auto&&...) const {
        return value;
    }
};

template <typename T>
concept LambdaExprLike = std::derived_from<std::decay_t<T>, LambdaExpr>;

template <typename T>
[[gnu::always_inline]] constexpr auto lambda_wrap(T&& t) {
    if constexpr (std::derived_from<std::decay_t<T>, LambdaExpr>) {
        return std::forward<T>(t);
    } else {
        return ConstExpr<std::decay_t<T>>{{}, std::forward<T>(t)};
    }
}

template <int id, typename T1, typename T2>
struct BinaryExpr;

template <int id, typename T>
struct UnaryExpr;

#define DEF_PLACEHOLDER(i) inline constexpr detail::Placeholder<i> _##i{};
#define DEF_BINARY(id, op)                                             \
    template <typename T1, typename T2>                                \
    struct BinaryExpr<id, T1, T2>: LambdaExpr {                        \
        T1 first;                                                      \
        T2 second;                                                     \
        template <typename... Args>                                    \
        [[gnu::always_inline]] auto operator()(Args&&... args) const { \
            return first(std::forward<Args>(args)...) op second(       \
                std::forward<Args>(args)...                            \
            );                                                         \
        }                                                              \
    };                                                                 \
    template <typename T1, typename T2>                                \
        requires LambdaExprLike<T1> || LambdaExprLike<T2>              \
    inline auto operator op(T1 first, T2 second) {                     \
        auto w1 = lambda_wrap(first);                                  \
        auto w2 = lambda_wrap(second);                                 \
        return BinaryExpr<id, decltype(w1), decltype(w2)>{             \
            {}, std::move(w1), std::move(w2)};                         \
    }
#define DEF_UNARY(id, op)                                              \
    template <typename T>                                              \
    struct UnaryExpr<id, T>: LambdaExpr {                              \
        T arg;                                                         \
        template <typename... Args>                                    \
        [[gnu::always_inline]] auto operator()(Args&&... args) const { \
            return op arg(std::forward<Args>(args)...);                \
        }                                                              \
    };                                                                 \
    template <typename T>                                              \
        requires LambdaExprLike<T>                                     \
    inline auto operator op(T arg) {                                   \
        auto w = lambda_wrap(arg);                                     \
        return UnaryExpr<id, decltype(w)>{{}, std::move(w)};           \
    }

DEF_BINARY(1, +) DEF_BINARY(2, -) DEF_BINARY(3, *)
DEF_BINARY(4, /) DEF_BINARY(5, %)
DEF_BINARY(6, ==) DEF_BINARY(7, !=) DEF_BINARY(8, <)
DEF_BINARY(9, >) DEF_BINARY(10, <=) DEF_BINARY(11, >=)
DEF_BINARY(12, &&) DEF_BINARY(13, ||) DEF_BINARY(14, <=>)
DEF_BINARY(15, &) DEF_BINARY(16, |) DEF_BINARY(17, ^)
DEF_BINARY(18, <<) DEF_BINARY(19, >>)

DEF_UNARY(1, !) DEF_UNARY(2, -) DEF_UNARY(3, +) DEF_UNARY(4, ~)

}  // namespace detail

inline namespace placeholders
{

DEF_PLACEHOLDER(1) DEF_PLACEHOLDER(2) DEF_PLACEHOLDER(3)
DEF_PLACEHOLDER(4) DEF_PLACEHOLDER(5) DEF_PLACEHOLDER(6)
DEF_PLACEHOLDER(7) DEF_PLACEHOLDER(8) DEF_PLACEHOLDER(9)

}  // namespace placeholders

}  // namespace cp

#pragma once

#include <concepts>
#include <tuple>
#include <type_traits>

namespace cp
{

namespace detail
{

template <typename F, typename S>
struct SignatureCheck: std::false_type {};

template <typename F, typename R, typename... Args>
    requires std::invocable<F, Args...> &&
    (std::same_as<R, void> ||
     std::convertible_to<std::invoke_result_t<F, Args...>, R>)
struct SignatureCheck<F, R(Args...)>: std::true_type {};

}  // namespace detail

template <typename F, typename S>
concept Fn = detail::SignatureCheck<const F&, S>::value;

template <typename F, typename S>
concept FnMut = detail::SignatureCheck<F&, S>::value;

template <typename F, typename S>
concept FnOnce = detail::SignatureCheck<F&&, S>::value;

template <typename T, typename R, typename S>
concept PairLike = (std::tuple_size_v<T> == 2) && requires(T p) {
    { get<0>(p) } -> std::convertible_to<R>;
    { get<1>(p) } -> std::convertible_to<S>;
};

template <typename T>
concept ArithmeticLike = requires(T x) {
    T{0};
    { x + x } -> std::convertible_to<T>;
    { x - x } -> std::convertible_to<T>;
    { x * x } -> std::convertible_to<T>;
    { x / x } -> std::convertible_to<T>;
    { -x } -> std::convertible_to<T>;
};

}  // namespace cp

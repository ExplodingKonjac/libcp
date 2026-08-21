#include <concepts>
#include <functional>
#include <tuple>

namespace cp
{

namespace detail
{

template <typename From, typename To>
concept returnable_as = std::same_as<To, void> || std::convertible_to<From, To>;

}  // namespace detail

template <class F, class R, class... Args>
concept Fn = std::invocable<const F&, Args...> &&
    detail::returnable_as<std::invoke_result_t<const F&, Args...>, R>;

template <class F, class R, class... Args>
concept FnMut = std::invocable<F&, Args...> &&
    detail::returnable_as<std::invoke_result_t<F&, Args...>, R>;

template <class F, class R, class... Args>
concept FnOnce = std::invocable<F&&, Args...> &&
    detail::returnable_as<std::invoke_result_t<F&&, Args...>, R>;

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

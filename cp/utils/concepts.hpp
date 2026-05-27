#include <concepts>

namespace cp
{

namespace detail
{

template <typename From, typename To>
concept ReturnableAs = std::same_as<To, void> || std::convertible_to<From, To>;

}  // namespace detail

template <class F, class R, class... Args>
concept Fn = std::invocable<const F&, Args...>
    && detail::ReturnableAs<std::invoke_result_t<const F&, Args...>, R>;

template <class F, class R, class... Args>
concept FnMut = std::invocable<F&, Args...>
    && detail::ReturnableAs<std::invoke_result_t<F&, Args...>, R>;

template <class F, class R, class... Args>
concept FnOnce = std::invocable<F&&, Args...>
    && detail::ReturnableAs<std::invoke_result_t<F&&, Args...>, R>;

}  // namespace cp

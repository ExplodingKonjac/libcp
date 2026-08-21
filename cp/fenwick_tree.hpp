#pragma once
#include <concepts>
#include <functional>
#include <memory>
#include <vector>

#include "def.hpp"
#include "utils/concepts.hpp"

namespace cp
{

namespace detail
{

template <typename T>
struct ZeroFn {
    constexpr auto operator()() const { return T{}; }
};

}  // namespace detail

template <
    typename T,
    Fn<T, T, T> PlusOp = std::plus<T>,
    Fn<T, T, T> MinusOp = std::minus<T>,
    Fn<T> ZeroFn = detail::ZeroFn<T>,
    typename Alloc = std::allocator<T>
>
    requires requires { typename std::allocator_traits<Alloc>; }
class FenwickTree {
public:
    explicit FenwickTree(
        usize n, PlusOp plus = {}, MinusOp minus = {}, ZeroFn zero = {}
    ):
        _t(n, zero()), _plus{plus}, _minus{minus}, _zero{zero} {}

    usize size() const { return _t.size(); }
    void add(usize p, T x) {
        _t[0] = _plus(_t[0], x);
        for (; p; p -= p & (-p)) _t[p] = _plus(_t[p], x);
    }
    T sum(usize l, usize r) const {
        T res{_zero()};
        if (l >= r) return res;
        if (l == 0) res = _t[0];
        else sufop(l, res, _plus);
        sufop(r, res, _minus);
        return res;
    }
    T pre_sum(usize p) const {
        if (p == 0) return _zero();
        T res{_t[0]};
        sufop(p, res, _minus);
        return res;
    }
    T suf_sum(usize p) const {
        if (p == 0) return _t[0];
        T res{_zero()};
        sufop(p, res, _plus);
        return res;
    }

private:
    void sufop(usize p, T& res, auto f) const {
        for (; p < _t.size(); p += p & (-p)) res = f(res, _t[p]);
    }

    std::vector<T, Alloc> _t{};
    [[no_unique_address]] PlusOp _plus{};
    [[no_unique_address]] MinusOp _minus{};
    [[no_unique_address]] ZeroFn _zero{};
};

}  // namespace cp

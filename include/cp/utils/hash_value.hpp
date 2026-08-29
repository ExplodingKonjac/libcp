#pragma once
#include <cassert>
#include <concepts>
#include <tuple>
#include <utility>

#include "cp/def.hpp"

namespace cp
{

template <i32... MOD>
class HashValue {
private:
    static constexpr i32 N = sizeof...(MOD);
    static constexpr i32 mods[] = {MOD...};
    i32 val_[N];

    template <std::integral T>
    static constexpr i32 normalize(T x, i32 mod) {
        auto val = x % mod;
        if constexpr (std::signed_integral<T>)
            if (val < 0) val += mod;
        return val;
    }

    static constexpr i32 mod_inv(i32 x, i32 mod) {
        i64 a = x, b = mod, s = 1, t = 0;
        while (b) {
            i64 q = a / b;
            std::swap(a -= q * b, b);
            std::swap(s -= q * t, t);
        }
        return normalize(s, mod);
    }

    static constexpr void apply_op(auto f) {
        auto helper = [&]<usize... Idx>(std::index_sequence<Idx...>) {
            (f.template operator()<Idx>(), ...);
        };
        helper(std::make_index_sequence<N>{});
    }

public:
    constexpr HashValue() = default;

    template <std::integral T>
    constexpr HashValue(T x) {
        apply_op([&]<usize I>() { val_[I] = normalize(x, mods[I]); });
    }
    template <std::integral... T>
        requires(sizeof...(T) == N && sizeof...(T) != 1)
    constexpr HashValue(T... x) {
        auto values = std::tuple<T...>(x...);
        apply_op([&]<usize I>() {
            val_[I] = normalize(std::get<I>(values), mods[I]);
        });
    }
    constexpr auto tuple() const {
        auto helper = [&]<usize... Idx>(std::index_sequence<Idx...>) {
            return std::tuple(val_[Idx]...);
        };
        return helper(std::make_index_sequence<N>{});
    }
    constexpr HashValue inv() const {
        HashValue res;
        apply_op([&]<usize I>() { res.val_[I] = mod_inv(val_[I], mods[I]); });
        return res;
    }
    constexpr HashValue& operator+=(const HashValue& rhs) {
        apply_op([&]<usize I>() {
            (val_[I] += rhs.val_[I]) >= mods[I] ? val_[I] -= mods[I] : 0;
        });
        return *this;
    }
    constexpr HashValue& operator-=(const HashValue& rhs) {
        apply_op([&]<usize I>() {
            (val_[I] -= rhs.val_[I]) < 0 ? val_[I] += mods[I] : 0;
        });
        return *this;
    }
    constexpr HashValue& operator*=(const HashValue& rhs) {
        apply_op([&]<usize I>() {
            val_[I] = i64(val_[I]) * rhs.val_[I] % mods[I];
        });
        return *this;
    }
    friend constexpr HashValue operator+(HashValue lhs, const HashValue& rhs) {
        return lhs += rhs;
    }
    friend constexpr HashValue operator-(HashValue lhs, const HashValue& rhs) {
        return lhs -= rhs;
    }
    friend constexpr HashValue operator*(HashValue lhs, const HashValue& rhs) {
        return lhs *= rhs;
    }
    constexpr auto operator<=>(const HashValue&) const = default;
};

}  // namespace cp

#pragma once
#include <concepts>
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

    static constexpr void apply_op(auto f) {
        auto helper = [&]<usize... Idx>(std::index_sequence<Idx...>) {
            (f.template operator()<Idx>(), ...);
        };
        helper(std::make_index_sequence<N>{});
    }

public:
    HashValue() = default;

    template <std::integral T>
    HashValue(T x) {
        apply_op([&]<usize I>() {
            val_[I] = x % mods[I];
            if (val_[I] < 0) val_[I] += mods[I];
        });
    }
    auto tuple() const {
        auto helper = [&]<usize... Idx>(std::index_sequence<Idx...>) {
            return std::tuple(val_[Idx]...);
        };
        return helper(std::make_index_sequence<N>{});
    }
    HashValue& operator+=(const HashValue& rhs) {
        apply_op([&]<usize I>() {
            (val_[I] += rhs.val_[I]) >= mods[I] ? val_[I] -= mods[I] : 0;
        });
        return *this;
    }
    HashValue& operator-=(const HashValue& rhs) {
        apply_op([&]<usize I>() {
            (val_[I] -= rhs.val_[I]) < 0 ? val_[I] += mods[I] : 0;
        });
        return *this;
    }
    HashValue& operator*=(const HashValue& rhs) {
        apply_op([&]<usize I>() {
            val_[I] = i64(val_[I]) * rhs.val_[I] % mods[I];
        });
        return *this;
    }
    friend HashValue operator+(HashValue lhs, const HashValue& rhs) {
        return lhs += rhs;
    }
    friend HashValue operator-(HashValue lhs, const HashValue& rhs) {
        return lhs -= rhs;
    }
    friend HashValue operator*(HashValue lhs, const HashValue& rhs) {
        return lhs *= rhs;
    }
    auto operator<=>(const HashValue&) const = default;
};

}  // namespace cp
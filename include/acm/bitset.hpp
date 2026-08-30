#pragma once

#include <algorithm>
#include <array>
#include <bit>

#include "def.hpp"

namespace acm
{

template <usize SIZE>
class Bitset {
    static constexpr usize B = 64, N = (SIZE + B - 1) / B;
    std::array<u64, N ? N : 1> a{};
    static constexpr u64 last_mask() {
        return SIZE % B ? (u64{1} << SIZE % B) - 1 : ~u64{};
    }
    void trim() {
        if constexpr (N) a[N - 1] &= last_mask();
    }

public:
    constexpr usize size() const { return SIZE; }
    constexpr usize length() const { return SIZE; }
    bool operator[](usize p) const { return a[p / B] >> (p % B) & 1; }
    void set_bit(usize p) { a[p / B] |= u64{1} << p % B; }
    void unset_bit(usize p) { a[p / B] &= ~(u64{1} << p % B); }
    void flip_bit(usize p) { a[p / B] ^= u64{1} << p % B; }
    void set_all() {
        a.fill(~u64{});
        trim();
    }
    void unset_all() { a.fill(0); }
    void flip_all() {
        for (auto& x: a) x = ~x;
        trim();
    }
    usize count() const {
        usize r = 0;
        for (u64 x: a) r += std::popcount(x);
        return r;
    }
    bool none() const { return !any(); }
    bool any() const {
        for (u64 x: a)
            if (x) return true;
        return false;
    }
    bool all() const { return count() == SIZE; }
    usize find_first_set(usize p = 0) const {
        if (p >= SIZE) return SIZE;
        usize i = p / B;
        u64 x = a[i] & (~u64{} << p % B);
        while (!x && i + 1 < N) {
            i++;
            x = a[i];
        }
        return x ? std::min(SIZE, i * B + usize(std::countr_zero(x))) : SIZE;
    }
    usize find_first_unset(usize p = 0) const {
        if (p >= SIZE) return SIZE;
        usize i = p / B;
        u64 x = ~a[i] & (~u64{} << p % B);
        while (!x && i + 1 < N) {
            i++;
            x = ~a[i];
        }
        return x ? std::min(SIZE, i * B + usize(std::countr_zero(x))) : SIZE;
    }
    void set_range(usize p, usize n) {
        if (p >= SIZE) return;
        n = std::min(n, SIZE - p);
        usize i = p / B, b = p % B;
        if (b) {
            usize k = std::min(n, B - b);
            a[i++] |= ((u64{1} << k) - 1) << b;
            n -= k;
        }
        while (n >= B) a[i++] = ~u64{}, n -= B;
        if (n) a[i] |= (u64{1} << n) - 1;
    }
    void unset_range(usize p, usize n) {
        if (p >= SIZE) return;
        n = std::min(n, SIZE - p);
        usize i = p / B, b = p % B;
        if (b) {
            usize k = std::min(n, B - b);
            a[i++] &= ~(((u64{1} << k) - 1) << b);
            n -= k;
        }
        while (n >= B) a[i++] = 0, n -= B;
        if (n) a[i] &= ~((u64{1} << n) - 1);
    }
    void flip_range(usize p, usize n) {
        if (p >= SIZE) return;
        n = std::min(n, SIZE - p);
        usize i = p / B, b = p % B;
        if (b) {
            usize k = std::min(n, B - b);
            a[i++] ^= ((u64{1} << k) - 1) << b;
            n -= k;
        }
        while (n >= B) a[i++] ^= ~u64{}, n -= B;
        if (n) a[i] ^= (u64{1} << n) - 1;
    }
    Bitset& operator&=(const Bitset& b) {
        for (usize i = 0; i < N; i++) a[i] &= b.a[i];
        return *this;
    }
    Bitset& operator|=(const Bitset& b) {
        for (usize i = 0; i < N; i++) a[i] |= b.a[i];
        return *this;
    }
    Bitset& operator^=(const Bitset& b) {
        for (usize i = 0; i < N; i++) a[i] ^= b.a[i];
        return *this;
    }
    Bitset& operator-=(const Bitset& b) {
        for (usize i = 0; i < N; i++) a[i] &= ~b.a[i];
        return *this;
    }
    friend Bitset operator~(Bitset x) {
        x.flip_all();
        return x;
    }
    friend Bitset operator&(Bitset x, const Bitset& y) { return x &= y; }
    friend Bitset operator|(Bitset x, const Bitset& y) { return x |= y; }
    friend Bitset operator^(Bitset x, const Bitset& y) { return x ^= y; }
    friend Bitset operator-(Bitset x, const Bitset& y) { return x -= y; }
    friend bool operator==(const Bitset&, const Bitset&) = default;
    friend bool operator<=(const Bitset& x, const Bitset& y) {
        return (x - y).none();
    }
    friend bool operator>=(const Bitset& x, const Bitset& y) { return y <= x; }
    friend bool operator<(const Bitset& x, const Bitset& y) {
        return x <= y && x != y;
    }
    friend bool operator>(const Bitset& x, const Bitset& y) { return y < x; }
    Bitset& operator<<=(usize s) {
        if (s >= SIZE) return unset_all(), *this;
        usize w = s / B, b = s % B;
        if (b) {
            for (usize i = N; i > w + 1; i--)
                a[i - 1] = a[i - 1 - w] << b | a[i - 2 - w] >> (B - b);
            a[w] = a[0] << b;
        } else {
            for (usize i = N; i > w; i--) a[i - 1] = a[i - 1 - w];
        }
        for (usize i = 0; i < w; i++) a[i] = 0;
        trim();
        return *this;
    }
    Bitset& operator>>=(usize s) {
        if (s >= SIZE) return unset_all(), *this;
        usize w = s / B, b = s % B;
        if (b) {
            for (usize i = 0; i + w + 1 < N; i++)
                a[i] = a[i + w] >> b | a[i + w + 1] << (B - b);
            a[N - w - 1] = a[N - 1] >> b;
        } else {
            for (usize i = 0; i + w < N; i++) a[i] = a[i + w];
        }
        for (usize i = N - w; i < N; i++) a[i] = 0;
        trim();
        return *this;
    }
    friend Bitset operator<<(Bitset x, usize s) { return x <<= s; }
    friend Bitset operator>>(Bitset x, usize s) { return x >>= s; }
};

}  // namespace acm

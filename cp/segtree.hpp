#pragma once

#include <algorithm>
#include <bit>
#include <concepts>
#include <memory>
#include <optional>
#include <stdexcept>

#include "def.hpp"

namespace cp
{

template <
    typename SemiGroup, typename Mult,
    typename Alloc = std::allocator<SemiGroup>>
    requires requires(SemiGroup x, Mult mul) {
        { mul(x, x) } -> std::same_as<SemiGroup>;
        typename std::allocator_traits<Alloc>;
    }
class SegTree {
public:
    SegTree() = default;
    SegTree(usize n, SemiGroup v = {}, Mult mul = {}, Alloc alloc = {}):
        n_{n},
        m_{std::bit_ceil(n)},
        mul_{std::move(mul)},
        alloc_{std::move(alloc)} {
        t_ = std::allocator_traits<Alloc>::allocate(alloc_, 2 * m_);
        for (usize i = 0; i < m_; i++) {
            std::allocator_traits<Alloc>::construct(alloc_, t_ + m_ + i, v);
        }
        for (usize i = m_ - 1; i > 0; i--) {
            std::allocator_traits<Alloc>::construct(
                alloc_, t_ + i, mul_(t_[i << 1], t_[i << 1 | 1])
            );
        }
    }
    void modify(usize p, SemiGroup v) {
        if (p >= n_) throw std::out_of_range("index exceed [0, n)");
        t_[m_ + p] = v;
        for (usize i = (m_ + p) >> 1; i; i >>= 1) {
            t_[i] = mul_(t_[i << 1], t_[i << 1 | 1]);
        }
    }
    SemiGroup query(usize l, usize r) const {
        if (l >= r) throw std::invalid_argument("empty query interval");
        if (r > n_) throw std::out_of_range("index exceed [0, n)");
        std::optional<SemiGroup> resl, resr;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
        for (l += m_, r += m_; l != r; l >>= 1, r >>= 1) {
            if (l & 1) resl = resl ? mul_(*resl, t_[l]) : t_[l], l++;
            if (r & 1) r--, resr = resr ? mul_(t_[r], *resr) : t_[r];
        }
#pragma GCC diagnostic pop
        return !resl ? *resr : !resr ? *resl : mul_(*resl, *resr);
    }
    SemiGroup all() const { return t_[1]; }

private:
    usize n_, m_;
    SemiGroup* t_;
    [[no_unique_address]] Mult mul_;
    [[no_unique_address]] Alloc alloc_;
};

}  // namespace cp

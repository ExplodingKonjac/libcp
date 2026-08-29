#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>

#include "def.hpp"
#include "utils/concepts.hpp"

namespace cp
{

template <
    typename SemiGroup,
    Fn<SemiGroup(SemiGroup, SemiGroup)> Mult = std::multiplies<SemiGroup>,
    typename Alloc = std::allocator<SemiGroup>
>
    requires requires { typename std::allocator_traits<Alloc>; }
class SegTree {
    using AllocTraits = std::allocator_traits<Alloc>;

public:
    SegTree() = default;

    template <FnMut<SemiGroup(usize)> F>
    SegTree(usize n, F gen, Mult mul = {}, Alloc alloc = {}):
        n_{n}, mult_{std::move(mul)}, alloc_{std::move(alloc)} {
        if (n_ > 0) {
            t_ = AllocTraits::allocate(alloc, 2 * n_);
            for (usize i = 0; i < n_; i++) {
                AllocTraits::construct(alloc_, t_ + n_ + i, gen(i));
            }
            for (usize i = n_ - 1; i > 0; i--) {
                auto tmp = mult_(t_[i << 1], t_[i << 1 | 1]);
                AllocTraits::construct(alloc_, t_ + i, std::move(tmp));
            }
        }
    }
    SegTree(usize n, SemiGroup v, Mult mul = {}, Alloc alloc = {}):
        SegTree(n, [v](auto) { return v; }, mul, alloc) {}
    ~SegTree() {
        for (usize i = 1; i < 2 * n_; i++) AllocTraits::destroy(alloc_, t_ + i);
        AllocTraits::deallocate(alloc_, t_, 2 * n_);
    }
    template <FnOnce<SemiGroup(SemiGroup)> F>
    bool update(usize p, F func) noexcept {
        if (p >= n_) return false;
        return modify(p, func(t_[n_ + p]));
    }
    bool modify(usize p, SemiGroup v) noexcept {
        if (p >= n_) return false;
        t_[n_ + p] = std::move(v);
        for (usize i = (n_ + p) >> 1; i; i >>= 1) {
            t_[i] = mult_(t_[i << 1], t_[i << 1 | 1]);
        }
        return true;
    }
    std::optional<SemiGroup> query(usize l, usize r) const noexcept {
        if (l >= r || r > n_) return std::nullopt;
        std::optional<SemiGroup> resl, resr;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
        for (l += n_, r += n_; l != r; l >>= 1, r >>= 1) {
            if (l & 1) resl = resl ? mult_(*resl, t_[l]) : t_[l], l++;
            if (r & 1) r--, resr = resr ? mult_(t_[r], *resr) : t_[r];
        }
#pragma GCC diagnostic pop
        return !resl ? resr : !resr ? resl : mult_(*resl, *resr);
    }
    SemiGroup all() const { return *query(0, n_); }

private:
    usize n_ = 0;
    SemiGroup* t_ = nullptr;
    [[no_unique_address]] Mult mult_{};
    [[no_unique_address]] Alloc alloc_{};
};

template <
    std::invocable<usize> F,
    typename SemiGroup = std::remove_cvref_t<std::invoke_result_t<F, usize>>,
    typename Mult = std::multiplies<SemiGroup>,
    typename Alloc = std::allocator<SemiGroup>
>
SegTree(usize, F, Mult = {}, Alloc = {}) -> SegTree<SemiGroup, Mult, Alloc>;

}  // namespace cp

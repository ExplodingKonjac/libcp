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
    typename Value,
    typename Operand = Value,
    typename Plus = std::plus<Value>,
    typename Mult = std::multiplies<>,
    typename Alloc = std::allocator<Value>
>
    requires FnMut<Plus, Value, Value, Value> &&
    FnMut<Mult, Value, Operand, Value> &&
    FnMut<Mult, Operand, Operand, Operand> &&
    requires { typename std::allocator_traits<Alloc>; }
class LazySegTree {
    using Tag = std::optional<Operand>;
    using ValueAlloc =
        typename std::allocator_traits<Alloc>::template rebind_alloc<Value>;
    using TagAlloc =
        typename std::allocator_traits<Alloc>::template rebind_alloc<Tag>;
    using ValueAllocTraits = std::allocator_traits<ValueAlloc>;
    using TagAllocTraits = std::allocator_traits<TagAlloc>;

public:
    LazySegTree() = default;

    template <FnMut<Value, usize> F>
    LazySegTree(
        usize n, F gen, Plus plus = {}, Mult mult = {}, Alloc alloc = {}
    ):
        n_{n},
        plus_{std::move(plus)},
        mult_{std::move(mult)},
        alloc_{std::move(alloc)},
        value_alloc_{alloc_},
        operand_alloc_{alloc_} {
        if (n_ == 0) return;
        cap_ = 4 * n_;
        t_ = ValueAllocTraits::allocate(value_alloc_, cap_);
        lz_ = TagAllocTraits::allocate(operand_alloc_, cap_);
        for (usize i = 0; i < cap_; i++) {
            TagAllocTraits::construct(operand_alloc_, lz_ + i);
        }
        build(1, 0, n_, gen);
    }
    LazySegTree(
        usize n, Value v, Plus plus = {}, Mult mult = {}, Alloc alloc = {}
    ):
        LazySegTree(n, [v](auto) { return v; }, plus, mult, alloc) {}
    ~LazySegTree() {
        if (!t_) return;
        destroy(1, 0, n_);
        for (usize i = 0; i < cap_; i++) {
            TagAllocTraits::destroy(operand_alloc_, lz_ + i);
        }
        ValueAllocTraits::deallocate(value_alloc_, t_, cap_);
        TagAllocTraits::deallocate(operand_alloc_, lz_, cap_);
    }

    template <FnOnce<Value, Value> F>
    bool update(usize p, F func) noexcept {
        if (p >= n_) return false;
        update_impl(p, std::forward<F>(func), 1, 0, n_);
        return true;
    }
    bool modify(usize p, Value v) noexcept {
        if (p >= n_) return false;
        update_impl(p, [v = std::move(v)](auto) { return v; }, 1, 0, n_);
        return true;
    }
    bool apply(usize l, usize r, Operand op) noexcept {
        if (l >= r || r > n_) return false;
        apply_impl(l, r, op, 1, 0, n_);
        return true;
    }
    std::optional<Value> query(usize l, usize r) noexcept {
        if (l >= r || r > n_) return std::nullopt;
        return query_impl(l, r, 1, 0, n_);
    }
    Value all() const { return t_[1]; }

private:
#define LC (i << 1)
#define RC (i << 1 | 1)
    template <FnMut<Value, usize> F>
    void build(usize i, usize l, usize r, F& gen) {
        if (r - l == 1) {
            ValueAllocTraits::construct(value_alloc_, t_ + i, gen(l));
            return;
        }
        usize mid = l + (r - l) / 2;
        build(LC, l, mid, gen);
        build(RC, mid, r, gen);
        ValueAllocTraits::construct(
            value_alloc_, t_ + i, plus_(t_[LC], t_[RC])
        );
    }
    void destroy(usize i, usize l, usize r) {
        if (r - l > 1) {
            usize mid = l + (r - l) / 2;
            destroy(LC, l, mid);
            destroy(RC, mid, r);
        }
        ValueAllocTraits::destroy(value_alloc_, t_ + i);
    }
    void pushup(usize i) { t_[i] = plus_(t_[LC], t_[RC]); }
    void pushtag(usize i, const Operand& op) {
        t_[i] = mult_(op, t_[i]);
        if (lz_[i]) lz_[i] = mult_(op, *lz_[i]);
        else lz_[i].emplace(op);
    }
    void pushdown(usize i) {
        if (!lz_[i]) return;
        pushtag(i << 1, *lz_[i]);
        pushtag(i << 1 | 1, *lz_[i]);
        lz_[i].reset();
    }
    template <FnOnce<Value, Value> F>
    void update_impl(usize pos, F&& func, usize i, usize l, usize r) {
        if (r - l == 1) {
            t_[i] = std::forward<F>(func)(t_[i]);
            lz_[i].reset();
            return;
        }
        pushdown(i);
        usize mid = l + (r - l) / 2;
        if (pos < mid) update_impl(pos, std::forward<F>(func), LC, l, mid);
        else update_impl(pos, std::forward<F>(func), RC, mid, r);
        pushup(i);
    }
    void apply_impl(
        usize lq, usize rq, const Operand& op, usize i, usize l, usize r
    ) {
        if (l >= lq && r <= rq) return pushtag(i, op);
        pushdown(i);
        usize mid = l + (r - l) / 2;
        if (mid > lq) apply_impl(lq, rq, op, LC, l, mid);
        if (mid < rq) apply_impl(lq, rq, op, RC, mid, r);
        pushup(i);
    }
    Value query_impl(usize lq, usize rq, usize i, usize l, usize r) {
        if (l >= lq && r <= rq) return t_[i];
        pushdown(i);
        usize mid = l + (r - l) / 2;
        if (rq <= mid) return query_impl(lq, rq, LC, l, mid);
        if (mid <= lq) return query_impl(lq, rq, RC, mid, r);
        return plus_(
            query_impl(lq, rq, LC, l, mid), query_impl(lq, rq, RC, mid, r)
        );
    }
#undef LC
#undef RC

    usize n_ = 0;
    usize cap_ = 0;
    Value* t_ = nullptr;
    Tag* lz_ = nullptr;
    [[no_unique_address]] Plus plus_;
    [[no_unique_address]] Mult mult_;
    [[no_unique_address]] Alloc alloc_;
    [[no_unique_address]] ValueAlloc value_alloc_{};
    [[no_unique_address]] TagAlloc operand_alloc_{};
};

template <
    std::invocable<usize> F,
    typename Value = std::remove_cvref_t<std::invoke_result_t<F, usize>>,
    typename Operand = Value,
    typename Plus = std::plus<Value>,
    typename Mult = std::multiplies<>,
    typename Alloc = std::allocator<Value>
>
LazySegTree(usize, F, Plus = {}, Mult = {}, Alloc = {})
    -> LazySegTree<Value, Operand, Plus, Mult, Alloc>;

}  // namespace cp

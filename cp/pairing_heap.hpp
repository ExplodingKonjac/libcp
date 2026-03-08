#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <memory>
#include <utility>

#include "def.hpp"

namespace cp
{

template <typename T, typename Compare = std::less<T>,
          typename Alloc = std::allocator<T>>
    requires requires(T x, T y, Compare cmp) {
        { cmp(x, y) } -> std::same_as<bool>;
        typename std::allocator_traits<Alloc>;
    }
class PairingHeap {
private:
    struct Node {
        T val;
        std::unique_ptr<Node> son;
        std::unique_ptr<Node> nxt;
        Node* pre;

        Node(T v): val{std::move(v)}, son{}, nxt{}, pre{} {}
        auto moveSon() {
            if (son) son->pre = nullptr;
            return std::move(son);
        }
    };

    class Iter {
        friend class PairingHeap;

    public:
        Iter() = default;
        const T& operator*() const { return node->val; }
        const T* operator->() const { return std::addressof(node->val); }

    private:
        Iter(Node* n): node{n} {}
        Node* node;
    };

    using NodeAlloc = std::allocator_traits<Alloc>::template rebind_alloc<Node>;
    using NodeAllocTrait =
        std::allocator_traits<Alloc>::template rebind_traits<Node>;

    template <typename... Args>
    auto allocNode(Args&&... val) {
        auto p = NodeAllocTrait::allocate(_alloc, sizeof(Node));
        NodeAllocTrait::construct(_alloc, p, std::forward<Args>(val)...);
        return std::unique_ptr<Node>(p);
    }
    auto merge(std::unique_ptr<Node> x, std::unique_ptr<Node> y) {
        if (!x) return y;
        if (!y) return x;
        if (_cmp(x->val, y->val)) x.swap(y);
        if (x->son) x->son->pre = y.get();
        y->pre = x.get();
        y->nxt = std::move(x->son);
        x->son = std::move(y);
        return x;
    }
    auto merges(std::unique_ptr<Node> x) {
        if (!x) return x;
        x->pre = nullptr;
        auto y = std::move(x->nxt);
        if (!y) return x;
        y->pre = nullptr;
        auto z = std::move(y->nxt);
        return merge(merge(std::move(x), std::move(y)), merges(std::move(z)));
    }
    auto extract(Node* x) {
        Node* y = x->pre;
        if (!y) return std::move(_rt);
        auto& p = (y->nxt.get() == x ? y->nxt : y->son);
        std::unique_ptr<Node> owning_x = std::move(p);
        owning_x->pre = nullptr;
        p = std::move(owning_x->nxt);
        if (p) p->pre = y;
        return owning_x;
    }

    std::unique_ptr<Node> _rt;
    usize _sz;
    Compare _cmp;
    NodeAlloc _alloc;

public:
    using point_iterator = Iter;

    PairingHeap() = default;
    PairingHeap(Alloc alloc): _rt{}, _sz{}, _cmp{}, _alloc{std::move(alloc)} {}
    PairingHeap(Compare cmp, Alloc alloc = {}):
        _rt{}, _sz{}, _cmp{std::move(cmp)}, _alloc{std::move(alloc)} {}

    template <typename... Args>
    point_iterator emplace(Args&&... args) {
        auto x = allocNode(std::forward<Args>(args)...);
        Iter it{x.get()};
        _rt = merge(std::move(_rt), std::move(x));
        _sz++;
        return it;
    }
    point_iterator push(T val) { return emplace(std::move(val)); }
    T pop() {
        auto old_rt = std::move(_rt);
        _rt = merges(old_rt->moveSon());
        _sz--;
        return std::move(old_rt)->val;
    }
    void join(PairingHeap& other) {
        _rt = merge(std::move(_rt), std::move(other._rt));
        _sz += std::exchange(other._sz, 0);
    }
    T erase(point_iterator it) {
        auto x = extract(it.node);
        _rt = merge(std::move(_rt), merges(x->moveSon()));
        _sz--;
        return std::move(x->val);
    }
    void modify(point_iterator it, T val) {
        auto x = extract(it.node);
        if (_cmp(x->val, val)) {
            x->val = val;
            _rt = merge(std::move(_rt), std::move(x));
        } else {
            x->val = val;
            auto tmp = merges(x->moveSon());
            _rt = merge(merge(std::move(_rt), std::move(tmp)), std::move(x));
        }
    }
    void clear() {
        _rt.reset(nullptr);
        _sz = 0;
    }

    usize size() const { return _sz; }
    bool empty() const { return !_sz; }
    const T& top() const { return _rt->val; }
};

}  // namespace cp

#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "def.hpp"

namespace cp
{

template <typename T, typename Compare = std::less<T>,
          typename Alloc = std::allocator<T>>
    requires requires(T x, Compare cmp) {
        { cmp(x, x) } -> std::same_as<bool>;
        typename std::allocator_traits<Alloc>;
    }
class PairingHeap {
private:
    struct Node {
        T val;
        Node* son;
        Node* nxt;
        Node* pre;

        Node(T v): val{std::move(v)}, son{}, nxt{}, pre{} {}
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
    Node* new_node(Args&&... val) const {
        auto p = NodeAllocTrait::allocate(_alloc, 1);
        NodeAllocTrait::construct(_alloc, p, std::forward<Args>(val)...);
        return p;
    }
    void del_node(Node* p) const {
        NodeAllocTrait::destroy(_alloc, p);
        NodeAllocTrait::deallocate(_alloc, p, 1);
    }
    void rdel_node(Node* p) const {
        std::vector<Node*> stk;
        if (p) stk.push_back(p);
        while (!stk.empty()) {
            auto u = stk.back();
            stk.pop_back();
            if (u->nxt) stk.push_back(u->nxt);
            if (u->son) stk.push_back(u->son);
            del_node(u);
        }
    }
    Node* clone_node(Node* src, Node* pre = nullptr) const {
        if (!src) return nullptr;
        Node* root = nullptr;
        struct Task {
            Node* src;
            Node** dst;
            Node* pre;
        };
        std::vector<Task> stk;
        stk.push_back({src, &root, pre});
        while (!stk.empty()) {
            auto [s, d, pr] = stk.back();
            stk.pop_back();
            auto p = new_node(T(s->val));
            p->pre = pr;
            *d = p;
            if (s->nxt) stk.push_back({s->nxt, &p->nxt, pr});
            if (s->son) stk.push_back({s->son, &p->son, p});
        }
        return root;
    }

    Node* merge(Node* x, Node* y) const {
        if (!x) return y;
        if (!y) return x;
        if (_cmp(x->val, y->val)) std::swap(x, y);
        if (x->son) x->son->pre = y;
        y->pre = x;
        y->nxt = x->son;
        x->son = y;
        return x;
    }
    Node* merges(Node* x) const {
        if (!x) return nullptr;
        std::vector<Node*> lst;
        while (x) {
            x->pre = nullptr;
            lst.push_back(x);
            x = std::exchange(x->nxt, nullptr);
        }
        usize cur = lst.size() & 1;
        for (usize i = cur; i < lst.size(); i += 2)
            lst[cur++] = merge(lst[i], lst[i + 1]);
        while (--cur) lst[cur - 1] = merge(lst[cur - 1], lst[cur]);
        return lst[0];
    }
    Node* extract(Node* x) {
        Node* y = x->pre;
        if (!y) return std::exchange(_rt, nullptr);
        auto& p = (y->nxt == x ? y->nxt : y->son);
        x->pre = nullptr;
        p = std::exchange(x->nxt, nullptr);
        if (p) p->pre = y;
        return x;
    }

    Node* _rt{nullptr};
    usize _sz{0};
    Compare _cmp{};
    mutable NodeAlloc _alloc{};

public:
    using point_iterator = Iter;

    PairingHeap(PairingHeap&& other) noexcept:
        _rt{std::exchange(other._rt, nullptr)},
        _sz{std::exchange(other._sz, 0)},
        _cmp(std::move(other._cmp)),
        _alloc(std::move(other._alloc)) {}
    PairingHeap(const PairingHeap& other):
        _rt{other.clone_node(other._rt)},
        _sz{other._sz},
        _cmp(other._cmp),
        _alloc(other._alloc) {}
    auto& operator=(PairingHeap&& other) noexcept {
        if (this != &other) {
            rdel_node(_rt);
            _rt = std::exchange(other._rt, nullptr);
            _sz = std::exchange(other._sz, 0);
            _cmp = std::move(other._cmp);
            _alloc = std::move(other._alloc);
        }
        return *this;
    }
    auto& operator=(const PairingHeap& other) {
        if (this != &other) {
            rdel_node(_rt);
            _rt = other.clone_node(other._rt);
            _sz = other._sz;
            _cmp = Compare(other._cmp);
            _alloc = NodeAlloc(other._alloc);
        }
        return *this;
    }
    ~PairingHeap() { rdel_node(_rt); }

    PairingHeap(Alloc alloc): _rt{}, _sz{}, _cmp{}, _alloc{std::move(alloc)} {}
    PairingHeap(Compare cmp, Alloc alloc = {}):
        _rt{}, _sz{}, _cmp{std::move(cmp)}, _alloc{std::move(alloc)} {}

    template <typename... Args>
    point_iterator emplace(Args&&... args) {
        auto x = new_node(std::forward<Args>(args)...);
        _rt = merge(_rt, x);
        _sz++;
        return point_iterator{x};
    }
    point_iterator push(T val) { return emplace(std::move(val)); }
    void join(PairingHeap& other) {
        _rt = merge(_rt, std::exchange(other._rt, nullptr));
        _sz += std::exchange(other._sz, 0);
    }
    void modify(point_iterator it, T val) {
        auto x = extract(it.node);
        if (_cmp(x->val, val)) {
            x->val = val;
        } else {
            auto tmp = merges(x->son);
            x->val = val;
            x->son = nullptr;
            x = merge(x, tmp);
        }
        _rt = merge(_rt, x);
    }
    T pop() {
        auto old_rt = _rt;
        _rt = merges(old_rt->son);
        _sz--;
        auto res = std::move(old_rt->val);
        del_node(old_rt);
        return res;
    }
    T erase(point_iterator it) {
        auto x = extract(it.node);
        _rt = merge(_rt, merges(x->son));
        _sz--;
        auto res = std::move(x->val);
        del_node(x);
        return res;
    }
    void clear() {
        rdel_node(_rt);
        _rt = nullptr;
        _sz = 0;
    }

    usize size() const { return _sz; }
    bool empty() const { return !_sz; }
    const T& top() const { return _rt->val; }
};

}  // namespace cp

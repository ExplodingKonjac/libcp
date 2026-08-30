#pragma once

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "def.hpp"

namespace acm
{

template <typename T, typename Compare = std::less<T>>
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

    template <typename... Args>
    Node* new_node(Args&&... val) const {
        return new Node(std::forward<Args>(val)...);
    }
    void del_node(Node* p) const { delete p; }
    void rdel_node(Node* p) const {
        if (!p) return;
        rdel_node(p->son);
        rdel_node(p->nxt);
        del_node(p);
    }
    Node* clone_node(Node* src, Node* pre = nullptr) const {
        if (!src) return nullptr;
        auto p = new_node(T(src->val));
        p->pre = pre;
        p->son = clone_node(src->son, p);
        p->nxt = clone_node(src->nxt, pre);
        return p;
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

public:
    using point_iterator = Iter;

    PairingHeap() = default;
    PairingHeap(PairingHeap&& other) noexcept:
        _rt{std::exchange(other._rt, nullptr)},
        _sz{std::exchange(other._sz, 0)},
        _cmp(std::move(other._cmp)) {}
    PairingHeap(const PairingHeap& other):
        _rt{other.clone_node(other._rt)}, _sz{other._sz}, _cmp(other._cmp) {}
    auto& operator=(PairingHeap&& other) noexcept {
        if (this != &other) {
            rdel_node(_rt);
            _rt = std::exchange(other._rt, nullptr);
            _sz = std::exchange(other._sz, 0);
            _cmp = std::move(other._cmp);
        }
        return *this;
    }
    auto& operator=(const PairingHeap& other) {
        if (this != &other) {
            rdel_node(_rt);
            _rt = other.clone_node(other._rt);
            _sz = other._sz;
            _cmp = Compare(other._cmp);
        }
        return *this;
    }
    ~PairingHeap() { rdel_node(_rt); }

    explicit PairingHeap(Compare cmp): _cmp{std::move(cmp)} {}

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
    void update(point_iterator it, T val) { modify(it, std::move(val)); }
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

}  // namespace acm

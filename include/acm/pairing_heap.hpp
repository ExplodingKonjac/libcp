#pragma once

#include <functional>
#include <utility>

#include "def.hpp"

namespace acm
{

template <typename T = i32, typename Compare = std::less<T>>
class PairingHeap {
    struct Node {
        T v;
        Node *s, *n, *p;

        template <typename... Args>
        explicit Node(Args&&... args):
            v(std::forward<Args>(args)...),
            s(nullptr),
            n(nullptr),
            p(nullptr) {}
    };

public:
    class point_iterator {
        friend class PairingHeap;

    public:
        point_iterator() = default;
        const T& operator*() const { return x->v; }
        const T* operator->() const { return &x->v; }

    private:
        explicit point_iterator(Node* x): x(x) {}
        Node* x = nullptr;
    };

private:
    Node* rt = nullptr;
    usize sz = 0;
    Compare cmp{};

    // 把 y 挂到 x 的兄弟链头部，调用前保证 x 的键不小于 y。
    Node* merge(Node* x, Node* y) const {
        if (!x) return y;
        if (!y) return x;
        if (cmp(x->v, y->v)) std::swap(x, y);
        if (x->s) x->s->p = y;
        y->p = x;
        y->n = x->s;
        x->s = y;
        return x;
    }
    // 两遍合并：先两两配对，再从右往左合并。
    Node* merges(Node* x) const {
        if (!x) return nullptr;
        if (!x->n) {
            x->p = nullptr;
            return x;
        }
        Node* y = x->n;
        Node* z = y->n;
        x->p = y->p = nullptr;
        x->n = y->n = nullptr;
        return merge(merge(x, y), merges(z));
    }
    // 从兄弟链摘下 x：x 是根时清空 rt，否则由父亲接替它的位置。
    Node* detach(Node* x) {
        Node* y = x->p;
        if (!y) return std::exchange(rt, nullptr);
        Node*& p = (y->n == x ? y->n : y->s);
        x->p = nullptr;
        p = std::exchange(x->n, nullptr);
        if (p) p->p = y;
        return x;
    }
    static void destroy(Node* x) {
        if (!x) return;
        destroy(x->s);
        destroy(x->n);
        delete x;
    }
    static Node* clone(Node* x, Node* p) {
        if (!x) return nullptr;
        Node* y = new Node(x->v);
        y->p = p;
        y->s = clone(x->s, y);
        y->n = clone(x->n, p);
        return y;
    }

public:
    PairingHeap() = default;
    PairingHeap(const PairingHeap& o):
        rt(clone(o.rt, nullptr)), sz(o.sz), cmp(o.cmp) {}
    PairingHeap(PairingHeap&& o) noexcept:
        rt(std::exchange(o.rt, nullptr)),
        sz(std::exchange(o.sz, 0)),
        cmp(std::move(o.cmp)) {}
    PairingHeap& operator=(const PairingHeap& o) {
        if (this != &o) {
            destroy(rt);
            rt = clone(o.rt, nullptr);
            sz = o.sz;
            cmp = o.cmp;
        }
        return *this;
    }
    PairingHeap& operator=(PairingHeap&& o) noexcept {
        if (this != &o) {
            destroy(rt);
            rt = std::exchange(o.rt, nullptr);
            sz = std::exchange(o.sz, 0);
            cmp = std::move(o.cmp);
        }
        return *this;
    }
    ~PairingHeap() { destroy(rt); }

    template <typename... Args>
    point_iterator emplace(Args&&... args) {
        Node* x = new Node(std::forward<Args>(args)...);
        rt = merge(rt, x);
        sz++;
        return point_iterator{x};
    }
    point_iterator push(T v) { return emplace(std::move(v)); }
    void join(PairingHeap& o) {
        rt = merge(rt, std::exchange(o.rt, nullptr));
        sz += std::exchange(o.sz, 0);
    }
    void modify(point_iterator it, T v) {
        Node* x = detach(it.x);
        if (cmp(x->v, v)) {
            x->v = std::move(v);
        } else {
            Node* y = merges(x->s);
            x->v = std::move(v);
            x->s = nullptr;
            x = merge(x, y);
        }
        rt = merge(rt, x);
    }
    T pop() {
        Node* x = rt;
        rt = merges(x->s);
        sz--;
        T v = std::move(x->v);
        delete x;
        return v;
    }
    T erase(point_iterator it) {
        Node* x = detach(it.x);
        rt = merge(rt, merges(x->s));
        sz--;
        T v = std::move(x->v);
        delete x;
        return v;
    }
    void clear() {
        destroy(rt);
        rt = nullptr;
        sz = 0;
    }

    const T& top() const { return rt->v; }
    usize size() const { return sz; }
    bool empty() const { return !sz; }
};

}  // namespace acm

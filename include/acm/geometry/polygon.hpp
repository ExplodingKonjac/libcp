#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "../geometry.hpp"

namespace acm
{

enum class PointPolygonRelation { outside, boundary, inside };

namespace detail
{

template <GeomScalar T>
Wide<T> area2(const std::vector<Vec2<T>>& p) {
    Wide<T> s{};
    for (usize i = 1; i + 1 < p.size(); i++) s += cross(p[0], p[i], p[i + 1]);
    return s;
}

template <GeomScalar T>
bool outside(Line2<T> l, Vec2<T> p) {
    return line_side(l, p) < 0;
}

}  // namespace detail

template <GeomScalar T>
struct Polygon {
    std::vector<Vec2<T>> p;

    Polygon() = default;
    explicit Polygon(std::vector<Vec2<T>> q): p(std::move(q)) {
        if (p.size() > 1 && p.front() == p.back()) p.pop_back();
        if (detail::area2(p) < 0) std::reverse(p.begin(), p.end());
    }

    const std::vector<Vec2<T>>& vertices() const { return p; }
    usize size() const { return p.size(); }
    const Vec2<T>& operator[](usize i) const { return p[i]; }

    std::vector<Segment2<T>> edges() const {
        std::vector<Segment2<T>> res;
        res.reserve(p.size());
        for (usize i = 0; i < p.size(); i++)
            res.push_back({p[i], p[(i + 1) % p.size()]});
        return res;
    }

    Real<T> area() const {
        using R = Real<T>;
        R s = R(detail::area2(p));
        return (s < 0 ? -s : s) / R{2};
    }

    bool is_convex() const {
        if (p.size() < 3) return true;
        for (usize i = 0; i < p.size(); i++)
            if (orientation(
                    p[i], p[(i + 1) % p.size()], p[(i + 2) % p.size()]
                ) < 0)
                return false;
        return true;
    }

    PointPolygonRelation relation(Vec2<T> q) const {
        if (p.empty()) return PointPolygonRelation::outside;
        if (p.size() == 1)
            return sgn(norm_sq(p[0] - q)) ? PointPolygonRelation::outside
                                          : PointPolygonRelation::boundary;
        bool in = false;
        for (usize i = 0; i < p.size(); i++) {
            Vec2<T> a = p[i], b = p[(i + 1) % p.size()];
            int s = orientation(a, b, q);
            if ((a.y <= q.y && q.y < b.y && s > 0) ||
                (b.y <= q.y && q.y < a.y && s < 0))
                in = !in;
            if (on_segment(a, b, q)) return PointPolygonRelation::boundary;
        }
        return in ? PointPolygonRelation::inside
                  : PointPolygonRelation::outside;
    }
};

template <GeomScalar T>
std::vector<Vec2<T>> convex_hull(std::vector<Vec2<T>> p) {
    usize n = p.size();
    if (n <= 1) return p;

    std::swap(p[0], *std::min_element(p.begin(), p.end()));
    std::sort(p.begin() + 1, p.end(), [&](Vec2<T> a, Vec2<T> b) {
        int c = orientation(p[0], a, b);
        if (c) return c > 0;
        return norm_sq(a - p[0]) < norm_sq(b - p[0]);
    });
    std::vector<Vec2<T>> h;
    h.reserve(n);
    for (auto q: p) {
        while (h.size() >= 2 && orientation(h[h.size() - 2], h.back(), q) <= 0)
            h.pop_back();
        h.push_back(q);
    }
    return h;
}

template <GeomScalar T>
Polygon<T> minkowski_sum(const Polygon<T>& a, const Polygon<T>& b) {
    assert(a.is_convex() && b.is_convex());
    if (a.size() == 0 || b.size() == 0) return {};
    if (a.size() == 1 && b.size() == 1) return Polygon<T>{{a[0] + b[0]}};

    auto get_edges = [&](const std::vector<Vec2<T>>& p) {
        usize s = std::min_element(p.begin(), p.end()) - p.begin();
        std::vector<Vec2<T>> e;
        e.reserve(p.size());
        for (usize i = s; i + 1 < p.size(); i++) e.push_back(p[i + 1] - p[i]);
        e.push_back(p[0] - p.back());
        for (usize i = 0; i < s; i++) e.push_back(p[i + 1] - p[i]);
        return std::pair{std::move(e), p[s]};
    };
    auto [e1, p0] = get_edges(a.vertices());
    auto [e2, q0] = get_edges(b.vertices());

    std::vector<Vec2<T>> res{p0 + q0};
    for (usize i = 0, j = 0; i != e1.size() || j != e2.size();) {
        int c = i == e1.size() ? 1
            : j == e2.size()   ? -1
                               : sgn(cross(e2[j], e1[i]));
        Vec2<T> d{};
        if (c <= 0) d += e1[i++];
        if (c >= 0) d += e2[j++];
        res.push_back(res.back() + d);
    }
    if (res.size() > 1) res.pop_back();
    return Polygon<T>{std::move(res)};
}

template <std::floating_point T>
std::optional<Polygon<T>> half_plane_intersection(std::vector<Line2<T>> ls) {
    if (ls.empty()) return std::nullopt;

    struct HP {
        Line2<T> l;
        T a;
    };
    std::vector<HP> h;
    h.reserve(ls.size());
    T pi = std::acos(T(-1));
    for (auto l: ls) {
        assert(l.valid());
        T a = angle(l.d);
        if (a < T{}) a += T{2} * pi;
        h.push_back({l, a});
    }
    std::sort(h.begin(), h.end(), [](const HP& x, const HP& y) {
        return x.a < y.a;
    });

    std::vector<Line2<T>> u;
    u.reserve(h.size());
    for (auto x: h) {
        auto l = x.l;
        if (!u.empty() && parallel(u.back(), l) && dot(u.back().d, l.d) > T{}) {
            if (detail::outside(l, u.back().p)) u.back() = l;
        } else {
            u.push_back(l);
        }
    }

    auto inter = [&](Line2<T> a, Line2<T> b) -> std::optional<Vec2<T>> {
        auto r = intersection(a, b);
        if (r.kind != LineIntersectionKind::point) return std::nullopt;
        return r.point;
    };

    std::deque<Line2<T>> q;
    for (auto l: u) {
        while (q.size() > 1) {
            auto x = inter(q[q.size() - 2], q.back());
            if (!x) return std::nullopt;
            if (!detail::outside(l, *x)) break;
            q.pop_back();
        }
        while (q.size() > 1) {
            auto x = inter(q[0], q[1]);
            if (!x) return std::nullopt;
            if (!detail::outside(l, *x)) break;
            q.pop_front();
        }
        q.push_back(l);
    }
    while (q.size() > 2) {
        auto x = inter(q[q.size() - 2], q.back());
        if (!x) return std::nullopt;
        if (!detail::outside(q.front(), *x)) break;
        q.pop_back();
    }
    while (q.size() > 2) {
        auto x = inter(q[0], q[1]);
        if (!x) return std::nullopt;
        if (!detail::outside(q.back(), *x)) break;
        q.pop_front();
    }
    if (q.size() < 3) return std::nullopt;

    std::vector<Vec2<T>> p;
    p.reserve(q.size());
    for (usize i = 0; i < q.size(); i++) {
        if (orientation(Vec2<T>{}, q[i].d, q[(i + 1) % q.size()].d) <= 0)
            return std::nullopt;
        auto x = inter(q[i], q[(i + 1) % q.size()]);
        if (!x) return std::nullopt;
        if (p.empty() || cmp(p.back(), *x) != 0) p.push_back(*x);
    }
    if (p.size() > 1 && !cmp(p.front(), p.back())) p.pop_back();
    if (p.size() < 3) return std::nullopt;
    return Polygon<T>(std::move(p));
}

}  // namespace acm

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "../geometry.hpp"

namespace cp
{

enum class HullMode { lower, upper, full };

enum class PointPolygonRelation { outside, boundary, inside };

namespace detail
{

template <GeometryScalar T>
geometry_wide_t<T> area2(const std::vector<Point2<T>>& p) {
    geometry_wide_t<T> s{};
    for (usize i = 1; i + 1 < p.size(); ++i) s += cross(p[0], p[i], p[i + 1]);
    return s;
}

template <bool up, GeometryScalar T>
std::vector<Point2<T>> chain(const std::vector<Point2<T>>& p) {
    std::vector<Point2<T>> res;
    res.reserve(p.size());
    const auto add = [&](Point2<T> q) {
        while (
            res.size() >= 2 &&
            orientation(res[res.size() - 2], res.back(), q) <= 0
        )
            res.pop_back();
        res.push_back(q);
    };
    if constexpr (up) {
        for (auto i = p.rbegin(); i != p.rend(); ++i) add(*i);
        std::reverse(res.begin(), res.end());
    } else {
        for (auto q: p) add(q);
    }
    return res;
}

template <std::floating_point T>
bool outside(Line2<T> l, Point2<T> p) {
    return line_side(l, p) < 0;
}

}  // namespace detail

template <GeometryScalar T>
class Polygon {
private:
    std::vector<Point2<T>> p_;

public:
    Polygon() = default;

    explicit Polygon(std::vector<Point2<T>> p): p_{std::move(p)} {
        if (p_.size() > 1 && p_.front() == p_.back()) p_.pop_back();
        if (detail::area2(p_) < 0) std::reverse(p_.begin(), p_.end());
    }

    const std::vector<Point2<T>>& vertices() const { return p_; }

    std::vector<std::pair<Point2<T>, Vec2<T>>> edges() const {
        std::vector<std::pair<Point2<T>, Vec2<T>>> res;
        res.reserve(p_.size());
        for (usize i = 0; i != p_.size(); ++i)
            res.push_back({p_[i], p_[(i + 1) % p_.size()] - p_[i]});
        return res;
    }

    usize size() const { return p_.size(); }

    const Point2<T>& operator[](usize i) const { return p_[i]; }

    geometry_real_t<T> area() const {
        using R = geometry_real_t<T>;
        return std::abs(static_cast<R>(detail::area2(p_))) / R{2};
    }

    bool is_convex() const {
        if (p_.size() < 3) return true;
        usize n = p_.size();
        for (usize i = 0; i < n; i++)
            if (orientation(p_[i], p_[(i + 1) % n], p_[(i + 2) % n]) < 0)
                return false;
        return true;
    }

    PointPolygonRelation relation(Point2<T> q) const {
        if (p_.empty()) return PointPolygonRelation::outside;
        if (p_.size() == 1)
            return sgn(norm_sq(p_[0] - q)) ? PointPolygonRelation::outside
                                           : PointPolygonRelation::boundary;
        bool in = false;
        for (usize i = 0; i < p_.size(); i++) {
            const auto a = p_[i], b = p_[(i + 1) % p_.size()];
            const int s = orientation(a, b, q);
            if ((a.y <= q.y && q.y < b.y && s > 0) ||
                (b.y <= q.y && q.y < a.y && s < 0))
                in = !in;
            if (on_segment(a, b, q)) return PointPolygonRelation::boundary;
        }
        return in ? PointPolygonRelation::inside
                  : PointPolygonRelation::outside;
    }
};

template <GeometryScalar T>
std::vector<Point2<T>> convex_hull(
    std::vector<Point2<T>> p, HullMode mode = HullMode::full
) {
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());
    if (p.size() <= 1) return p;

    if (mode == HullMode::lower) return detail::chain<false>(p);
    if (mode == HullMode::upper) return detail::chain<true>(p);

    auto lo = detail::chain<false>(p);
    auto up = detail::chain<true>(p);
    lo.reserve(lo.size() + up.size() - 2);
    for (usize i = up.size() - 1; i > 1; i--) lo.push_back(up[i - 1]);
    return lo;
}

template <GeometryScalar T>
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
        return std::pair{e, p[s]};
    };
    const auto [e1, p0] = get_edges(a.vertices());
    const auto [e2, q0] = get_edges(b.vertices());

    std::vector<Point2<T>> res{p0 + q0};
    for (usize i = 0, j = 0; i != e1.size() || j != e2.size();) {
        const auto c = i == e1.size() ? 1
            : j == e2.size()          ? -1
                                      : sgn(cross(e2[j], e1[i]));
        Vec2<T> d{};
        if (c <= 0) d += e1[i++];
        if (c >= 0) d += e2[j++];
        res.push_back(res.back() + d);
    }
    if (res.size() > 1) res.pop_back();
    return Polygon<T>{std::move(res)};
}

template <GeometryScalar T>
std::optional<Polygon<geometry_real_t<T>>> half_plane_intersection(
    std::vector<Line2<T>> ls
) {
    using R = geometry_real_t<T>;
    if (ls.empty()) return std::nullopt;

    struct HP {
        Line2<R> l;
        R a;
    };
    std::vector<HP> h;
    h.reserve(ls.size());
    const R pi = std::acos(R{-1});
    for (auto l: ls) {
        assert(l.is_valid());
        Line2<R> q{l.point.template cast<R>(), l.direction.template cast<R>()};
        R a = angle(q.direction);
        if (a < R{}) a += R{2} * pi;
        h.push_back({q, a});
    }
    std::sort(h.begin(), h.end(), [](const HP& x, const HP& y) {
        return x.a < y.a;
    });

    std::vector<Line2<R>> u;
    u.reserve(h.size());
    for (auto x: h) {
        auto l = x.l;
        if (!u.empty() &&
            parallel(u.back(), l) &&
            dot(u.back().direction, l.direction) > R{}) {
            if (detail::outside(l, u.back().point)) u.back() = l;
        } else {
            u.push_back(l);
        }
    }

    const auto inter = [&](Line2<R> a, Line2<R> b) -> std::optional<Point2<R>> {
        const auto r = intersection(a, b);
        if (r.kind != LineIntersectionKind::point) return std::nullopt;
        return r.point;
    };

    std::deque<Line2<R>> q;
    for (auto l: u) {
        while (q.size() > 1) {
            const auto p = inter(q[q.size() - 2], q.back());
            if (!p) return std::nullopt;
            if (!detail::outside(l, *p)) break;
            q.pop_back();
        }
        while (q.size() > 1) {
            const auto p = inter(q[0], q[1]);
            if (!p) return std::nullopt;
            if (!detail::outside(l, *p)) break;
            q.pop_front();
        }
        q.push_back(l);
    }
    while (q.size() > 2) {
        const auto p = inter(q[q.size() - 2], q.back());
        if (!p) return std::nullopt;
        if (!detail::outside(q.front(), *p)) break;
        q.pop_back();
    }
    while (q.size() > 2) {
        const auto p = inter(q[0], q[1]);
        if (!p) return std::nullopt;
        if (!detail::outside(q.back(), *p)) break;
        q.pop_front();
    }
    if (q.size() < 3) return std::nullopt;

    std::vector<Point2<R>> p;
    p.reserve(q.size());
    for (usize i = 0; i != q.size(); ++i) {
        if (orientation(
                Point2<R>{}, q[i].direction, q[(i + 1) % q.size()].direction
            ) <= 0)
            return std::nullopt;
        const auto x = inter(q[i], q[(i + 1) % q.size()]);
        if (!x) return std::nullopt;
        if (p.empty() || !almost_equal(p.back(), *x)) p.push_back(*x);
    }
    if (p.size() > 1 && almost_equal(p.front(), p.back())) p.pop_back();
    if (p.size() < 3) return std::nullopt;
    return Polygon{std::move(p)};
}

}  // namespace cp

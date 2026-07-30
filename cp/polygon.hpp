#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "geometry.hpp"

namespace cp
{

enum class HullMode {
    lower,
    upper,
    full,
};

enum class PointPolygonRelation {
    outside,
    boundary,
    inside,
};

namespace detail
{

template <GeometryScalar T>
geometry_wide_t<T> area2(const std::vector<Point2<T>>& p) {
    geometry_wide_t<T> s{};
    for (usize i = 0; i != p.size(); ++i)
        s = wide_add<T>(s, cross(p[i], p[(i + 1) % p.size()]));
    return s;
}

template <GeometryScalar T>
std::vector<Point2<T>> chain(
    const std::vector<Point2<T>>& p,
    bool up,
    GeometryTolerance<geometry_real_t<T>> eps
) {
    std::vector<Point2<T>> res;
    res.reserve(p.size());
    const auto add = [&](Point2<T> q) {
        while (
            res.size() >= 2 &&
            orientation(res[res.size() - 2], res.back(), q, eps) <= 0
        )
            res.pop_back();
        res.push_back(q);
    };
    if (up) {
        for (auto i = p.rbegin(); i != p.rend(); ++i) add(*i);
        std::reverse(res.begin(), res.end());
    } else {
        for (auto q: p) add(q);
    }
    return res;
}

template <GeometryScalar T>
int dir_cmp(Vec2<T> a, Vec2<T> b, GeometryTolerance<geometry_real_t<T>> eps) {
    const auto half = [](Vec2<T> v) {
        return v.y < T{} || (v.y == T{} && v.x < T{});
    };
    if (half(a) != half(b)) return half(a) ? 1 : -1;
    const int s = orientation(Point2<T>{}, a, b, eps);
    if (s != 0) return s > 0 ? -1 : 1;
    return 0;
}

template <std::floating_point T>
bool outside(Line2<T> l, Point2<T> p, GeometryTolerance<T> eps) {
    return line_side(l, p, eps) < 0;
}

template <GeometryScalar T>
std::vector<Vec2<T>> edge_dirs(
    std::vector<Point2<T>>& p, GeometryTolerance<geometry_real_t<T>> eps
) {
    const auto s =
        std::min_element(p.begin(), p.end(), [](Point2<T> a, Point2<T> b) {
            return a.y < b.y || (a.y == b.y && a.x < b.x);
        });
    std::rotate(p.begin(), s, p.end());

    std::vector<Vec2<T>> e;
    e.reserve(p.size());
    for (usize i = 0; i != p.size(); ++i) {
        auto d = p[(i + 1) % p.size()] - p[i];
        if (d == Vec2<T>{}) continue;
        if (!e.empty() && dir_cmp(e.back(), d, eps) == 0) e.back() += d;
        else e.push_back(d);
    }
    return e;
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

    bool is_convex(GeometryTolerance<geometry_real_t<T>> eps = {}) const {
        if (p_.size() < 3) return true;
        for (usize i = 0; i != p_.size(); ++i)
            if (orientation(
                    p_[i], p_[(i + 1) % p_.size()], p_[(i + 2) % p_.size()], eps
                ) < 0)
                return false;
        return true;
    }

    PointPolygonRelation relation(
        Point2<T> q, GeometryTolerance<geometry_real_t<T>> eps = {}
    ) const {
        if (p_.empty()) return PointPolygonRelation::outside;
        for (usize i = 0; i != p_.size(); ++i)
            if (on_segment(p_[i], p_[(i + 1) % p_.size()], q, eps))
                return PointPolygonRelation::boundary;
        if (p_.size() < 3) return PointPolygonRelation::outside;

        bool in = false;
        for (usize i = 0; i != p_.size(); ++i) {
            const auto a = p_[i], b = p_[(i + 1) % p_.size()];
            const int s = orientation(a, b, q, eps);
            if ((a.y <= q.y && q.y < b.y && s > 0) ||
                (b.y <= q.y && q.y < a.y && s < 0))
                in = !in;
        }
        return in ? PointPolygonRelation::inside
                  : PointPolygonRelation::outside;
    }
};

template <GeometryScalar T>
std::vector<Point2<T>> convex_hull(
    std::vector<Point2<T>> p,
    HullMode mode = HullMode::full,
    GeometryTolerance<geometry_real_t<T>> eps = {}
) {
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());
    if (p.size() <= 1) return p;

    auto lo = detail::chain(p, false, eps);
    if (mode == HullMode::lower) return lo;
    auto up = detail::chain(p, true, eps);
    if (mode == HullMode::upper) return up;

    lo.reserve(lo.size() + up.size() - 2);
    for (usize i = up.size() - 1; i-- > 1;) lo.push_back(up[i]);
    return lo;
}

template <GeometryScalar T>
Polygon<T> minkowski_sum(
    const Polygon<T>& a,
    const Polygon<T>& b,
    GeometryTolerance<geometry_real_t<T>> eps = {}
) {
    assert(a.is_convex(eps) && b.is_convex(eps));
    if (a.size() == 0 || b.size() == 0) return {};
    if (a.size() == 1 && b.size() == 1)
        return Polygon<T>{std::vector<Point2<T>>{a[0] + b[0]}};

    auto p = a.vertices(), q = b.vertices();
    const auto x = detail::edge_dirs(p, eps);
    const auto y = detail::edge_dirs(q, eps);
    usize i = 0, j = 0;
    std::vector<Point2<T>> res{p[0] + q[0]};
    while (i != x.size() || j != y.size()) {
        const int c = i == x.size() ? 1
            : j == y.size()         ? -1
                                    : detail::dir_cmp(x[i], y[j], eps);
        Vec2<T> d{};
        if (c <= 0) d += x[i++];
        if (c >= 0) d += y[j++];
        res.push_back(res.back() + d);
    }
    res.pop_back();
    return Polygon<T>{std::move(res)};
}

template <GeometryScalar T>
std::optional<Polygon<geometry_real_t<T>>> half_plane_intersection(
    std::vector<Line2<T>> ls, GeometryTolerance<geometry_real_t<T>> eps = {}
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
            parallel(u.back(), l, eps) &&
            dot(u.back().direction, l.direction) > R{}) {
            if (detail::outside(l, u.back().point, eps)) u.back() = l;
        } else {
            u.push_back(l);
        }
    }

    const auto inter = [&](Line2<R> a, Line2<R> b) -> std::optional<Point2<R>> {
        const auto r = intersection(a, b, eps);
        if (r.kind != LineIntersectionKind::point) return std::nullopt;
        return r.point;
    };

    std::deque<Line2<R>> q;
    for (auto l: u) {
        while (q.size() > 1) {
            const auto p = inter(q[q.size() - 2], q.back());
            if (!p) return std::nullopt;
            if (!detail::outside(l, *p, eps)) break;
            q.pop_back();
        }
        while (q.size() > 1) {
            const auto p = inter(q[0], q[1]);
            if (!p) return std::nullopt;
            if (!detail::outside(l, *p, eps)) break;
            q.pop_front();
        }
        q.push_back(l);
    }
    while (q.size() > 2) {
        const auto p = inter(q[q.size() - 2], q.back());
        if (!p) return std::nullopt;
        if (!detail::outside(q.front(), *p, eps)) break;
        q.pop_back();
    }
    while (q.size() > 2) {
        const auto p = inter(q[0], q[1]);
        if (!p) return std::nullopt;
        if (!detail::outside(q.back(), *p, eps)) break;
        q.pop_front();
    }
    if (q.size() < 3) return std::nullopt;

    std::vector<Point2<R>> p;
    p.reserve(q.size());
    for (usize i = 0; i != q.size(); ++i) {
        if (orientation(
                Point2<R>{}, q[i].direction, q[(i + 1) % q.size()].direction,
                eps
            ) <= 0)
            return std::nullopt;
        const auto x = inter(q[i], q[(i + 1) % q.size()]);
        if (!x) return std::nullopt;
        p.push_back(*x);
    }
    Polygon<R> res{std::move(p)};
    if (almost_equal(res.area(), R{}, eps)) return std::nullopt;
    return res;
}

}  // namespace cp

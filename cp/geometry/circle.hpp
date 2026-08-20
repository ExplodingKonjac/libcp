#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include "../geometry.hpp"

namespace cp
{

template <GeometryScalar T>
struct Circle2 {
    Point2<T> center{};
    T radius{};
    constexpr bool is_valid() const { return radius >= T{}; }
};

template <GeometryScalar T>
constexpr bool on_circle(Circle2<T> c, Point2<T> p) {
    assert(c.is_valid());
    if constexpr (std::floating_point<T>)
        return almost_equal(distance(c.center, p), c.radius);
    using W = geometry_wide_t<T>;
    return distance_sq(c.center, p) == (W)c.radius * (W)c.radius;
}

template <GeometryScalar T>
constexpr bool contains(Circle2<T> c, Point2<T> p) {
    assert(c.is_valid());
    if constexpr (std::floating_point<T>) {
        T d = distance(c.center, p);
        return d < c.radius || almost_equal(d, c.radius);
    }
    using W = geometry_wide_t<T>;
    return distance_sq(c.center, p) <= (W)c.radius * (W)c.radius;
}

template <GeometryScalar T>
geometry_real_t<T> distance_to_circle(Point2<T> p, Circle2<T> c) {
    using R = geometry_real_t<T>;
    assert(c.is_valid());
    return std::abs(distance(p, c.center) - (R)c.radius);
}

template <GeometryScalar T>
geometry_real_t<T> distance_to_disk(Point2<T> p, Circle2<T> c) {
    using R = geometry_real_t<T>;
    assert(c.is_valid());
    return std::max<R>(R{}, distance(p, c.center) - (R)c.radius);
}

enum class PointIntersectionKind { none, one, two, coincident };

template <std::floating_point T>
struct PointIntersection2 {
    PointIntersectionKind kind{PointIntersectionKind::none};
    std::array<Point2<T>, 2> points{};
    constexpr usize count() const {
        return kind == PointIntersectionKind::one ? 1
            : kind == PointIntersectionKind::two  ? 2
                                                  : 0;
    }
};

template <GeometryScalar T>
PointIntersection2<geometry_real_t<T>> intersection(Line2<T> l, Circle2<T> c) {
    using R = geometry_real_t<T>;
    assert(l.is_valid() && c.is_valid());
    auto o = c.center.template cast<R>(), p = project(c.center, l);
    auto d = l.direction.template cast<R>();
    R r = (R)c.radius, x = distance(o, p);
    int s = cmp(x, r);
    if (s > 0) return {};
    if (!s) return {PointIntersectionKind::one, {p, {}}};
    R h = std::sqrt(std::max<R>(R{}, r - x)) * std::sqrt(r + x);
    d /= std::hypot(d.x, d.y);
    return {PointIntersectionKind::two, {p - d * h, p + d * h}};
}

template <GeometryScalar T>
PointIntersection2<geometry_real_t<T>> intersection(
    Circle2<T> a, Circle2<T> b
) {
    using R = geometry_real_t<T>;
    assert(a.is_valid() && b.is_valid());
    auto p = a.center.template cast<R>(), q = b.center.template cast<R>();
    R x = (R)a.radius, y = (R)b.radius;
    auto d = q - p;
    R z = distance(p, q);
    if (!cmp(z, R{})) {
        if (cmp(x, y)) return {};
        if (!cmp(x, R{})) return {PointIntersectionKind::one, {p, {}}};
        return {PointIntersectionKind::coincident, {}};
    }
    R sum = x + y, dif = std::abs(x - y);
    if (cmp(z, sum) > 0 || cmp(z, dif) < 0) return {};
    R t = ((x - y) * ((x + y) / z) + z) / R{2};
    auto m = p + d * (t / z);
    if (!cmp(z, sum) || !cmp(z, dif))
        return {PointIntersectionKind::one, {m, {}}};
    R h =
        std::sqrt(std::max<R>(R{}, x - t)) * std::sqrt(std::max<R>(R{}, x + t));
    auto v = Vec2{-d.y, d.x} * (h / z);
    return {PointIntersectionKind::two, {m + v, m - v}};
}

namespace detail
{

template <std::floating_point T>
Circle2<T> circle_from_pair(Point2<T> a, Point2<T> b) {
    const Point2<T> center{std::midpoint(a.x, b.x), std::midpoint(a.y, b.y)};
    return {center, distance(center, a)};
}

}  // namespace detail

template <GeometryScalar T>
Circle2<geometry_real_t<T>> circle_from(Point2<T> a, Point2<T> b, Point2<T> c) {
    using R = geometry_real_t<T>;
    const auto x = a.template cast<R>();
    const auto y = b.template cast<R>();
    const auto z = c.template cast<R>();
    const auto u = y - x, v = z - x;
    const R scale =
        std::max({std::abs(u.x), std::abs(u.y), std::abs(v.x), std::abs(v.y)});
    if (scale == R{}) return detail::circle_from_pair(x, y);
    const auto su = u / scale, sv = v / scale;
    const R d = R{2} * cross(su, sv);
    if (almost_equal(d, R{})) {
        const R xy = distance(x, y);
        const R xz = distance(x, z);
        const R yz = distance(y, z);
        if (xy >= xz && xy >= yz) return detail::circle_from_pair(x, y);
        if (xz >= yz) return detail::circle_from_pair(x, z);
        return detail::circle_from_pair(y, z);
    }

    const R u2 = norm_sq(su), v2 = norm_sq(sv);
    const Point2<R> center = x +
        Vec2<R>{
            (u2 * sv.y - v2 * su.y) / d,
            (su.x * v2 - sv.x * u2) / d,
        } * scale;
    return {center, distance(center, x)};
}

template <GeometryScalar T>
std::optional<Circle2<geometry_real_t<T>>> minimum_enclosing_circle(
    std::vector<Point2<T>> points
) {
    using R = geometry_real_t<T>;
    if (points.empty()) return std::nullopt;

    std::random_device random;
    std::mt19937_64 engine(random());
    std::shuffle(points.begin(), points.end(), engine);

    std::optional<Circle2<R>> circle;
    for (usize i = 0; i != points.size(); ++i) {
        const auto a = points[i].template cast<R>();
        if (circle && contains(*circle, a)) continue;
        circle = Circle2<R>{a, R{}};
        for (usize j = 0; j != i; ++j) {
            const auto b = points[j].template cast<R>();
            if (contains(*circle, b)) continue;
            circle = detail::circle_from_pair(a, b);
            for (usize k = 0; k != j; ++k) {
                const auto c = points[k].template cast<R>();
                if (contains(*circle, c)) continue;
                circle = circle_from(a, b, c);
            }
        }
    }
    return circle;
}

}  // namespace cp

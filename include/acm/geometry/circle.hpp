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

namespace acm
{

template <GeometryScalar T>
struct Circle2 {
    Vec2<T> center{};
    T radius{};
    constexpr bool is_valid() const { return radius >= T{}; }
};

template <GeometryScalar T>
constexpr bool on_circle(Circle2<T> c, Vec2<T> p) {
    assert(c.is_valid());
    if constexpr (std::floating_point<T>)
        return almost_equal(distance(c.center, p), c.radius);
    return !cmp(distance_sq(c.center, p), c.radius * c.radius);
}

template <GeometryScalar T>
constexpr bool contains(Circle2<T> c, Vec2<T> p) {
    assert(c.is_valid());
    if constexpr (std::floating_point<T>)
        return cmp(distance(c.center, p), c.radius) <= 0;
    return cmp(distance_sq(c.center, p), c.radius * c.radius) <= 0;
}

template <GeometryScalar T>
T distance_to_circle(Vec2<T> p, Circle2<T> c) {
    assert(c.is_valid());
    return std::abs(distance(p, c.center) - c.radius);
}

template <GeometryScalar T>
T distance_to_disk(Vec2<T> p, Circle2<T> c) {
    assert(c.is_valid());
    return std::max(T{}, distance(p, c.center) - c.radius);
}

enum class PointIntersectionKind { none, one, two, coincident };

template <GeometryScalar T>
struct PointIntersection2 {
    PointIntersectionKind kind{PointIntersectionKind::none};
    std::array<Vec2<T>, 2> points{};
    constexpr usize count() const {
        return kind == PointIntersectionKind::one ? 1
            : kind == PointIntersectionKind::two  ? 2
                                                  : 0;
    }
};

template <GeometryScalar T>
PointIntersection2<T> intersection(Line2<T> l, Circle2<T> c) {
    assert(l.is_valid() && c.is_valid());
    Vec2<T> p = project(c.center, l), d = l.direction;
    T x = distance(c.center, p);
    int s = cmp(x, c.radius);
    if (s > 0) return {};
    if (!s) return {PointIntersectionKind::one, {p, {}}};
    T h = T(std::sqrt(std::max(T{}, c.radius - x)) * std::sqrt(c.radius + x));
    d /= norm(d);
    return {PointIntersectionKind::two, {p - d * h, p + d * h}};
}

template <GeometryScalar T>
PointIntersection2<T> intersection(Circle2<T> a, Circle2<T> b) {
    assert(a.is_valid() && b.is_valid());
    Vec2<T> d = b.center - a.center;
    T z = distance(a.center, b.center);
    if (!cmp(z, T{})) {
        if (cmp(a.radius, b.radius)) return {};
        if (!cmp(a.radius, T{}))
            return {PointIntersectionKind::one, {a.center, {}}};
        return {PointIntersectionKind::coincident, {}};
    }
    T sum = a.radius + b.radius, dif = std::abs(a.radius - b.radius);
    if (cmp(z, sum) > 0 || cmp(z, dif) < 0) return {};
    T t = ((a.radius - b.radius) * ((a.radius + b.radius) / z) + z) / T{2};
    Vec2<T> m = a.center + d * (t / z);
    if (!cmp(z, sum) || !cmp(z, dif))
        return {PointIntersectionKind::one, {m, {}}};
    T h =
        T(std::sqrt(std::max(T{}, a.radius - t)) *
          std::sqrt(std::max(T{}, a.radius + t)));
    Vec2<T> v{-d.y, d.x};
    v *= h / z;
    return {PointIntersectionKind::two, {m + v, m - v}};
}

namespace detail
{

template <GeometryScalar T>
Circle2<T> circle_from_pair(Vec2<T> a, Vec2<T> b) {
    Vec2<T> center{std::midpoint(a.x, b.x), std::midpoint(a.y, b.y)};
    return {center, distance(center, a)};
}

}  // namespace detail

template <GeometryScalar T>
Circle2<T> circle_from(Vec2<T> a, Vec2<T> b, Vec2<T> c) {
    Vec2<T> u = b - a, v = c - a;
    T scale =
        std::max({std::abs(u.x), std::abs(u.y), std::abs(v.x), std::abs(v.y)});
    if (!scale) return detail::circle_from_pair(a, b);
    Vec2<T> su = u / scale, sv = v / scale;
    T d = T{2} * cross(su, sv);
    if (!sgn(d)) {
        T ab = distance(a, b), ac = distance(a, c), bc = distance(b, c);
        if (ab >= ac && ab >= bc) return detail::circle_from_pair(a, b);
        if (ac >= bc) return detail::circle_from_pair(a, c);
        return detail::circle_from_pair(b, c);
    }
    T u2 = norm_sq(su), v2 = norm_sq(sv);
    Vec2<T> center = a +
        Vec2<T>{
            (u2 * sv.y - v2 * su.y) / d,
            (su.x * v2 - sv.x * u2) / d,
        } * scale;
    return {center, distance(center, a)};
}

template <GeometryScalar T>
std::optional<Circle2<T>> minimum_enclosing_circle(std::vector<Vec2<T>> p) {
    if (p.empty()) return std::nullopt;
    std::random_device random;
    std::mt19937_64 engine(random());
    std::shuffle(p.begin(), p.end(), engine);
    std::optional<Circle2<T>> circle;
    for (usize i = 0; i != p.size(); i++) {
        if (circle && contains(*circle, p[i])) continue;
        circle = Circle2<T>{p[i], T{}};
        for (usize j = 0; j != i; j++) {
            if (contains(*circle, p[j])) continue;
            circle = detail::circle_from_pair(p[i], p[j]);
            for (usize k = 0; k != j; k++) {
                if (contains(*circle, p[k])) continue;
                circle = circle_from(p[i], p[j], p[k]);
            }
        }
    }
    return circle;
}

}  // namespace acm

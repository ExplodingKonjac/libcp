#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <optional>
#include <random>
#include <vector>

#include "../geometry.hpp"

namespace acm
{

template <GeomScalar T>
struct Circle2 {
    Vec2<T> o{};
    T r{};

    constexpr bool valid() const { return r >= T{}; }
};

template <GeomScalar T>
constexpr bool on_circle(Circle2<T> c, Vec2<T> p) {
    assert(c.valid());
    if constexpr (std::floating_point<T>)
        return !cmp(distance(c.o, p), c.r);
    else {
        using W = Wide<T>;
        return distance_sq(c.o, p) == W(c.r) * W(c.r);
    }
}

template <GeomScalar T>
constexpr bool contains(Circle2<T> c, Vec2<T> p) {
    assert(c.valid());
    if constexpr (std::floating_point<T>)
        return cmp(distance(c.o, p), c.r) <= 0;
    else {
        using W = Wide<T>;
        return distance_sq(c.o, p) <= W(c.r) * W(c.r);
    }
}

template <std::floating_point T>
T distance_to_circle(Vec2<T> p, Circle2<T> c) {
    assert(c.valid());
    return std::abs(distance(p, c.o) - c.r);
}

template <std::floating_point T>
T distance_to_disk(Vec2<T> p, Circle2<T> c) {
    assert(c.valid());
    return std::max(T{}, distance(p, c.o) - c.r);
}

enum class PointIntersectionKind { none, one, two, coincident };

template <std::floating_point T>
struct PointIntersection2 {
    PointIntersectionKind kind = PointIntersectionKind::none;
    std::array<Vec2<T>, 2> points{};

    usize count() const {
        return kind == PointIntersectionKind::one ? 1
            : kind == PointIntersectionKind::two  ? 2
                                                  : 0;
    }
};

template <std::floating_point T>
PointIntersection2<T> intersection(Line2<T> l, Circle2<T> c) {
    assert(l.valid() && c.valid());
    Vec2<T> p = project(c.o, l);
    T x = distance(c.o, p);
    int s = cmp(x, c.r);
    if (s > 0) return {};
    if (!s) return {PointIntersectionKind::one, {p, {}}};
    T h = std::sqrt(std::max(T{}, c.r - x)) * std::sqrt(c.r + x);
    Vec2<T> d = l.d / norm(l.d);
    return {PointIntersectionKind::two, {p - d * h, p + d * h}};
}

template <std::floating_point T>
PointIntersection2<T> intersection(Circle2<T> a, Circle2<T> b) {
    assert(a.valid() && b.valid());
    Vec2<T> d = b.o - a.o;
    T z = distance(a.o, b.o);
    if (!cmp(z, T{})) {
        if (cmp(a.r, b.r)) return {};
        if (!cmp(a.r, T{})) return {PointIntersectionKind::one, {a.o, {}}};
        return {PointIntersectionKind::coincident, {}};
    }
    T sum = a.r + b.r, dif = std::abs(a.r - b.r);
    if (cmp(z, sum) > 0 || cmp(z, dif) < 0) return {};
    T t = ((a.r - b.r) * ((a.r + b.r) / z) + z) / T{2};
    Vec2<T> m = a.o + d * (t / z);
    if (!cmp(z, sum) || !cmp(z, dif))
        return {PointIntersectionKind::one, {m, {}}};
    T h = std::sqrt(std::max(T{}, a.r - t)) * std::sqrt(std::max(T{}, a.r + t));
    Vec2<T> v{-d.y, d.x};
    v *= h / z;
    return {PointIntersectionKind::two, {m + v, m - v}};
}

namespace detail
{

template <std::floating_point T>
Circle2<T> circle_from_pair(Vec2<T> a, Vec2<T> b) {
    Vec2<T> o{(a.x + b.x) / T{2}, (a.y + b.y) / T{2}};
    return {o, distance(o, a)};
}

}  // namespace detail

template <std::floating_point T>
Circle2<T> circle_from(Vec2<T> a, Vec2<T> b, Vec2<T> c) {
    Vec2<T> u = b - a, v = c - a;
    T scale =
        std::max({std::abs(u.x), std::abs(u.y), std::abs(v.x), std::abs(v.y)});
    if (scale == T{}) return detail::circle_from_pair(a, b);
    Vec2<T> su = u / scale, sv = v / scale;
    T d = T{2} * cross(su, sv);
    if (!cmp(d, T{})) {
        T ab = distance(a, b), ac = distance(a, c), bc = distance(b, c);
        if (ab >= ac && ab >= bc) return detail::circle_from_pair(a, b);
        if (ac >= bc) return detail::circle_from_pair(a, c);
        return detail::circle_from_pair(b, c);
    }
    T u2 = norm_sq(su), v2 = norm_sq(sv);
    Vec2<T> o = a +
        Vec2<T>{
            (u2 * sv.y - v2 * su.y) / d,
            (su.x * v2 - sv.x * u2) / d,
        } * scale;
    return {o, distance(o, a)};
}

template <std::floating_point T>
std::optional<Circle2<T>> minimum_enclosing_circle(std::vector<Vec2<T>> p) {
    if (p.empty()) return std::nullopt;
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::shuffle(p.begin(), p.end(), rng);

    std::optional<Circle2<T>> c;
    for (usize i = 0; i < p.size(); i++) {
        if (c && contains(*c, p[i])) continue;
        c = Circle2<T>{p[i], T{}};
        for (usize j = 0; j < i; j++) {
            if (contains(*c, p[j])) continue;
            c = detail::circle_from_pair(p[i], p[j]);
            for (usize k = 0; k < j; k++) {
                if (contains(*c, p[k])) continue;
                c = circle_from(p[i], p[j], p[k]);
            }
        }
    }
    return c;
}

}  // namespace acm

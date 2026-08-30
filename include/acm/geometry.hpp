#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <optional>
#include <type_traits>

#include "def.hpp"

namespace acm
{

#ifdef ACM_GEOMETRY_EPS
inline constexpr long double geometry_eps = ACM_GEOMETRY_EPS;
static_assert(geometry_eps >= 0, "ACM_GEOMETRY_EPS must be nonnegative");
#else
inline constexpr long double geometry_eps = 1e-9L;
#endif

template <typename T>
concept GeometryScalar = (std::signed_integral<T> || std::floating_point<T>) &&
    !std::same_as<std::remove_cv_t<T>, bool>;

template <std::floating_point T>
constexpr bool almost_equal(T a, T b) {
    return std::abs(a - b) <= geometry_eps;
}

template <GeometryScalar T>
constexpr int cmp(T a, T b) {
    if constexpr (std::floating_point<T>)
        if (almost_equal(a, b)) return 0;
    return (a > b) - (a < b);
}

template <GeometryScalar T>
constexpr int sgn(T x) {
    return cmp(x, T{});
}

template <GeometryScalar T>
struct Vec2 {
    T x{}, y{};

    constexpr Vec2() = default;
    constexpr Vec2(T x, T y): x{x}, y{y} {}
    template <GeometryScalar U>
    constexpr Vec2<U> cast() const {
        return {(U)x, (U)y};
    }
    constexpr Vec2 operator+() const { return *this; }
    constexpr Vec2 operator-() const { return {-x, -y}; }
    constexpr Vec2& operator+=(Vec2 a) { return x += a.x, y += a.y, *this; }
    constexpr Vec2& operator-=(Vec2 a) { return x -= a.x, y -= a.y, *this; }
    constexpr Vec2& operator*=(T k) { return x *= k, y *= k, *this; }
    constexpr Vec2& operator/=(T k) { return x /= k, y /= k, *this; }
    friend constexpr Vec2 operator+(Vec2 a, Vec2 b) { return a += b; }
    friend constexpr Vec2 operator-(Vec2 a, Vec2 b) { return a -= b; }
    friend constexpr Vec2 operator*(Vec2 a, T k) { return a *= k; }
    friend constexpr Vec2 operator*(T k, Vec2 a) { return a *= k; }
    friend constexpr Vec2 operator/(Vec2 a, T k) { return a /= k; }
    constexpr auto operator<=>(const Vec2&) const = default;
};

template <std::floating_point T>
constexpr bool almost_equal(Vec2<T> a, Vec2<T> b) {
    return almost_equal(a.x, b.x) && almost_equal(a.y, b.y);
}

template <GeometryScalar T>
constexpr T dot(Vec2<T> a, Vec2<T> b) {
    return a.x * b.x + a.y * b.y;
}

template <GeometryScalar T>
constexpr T cross(Vec2<T> a, Vec2<T> b) {
    return a.x * b.y - a.y * b.x;
}

template <GeometryScalar T>
constexpr T cross(Vec2<T> o, Vec2<T> a, Vec2<T> b) {
    return cross(a - o, b - o);
}

template <GeometryScalar T>
constexpr T norm_sq(Vec2<T> a) {
    return dot(a, a);
}

template <GeometryScalar T>
constexpr T distance_sq(Vec2<T> a, Vec2<T> b) {
    return norm_sq(a - b);
}

template <GeometryScalar T>
T norm(Vec2<T> a) {
    return T(std::hypot(a.x, a.y));
}

template <GeometryScalar T>
T distance(Vec2<T> a, Vec2<T> b) {
    return norm(a - b);
}

template <GeometryScalar T>
T angle(Vec2<T> a) {
    return T(std::atan2(a.y, a.x));
}

template <GeometryScalar T>
Vec2<T> rotate(Vec2<T> a, T r) {
    T c = T(std::cos(r)), s = T(std::sin(r));
    return {a.x * c - a.y * s, a.x * s + a.y * c};
}

template <GeometryScalar T>
std::optional<Vec2<T>> normalized(Vec2<T> a) {
    T d = norm(a);
    if (!sgn(d)) return std::nullopt;
    return a / d;
}

template <GeometryScalar T>
struct Line2 {
    Vec2<T> point{}, direction{};

    static constexpr Line2 through(Vec2<T> a, Vec2<T> b) { return {a, b - a}; }
    constexpr bool is_valid() const {
        return direction.x != T{} || direction.y != T{};
    }
};

template <GeometryScalar T>
struct Segment2 {
    Vec2<T> a{}, b{};

    constexpr Line2<T> line() const {
        assert(a != b);
        return Line2<T>::through(a, b);
    }
};

namespace detail
{

template <std::floating_point T>
constexpr T max_norm(T x, T y) {
    return std::max(std::abs(x), std::abs(y));
}

template <std::floating_point T>
constexpr int unit_det_sgn(T ax, T ay, T bx, T by) {
    T an = max_norm(ax, ay), bn = max_norm(bx, by);
    if (!an || !bn) return 0;
    return sgn(ax / an * (by / bn) - ay / an * (bx / bn));
}

}  // namespace detail

template <GeometryScalar T>
constexpr int line_side(Line2<T> l, Vec2<T> p) {
    Vec2<T> d = p - l.point;
    if constexpr (std::floating_point<T>) {
        T n = detail::max_norm(l.direction.x, l.direction.y);
        return sgn(l.direction.x / n * d.y - l.direction.y / n * d.x);
    }
    return sgn(cross(l.direction, d));
}

template <GeometryScalar T>
constexpr int orientation(Vec2<T> o, Vec2<T> a, Vec2<T> b) {
    Vec2<T> x = a - o, y = b - o;
    if constexpr (std::floating_point<T>)
        return detail::unit_det_sgn(x.x, x.y, y.x, y.y);
    return sgn(cross(x, y));
}

template <GeometryScalar T>
constexpr bool parallel(Line2<T> a, Line2<T> b) {
    assert(a.is_valid() && b.is_valid());
    return orientation(Vec2<T>{}, a.direction, b.direction) == 0;
}

template <GeometryScalar T>
constexpr bool perpendicular(Line2<T> a, Line2<T> b) {
    assert(a.is_valid() && b.is_valid());
    if constexpr (std::floating_point<T>) {
        T x = detail::max_norm(a.direction.x, a.direction.y);
        T y = detail::max_norm(b.direction.x, b.direction.y);
        return !sgn(
            a.direction.x / x * (b.direction.x / y) +
            a.direction.y / x * (b.direction.y / y)
        );
    }
    return !dot(a.direction, b.direction);
}

template <GeometryScalar T>
constexpr bool on_line(Line2<T> l, Vec2<T> p) {
    assert(l.is_valid());
    return !line_side(l, p);
}

template <GeometryScalar T>
constexpr bool on_segment(Vec2<T> a, Vec2<T> b, Vec2<T> p) {
    if (orientation(a, b, p)) return false;
    auto in = [&](T x, T l, T r) {
        if (l > r) std::swap(l, r);
        if constexpr (std::floating_point<T>)
            return l - geometry_eps <= x && x <= r + geometry_eps;
        return l <= x && x <= r;
    };
    return in(p.x, a.x, b.x) && in(p.y, a.y, b.y);
}

template <GeometryScalar T>
Vec2<T> project(Vec2<T> p, Line2<T> l) {
    assert(l.is_valid());
    return l.point +
        l.direction * (dot(p - l.point, l.direction) / norm_sq(l.direction));
}

template <GeometryScalar T>
Vec2<T> reflect(Vec2<T> p, Line2<T> l) {
    return project(p, l) * T{2} - p;
}

template <GeometryScalar T>
T distance(Vec2<T> p, Line2<T> l) {
    return distance(p, project(p, l));
}

enum class LineIntersectionKind { none, point, coincident };

template <GeometryScalar T>
struct LineIntersection2 {
    LineIntersectionKind kind{LineIntersectionKind::none};
    Vec2<T> point{};
};

template <GeometryScalar T>
LineIntersection2<T> intersection(Line2<T> a, Line2<T> b) {
    assert(a.is_valid() && b.is_valid());
    if (parallel(a, b))
        return {on_line(a, b.point) ? LineIntersectionKind::coincident
                                    : LineIntersectionKind::none,
                {}};
    T t =
        cross(b.point - a.point, b.direction) / cross(a.direction, b.direction);
    return {LineIntersectionKind::point, a.point + a.direction * t};
}

template <GeometryScalar T>
LineIntersection2<T> intersection(Segment2<T> a, Segment2<T> b) {
    const bool ap = a.a == a.b, bp = b.a == b.b;
    if (ap && bp)
        return on_segment(a.a, a.b, b.a)
            ? LineIntersection2<T>{LineIntersectionKind::point, a.a}
            : LineIntersection2<T>{};
    if (ap)
        return on_segment(b.a, b.b, a.a)
            ? LineIntersection2<T>{LineIntersectionKind::point, a.a}
            : LineIntersection2<T>{};
    if (bp)
        return on_segment(a.a, a.b, b.a)
            ? LineIntersection2<T>{LineIntersectionKind::point, b.a}
            : LineIntersection2<T>{};
    auto r = intersection(a.line(), b.line());
    if (r.kind == LineIntersectionKind::point)
        return orientation(a.a, a.b, b.a) * orientation(a.a, a.b, b.b) <= 0 &&
                orientation(b.a, b.b, a.a) * orientation(b.a, b.b, a.b) <= 0
            ? r
            : LineIntersection2<T>{};
    if (r.kind == LineIntersectionKind::none) return r;
    T dx = a.b.x - a.a.x, dy = a.b.y - a.a.y;
    bool use_x = std::abs(dx) >= std::abs(dy);
    auto coord = [use_x](Vec2<T> p) { return use_x ? p.x : p.y; };
    T lo = std::max(
        std::min(coord(a.a), coord(a.b)), std::min(coord(b.a), coord(b.b))
    );
    T hi = std::min(
        std::max(coord(a.a), coord(a.b)), std::max(coord(b.a), coord(b.b))
    );
    if (cmp(hi, lo) < 0) return {};
    if (!cmp(hi, lo))
        for (auto p: {a.a, a.b, b.a, b.b})
            if (!cmp(coord(p), lo)) return {LineIntersectionKind::point, p};
    return {LineIntersectionKind::coincident, {}};
}

template <typename Lhs, typename Rhs, typename... Args>
bool intersects(Lhs a, Rhs b, Args... args) {
    return intersection(a, b, args...).kind != LineIntersectionKind::none;
}

}  // namespace acm

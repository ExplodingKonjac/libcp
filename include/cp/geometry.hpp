#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <limits>
#include <optional>
#include <type_traits>

#include "def.hpp"

namespace cp
{

#ifdef CP_GEOMETRY_EPS
inline constexpr long double geometry_eps = CP_GEOMETRY_EPS;
static_assert(geometry_eps >= 0, "CP_GEOMETRY_EPS must be nonnegative");
#else
inline constexpr long double geometry_eps = 1e-9L;
#endif

template <typename T>
concept GeometryScalar = (std::signed_integral<T> || std::floating_point<T>) &&
    !std::same_as<std::remove_cv_t<T>, bool>;
template <GeometryScalar T>
using geometry_wide_t = std::conditional_t<
    std::floating_point<T>,
    T,
    std::conditional_t<(sizeof(T) <= 2), i64, i128>
>;

template <GeometryScalar T>
using geometry_real_t =
    std::conditional_t<std::floating_point<T>, T, long double>;

template <std::floating_point T>
inline constexpr bool almost_equal(T a, T b) {
    return std::abs(a - b) <= geometry_eps;
}

template <GeometryScalar T>
inline constexpr int cmp(T a, T b) {
    if constexpr (std::floating_point<T>)
        if (almost_equal(a, b)) return 0;
    return (a > b) - (a < b);
}

template <GeometryScalar T>
inline constexpr int sgn(T x) {
    return cmp(x, T{});
}
inline constexpr int sgn(i128 x) { return x > 0 ? 1 : x < 0 ? -1 : 0; }

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
    constexpr Vec2& operator/=(T k)
        requires std::floating_point<T>
    {
        return x /= k, y /= k, *this;
    }
    friend constexpr Vec2 operator+(Vec2 a, Vec2 b) { return a += b; }
    friend constexpr Vec2 operator-(Vec2 a, Vec2 b) { return a -= b; }
    friend constexpr Vec2 operator*(Vec2 a, T k) { return a *= k; }
    friend constexpr Vec2 operator*(T k, Vec2 a) { return a *= k; }
    friend constexpr Vec2 operator/(Vec2 a, T k)
        requires std::floating_point<T>
    {
        return a /= k;
    }
    constexpr auto operator<=>(const Vec2&) const = default;
};

template <GeometryScalar T>
using Point2 = Vec2<T>;

template <std::floating_point T>
inline constexpr bool almost_equal(Vec2<T> a, Vec2<T> b) {
    return almost_equal(a.x, b.x) && almost_equal(a.y, b.y);
}

template <GeometryScalar T>
inline constexpr geometry_wide_t<T> dot(Vec2<T> a, Vec2<T> b) {
    using W = geometry_wide_t<T>;
    return (W)a.x * (W)b.x + (W)a.y * (W)b.y;
}

template <GeometryScalar T>
inline constexpr geometry_wide_t<T> cross(Vec2<T> a, Vec2<T> b) {
    using W = geometry_wide_t<T>;
    return (W)a.x * (W)b.y - (W)a.y * (W)b.x;
}

template <GeometryScalar T>
inline constexpr geometry_wide_t<T> cross(
    Point2<T> o, Point2<T> a, Point2<T> b
) {
    using W = geometry_wide_t<T>;
    W ax = (W)a.x - (W)o.x, ay = (W)a.y - (W)o.y;
    W bx = (W)b.x - (W)o.x, by = (W)b.y - (W)o.y;
    return ax * by - ay * bx;
}

template <GeometryScalar T>
inline constexpr geometry_wide_t<T> norm_sq(Vec2<T> a) {
    return dot(a, a);
}

template <GeometryScalar T>
inline constexpr geometry_wide_t<T> distance_sq(Point2<T> a, Point2<T> b) {
    using W = geometry_wide_t<T>;
    W x = (W)a.x - (W)b.x, y = (W)a.y - (W)b.y;
    return x * x + y * y;
}

template <GeometryScalar T>
inline geometry_real_t<T> norm(Vec2<T> a) {
    using R = geometry_real_t<T>;
    return std::hypot((R)a.x, (R)a.y);
}

template <GeometryScalar T>
inline geometry_real_t<T> distance(Point2<T> a, Point2<T> b) {
    using R = geometry_real_t<T>;
    return std::hypot((R)a.x - (R)b.x, (R)a.y - (R)b.y);
}

template <GeometryScalar T>
inline geometry_real_t<T> angle(Vec2<T> a) {
    using R = geometry_real_t<T>;
    return std::atan2((R)a.y, (R)a.x);
}

template <GeometryScalar T>
inline Vec2<geometry_real_t<T>> rotate(Vec2<T> a, geometry_real_t<T> r) {
    using R = geometry_real_t<T>;
    R c = std::cos(r), s = std::sin(r), x = (R)a.x, y = (R)a.y;
    return {x * c - y * s, x * s + y * c};
}

template <GeometryScalar T>
inline std::optional<Vec2<geometry_real_t<T>>> normalized(Vec2<T> a) {
    using R = geometry_real_t<T>;
    R d = norm(a);
    if (almost_equal(d, R{})) return std::nullopt;
    return a.template cast<R>() / d;
}

template <GeometryScalar T>
struct Line2 {
    Point2<T> point{};
    Vec2<T> direction{};

    static constexpr Line2 through(Point2<T> a, Point2<T> b) {
        if constexpr (std::floating_point<T>) return {a, b - a};
        using W = geometry_wide_t<T>;
        W x = (W)b.x - (W)a.x, y = (W)b.y - (W)a.y;
        assert(x >= (W)std::numeric_limits<T>::min());
        assert(x <= (W)std::numeric_limits<T>::max());
        assert(y >= (W)std::numeric_limits<T>::min());
        assert(y <= (W)std::numeric_limits<T>::max());
        return {a, {(T)x, (T)y}};
    }
    constexpr bool is_valid() const {
        return direction.x != T{} || direction.y != T{};
    }
};

template <GeometryScalar T>
struct Segment2 {
    Point2<T> a{}, b{};

    constexpr Line2<T> line() const {
        assert(a != b);
        return Line2<T>::through(a, b);
    }
};

namespace detail
{

// Scale a direction so that its larger component has magnitude one;
// a zero vector stays zero.
template <std::floating_point T>
inline constexpr Vec2<T> max_norm_scale(Vec2<T> v) {
    T n = std::max(std::abs(v.x), std::abs(v.y));
    return n == T{} ? Vec2<T>{} : Vec2<T>{v.x / n, v.y / n};
}

template <GeometryScalar T>
inline constexpr int det_sgn(
    geometry_wide_t<T> ax,
    geometry_wide_t<T> ay,
    geometry_wide_t<T> bx,
    geometry_wide_t<T> by
) {
    using W = geometry_wide_t<T>;
    W x = ax * by, y = ay * bx;
    if constexpr (std::floating_point<T>) return cmp(x, y);
    return (x > y) - (x < y);
}

}  // namespace detail

template <GeometryScalar T>
inline constexpr int line_side(Line2<T> l, Point2<T> p) {
    if constexpr (std::floating_point<T>)
        return sgn(cross(detail::max_norm_scale(l.direction), p - l.point));
    else {
        using W = geometry_wide_t<T>;
        W x = (W)p.x - (W)l.point.x, y = (W)p.y - (W)l.point.y;
        return detail::det_sgn<T>((W)l.direction.x, (W)l.direction.y, x, y);
    }
}

template <GeometryScalar T>
inline constexpr int orientation(Point2<T> o, Point2<T> a, Point2<T> b) {
    if constexpr (std::floating_point<T>)
        return sgn(
            cross(detail::max_norm_scale(a - o), detail::max_norm_scale(b - o))
        );
    else {
        using W = geometry_wide_t<T>;
        W ax = (W)a.x - (W)o.x, ay = (W)a.y - (W)o.y;
        W bx = (W)b.x - (W)o.x, by = (W)b.y - (W)o.y;
        return detail::det_sgn<T>(ax, ay, bx, by);
    }
}

template <GeometryScalar T>
inline constexpr bool parallel(Line2<T> a, Line2<T> b) {
    assert(a.is_valid() && b.is_valid());
    return orientation(Point2<T>{}, a.direction, b.direction) == 0;
}

template <GeometryScalar T>
inline constexpr bool perpendicular(Line2<T> a, Line2<T> b) {
    assert(a.is_valid() && b.is_valid());
    if constexpr (std::floating_point<T>)
        return !sgn(
            dot(detail::max_norm_scale(a.direction),
                detail::max_norm_scale(b.direction))
        );
    else {
        using W = geometry_wide_t<T>;
        return (W)a.direction.x * (W)b.direction.x +
            (W)a.direction.y * (W)b.direction.y ==
            0;
    }
}

template <GeometryScalar T>
inline constexpr bool on_line(Line2<T> l, Point2<T> p) {
    assert(l.is_valid());
    return line_side(l, p) == 0;
}

template <GeometryScalar T>
inline constexpr bool on_segment(Point2<T> a, Point2<T> b, Point2<T> p) {
    if (orientation(a, b, p)) return false;
    auto in = [&](T x, T l, T r) {
        if (l > r) std::swap(l, r);
        if constexpr (std::floating_point<T>) {
            return l - geometry_eps <= x && x <= r + geometry_eps;
        }
        return l <= x && x <= r;
    };
    return in(p.x, a.x, b.x) && in(p.y, a.y, b.y);
}

template <GeometryScalar T>
inline Point2<geometry_real_t<T>> project(Point2<T> p, Line2<T> l) {
    using R = geometry_real_t<T>;
    assert(l.is_valid());
    auto a = l.point.template cast<R>(), d = l.direction.template cast<R>();
    d /= std::hypot(d.x, d.y);
    return a + d * dot(p.template cast<R>() - a, d);
}

template <GeometryScalar T>
inline Point2<geometry_real_t<T>> reflect(Point2<T> p, Line2<T> l) {
    using R = geometry_real_t<T>;
    return project(p, l) * R{2} - p.template cast<R>();
}

template <GeometryScalar T>
inline geometry_real_t<T> distance(Point2<T> p, Line2<T> l) {
    using R = geometry_real_t<T>;
    return distance(p.template cast<R>(), project(p, l));
}

enum class LineIntersectionKind { none, point, coincident };

template <std::floating_point T>
struct LineIntersection2 {
    LineIntersectionKind kind{LineIntersectionKind::none};
    Point2<T> point{};
};

template <GeometryScalar T>
inline LineIntersection2<geometry_real_t<T>> intersection(
    Line2<T> a, Line2<T> b
) {
    using R = geometry_real_t<T>;
    assert(a.is_valid() && b.is_valid());
    if (parallel(a, b))
        return {on_line(a, b.point) ? LineIntersectionKind::coincident
                                    : LineIntersectionKind::none,
                {}};
    // Both directions are max-norm scaled, so the denominator stays O(1) for
    // any direction magnitude; the parameter is measured in units of u, which
    // cancels the scaling factor of the direction exactly.
    const auto p = a.point.template cast<R>();
    const auto q = b.point.template cast<R>();
    const auto u = detail::max_norm_scale(a.direction.template cast<R>());
    const auto v = detail::max_norm_scale(b.direction.template cast<R>());
    return {LineIntersectionKind::point,
            p + u * (cross(q - p, v) / cross(u, v))};
}

template <GeometryScalar T>
inline LineIntersection2<geometry_real_t<T>> intersection(
    Segment2<T> a, Segment2<T> b
) {
    using R = geometry_real_t<T>;
    if (a.a == a.b)
        return on_segment(b.a, b.b, a.a)
            ? LineIntersection2<R>{LineIntersectionKind::point,
                                   a.a.template cast<R>()}
            : LineIntersection2<R>{};
    if (b.a == b.b)
        return on_segment(a.a, a.b, b.a)
            ? LineIntersection2<R>{LineIntersectionKind::point,
                                   b.a.template cast<R>()}
            : LineIntersection2<R>{};

    const auto x = a.a.template cast<R>(), y = a.b.template cast<R>();
    const auto u = b.a.template cast<R>(), v = b.b.template cast<R>();
    const auto r =
        intersection(Line2<R>::through(x, y), Line2<R>::through(u, v));
    if (r.kind == LineIntersectionKind::none) return r;
    int s1 = orientation(a.a, a.b, b.a), s2 = orientation(a.a, a.b, b.b);
    int s3 = orientation(b.a, b.b, a.a), s4 = orientation(b.a, b.b, a.b);
    if (r.kind == LineIntersectionKind::point)
        return s1 * s2 <= 0 && s3 * s4 <= 0 ? r : LineIntersection2<R>{};

    // Coincident lines: both segments share a direction, so their overlap is
    // a one-dimensional interval on the dominant axis of a.
    const R dx = y.x - x.x, dy = y.y - x.y;
    const bool use_x = std::abs(dx) >= std::abs(dy);
    const auto coord = [use_x](Point2<R> p) { return use_x ? p.x : p.y; };
    const R a1 = coord(x), a2 = coord(y), b1 = coord(u), b2 = coord(v);
    const R lo = std::max(std::min(a1, a2), std::min(b1, b2));
    const R hi = std::min(std::max(a1, a2), std::max(b1, b2));
    if (cmp(hi, lo) < 0) return {};
    if (cmp(hi, lo) > 0) return {LineIntersectionKind::coincident, {}};
    for (const auto p: {x, y, u, v})
        if (cmp(coord(p), lo) == 0) return {LineIntersectionKind::point, p};
    return {LineIntersectionKind::coincident, {}};
}

template <typename Lhs, typename Rhs, typename... Args>
inline bool intersects(Lhs a, Rhs b, Args... args) {
    return intersection(a, b, args...).kind !=
        decltype(intersection(a, b, args...)){}.kind;
}

}  // namespace cp

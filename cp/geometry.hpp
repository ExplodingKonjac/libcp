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
constexpr int sgn(i128 x) { return x > 0 ? 1 : x < 0 ? -1 : 0; }

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
constexpr bool almost_equal(Vec2<T> a, Vec2<T> b) {
    return almost_equal(a.x, b.x) && almost_equal(a.y, b.y);
}

template <GeometryScalar T>
constexpr geometry_wide_t<T> dot(Vec2<T> a, Vec2<T> b) {
    using W = geometry_wide_t<T>;
    return (W)a.x * (W)b.x + (W)a.y * (W)b.y;
}

template <GeometryScalar T>
constexpr geometry_wide_t<T> cross(Vec2<T> a, Vec2<T> b) {
    using W = geometry_wide_t<T>;
    return (W)a.x * (W)b.y - (W)a.y * (W)b.x;
}

template <GeometryScalar T>
constexpr geometry_wide_t<T> cross(Point2<T> o, Point2<T> a, Point2<T> b) {
    using W = geometry_wide_t<T>;
    W ax = (W)a.x - (W)o.x, ay = (W)a.y - (W)o.y;
    W bx = (W)b.x - (W)o.x, by = (W)b.y - (W)o.y;
    return ax * by - ay * bx;
}

template <GeometryScalar T>
constexpr geometry_wide_t<T> norm_sq(Vec2<T> a) {
    return dot(a, a);
}

template <GeometryScalar T>
constexpr geometry_wide_t<T> distance_sq(Point2<T> a, Point2<T> b) {
    using W = geometry_wide_t<T>;
    W x = (W)a.x - (W)b.x, y = (W)a.y - (W)b.y;
    return x * x + y * y;
}

template <GeometryScalar T>
geometry_real_t<T> norm(Vec2<T> a) {
    return std::sqrt((geometry_real_t<T>)norm_sq(a));
}

template <GeometryScalar T>
geometry_real_t<T> distance(Point2<T> a, Point2<T> b) {
    return std::sqrt((geometry_real_t<T>)distance_sq(a, b));
}

template <GeometryScalar T>
geometry_real_t<T> angle(Vec2<T> a) {
    using R = geometry_real_t<T>;
    return std::atan2((R)a.y, (R)a.x);
}

template <GeometryScalar T>
Vec2<geometry_real_t<T>> rotate(Vec2<T> a, geometry_real_t<T> r) {
    using R = geometry_real_t<T>;
    R c = std::cos(r), s = std::sin(r), x = (R)a.x, y = (R)a.y;
    return {x * c - y * s, x * s + y * c};
}

template <GeometryScalar T>
std::optional<Vec2<geometry_real_t<T>>> normalized(Vec2<T> a) {
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

namespace detail
{

template <GeometryScalar T>
constexpr int det_sgn(
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
constexpr int line_side(Line2<T> l, Point2<T> p) {
    using W = geometry_wide_t<T>;
    W x = (W)p.x - (W)l.point.x, y = (W)p.y - (W)l.point.y;
    W a = (W)l.direction.x, b = (W)l.direction.y;
    return detail::det_sgn<T>(a, b, x, y);
}

template <GeometryScalar T>
constexpr int orientation(Point2<T> o, Point2<T> a, Point2<T> b) {
    using W = geometry_wide_t<T>;
    W ax = (W)a.x - (W)o.x, ay = (W)a.y - (W)o.y;
    W bx = (W)b.x - (W)o.x, by = (W)b.y - (W)o.y;
    return detail::det_sgn<T>(ax, ay, bx, by);
}

template <GeometryScalar T>
constexpr bool parallel(Line2<T> a, Line2<T> b) {
    assert(a.is_valid() && b.is_valid());
    return orientation(Point2<T>{}, a.direction, b.direction) == 0;
}

template <GeometryScalar T>
constexpr bool perpendicular(Line2<T> a, Line2<T> b) {
    using W = geometry_wide_t<T>;
    assert(a.is_valid() && b.is_valid());
    W x = (W)a.direction.x * (W)b.direction.x;
    W y = (W)a.direction.y * (W)b.direction.y;
    if constexpr (std::floating_point<T>) return cmp(x, -y) == 0;
    return x + y == 0;
}

template <GeometryScalar T>
constexpr bool on_line(Line2<T> l, Point2<T> p) {
    assert(l.is_valid());
    return line_side(l, p) == 0;
}

template <GeometryScalar T>
constexpr bool on_segment(Point2<T> a, Point2<T> b, Point2<T> p) {
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
Point2<geometry_real_t<T>> project(Point2<T> p, Line2<T> l) {
    using R = geometry_real_t<T>;
    assert(l.is_valid());
    auto a = l.point.template cast<R>(), d = l.direction.template cast<R>();
    return a + d * (dot(p.template cast<R>() - a, d) / dot(d, d));
}

template <GeometryScalar T>
Point2<geometry_real_t<T>> reflect(Point2<T> p, Line2<T> l) {
    using R = geometry_real_t<T>;
    return project(p, l) * R{2} - p.template cast<R>();
}

template <GeometryScalar T>
geometry_real_t<T> distance(Point2<T> p, Line2<T> l) {
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
LineIntersection2<geometry_real_t<T>> intersection(Line2<T> a, Line2<T> b) {
    using R = geometry_real_t<T>;
    assert(a.is_valid() && b.is_valid());
    if (parallel(a, b))
        return {on_line(a, b.point) ? LineIntersectionKind::coincident
                                    : LineIntersectionKind::none,
                {}};
    auto p = a.point.template cast<R>(), d = a.direction.template cast<R>();
    auto q = b.point.template cast<R>(), e = b.direction.template cast<R>();
    R t = cross(q - p, e) / cross(d, e);
    return {LineIntersectionKind::point, p + d * t};
}

template <typename Lhs, typename Rhs, typename... Args>
bool intersects(Lhs a, Rhs b, Args... args) {
    return intersection(a, b, args...).kind !=
        decltype(intersection(a, b, args...)){}.kind;
}

}  // namespace cp

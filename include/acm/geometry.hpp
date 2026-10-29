#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

#include "def.hpp"

namespace acm
{

#ifdef ACM_EPS
inline constexpr long double eps = ACM_EPS;
#else
inline constexpr long double eps = 1e-9L;
#endif

template <typename T>
concept GeomScalar = (std::signed_integral<T> || std::floating_point<T> ||
                      std::same_as<std::remove_cv_t<T>, i128>) &&
    !std::same_as<std::remove_cv_t<T>, bool>;

template <GeomScalar T>
using Wide = std::conditional_t<
    std::is_integral_v<T>, std::conditional_t<(sizeof(T) <= 4), i64, i128>, T
>;

template <GeomScalar T>
using Real = std::conditional_t<std::is_integral_v<T>, long double, T>;

template <std::floating_point T>
constexpr bool almost_equal(T a, T b) {
    return std::abs(a - b) <= T(eps);
}

template <GeomScalar T>
constexpr int cmp(T a, T b) {
    if constexpr (std::floating_point<T>)
        if (almost_equal(a, b)) return 0;
    return (a > b) - (a < b);
}

template <GeomScalar T>
constexpr int sgn(T x) {
    return cmp(x, T{});
}

template <GeomScalar T>
struct Vec2 {
    T x{}, y{};

    constexpr Vec2() = default;
    constexpr Vec2(T x, T y): x(x), y(y) {}

    template <GeomScalar U>
    constexpr Vec2<U> cast() const {
        return Vec2<U>{(U)x, (U)y};
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

    friend constexpr bool operator==(Vec2 a, Vec2 b) {
        return a.x == b.x && a.y == b.y;
    }
    friend constexpr bool operator!=(Vec2 a, Vec2 b) { return !(a == b); }
    friend constexpr bool operator<(Vec2 a, Vec2 b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    }
};

template <GeomScalar T>
using Point2 = Vec2<T>;

template <std::floating_point T>
constexpr bool almost_equal(Vec2<T> a, Vec2<T> b) {
    return almost_equal(a.x, b.x) && almost_equal(a.y, b.y);
}

template <GeomScalar T>
constexpr Wide<T> dot(Vec2<T> a, Vec2<T> b) {
    using W = Wide<T>;
    return W(a.x) * W(b.x) + W(a.y) * W(b.y);
}

template <GeomScalar T>
constexpr Wide<T> cross(Vec2<T> a, Vec2<T> b) {
    using W = Wide<T>;
    return W(a.x) * W(b.y) - W(a.y) * W(b.x);
}

template <GeomScalar T>
constexpr Wide<T> cross(Vec2<T> o, Vec2<T> a, Vec2<T> b) {
    using W = Wide<T>;
    return (W(a.x) - W(o.x)) * (W(b.y) - W(o.y)) -
        (W(a.y) - W(o.y)) * (W(b.x) - W(o.x));
}

template <GeomScalar T>
constexpr Wide<T> norm_sq(Vec2<T> a) {
    return dot(a, a);
}

template <GeomScalar T>
constexpr Wide<T> distance_sq(Vec2<T> a, Vec2<T> b) {
    return norm_sq(a - b);
}

template <std::floating_point T>
T norm(Vec2<T> a) {
    return std::hypot(a.x, a.y);
}

template <std::floating_point T>
T distance(Vec2<T> a, Vec2<T> b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

template <std::floating_point T>
T angle(Vec2<T> a) {
    return std::atan2(a.y, a.x);
}

template <std::floating_point T>
Vec2<T> rotate(Vec2<T> a, T r) {
    T c = std::cos(r), s = std::sin(r);
    return {a.x * c - a.y * s, a.x * s + a.y * c};
}

template <std::floating_point T>
std::optional<Vec2<T>> normalized(Vec2<T> a) {
    T d = norm(a);
    if (almost_equal(d, T{})) return std::nullopt;
    return a / d;
}

template <GeomScalar T>
struct Line2 {
    Vec2<T> p{}, d{};

    static constexpr Line2 through(Vec2<T> a, Vec2<T> b) {
        return {a, b - a};
    }
    constexpr bool valid() const { return d.x != T{} || d.y != T{}; }
};

template <GeomScalar T>
struct Segment2 {
    Vec2<T> a{}, b{};

    constexpr Line2<T> line() const {
        assert(a != b);
        return Line2<T>::through(a, b);
    }
};

template <GeomScalar T>
constexpr int line_side(Line2<T> l, Vec2<T> q) {
    if constexpr (std::floating_point<T>) {
        T n = std::max(std::abs(l.d.x), std::abs(l.d.y));
        if (n == T{}) return 0;
        return sgn(
            l.d.x / n * (q.y - l.p.y) - l.d.y / n * (q.x - l.p.x)
        );
    } else {
        using W = Wide<T>;
        W x = W(q.x) - W(l.p.x), y = W(q.y) - W(l.p.y);
        return sgn(W(l.d.x) * y - W(l.d.y) * x);
    }
}

template <GeomScalar T>
constexpr int orientation(Vec2<T> o, Vec2<T> a, Vec2<T> b) {
    if constexpr (std::floating_point<T>) {
        T ax = a.x - o.x, ay = a.y - o.y;
        T bx = b.x - o.x, by = b.y - o.y;
        T an = std::max(std::abs(ax), std::abs(ay));
        T bn = std::max(std::abs(bx), std::abs(by));
        if (an == T{} || bn == T{}) return 0;
        return sgn(ax / an * (by / bn) - ay / an * (bx / bn));
    } else {
        return sgn(cross(o, a, b));
    }
}

template <GeomScalar T>
constexpr bool parallel(Line2<T> a, Line2<T> b) {
    assert(a.valid() && b.valid());
    return orientation(Vec2<T>{}, a.d, b.d) == 0;
}

template <GeomScalar T>
constexpr bool perpendicular(Line2<T> a, Line2<T> b) {
    assert(a.valid() && b.valid());
    if constexpr (std::floating_point<T>) {
        T x = std::max(std::abs(a.d.x), std::abs(a.d.y));
        T y = std::max(std::abs(b.d.x), std::abs(b.d.y));
        return sgn(a.d.x / x * (b.d.x / y) + a.d.y / x * (b.d.y / y)) == 0;
    } else {
        return dot(a.d, b.d) == 0;
    }
}

template <GeomScalar T>
constexpr bool on_line(Line2<T> l, Vec2<T> p) {
    assert(l.valid());
    return line_side(l, p) == 0;
}

template <GeomScalar T>
constexpr bool on_segment(Vec2<T> a, Vec2<T> b, Vec2<T> p) {
    if (orientation(a, b, p)) return false;
    auto in = [&](T x, T l, T r) {
        if (l > r) std::swap(l, r);
        if constexpr (std::floating_point<T>)
            return l - T(eps) <= x && x <= r + T(eps);
        return l <= x && x <= r;
    };
    return in(p.x, a.x, b.x) && in(p.y, a.y, b.y);
}

template <std::floating_point T>
Vec2<T> project(Vec2<T> p, Line2<T> l) {
    assert(l.valid());
    return l.p + l.d * (dot(p - l.p, l.d) / dot(l.d, l.d));
}

template <std::floating_point T>
Vec2<T> reflect(Vec2<T> p, Line2<T> l) {
    return project(p, l) * T{2} - p;
}

template <std::floating_point T>
T distance(Vec2<T> p, Line2<T> l) {
    return distance(p, project(p, l));
}

enum class LineIntersectionKind { none, point, coincident };

template <std::floating_point T>
struct LineIntersection2 {
    LineIntersectionKind kind = LineIntersectionKind::none;
    Vec2<T> point{};
};

template <std::floating_point T>
LineIntersection2<T> intersection(Line2<T> a, Line2<T> b) {
    assert(a.valid() && b.valid());
    if (parallel(a, b))
        return {
            on_line(a, b.p) ? LineIntersectionKind::coincident
                            : LineIntersectionKind::none,
            {},
        };
    T t = cross(b.p - a.p, b.d) / cross(a.d, b.d);
    return {LineIntersectionKind::point, a.p + a.d * t};
}

template <std::floating_point T>
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
    if (cmp(hi, lo) == 0)
        for (auto p: {a.a, a.b, b.a, b.b})
            if (cmp(coord(p), lo) == 0)
                return {LineIntersectionKind::point, p};
    return {LineIntersectionKind::coincident, {}};
}

template <std::floating_point T>
bool intersects(Line2<T> a, Line2<T> b) {
    return intersection(a, b).kind != LineIntersectionKind::none;
}

template <std::floating_point T>
bool intersects(Segment2<T> a, Segment2<T> b) {
    return intersection(a, b).kind != LineIntersectionKind::none;
}

}  // namespace acm

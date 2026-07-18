#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <limits>
#include <optional>
#include <type_traits>

#include "def.hpp"

namespace cp
{

template <typename T>
concept GeometryScalar = (std::signed_integral<T> || std::floating_point<T>)
    && !std::same_as<std::remove_cv_t<T>, bool>;

template <GeometryScalar T>
using geometry_wide_t = std::conditional_t<
    std::floating_point<T>,
    T,
    std::conditional_t<(sizeof(T) <= 2), i64, i128>
>;

template <GeometryScalar T>
using geometry_real_t =
    std::conditional_t<std::floating_point<T>, T, long double>;

namespace detail
{

template <std::floating_point T>
consteval T default_geometry_tolerance() {
    if constexpr (std::same_as<T, float>) return static_cast<T>(1e-5L);
    if constexpr (std::same_as<T, double>) return static_cast<T>(1e-9L);
    return static_cast<T>(1e-12L);
}

template <GeometryScalar T>
constexpr geometry_wide_t<T> wide_add(
    geometry_wide_t<T> lhs, geometry_wide_t<T> rhs
) {
    using W = geometry_wide_t<T>;
    if constexpr (std::floating_point<T> || sizeof(T) < sizeof(i64)) {
        return lhs + rhs;
    } else {
        W result{};
        if (!__builtin_add_overflow(lhs, rhs, &result)) return result;
        return rhs < 0 ? std::numeric_limits<W>::min()
                       : std::numeric_limits<W>::max();
    }
}

template <GeometryScalar T>
constexpr geometry_wide_t<T> wide_multiply(
    geometry_wide_t<T> lhs, geometry_wide_t<T> rhs
) {
    using W = geometry_wide_t<T>;
    if constexpr (std::floating_point<T> || sizeof(T) < sizeof(i64)) {
        return lhs * rhs;
    } else {
        W result{};
        if (!__builtin_mul_overflow(lhs, rhs, &result)) return result;
        return (lhs < 0) == (rhs < 0) ? std::numeric_limits<W>::max()
                                      : std::numeric_limits<W>::min();
    }
}

struct SignedMagnitude {
    bool negative{};
    u128 magnitude{};
};

constexpr u128 unsigned_magnitude(i128 value) {
    return value < 0 ? static_cast<u128>(-(value + 1)) + 1
                     : static_cast<u128>(value);
}

constexpr SignedMagnitude signed_product(i128 lhs, i128 rhs) {
    return {(lhs < 0) != (rhs < 0),
            unsigned_magnitude(lhs) * unsigned_magnitude(rhs)};
}

constexpr i128 signed_from_magnitude(SignedMagnitude value) {
    const u128 positive_limit =
        static_cast<u128>(std::numeric_limits<i128>::max());
    const u128 negative_limit = positive_limit + 1;
    const u128 limit = value.negative ? negative_limit : positive_limit;
    if (value.magnitude > limit)
        return value.negative ? std::numeric_limits<i128>::min()
                              : std::numeric_limits<i128>::max();
    if (!value.negative) return static_cast<i128>(value.magnitude);
    if (value.magnitude == negative_limit)
        return std::numeric_limits<i128>::min();
    return -static_cast<i128>(value.magnitude);
}

constexpr i128 subtract_products(SignedMagnitude lhs, SignedMagnitude rhs) {
    if (lhs.negative == rhs.negative) {
        if (lhs.magnitude >= rhs.magnitude)
            return signed_from_magnitude(
                {lhs.negative, lhs.magnitude - rhs.magnitude}
            );
        return signed_from_magnitude(
            {!lhs.negative, rhs.magnitude - lhs.magnitude}
        );
    }

    u128 magnitude{};
    if (__builtin_add_overflow(lhs.magnitude, rhs.magnitude, &magnitude))
        return lhs.negative ? std::numeric_limits<i128>::min()
                            : std::numeric_limits<i128>::max();
    return signed_from_magnitude({lhs.negative, magnitude});
}

template <GeometryScalar T>
constexpr geometry_wide_t<T> wide_determinant(
    geometry_wide_t<T> ax,
    geometry_wide_t<T> ay,
    geometry_wide_t<T> bx,
    geometry_wide_t<T> by
) {
    if constexpr (std::floating_point<T> || sizeof(T) < sizeof(i64)) {
        return ax * by - ay * bx;
    } else {
        return subtract_products(
            signed_product(ax, by), signed_product(ay, bx)
        );
    }
}

}  // namespace detail

template <std::floating_point T>
struct GeometryTolerance {
    T absolute{detail::default_geometry_tolerance<T>()};
    T relative{detail::default_geometry_tolerance<T>()};
};

template <std::floating_point T>
constexpr bool almost_equal(T lhs, T rhs, GeometryTolerance<T> tolerance = {}) {
    const T scale = std::max<T>({T{1}, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs)
        <= tolerance.absolute
        + tolerance.relative
        * scale;
}

template <GeometryScalar T>
struct Vec2 {
    T x{}, y{};

    constexpr Vec2() = default;
    constexpr Vec2(T x, T y): x{x}, y{y} {}

    template <GeometryScalar U>
    constexpr Vec2<U> cast() const {
        return {static_cast<U>(x), static_cast<U>(y)};
    }

    constexpr Vec2 operator+() const { return *this; }
    constexpr Vec2 operator-() const { return {-x, -y}; }

    constexpr Vec2& operator+=(Vec2 rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }
    constexpr Vec2& operator-=(Vec2 rhs) {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }
    constexpr Vec2& operator*=(T scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }
    constexpr Vec2& operator/=(T scalar)
        requires std::floating_point<T>
    {
        x /= scalar;
        y /= scalar;
        return *this;
    }

    friend constexpr Vec2 operator+(Vec2 lhs, Vec2 rhs) { return lhs += rhs; }
    friend constexpr Vec2 operator-(Vec2 lhs, Vec2 rhs) { return lhs -= rhs; }
    friend constexpr Vec2 operator*(Vec2 value, T scalar) {
        return value *= scalar;
    }
    friend constexpr Vec2 operator*(T scalar, Vec2 value) {
        return value *= scalar;
    }
    friend constexpr Vec2 operator/(Vec2 value, T scalar)
        requires std::floating_point<T>
    {
        return value /= scalar;
    }

    constexpr auto operator<=>(const Vec2&) const = default;
};

template <GeometryScalar T>
using Point2 = Vec2<T>;

template <std::floating_point T>
constexpr bool almost_equal(
    Vec2<T> lhs, Vec2<T> rhs, GeometryTolerance<T> tolerance = {}
) {
    return almost_equal(lhs.x, rhs.x, tolerance)
        && almost_equal(lhs.y, rhs.y, tolerance);
}

template <GeometryScalar T>
constexpr geometry_wide_t<T> dot(Vec2<T> lhs, Vec2<T> rhs) {
    using W = geometry_wide_t<T>;
    return detail::wide_add<T>(
        detail::wide_multiply<T>(static_cast<W>(lhs.x), static_cast<W>(rhs.x)),
        detail::wide_multiply<T>(static_cast<W>(lhs.y), static_cast<W>(rhs.y))
    );
}

template <GeometryScalar T>
constexpr geometry_wide_t<T> cross(Vec2<T> lhs, Vec2<T> rhs) {
    using W = geometry_wide_t<T>;
    return detail::wide_determinant<T>(
        static_cast<W>(lhs.x), static_cast<W>(lhs.y), static_cast<W>(rhs.x),
        static_cast<W>(rhs.y)
    );
}

template <GeometryScalar T>
constexpr geometry_wide_t<T> cross(
    Point2<T> origin, Point2<T> lhs, Point2<T> rhs
) {
    using W = geometry_wide_t<T>;
    const W lhs_x = static_cast<W>(lhs.x) - static_cast<W>(origin.x);
    const W lhs_y = static_cast<W>(lhs.y) - static_cast<W>(origin.y);
    const W rhs_x = static_cast<W>(rhs.x) - static_cast<W>(origin.x);
    const W rhs_y = static_cast<W>(rhs.y) - static_cast<W>(origin.y);
    return detail::wide_determinant<T>(lhs_x, lhs_y, rhs_x, rhs_y);
}

template <GeometryScalar T>
constexpr geometry_wide_t<T> norm_sq(Vec2<T> value) {
    return dot(value, value);
}

template <GeometryScalar T>
constexpr geometry_wide_t<T> distance_sq(Point2<T> lhs, Point2<T> rhs) {
    using W = geometry_wide_t<T>;
    const W dx = static_cast<W>(lhs.x) - static_cast<W>(rhs.x);
    const W dy = static_cast<W>(lhs.y) - static_cast<W>(rhs.y);
    return detail::wide_add<T>(
        detail::wide_multiply<T>(dx, dx), detail::wide_multiply<T>(dy, dy)
    );
}

template <GeometryScalar T>
geometry_real_t<T> norm(Vec2<T> value) {
    using R = geometry_real_t<T>;
    return std::sqrt(static_cast<R>(norm_sq(value)));
}

template <GeometryScalar T>
geometry_real_t<T> distance(Point2<T> lhs, Point2<T> rhs) {
    using R = geometry_real_t<T>;
    return std::sqrt(static_cast<R>(distance_sq(lhs, rhs)));
}

template <GeometryScalar T>
constexpr Vec2<T> perp_ccw(Vec2<T> value) {
    return {-value.y, value.x};
}

template <GeometryScalar T>
constexpr Vec2<T> perp_cw(Vec2<T> value) {
    return {value.y, -value.x};
}

template <GeometryScalar T>
geometry_real_t<T> angle(Vec2<T> value) {
    using R = geometry_real_t<T>;
    return std::atan2(static_cast<R>(value.y), static_cast<R>(value.x));
}

template <GeometryScalar T>
Vec2<geometry_real_t<T>> rotate(Vec2<T> value, geometry_real_t<T> radians) {
    using R = geometry_real_t<T>;
    const R cosine = std::cos(radians);
    const R sine = std::sin(radians);
    const R x = static_cast<R>(value.x);
    const R y = static_cast<R>(value.y);
    return {x * cosine - y * sine, x * sine + y * cosine};
}

template <GeometryScalar T>
std::optional<Vec2<geometry_real_t<T>>> normalized(
    Vec2<T> value, GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    using R = geometry_real_t<T>;
    const R length = norm(value);
    if (almost_equal(length, R{}, tolerance)) return std::nullopt;
    return value.template cast<R>() / length;
}

template <GeometryScalar T>
struct Line2 {
    Point2<T> point{};
    Vec2<T> direction{};

    static constexpr Line2 through(Point2<T> first, Point2<T> second) {
        if constexpr (std::floating_point<T>) {
            return {first, second - first};
        } else {
            using W = geometry_wide_t<T>;
            const W dx = static_cast<W>(second.x) - static_cast<W>(first.x);
            const W dy = static_cast<W>(second.y) - static_cast<W>(first.y);
            assert(dx >= static_cast<W>(std::numeric_limits<T>::min()));
            assert(dx <= static_cast<W>(std::numeric_limits<T>::max()));
            assert(dy >= static_cast<W>(std::numeric_limits<T>::min()));
            assert(dy <= static_cast<W>(std::numeric_limits<T>::max()));
            return {first, {static_cast<T>(dx), static_cast<T>(dy)}};
        }
    }
    constexpr bool is_valid() const {
        return direction.x != T{} || direction.y != T{};
    }
};

template <GeometryScalar T>
struct Segment2 {
    Point2<T> first{};
    Point2<T> second{};

    constexpr bool is_degenerate() const { return first == second; }
};

template <GeometryScalar T>
struct Circle2 {
    Point2<T> center{};
    T radius{};

    constexpr bool is_valid() const { return radius >= T{}; }
};

namespace detail
{

template <GeometryScalar T>
using tolerance_t = GeometryTolerance<geometry_real_t<T>>;

template <GeometryScalar T>
constexpr geometry_wide_t<T> cross_scale(
    geometry_wide_t<T> ax,
    geometry_wide_t<T> ay,
    geometry_wide_t<T> bx,
    geometry_wide_t<T> by
) {
    using W = geometry_wide_t<T>;
    if constexpr (std::floating_point<T>) {
        return std::max<W>({W{1}, std::abs(ax * by), std::abs(ay * bx)});
    } else {
        return W{1};
    }
}

template <GeometryScalar T>
constexpr geometry_wide_t<T> dot_scale(
    geometry_wide_t<T> ax,
    geometry_wide_t<T> ay,
    geometry_wide_t<T> bx,
    geometry_wide_t<T> by
) {
    using W = geometry_wide_t<T>;
    if constexpr (std::floating_point<T>) {
        return std::max<W>({W{1}, std::abs(ax * bx), std::abs(ay * by)});
    } else {
        return W{1};
    }
}

template <GeometryScalar T>
constexpr int classify(
    geometry_wide_t<T> value, geometry_wide_t<T> scale, tolerance_t<T> tolerance
) {
    if constexpr (std::floating_point<T>) {
        const auto limit = tolerance.absolute + tolerance.relative * scale;
        if (std::abs(value) <= limit) return 0;
    } else if (value == 0) {
        return 0;
    }
    return value < 0 ? -1 : 1;
}

template <GeometryScalar T>
constexpr int line_side(
    Line2<T> line, Point2<T> point, tolerance_t<T> tolerance
) {
    using W = geometry_wide_t<T>;
    const W dx = static_cast<W>(point.x) - static_cast<W>(line.point.x);
    const W dy = static_cast<W>(point.y) - static_cast<W>(line.point.y);
    const W lx = static_cast<W>(line.direction.x);
    const W ly = static_cast<W>(line.direction.y);
    return classify<T>(
        wide_determinant<T>(lx, ly, dx, dy), cross_scale<T>(lx, ly, dx, dy),
        tolerance
    );
}

template <GeometryScalar T>
constexpr bool between(T value, T lhs, T rhs, tolerance_t<T> tolerance) {
    if (lhs > rhs) std::swap(lhs, rhs);
    if constexpr (std::floating_point<T>) {
        const T scale =
            std::max<T>({T{1}, std::abs(value), std::abs(lhs), std::abs(rhs)});
        const T margin = tolerance.absolute + tolerance.relative * scale;
        return value >= lhs - margin && value <= rhs + margin;
    } else {
        return value >= lhs && value <= rhs;
    }
}

template <std::floating_point R>
constexpr bool point_less(Point2<R> lhs, Point2<R> rhs) {
    return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
}

}  // namespace detail

template <GeometryScalar T>
constexpr int orientation(
    Point2<T> origin,
    Point2<T> lhs,
    Point2<T> rhs,
    GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    using W = geometry_wide_t<T>;
    const W lhs_x = static_cast<W>(lhs.x) - static_cast<W>(origin.x);
    const W lhs_y = static_cast<W>(lhs.y) - static_cast<W>(origin.y);
    const W rhs_x = static_cast<W>(rhs.x) - static_cast<W>(origin.x);
    const W rhs_y = static_cast<W>(rhs.y) - static_cast<W>(origin.y);
    return detail::classify<T>(
        detail::wide_determinant<T>(lhs_x, lhs_y, rhs_x, rhs_y),
        detail::cross_scale<T>(lhs_x, lhs_y, rhs_x, rhs_y), tolerance
    );
}

template <GeometryScalar T>
constexpr bool parallel(
    Line2<T> lhs,
    Line2<T> rhs,
    GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    using W = geometry_wide_t<T>;
    assert(lhs.is_valid() && rhs.is_valid());
    const W lx = static_cast<W>(lhs.direction.x);
    const W ly = static_cast<W>(lhs.direction.y);
    const W rx = static_cast<W>(rhs.direction.x);
    const W ry = static_cast<W>(rhs.direction.y);
    return detail::classify<T>(
               detail::wide_determinant<T>(lx, ly, rx, ry),
               detail::cross_scale<T>(lx, ly, rx, ry), tolerance
           )
        == 0;
}

template <GeometryScalar T>
constexpr bool perpendicular(
    Line2<T> lhs,
    Line2<T> rhs,
    GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    using W = geometry_wide_t<T>;
    assert(lhs.is_valid() && rhs.is_valid());
    const W lx = static_cast<W>(lhs.direction.x);
    const W ly = static_cast<W>(lhs.direction.y);
    const W rx = static_cast<W>(rhs.direction.x);
    const W ry = static_cast<W>(rhs.direction.y);
    return detail::classify<T>(
               detail::wide_add<T>(
                   detail::wide_multiply<T>(lx, rx),
                   detail::wide_multiply<T>(ly, ry)
               ),
               detail::dot_scale<T>(lx, ly, rx, ry), tolerance
           )
        == 0;
}

template <GeometryScalar T>
constexpr bool on_line(
    Line2<T> line,
    Point2<T> point,
    GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    assert(line.is_valid());
    return detail::line_side(line, point, tolerance) == 0;
}

template <GeometryScalar T>
constexpr bool on_segment(
    Segment2<T> segment,
    Point2<T> point,
    GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    return orientation(segment.first, segment.second, point, tolerance)
        == 0
        && detail::between(
               point.x, segment.first.x, segment.second.x, tolerance
        )
        && detail::between(
               point.y, segment.first.y, segment.second.y, tolerance
        );
}

template <GeometryScalar T>
constexpr bool on_circle(
    Circle2<T> circle,
    Point2<T> point,
    GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    assert(circle.is_valid());
    if constexpr (std::floating_point<T>) {
        return almost_equal(
            distance(circle.center, point), circle.radius, tolerance
        );
    } else {
        using W = geometry_wide_t<T>;
        const W radius = static_cast<W>(circle.radius);
        return distance_sq(circle.center, point)
            == detail::wide_multiply<T>(radius, radius);
    }
}

template <GeometryScalar T>
constexpr bool contains(
    Circle2<T> circle,
    Point2<T> point,
    GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    assert(circle.is_valid());
    if constexpr (std::floating_point<T>) {
        const T center_distance = distance(circle.center, point);
        return center_distance
            < circle.radius
            || almost_equal(center_distance, circle.radius, tolerance);
    } else {
        using W = geometry_wide_t<T>;
        const W radius = static_cast<W>(circle.radius);
        return distance_sq(circle.center, point)
            <= detail::wide_multiply<T>(radius, radius);
    }
}

template <GeometryScalar T>
Point2<geometry_real_t<T>> project(Point2<T> point, Line2<T> line) {
    using R = geometry_real_t<T>;
    assert(line.is_valid());
    const auto origin = line.point.template cast<R>();
    const auto direction = line.direction.template cast<R>();
    const auto delta = point.template cast<R>() - origin;
    return origin
        + direction
        * (dot(delta, direction) / dot(direction, direction));
}

template <GeometryScalar T>
Point2<geometry_real_t<T>> reflect(Point2<T> point, Line2<T> line) {
    using R = geometry_real_t<T>;
    const auto projected = project(point, line);
    return projected * R{2} - point.template cast<R>();
}

template <GeometryScalar T>
Point2<geometry_real_t<T>> closest_point(Point2<T> point, Segment2<T> segment) {
    using R = geometry_real_t<T>;
    const auto first = segment.first.template cast<R>();
    const auto second = segment.second.template cast<R>();
    const auto direction = second - first;
    const R length_squared = dot(direction, direction);
    if (length_squared == R{}) return first;
    const R parameter = std::clamp(
        dot(point.template cast<R>() - first, direction) / length_squared, R{},
        R{1}
    );
    return first + direction * parameter;
}

template <GeometryScalar T>
geometry_real_t<T> distance(Point2<T> point, Line2<T> line) {
    return distance(
        point.template cast<geometry_real_t<T>>(), project(point, line)
    );
}

template <GeometryScalar T>
geometry_real_t<T> distance(Point2<T> point, Segment2<T> segment) {
    return distance(
        point.template cast<geometry_real_t<T>>(), closest_point(point, segment)
    );
}

template <GeometryScalar T>
geometry_real_t<T> distance_to_circle(Point2<T> point, Circle2<T> circle) {
    using R = geometry_real_t<T>;
    assert(circle.is_valid());
    return std::abs(
        distance(point, circle.center) - static_cast<R>(circle.radius)
    );
}

template <GeometryScalar T>
geometry_real_t<T> distance_to_disk(Point2<T> point, Circle2<T> circle) {
    using R = geometry_real_t<T>;
    assert(circle.is_valid());
    return std::max<R>(
        R{}, distance(point, circle.center) - static_cast<R>(circle.radius)
    );
}

enum class LineIntersectionKind {
    none,
    point,
    coincident,
};

template <std::floating_point T>
struct LineIntersection2 {
    LineIntersectionKind kind{LineIntersectionKind::none};
    Point2<T> point{};
};

enum class SegmentIntersectionKind {
    none,
    point,
    overlap,
};

template <std::floating_point T>
struct SegmentIntersection2 {
    SegmentIntersectionKind kind{SegmentIntersectionKind::none};
    Point2<T> first{};
    Point2<T> second{};
};

enum class PointIntersectionKind {
    none,
    one,
    two,
    coincident,
};

template <std::floating_point T>
struct PointIntersection2 {
    PointIntersectionKind kind{PointIntersectionKind::none};
    std::array<Point2<T>, 2> points{};

    constexpr usize count() const {
        if (kind == PointIntersectionKind::one) return 1;
        if (kind == PointIntersectionKind::two) return 2;
        return 0;
    }
};

template <GeometryScalar T>
LineIntersection2<geometry_real_t<T>> intersection(
    Line2<T> lhs,
    Line2<T> rhs,
    GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    using R = geometry_real_t<T>;
    assert(lhs.is_valid() && rhs.is_valid());
    if (parallel(lhs, rhs, tolerance)) {
        return {on_line(lhs, rhs.point, tolerance)
                    ? LineIntersectionKind::coincident
                    : LineIntersectionKind::none,
                {}};
    }

    const auto lhs_point = lhs.point.template cast<R>();
    const auto lhs_direction = lhs.direction.template cast<R>();
    const auto rhs_point = rhs.point.template cast<R>();
    const auto rhs_direction = rhs.direction.template cast<R>();
    const R parameter = cross(rhs_point - lhs_point, rhs_direction)
        / cross(lhs_direction, rhs_direction);
    return {LineIntersectionKind::point, lhs_point + lhs_direction * parameter};
}

template <GeometryScalar T>
SegmentIntersection2<geometry_real_t<T>> intersection(
    Line2<T> line,
    Segment2<T> segment,
    GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    using R = geometry_real_t<T>;
    assert(line.is_valid());
    if (segment.is_degenerate()) {
        if (!on_line(line, segment.first, tolerance)) return {};
        const auto point = segment.first.template cast<R>();
        return {SegmentIntersectionKind::point, point, point};
    }

    const int first_side = detail::line_side(line, segment.first, tolerance);
    const int second_side = detail::line_side(line, segment.second, tolerance);
    if (first_side == 0 && second_side == 0) {
        auto first = segment.first.template cast<R>();
        auto second = segment.second.template cast<R>();
        if (detail::point_less(second, first)) std::swap(first, second);
        return {SegmentIntersectionKind::overlap, first, second};
    }
    if (first_side == 0) {
        const auto point = segment.first.template cast<R>();
        return {SegmentIntersectionKind::point, point, point};
    }
    if (second_side == 0) {
        const auto point = segment.second.template cast<R>();
        return {SegmentIntersectionKind::point, point, point};
    }
    if (first_side == second_side) return {};

    const auto line_point = line.point.template cast<R>();
    const auto line_direction = line.direction.template cast<R>();
    const auto segment_first = segment.first.template cast<R>();
    const auto segment_direction =
        segment.second.template cast<R>() - segment_first;
    const R parameter = cross(segment_first - line_point, segment_direction)
        / cross(line_direction, segment_direction);
    const auto point = line_point + line_direction * parameter;
    return {SegmentIntersectionKind::point, point, point};
}

template <GeometryScalar T>
SegmentIntersection2<geometry_real_t<T>> intersection(
    Segment2<T> lhs,
    Segment2<T> rhs,
    GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    using R = geometry_real_t<T>;
    if (lhs.is_degenerate()) {
        if (!on_segment(rhs, lhs.first, tolerance)) return {};
        const auto point = lhs.first.template cast<R>();
        return {SegmentIntersectionKind::point, point, point};
    }
    if (rhs.is_degenerate()) {
        if (!on_segment(lhs, rhs.first, tolerance)) return {};
        const auto point = rhs.first.template cast<R>();
        return {SegmentIntersectionKind::point, point, point};
    }

    const int lhs_first =
        orientation(lhs.first, lhs.second, rhs.first, tolerance);
    const int lhs_second =
        orientation(lhs.first, lhs.second, rhs.second, tolerance);
    const int rhs_first =
        orientation(rhs.first, rhs.second, lhs.first, tolerance);
    const int rhs_second =
        orientation(rhs.first, rhs.second, lhs.second, tolerance);

    std::array<Point2<R>, 4> candidates{};
    usize count = 0;
    const auto add_candidate = [&](Point2<T> point) {
        if (!on_segment(lhs, point, tolerance)
            || !on_segment(rhs, point, tolerance))
            return;
        const auto real_point = point.template cast<R>();
        for (usize i = 0; i != count; ++i)
            if (almost_equal(candidates[i], real_point, tolerance)) return;
        candidates[count++] = real_point;
    };

    if (lhs_first == 0) add_candidate(rhs.first);
    if (lhs_second == 0) add_candidate(rhs.second);
    if (rhs_first == 0) add_candidate(lhs.first);
    if (rhs_second == 0) add_candidate(lhs.second);

    if (count != 0) {
        Point2<R> first = candidates[0];
        Point2<R> last = candidates[0];
        for (usize i = 1; i != count; ++i) {
            if (detail::point_less(candidates[i], first)) first = candidates[i];
            if (detail::point_less(last, candidates[i])) last = candidates[i];
        }
        if (count == 1) return {SegmentIntersectionKind::point, first, first};
        return {SegmentIntersectionKind::overlap, first, last};
    }

    if (lhs_first == lhs_second || rhs_first == rhs_second) return {};

    const auto lhs_point = lhs.first.template cast<R>();
    const auto lhs_direction = lhs.second.template cast<R>() - lhs_point;
    const auto rhs_point = rhs.first.template cast<R>();
    const auto rhs_direction = rhs.second.template cast<R>() - rhs_point;
    const R parameter = cross(rhs_point - lhs_point, rhs_direction)
        / cross(lhs_direction, rhs_direction);
    const auto point = lhs_point + lhs_direction * parameter;
    return {SegmentIntersectionKind::point, point, point};
}

template <GeometryScalar T>
PointIntersection2<geometry_real_t<T>> intersection(
    Line2<T> line,
    Circle2<T> circle,
    GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    using R = geometry_real_t<T>;
    assert(line.is_valid() && circle.is_valid());
    const auto point = line.point.template cast<R>();
    const auto direction = line.direction.template cast<R>();
    const auto center = circle.center.template cast<R>();
    const R radius = static_cast<R>(circle.radius);
    const R direction_norm_sq = dot(direction, direction);
    const auto base = point
        + direction
        * (dot(center - point, direction) / direction_norm_sq);
    const R center_distance = distance(base, center);
    if (center_distance
        > radius
        && !almost_equal(center_distance, radius, tolerance))
        return {};
    if (almost_equal(center_distance, radius, tolerance))
        return {PointIntersectionKind::one, {base, {}}};

    const R half_chord = std::sqrt(
        std::max<R>(R{}, radius * radius - center_distance * center_distance)
    );
    const auto unit_direction = direction / std::sqrt(direction_norm_sq);
    const auto first = base - unit_direction * half_chord;
    const auto second = base + unit_direction * half_chord;
    return {PointIntersectionKind::two, {first, second}};
}

template <GeometryScalar T>
PointIntersection2<geometry_real_t<T>> intersection(
    Segment2<T> segment,
    Circle2<T> circle,
    GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    using R = geometry_real_t<T>;
    assert(circle.is_valid());
    if (segment.is_degenerate()) {
        if (!on_circle(circle, segment.first, tolerance)) return {};
        return {PointIntersectionKind::one,
                {segment.first.template cast<R>(), {}}};
    }

    const auto real_first = segment.first.template cast<R>();
    const auto real_second = segment.second.template cast<R>();
    const auto line_result = intersection(
        Line2<R>{real_first, real_second - real_first},
        Circle2<R>{circle.center.template cast<R>(),
                   static_cast<R>(circle.radius)},
        tolerance
    );
    PointIntersection2<R> result{};
    usize result_count = 0;
    for (usize i = 0; i != line_result.count(); ++i) {
        const auto point = line_result.points[i];
        const Segment2<R> real_segment{real_first, real_second};
        if (!on_segment(real_segment, point, tolerance)) continue;
        if (result_count
            == 1
            && almost_equal(result.points[0], point, tolerance))
            continue;
        assert(result_count < result.points.size());
        result.points[result_count++] = point;
    }
    result.kind = result_count == 0 ? PointIntersectionKind::none
        : result_count == 1         ? PointIntersectionKind::one
                                    : PointIntersectionKind::two;
    return result;
}

template <GeometryScalar T>
PointIntersection2<geometry_real_t<T>> intersection(
    Circle2<T> lhs,
    Circle2<T> rhs,
    GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    using R = geometry_real_t<T>;
    assert(lhs.is_valid() && rhs.is_valid());
    const auto lhs_center = lhs.center.template cast<R>();
    const auto rhs_center = rhs.center.template cast<R>();
    const R lhs_radius = static_cast<R>(lhs.radius);
    const R rhs_radius = static_cast<R>(rhs.radius);
    const auto delta = rhs_center - lhs_center;
    const R squared_distance = dot(delta, delta);
    const R center_distance = std::sqrt(squared_distance);

    if (almost_equal(center_distance, R{}, tolerance)) {
        if (!almost_equal(lhs_radius, rhs_radius, tolerance)) return {};
        if (almost_equal(lhs_radius, R{}, tolerance))
            return {PointIntersectionKind::one, {lhs_center, {}}};
        return {PointIntersectionKind::coincident, {}};
    }

    const R radius_sum = lhs_radius + rhs_radius;
    const R radius_difference = std::abs(lhs_radius - rhs_radius);
    if ((center_distance
         > radius_sum
         && !almost_equal(center_distance, radius_sum, tolerance))
        || (center_distance
            < radius_difference
            && !almost_equal(center_distance, radius_difference, tolerance)))
        return {};

    const R along =
        (lhs_radius * lhs_radius - rhs_radius * rhs_radius + squared_distance)
        / (R{2} * center_distance);
    const R height_squared = lhs_radius * lhs_radius - along * along;
    const auto base = lhs_center + delta * (along / center_distance);
    if (almost_equal(center_distance, radius_sum, tolerance)
        || almost_equal(center_distance, radius_difference, tolerance))
        return {PointIntersectionKind::one, {base, {}}};

    const R height = std::sqrt(std::max<R>(R{}, height_squared));
    const auto offset = perp_ccw(delta) * (height / center_distance);
    return {PointIntersectionKind::two, {base + offset, base - offset}};
}

template <typename Lhs, typename Rhs, typename... Args>
bool intersects(Lhs lhs, Rhs rhs, Args... args) {
    return intersection(lhs, rhs, args...).kind
        != decltype(intersection(lhs, rhs, args...)){}.kind;
}

template <GeometryScalar T>
geometry_real_t<T> distance(
    Segment2<T> lhs,
    Segment2<T> rhs,
    GeometryTolerance<geometry_real_t<T>> tolerance = {}
) {
    using R = geometry_real_t<T>;
    if (intersects(lhs, rhs, tolerance)) return R{};
    return std::min<R>({distance(lhs.first, rhs), distance(lhs.second, rhs),
                        distance(rhs.first, lhs), distance(rhs.second, lhs)});
}

}  // namespace cp

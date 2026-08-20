#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <iostream>
#include <type_traits>

#include "cp/geometry.hpp"

using namespace cp;

namespace
{

template <typename T>
bool near(T lhs, T rhs, T eps = static_cast<T>(1e-9L)) {
    return std::abs(lhs - rhs) <=
        eps * std::max<T>({1, std::abs(lhs), std::abs(rhs)});
}

template <std::floating_point T>
void require_point_near(
    Point2<T> actual, Point2<T> expected, T eps = static_cast<T>(1e-9L)
) {
    assert(near(actual.x, expected.x, eps));
    assert(near(actual.y, expected.y, eps));
}

template <typename T>
concept HasVectorDivision =
    requires(Vec2<T> value, T scalar) { value / scalar; };

static_assert(GeometryScalar<int>);
static_assert(GeometryScalar<long double>);
static_assert(!GeometryScalar<unsigned>);
static_assert(!GeometryScalar<bool>);
static_assert(std::same_as<geometry_wide_t<i16>, i64>);
static_assert(std::same_as<geometry_wide_t<i32>, i128>);
static_assert(std::same_as<geometry_wide_t<i64>, i128>);
static_assert(std::same_as<geometry_real_t<i64>, long double>);
static_assert(std::same_as<geometry_real_t<double>, double>);
static_assert(std::same_as<Point2<int>, Vec2<int>>);
static_assert(!HasVectorDivision<int>);
static_assert(HasVectorDivision<double>);

constexpr Vec2<int> constexpr_vector_test() {
    Vec2<int> value{2, 3};
    value += {1, -1};
    value *= 2;
    return value;
}

static_assert(constexpr_vector_test() == Vec2<int>{6, 4});
static_assert(Vec2<int>{3, 4}.cast<long double>() == Vec2<long double>{3, 4});
static_assert(dot(Vec2<int>{2, 3}, Vec2<int>{4, 5}) == 23);
static_assert(cross(Vec2<int>{2, 3}, Vec2<int>{4, 5}) == -2);
static_assert(sgn(-7) == -1);
static_assert(sgn(0) == 0);
static_assert(sgn(7) == 1);
static_assert(cmp(2, 3) == -1);
static_assert(cmp(3, 3) == 0);
static_assert(cmp(4, 3) == 1);
static_assert(
    orientation(Point2<int>{0, 0}, Point2<int>{2, 0}, Point2<int>{1, 1}) == 1
);
static_assert(
    orientation(
        Point2<double>{0, 0}, Point2<double>{1, 0}, Point2<double>{0, 1}
    ) == 1
);
static_assert(
    line_side(Line2<double>{{0, 0}, {1, 0}}, Point2<double>{0, 1}) == 1
);
static_assert(
    perpendicular(Line2<double>{{0, 0}, {1, 0}}, Line2<double>{{0, 0}, {0, 1}})
);

void test_vector_arithmetic() {
    const Vec2<i32> a{3, 4};
    const Vec2<i32> b{-2, 5};

    assert(a + b == Vec2<i32>(1, 9));
    assert(a - b == Vec2<i32>(5, -1));
    assert(-a == Vec2<i32>(-3, -4));
    assert(a * 3 == Vec2<i32>(9, 12));
    assert(3 * a == Vec2<i32>(9, 12));
    assert(dot(a, b) == 14);
    assert(cross(a, b) == 23);
    assert(norm_sq(a) == 25);
    assert(distance_sq(a, b) == 26);
    assert(near(norm(a), 5.0L));
    assert(near(distance(a, b), std::sqrt(26.0L)));
    assert(near(norm(Vec2<float>{1e20F, 1e20F}), std::sqrt(2.0F) * 1e20F));
    assert(near(angle(Vec2<double>{0, 1}), std::acos(-1.0) / 2));

    const auto rotated = rotate(Vec2<i32>{1, 0}, std::acos(-1.0L) / 2);
    require_point_near(rotated, Vec2<long double>{0, 1}, 1e-12L);

    const auto unit = normalized(a);
    assert(unit.has_value());
    require_point_near(*unit, Vec2<long double>{0.6L, 0.8L}, 1e-12L);
    assert(!normalized(Vec2<double>{0, 0}).has_value());
}

void test_widened_integral_arithmetic() {
    constexpr i32 big = 2'000'000'000;
    const Point2<i32> origin{-big, -big};
    const Point2<i32> x_axis{big, -big};
    const Point2<i32> upper{-big, big};

    const i128 expected_cross = i128{4'000'000'000LL} * i128{4'000'000'000LL};
    assert(cross(origin, x_axis, upper) == expected_cross);
    assert(orientation(origin, x_axis, upper) == 1);
    assert(
        distance_sq(origin, x_axis) ==
        i128{4'000'000'000LL} * i128{4'000'000'000LL}
    );

    constexpr i64 m = 4'000'000'000LL;
    assert(
        distance_sq(Point2<i64>{-m, 0}, Point2<i64>{m, 0}) ==
        (i128)(2 * m) * (2 * m)
    );
}

void test_tolerance() {
    static_assert(geometry_eps == 1e-9L);

    assert(almost_equal(1.0, 1.0 + 5e-10));
    assert(!almost_equal(1.0, 1.0 + 5e-7));
    assert(!almost_equal(1'000'000'000.0, 1'000'000'000.5));
    assert(sgn(5e-10) == 0);
    assert(sgn(1e-5) == 1);
    assert(cmp(1.0, 1.0 + 5e-10) == 0);
    assert(cmp(1.0, 1.0 + 1e-5) == -1);
    assert(cmp(1'000'000'000.0, 1'000'000'000.5) == -1);
    assert(
        almost_equal(Vec2<double>{1, 2}, Vec2<double>{1 + 5e-10, 2 - 5e-10})
    );

    assert(
        orientation(
            Point2<double>{0, 0}, Point2<double>{1, 1},
            Point2<double>{2, 2 + 1e-10}
        ) == 0
    );
    assert(
        orientation(
            Point2<double>{0, 0}, Point2<double>{1, 1},
            Point2<double>{2, 2 + 1e-5}
        ) == 1
    );
    assert(
        orientation(
            Point2<double>{0, 0}, Point2<double>{1e-5, 0},
            Point2<double>{0, 1e-5}
        ) == 1
    );
    assert(
        orientation(
            Point2<double>{0, 0}, Point2<double>{1e5, 0}, Point2<double>{0, 1e5}
        ) == 1
    );
}

void test_primitives_and_predicates() {
    const auto line = Line2<int>::through({0, 0}, {4, 0});
    assert(line.is_valid());
    assert((!Line2<int>{{0, 0}, {0, 0}}.is_valid()));

    assert(on_line(line, Point2<int>{2, 0}));
    assert(!on_line(line, Point2<int>{2, 1}));
    assert(on_segment(Point2<int>{0, 0}, Point2<int>{4, 0}, {0, 0}));
    assert(on_segment(Point2<int>{0, 0}, Point2<int>{4, 0}, {2, 0}));
    assert(!on_segment(Point2<int>{0, 0}, Point2<int>{4, 0}, {5, 0}));
    assert(!on_segment(Point2<int>{0, 0}, Point2<int>{4, 0}, {2, 1}));
    assert(on_segment(
        Point2<double>{0, 0}, Point2<double>{2, 0}, Point2<double>{2 + 5e-10, 0}
    ));
    assert(!on_segment(
        Point2<double>{0, 0}, Point2<double>{2, 0}, Point2<double>{2 + 1e-5, 0}
    ));

    assert(parallel(Line2<int>{{0, 0}, {1, 2}}, Line2<int>{{3, 4}, {-2, -4}}));
    assert(
        perpendicular(Line2<int>{{0, 0}, {1, 2}}, Line2<int>{{3, 4}, {-2, 1}})
    );
}

void test_projection_and_distances() {
    const Line2<int> line{{0, 0}, {2, 0}};
    require_point_near(project(Point2<int>{3, 4}, line), {3, 0});
    require_point_near(reflect(Point2<int>{3, 4}, line), {3, -4});
    assert(near(distance(Point2<int>{3, 4}, line), 4.0L));
}

void test_line_intersections() {
    const auto crossing =
        intersection(Line2<int>{{0, 0}, {1, 1}}, Line2<int>{{0, 2}, {1, -1}});
    assert(crossing.kind == LineIntersectionKind::point);
    require_point_near(crossing.point, Point2<long double>{1, 1});

    const auto parallel_result =
        intersection(Line2<int>{{0, 0}, {1, 0}}, Line2<int>{{0, 1}, {1, 0}});
    assert(parallel_result.kind == LineIntersectionKind::none);

    const auto coincident =
        intersection(Line2<int>{{0, 0}, {1, 0}}, Line2<int>{{2, 0}, {-3, 0}});
    assert(coincident.kind == LineIntersectionKind::coincident);
    assert(intersects(Line2<int>{{0, 0}, {1, 0}}, Line2<int>{{2, 0}, {-3, 0}}));
}

void test_segment_intersections() {
    const auto crossing = intersection(
        Segment2<int>{{0, 0}, {2, 2}}, Segment2<int>{{0, 2}, {2, 0}}
    );
    assert(crossing.kind == LineIntersectionKind::point);
    require_point_near(crossing.point, Point2<long double>{1, 1});
    assert(
        intersection(
            Segment2<int>{{0, 0}, {1, 0}}, Segment2<int>{{2, 0}, {3, 0}}
        )
            .kind == LineIntersectionKind::none
    );
    assert(
        intersection(
            Segment2<int>{{0, 0}, {3, 0}}, Segment2<int>{{2, 0}, {5, 0}}
        )
            .kind == LineIntersectionKind::coincident
    );
    assert(
        intersection(
            Segment2<int>{{0, 0}, {3, 0}}, Segment2<int>{{3, 0}, {5, 0}}
        )
            .kind == LineIntersectionKind::point
    );
    assert(
        intersection(
            Segment2<int>{{1, 1}, {1, 1}}, Segment2<int>{{0, 0}, {2, 2}}
        )
            .kind == LineIntersectionKind::point
    );
    assert(
        intersection(
            Segment2<int>{{3, 3}, {3, 3}}, Segment2<int>{{0, 0}, {2, 2}}
        )
            .kind == LineIntersectionKind::none
    );
    assert(
        intersection(
            Segment2<int>{{1, 1}, {1, 1}}, Segment2<int>{{1, 1}, {1, 1}}
        )
            .kind == LineIntersectionKind::point
    );
    assert(
        intersection(
            Segment2<int>{{1, 1}, {1, 1}}, Segment2<int>{{2, 2}, {2, 2}}
        )
            .kind == LineIntersectionKind::none
    );
    assert(
        intersection(
            Segment2<double>{{0, 0}, {0, 0}},
            Segment2<double>{{5e-10, 0}, {5e-10, 0}}
        )
            .kind == LineIntersectionKind::point
    );
    assert(
        intersection(
            Segment2<int>{{0, 0}, {3, 0}}, Segment2<int>{{0, 0}, {3, 0}}
        )
            .kind == LineIntersectionKind::coincident
    );
    const auto translated = intersection(
        Segment2<double>{{1e15, 1e15}, {1e15 + 4, 1e15}},
        Segment2<double>{{1e15 + 2, 1e15 - 2}, {1e15 + 2, 1e15 + 2}}
    );
    assert(translated.kind == LineIntersectionKind::point);
    require_point_near(translated.point, {1e15 + 2, 1e15}, 1e-9);
    const auto huge_direction = intersection(
        Line2<double>{{0, 0}, {1e155, 0}}, Line2<double>{{0, -1}, {0, 1e155}}
    );
    assert(huge_direction.kind == LineIntersectionKind::point);
    require_point_near(huge_direction.point, {0, 0});
    const auto tiny_direction = intersection(
        Line2<double>{{0, 0}, {1e-155, 0}}, Line2<double>{{0, -1}, {0, 1e-155}}
    );
    assert(tiny_direction.kind == LineIntersectionKind::point);
    require_point_near(tiny_direction.point, {0, 0});
}

void test_properties() {
    const Line2<double> line{{1, 2}, {3, -1}};
    const Point2<double> point{4, 7};
    const auto projected = project(point, line);
    assert(on_line(line, projected));
    assert(perpendicular(line, Line2<double>{projected, point - projected}));

    const auto reflected = reflect(point, line);
    require_point_near((point + reflected) / 2.0, projected);

    const auto rotated = rotate(Vec2<double>{3, 4}, 0.731);
    assert(near(norm(rotated), 5.0));
}

}  // namespace

int main() {
    test_vector_arithmetic();
    test_widened_integral_arithmetic();
    test_tolerance();
    test_primitives_and_predicates();
    test_projection_and_distances();
    test_line_intersections();
    test_segment_intersections();
    test_properties();
    std::cout << "All geometry tests passed!\n";
}

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <iostream>
#include <limits>
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
static_assert(
    orientation(Point2<int>{0, 0}, Point2<int>{2, 0}, Point2<int>{1, 1}) == 1
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
    assert(perp_ccw(a) == Vec2<i32>(-4, 3));
    assert(perp_cw(a) == Vec2<i32>(4, -3));
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
    const GeometryTolerance<double> defaults{};
    assert(near(defaults.absolute, 1e-9));
    assert(near(defaults.relative, 1e-9));

    assert(almost_equal(1.0, 1.0 + 5e-10));
    assert(!almost_equal(1.0, 1.0 + 5e-7));
    assert(almost_equal(
        1'000'000'000.0, 1'000'000'000.5, GeometryTolerance<double>{1e-12, 1e-9}
    ));
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

    const Circle2<double> point_circle{{0, 0}, 0};
    assert(!on_circle(point_circle, Point2<double>{1e-5, 0}));
    assert(!contains(point_circle, Point2<double>{1e-5, 0}));
}

void test_primitives_and_predicates() {
    const auto line = Line2<int>::through({0, 0}, {4, 0});
    assert(line.is_valid());
    assert((!Line2<int>{{0, 0}, {0, 0}}.is_valid()));

    const Segment2<int> segment{{0, 0}, {4, 0}};
    assert(!segment.is_degenerate());
    assert((Segment2<int>{{2, 2}, {2, 2}}.is_degenerate()));

    assert((Circle2<double>{{0, 0}, 2}.is_valid()));
    assert((!Circle2<double>{{0, 0}, -1}.is_valid()));

    assert(on_line(line, Point2<int>{2, 0}));
    assert(!on_line(line, Point2<int>{2, 1}));
    assert(on_segment(segment, Point2<int>{0, 0}));
    assert(on_segment(segment, Point2<int>{2, 0}));
    assert(!on_segment(segment, Point2<int>{5, 0}));

    const Circle2<int> circle{{0, 0}, 5};
    assert(on_circle(circle, Point2<int>{3, 4}));
    assert(contains(circle, Point2<int>{0, 0}));
    assert(contains(circle, Point2<int>{3, 4}));
    assert(!contains(circle, Point2<int>{6, 0}));

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

    const Segment2<int> segment{{0, 0}, {4, 0}};
    require_point_near(closest_point(Point2<int>{2, 3}, segment), {2, 0});
    require_point_near(closest_point(Point2<int>{-2, 3}, segment), {0, 0});
    assert(near(distance(Point2<int>{2, 3}, segment), 3.0L));
    assert(near(distance(Point2<int>{-2, 3}, segment), std::sqrt(13.0L)));

    const Segment2<int> other{{6, -2}, {6, 2}};
    assert(near(distance(segment, other), 2.0L));
    assert(near(distance(segment, Segment2<int>{{2, -2}, {2, 2}}), 0.0L));

    const Circle2<int> circle{{0, 0}, 5};
    assert(near(distance_to_circle(Point2<int>{0, 0}, circle), 5.0L));
    assert(near(distance_to_circle(Point2<int>{3, 4}, circle), 0.0L));
    assert(near(distance_to_disk(Point2<int>{0, 0}, circle), 0.0L));
    assert(near(distance_to_disk(Point2<int>{8, 0}, circle), 3.0L));
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
        Segment2<int>{{0, 0}, {4, 4}}, Segment2<int>{{0, 4}, {4, 0}}
    );
    assert(crossing.kind == SegmentIntersectionKind::point);
    require_point_near(crossing.first, Point2<long double>{2, 2});
    require_point_near(crossing.second, Point2<long double>{2, 2});

    const auto touching = intersection(
        Segment2<int>{{0, 0}, {2, 0}}, Segment2<int>{{2, 0}, {3, 1}}
    );
    assert(touching.kind == SegmentIntersectionKind::point);
    require_point_near(touching.first, Point2<long double>{2, 0});

    const auto overlap = intersection(
        Segment2<int>{{4, 0}, {0, 0}}, Segment2<int>{{2, 0}, {6, 0}}
    );
    assert(overlap.kind == SegmentIntersectionKind::overlap);
    require_point_near(overlap.first, Point2<long double>{2, 0});
    require_point_near(overlap.second, Point2<long double>{4, 0});

    const auto disjoint = intersection(
        Segment2<int>{{0, 0}, {1, 0}}, Segment2<int>{{2, 0}, {3, 0}}
    );
    assert(disjoint.kind == SegmentIntersectionKind::none);

    const auto point_on_segment = intersection(
        Segment2<int>{{2, 0}, {2, 0}}, Segment2<int>{{0, 0}, {4, 0}}
    );
    assert(point_on_segment.kind == SegmentIntersectionKind::point);
    require_point_near(point_on_segment.first, Point2<long double>{2, 0});
    assert(
        intersection(
            Segment2<int>{{0, 0}, {4, 0}}, Segment2<int>{{2, 0}, {2, 0}}
        )
            .kind == SegmentIntersectionKind::point
    );

    const auto line_overlap =
        intersection(Line2<int>{{0, 0}, {1, 0}}, Segment2<int>{{3, 0}, {1, 0}});
    assert(line_overlap.kind == SegmentIntersectionKind::overlap);
    require_point_near(line_overlap.first, Point2<long double>{1, 0});
    require_point_near(line_overlap.second, Point2<long double>{3, 0});

    const auto line_crossing = intersection(
        Line2<int>{{0, 0}, {1, 0}}, Segment2<int>{{2, -1}, {2, 1}}
    );
    assert(line_crossing.kind == SegmentIntersectionKind::point);
    require_point_near(line_crossing.first, Point2<long double>{2, 0});
    assert(
        intersection(Line2<int>{{0, 0}, {1, 0}}, Segment2<int>{{2, 1}, {2, 2}})
            .kind == SegmentIntersectionKind::none
    );
    assert(
        intersection(Line2<int>{{0, 0}, {1, 0}}, Segment2<int>{{2, 0}, {2, 0}})
            .kind == SegmentIntersectionKind::point
    );
}

void test_line_and_segment_circle_intersections() {
    const Circle2<int> circle{{0, 0}, 5};

    const auto none = intersection(Line2<int>{{0, 6}, {1, 0}}, circle);
    assert(none.kind == PointIntersectionKind::none);
    assert(none.count() == 0);

    const auto tangent = intersection(Line2<int>{{0, 5}, {1, 0}}, circle);
    assert(tangent.kind == PointIntersectionKind::one);
    assert(tangent.count() == 1);
    require_point_near(tangent.points[0], Point2<long double>{0, 5});

    const auto two = intersection(Line2<int>{{0, 0}, {1, 0}}, circle);
    assert(two.kind == PointIntersectionKind::two);
    assert(two.count() == 2);
    require_point_near(two.points[0], Point2<long double>{-5, 0});
    require_point_near(two.points[1], Point2<long double>{5, 0});

    const auto clipped = intersection(Segment2<int>{{-6, 0}, {0, 0}}, circle);
    assert(clipped.kind == PointIntersectionKind::one);
    require_point_near(clipped.points[0], Point2<long double>{-5, 0});

    assert(
        intersection(Segment2<int>{{-6, 0}, {6, 0}}, circle).kind ==
        PointIntersectionKind::two
    );
    assert(
        intersection(Segment2<int>{{-2, 5}, {2, 5}}, circle).kind ==
        PointIntersectionKind::one
    );
    assert(
        intersection(Segment2<int>{{3, 4}, {3, 4}}, circle).kind ==
        PointIntersectionKind::one
    );

    const auto wide_segment = intersection(
        Segment2<i64>{{std::numeric_limits<i64>::min(), 0},
                      {std::numeric_limits<i64>::max(), 0}},
        Circle2<i64>{{0, 0}, 1}
    );
    assert(wide_segment.kind == PointIntersectionKind::two);
    require_point_near(wide_segment.points[0], Point2<long double>{-1, 0});
    require_point_near(wide_segment.points[1], Point2<long double>{1, 0});
}

void test_circle_intersections() {
    const Circle2<int> first{{0, 0}, 5};

    assert(
        intersection(first, Circle2<int>{{20, 0}, 5}).kind ==
        PointIntersectionKind::none
    );
    assert(
        intersection(first, Circle2<int>{{1, 0}, 1}).kind ==
        PointIntersectionKind::none
    );

    const auto external_tangent = intersection(first, Circle2<int>{{10, 0}, 5});
    assert(external_tangent.kind == PointIntersectionKind::one);
    require_point_near(external_tangent.points[0], Point2<long double>{5, 0});

    const auto internal_tangent = intersection(first, Circle2<int>{{3, 0}, 2});
    assert(internal_tangent.kind == PointIntersectionKind::one);
    require_point_near(internal_tangent.points[0], Point2<long double>{5, 0});

    const auto two = intersection(first, Circle2<int>{{6, 0}, 5});
    assert(two.kind == PointIntersectionKind::two);
    require_point_near(two.points[0], Point2<long double>{3, 4});
    require_point_near(two.points[1], Point2<long double>{3, -4});

    assert(
        intersection(first, Circle2<int>{{0, 0}, 4}).kind ==
        PointIntersectionKind::none
    );
    const auto coincident = intersection(first, Circle2<int>{{0, 0}, 5});
    assert(coincident.kind == PointIntersectionKind::coincident);
    assert(coincident.count() == 0);

    const auto same_point =
        intersection(Circle2<int>{{2, 3}, 0}, Circle2<int>{{2, 3}, 0});
    assert(same_point.kind == PointIntersectionKind::one);
    require_point_near(same_point.points[0], Point2<long double>{2, 3});

    const auto nearby =
        intersection(Circle2<double>{{0, 0}, 1}, Circle2<double>{{1e-5, 0}, 1});
    assert(nearby.kind == PointIntersectionKind::two);
}

void test_properties() {
    const Segment2<double> first{{0, 0}, {4, 1}};
    const Segment2<double> second{{5, -2}, {5, 3}};
    assert(near(distance(first, second), distance(second, first)));
    assert(
        intersects(first, second) ==
        (intersection(first, second).kind != SegmentIntersectionKind::none)
    );

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
    test_line_and_segment_circle_intersections();
    test_circle_intersections();
    test_properties();
    std::cout << "All geometry tests passed!\n";
}

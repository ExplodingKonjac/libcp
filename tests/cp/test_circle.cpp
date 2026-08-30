#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "cp/geometry/circle.hpp"

using namespace cp;

namespace
{

template <typename T>
bool near(T lhs, T rhs, T eps = static_cast<T>(1e-9L)) {
    return std::abs(lhs - rhs) <=
        eps * std::max<T>({T{1}, std::abs(lhs), std::abs(rhs)});
}

template <std::floating_point T>
void require_point_near(
    Point2<T> actual, Point2<T> expected, T eps = static_cast<T>(1e-9L)
) {
    assert(near(actual.x, expected.x, eps));
    assert(near(actual.y, expected.y, eps));
}

void test_circle_primitives() {
    assert((Circle2<double>{{0, 0}, 2}.is_valid()));
    assert((!Circle2<double>{{0, 0}, -1}.is_valid()));

    const Circle2<int> circle{{0, 0}, 5};
    assert(on_circle(circle, Point2<int>{3, 4}));
    assert(contains(circle, Point2<int>{0, 0}));
    assert(contains(circle, Point2<int>{3, 4}));
    assert(!contains(circle, Point2<int>{6, 0}));

    const Circle2<double> point_circle{{0, 0}, 0};
    assert(!on_circle(point_circle, Point2<double>{1e-5, 0}));
    assert(!contains(point_circle, Point2<double>{1e-5, 0}));
}

void test_circle_distances() {
    const Circle2<int> circle{{0, 0}, 5};
    assert(near(distance_to_circle(Point2<int>{0, 0}, circle), 5.0L));
    assert(near(distance_to_circle(Point2<int>{3, 4}, circle), 0.0L));
    assert(near(distance_to_disk(Point2<int>{0, 0}, circle), 0.0L));
    assert(near(distance_to_disk(Point2<int>{8, 0}, circle), 3.0L));
}

void test_line_circle_intersections() {
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

    const double large_radius = 1e9;
    const double larger_radius = large_radius + 1e-6;
    const auto close_radii = intersection(
        Circle2<double>{{0, 0}, larger_radius},
        Circle2<double>{{1, 0}, large_radius}
    );
    const double expected_x = ((larger_radius - large_radius) *
                                   ((larger_radius + large_radius) / 1.0) +
                               1.0) /
        2.0;
    assert(close_radii.kind == PointIntersectionKind::two);
    assert(near(close_radii.points[0].x, expected_x, 1e-12));

    const auto huge = intersection(
        Circle2<double>{{0, 0}, 1e155}, Circle2<double>{{1e155, 0}, 1e155}
    );
    assert(huge.kind == PointIntersectionKind::two);
    assert(near(huge.points[0].x / 1e155, 0.5, 1e-12));
    assert(
        near(std::abs(huge.points[0].y) / 1e155, std::sqrt(3.0) / 2.0, 1e-12)
    );
}

void test_circle_from_three_points() {
    static_assert(
        std::same_as<
            decltype(circle_from(Point2<int>{}, Point2<int>{}, Point2<int>{})),
            Circle2<long double>
        >
    );

    const auto triangle =
        circle_from(Point2<int>{0, 0}, Point2<int>{4, 0}, Point2<int>{0, 3});
    require_point_near(triangle.center, Point2<long double>{2, 1.5L});
    assert(near(triangle.radius, 2.5L));

    const auto collinear =
        circle_from(Point2<int>{-3, 2}, Point2<int>{5, 2}, Point2<int>{0, 2});
    require_point_near(collinear.center, Point2<long double>{1, 2});
    assert(near(collinear.radius, 4.0L));

    const auto tiny = circle_from(
        Point2<double>{0, 0}, Point2<double>{1e-5, 0}, Point2<double>{0, 1e-5}
    );
    assert(std::abs(tiny.center.x - 5e-6) <= 1e-15);
    assert(std::abs(tiny.center.y - 5e-6) <= 1e-15);

    const auto huge = circle_from(
        Point2<double>{0, 0}, Point2<double>{1e155, 0}, Point2<double>{0, 1e155}
    );
    assert(near(huge.center.x / 1e155, 0.5, 1e-12));
    assert(near(huge.center.y / 1e155, 0.5, 1e-12));
    assert(near(huge.radius / 1e155, std::sqrt(0.5), 1e-12));
}

void test_minimum_enclosing_circle_degenerate_inputs() {
    static_assert(
        std::same_as<
            decltype(minimum_enclosing_circle(std::vector<Point2<int>>{})),
            std::optional<Circle2<long double>>
        >
    );

    assert(!minimum_enclosing_circle(std::vector<Point2<int>>{}).has_value());

    const auto singleton =
        minimum_enclosing_circle(std::vector<Point2<int>>{{3, -4}});
    assert(singleton.has_value());
    require_point_near(singleton->center, Point2<long double>{3, -4});
    assert(near(singleton->radius, 0.0L));

    const auto duplicate = minimum_enclosing_circle(
        std::vector<Point2<double>>{{2, 1}, {2, 1}, {2, 1}}
    );
    assert(duplicate.has_value());
    require_point_near(duplicate->center, Point2<double>{2, 1});
    assert(near(duplicate->radius, 0.0));

    const auto pair =
        minimum_enclosing_circle(std::vector<Point2<int>>{{-2, 1}, {4, 1}});
    assert(pair.has_value());
    require_point_near(pair->center, Point2<long double>{1, 1});
    assert(near(pair->radius, 3.0L));

    const auto collinear = minimum_enclosing_circle(
        std::vector<Point2<int>>{{-3, 2}, {0, 2}, {5, 2}, {1, 2}}
    );
    assert(collinear.has_value());
    require_point_near(collinear->center, Point2<long double>{1, 2});
    assert(near(collinear->radius, 4.0L));
}

void test_minimum_enclosing_circle_triangles() {
    const auto right = minimum_enclosing_circle(
        std::vector<Point2<double>>{{0, 0}, {0, 2}, {2, 0}}
    );
    assert(right.has_value());
    require_point_near(right->center, Point2<double>{1, 1});
    assert(near(right->radius, std::sqrt(2.0)));

    const auto obtuse = minimum_enclosing_circle(
        std::vector<Point2<double>>{{-2, 0}, {2, 0}, {0, 1}}
    );
    assert(obtuse.has_value());
    require_point_near(obtuse->center, Point2<double>{0, 0});
    assert(near(obtuse->radius, 2.0));

    constexpr double offset = 1e15;
    const auto translated = minimum_enclosing_circle(
        std::vector<Point2<double>>{
            {offset, offset}, {offset + 4, offset}, {offset + 2, offset + 4}}
    );
    assert(translated.has_value());
    assert(std::abs(translated->center.x - (offset + 2)) <= 0.125);
    assert(std::abs(translated->center.y - (offset + 1.5)) <= 0.125);
    assert(near(translated->radius, 2.5));

    assert(contains(Circle2<double>{{0, 0}, 1}, Point2<double>{1 + 5e-10, 0}));
    assert(!contains(Circle2<double>{{0, 0}, 1}, Point2<double>{1 + 5e-8, 0}));
}

Circle2<double> reference_pair_circle(Point2<double> a, Point2<double> b) {
    const auto center = (a + b) / 2.0;
    return {center, distance(center, a)};
}

std::optional<Circle2<double>> reference_triple_circle(
    Point2<double> a, Point2<double> b, Point2<double> c
) {
    const double d = 2 * cross(a, b, c);
    if (std::abs(d) < 1e-12) return std::nullopt;
    const double a2 = norm_sq(a), b2 = norm_sq(b), c2 = norm_sq(c);
    const Point2<double> center{
        (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d,
        (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d,
    };
    return Circle2<double>{center, distance(center, a)};
}

bool reference_contains_all(
    Circle2<double> circle, const std::vector<Point2<double>>& points
) {
    return std::ranges::all_of(points, [&](Point2<double> point) {
        return distance(circle.center, point) <= circle.radius + 1e-8;
    });
}

Circle2<double> brute_force_minimum_circle(
    const std::vector<Point2<double>>& points
) {
    Circle2<double> best{{}, std::numeric_limits<double>::infinity()};
    const auto consider = [&](Circle2<double> candidate) {
        if (candidate.radius < best.radius &&
            reference_contains_all(candidate, points))
            best = candidate;
    };

    for (usize i = 0; i != points.size(); ++i) {
        consider({points[i], 0});
        for (usize j = i + 1; j != points.size(); ++j) {
            consider(reference_pair_circle(points[i], points[j]));
            for (usize k = j + 1; k != points.size(); ++k)
                if (const auto circle = reference_triple_circle(
                        points[i], points[j], points[k]
                    ))
                    consider(*circle);
        }
    }
    return best;
}

void test_minimum_enclosing_circle_against_brute_force() {
    std::mt19937 random(0xC1AC1E);
    std::uniform_int_distribution<int> coordinate(-5, 5);

    for (int size = 1; size <= 8; ++size) {
        for (int trial = 0; trial != 100; ++trial) {
            std::vector<Point2<double>> points(size);
            for (auto& point: points)
                point = {
                    static_cast<double>(coordinate(random)),
                    static_cast<double>(coordinate(random)),
                };

            const auto actual = minimum_enclosing_circle(points);
            const auto expected = brute_force_minimum_circle(points);
            assert(actual.has_value());
            assert(near(actual->radius, expected.radius, 1e-8));
            assert(reference_contains_all(*actual, points));
        }
    }
}

}  // namespace

int main() {
    test_circle_primitives();
    test_circle_distances();
    test_line_circle_intersections();
    test_circle_intersections();
    test_circle_from_three_points();
    test_minimum_enclosing_circle_degenerate_inputs();
    test_minimum_enclosing_circle_triangles();
    test_minimum_enclosing_circle_against_brute_force();
    std::cout << "All circle tests passed!\n";
}

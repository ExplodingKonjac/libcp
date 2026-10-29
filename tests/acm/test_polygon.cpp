#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "acm/geometry/polygon.hpp"

using namespace acm;

namespace
{

template <typename T>
bool near(T lhs, T rhs, T eps = static_cast<T>(1e-9)) {
    return std::abs(lhs - rhs) <=
        eps * std::max<T>({T{1}, std::abs(lhs), std::abs(rhs)});
}

template <typename T>
void require_points(
    const std::vector<Vec2<T>>& actual, const std::vector<Vec2<T>>& expected
) {
    assert(actual == expected);
}

void test_polygon_basics() {
    const Polygon<int> polygon{
        std::vector<Vec2<int>>{{0, 0}, {0, 2}, {2, 2}, {2, 0}, {0, 0}}};
    assert(polygon.size() == 4);
    assert(polygon.vertices().front() != polygon.vertices().back());
    assert(polygon.area() == 4);

    const auto edges = polygon.edges();
    assert(edges.size() == polygon.size());
    for (usize i = 0; i != polygon.size(); i++) {
        assert(edges[i].a == polygon[i]);
        assert(edges[i].b == polygon[(i + 1) % polygon.size()]);
        assert(
            orientation(
                polygon[i], polygon[(i + 1) % polygon.size()],
                polygon[(i + 2) % polygon.size()]
            ) >= 0
        );
    }

    const Polygon<int> empty;
    assert(empty.size() == 0);
    assert(empty.edges().empty());
    assert(empty.area() == 0);

    const Polygon<int> point{std::vector<Vec2<int>>{{3, 4}}};
    assert(point.edges().size() == 1);
    assert(point.edges()[0].a == Vec2<int>(3, 4));
    assert(point.edges()[0].b == Vec2<int>(3, 4));

    const Polygon<int> two_points{std::vector<Vec2<int>>{{0, 0}, {2, 0}}};
    assert(two_points.edges().size() == 2);
    assert(two_points.edges()[0].b == Vec2<int>(2, 0));
    assert(two_points.edges()[1].b == Vec2<int>(0, 0));

    const Polygon<double> triangle{
        std::vector<Vec2<double>>{{0, 0}, {0.5, 0}, {0, 0.5}}};
    assert(near(triangle.area(), 0.125));

    constexpr double offset = 1e15;
    const Polygon<double> translated{
        std::vector<Vec2<double>>{{offset, offset},
                                  {offset, offset + 3},
                                  {offset + 4, offset + 3},
                                  {offset + 4, offset}}};
    assert(near(translated.area(), 12.0, 1e-12));
    assert(orientation(translated[0], translated[1], translated[2]) > 0);
}

void test_convexity() {
    const Polygon<int> convex{
        std::vector<Vec2<int>>{{0, 0}, {2, 0}, {2, 1}, {2, 2}, {0, 2}}};
    assert(convex.is_convex());

    const Polygon<int> concave{
        std::vector<Vec2<int>>{{0, 0}, {4, 0}, {4, 4}, {2, 2}, {0, 4}}};
    assert(!concave.is_convex());
    assert(Polygon<int>{}.is_convex());
    assert((Polygon<int>{std::vector<Vec2<int>>{{0, 0}, {1, 0}}}.is_convex()));
}

void test_point_relation() {
    const Polygon<int> square{
        std::vector<Vec2<int>>{{0, 0}, {0, 4}, {4, 4}, {4, 0}}};
    assert(square.relation({2, 2}) == PointPolygonRelation::inside);
    assert(square.relation({5, 2}) == PointPolygonRelation::outside);
    assert(square.relation({0, 0}) == PointPolygonRelation::boundary);
    assert(square.relation({2, 0}) == PointPolygonRelation::boundary);
    assert(square.relation({4, 3}) == PointPolygonRelation::boundary);

    const Polygon<int> concave{
        std::vector<Vec2<int>>{{0, 0}, {4, 0}, {4, 4}, {2, 2}, {0, 4}}};
    assert(concave.relation({2, 1}) == PointPolygonRelation::inside);
    assert(concave.relation({2, 3}) == PointPolygonRelation::outside);
    assert(concave.relation({3, 3}) == PointPolygonRelation::boundary);

    const Polygon<int> triangle{std::vector<Vec2<int>>{{0, 0}, {4, 2}, {0, 4}}};
    assert(triangle.relation({1, 2}) == PointPolygonRelation::inside);
    assert(triangle.relation({-1, 2}) == PointPolygonRelation::outside);

    assert(Polygon<int>{}.relation({0, 0}) == PointPolygonRelation::outside);
    const Polygon<int> point{std::vector<Vec2<int>>{{1, 1}}};
    assert(point.relation({1, 1}) == PointPolygonRelation::boundary);
    assert(point.relation({1, 2}) == PointPolygonRelation::outside);
    const Polygon<int> segment{std::vector<Vec2<int>>{{0, 0}, {4, 0}}};
    assert(segment.relation({2, 0}) == PointPolygonRelation::boundary);
    assert(segment.relation({2, 1}) == PointPolygonRelation::outside);

    const Polygon<double> floating{
        std::vector<Vec2<double>>{{0, 0}, {2, 0}, {2, 2}, {0, 2}}};
    assert(floating.relation({-4e-10, 1}) == PointPolygonRelation::boundary);
}

void test_convex_hull() {
    const std::vector<Vec2<int>> points{
        {0, 0}, {1, 0}, {2, 0}, {2, 1}, {2, 2}, {1, 2},
        {0, 2}, {0, 1}, {1, 1}, {0, 0}, {2, 2},
    };
    require_points(
        convex_hull(points, HullMode::lower),
        std::vector<Vec2<int>>{{0, 0}, {2, 0}, {2, 2}}
    );
    require_points(
        convex_hull(points, HullMode::upper),
        std::vector<Vec2<int>>{{0, 0}, {0, 2}, {2, 2}}
    );
    require_points(
        convex_hull(points),
        std::vector<Vec2<int>>{{0, 0}, {2, 0}, {2, 2}, {0, 2}}
    );

    const std::vector<Vec2<int>> collinear{{2, 0}, {0, 0}, {1, 0}, {1, 0}};
    require_points(
        convex_hull(collinear), std::vector<Vec2<int>>{{0, 0}, {2, 0}}
    );
    assert(convex_hull(std::vector<Vec2<int>>{}).empty());
    require_points(
        convex_hull(std::vector<Vec2<int>>{{3, 4}}),
        std::vector<Vec2<int>>{{3, 4}}
    );
}

template <typename T>
std::vector<Vec2<T>> canonical_vertices(const Polygon<T>& polygon) {
    auto result = polygon.vertices();
    if (result.empty()) return result;
    const auto start = std::min_element(
        result.begin(), result.end(), [](Vec2<T> lhs, Vec2<T> rhs) {
            return lhs.y < rhs.y || (lhs.y == rhs.y && lhs.x < rhs.x);
        }
    );
    std::rotate(result.begin(), start, result.end());
    return result;
}

void test_minkowski_sum() {
    const Polygon<int> first{
        std::vector<Vec2<int>>{{0, 0}, {1, 0}, {2, 0}, {2, 1}, {0, 1}}};
    const Polygon<int> second{
        std::vector<Vec2<int>>{{0, 0}, {1, 0}, {1, 2}, {0, 2}}};
    require_points(
        canonical_vertices(minkowski_sum(first, second)),
        std::vector<Vec2<int>>{{0, 0}, {2, 0}, {3, 0}, {3, 3}, {0, 3}}
    );

    const Polygon<int> triangle{std::vector<Vec2<int>>{{0, 0}, {1, 0}, {0, 1}}};
    require_points(
        canonical_vertices(minkowski_sum(triangle, triangle)),
        std::vector<Vec2<int>>{{0, 0}, {2, 0}, {0, 2}}
    );

    const Polygon<int> point{std::vector<Vec2<int>>{{5, -1}}};
    require_points(
        canonical_vertices(minkowski_sum(first, point)),
        std::vector<Vec2<int>>{{5, -1}, {6, -1}, {7, -1}, {7, 0}, {5, 0}}
    );

    const Polygon<int> horizontal{std::vector<Vec2<int>>{{0, 0}, {2, 0}}};
    const Polygon<int> vertical{std::vector<Vec2<int>>{{0, 0}, {0, 3}}};
    require_points(
        canonical_vertices(minkowski_sum(horizontal, vertical)),
        std::vector<Vec2<int>>{{0, 0}, {2, 0}, {2, 3}, {0, 3}}
    );

    const Polygon<int> parallel{std::vector<Vec2<int>>{{1, 0}, {4, 0}}};
    require_points(
        canonical_vertices(minkowski_sum(horizontal, parallel)),
        std::vector<Vec2<int>>{{1, 0}, {6, 0}}
    );

    assert(minkowski_sum(Polygon<int>{}, first).vertices().empty());
}

void test_half_plane_intersection() {
    const std::vector<Line2<double>> box{
        {{2, 0}, {0, 1}}, {{0, 3}, {-1, 0}}, {{-1, 0}, {0, -1}},
        {{0, 0}, {1, 0}}, {{0, 0}, {0, -1}},
    };
    const auto bounded = half_plane_intersection(box);
    assert(bounded.has_value());
    assert(near(bounded->area(), 6.0, 1e-12));
    assert(bounded->relation({1, 1}) == PointPolygonRelation::inside);
    assert(bounded->relation({0, 2}) == PointPolygonRelation::boundary);

    const std::vector<Line2<double>> box_with_redundant_corner{
        {{0, 0}, {1, 0}},  {{1, 0}, {0, 1}},  {{0, 1}, {-1, 0}},
        {{0, 0}, {0, -1}}, {{0, 0}, {1, -1}},
    };
    const auto without_duplicate =
        half_plane_intersection(box_with_redundant_corner);
    assert(without_duplicate.has_value());
    assert(without_duplicate->size() == 4);
    for (usize i = 0; i != without_duplicate->size(); i++)
        assert(!almost_equal(
            (*without_duplicate)[i],
            (*without_duplicate)[(i + 1) % without_duplicate->size()]
        ));

    const std::vector<Line2<double>> wedge{
        {{0, 0}, {0, -1}},
        {{0, 0}, {1, 0}},
    };
    assert(!half_plane_intersection(wedge).has_value());
    assert(!half_plane_intersection(std::vector<Line2<double>>{}).has_value());

    const std::vector<Line2<double>> contradictory{
        {{1, 0}, {0, -1}},
        {{0, 0}, {0, 1}},
        {{0, -1}, {1, 0}},
        {{0, 1}, {-1, 0}},
    };
    assert(!half_plane_intersection(contradictory).has_value());

    const std::vector<Line2<double>> segment{
        {{0, 0}, {0, -1}},
        {{0, 0}, {0, 1}},
        {{0, 0}, {1, 0}},
        {{0, 1}, {-1, 0}},
    };
    assert(!half_plane_intersection(segment).has_value());
}

}  // namespace

int main() {
    test_polygon_basics();
    test_convexity();
    test_point_relation();
    test_convex_hull();
    test_minkowski_sum();
    test_half_plane_intersection();
    std::cout << "All polygon tests passed!\n";
}

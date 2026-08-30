#include <cassert>

#define ACM_GEOMETRY_EPS 1e-4L
#include "acm/geometry.hpp"

int main() {
    static_assert(acm::geometry_eps == 1e-4L);
    assert(acm::almost_equal(1.0, 1.0 + 5e-5));
    assert(!acm::almost_equal(1.0, 1.0 + 5e-4));
}

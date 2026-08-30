#include <cassert>

#define CP_GEOMETRY_EPS 1e-4L
#include "cp/geometry.hpp"

int main() {
    static_assert(cp::geometry_eps == 1e-4L);
    assert(cp::almost_equal(1.0, 1.0 + 5e-5));
    assert(!cp::almost_equal(1.0, 1.0 + 5e-4));
}

#include <array>
#include <string>
#include <tuple>

#include "cp/utils/concepts.hpp"

struct ConstFn {
    int operator()(int) const;
};

struct MutableFn {
    int operator()(int);
};

struct OnceFn {
    int operator()(int) &&;
};

struct WrongReturnFn {
    void operator()(int) const;
};

static_assert(cp::fn<ConstFn, int, int>);
static_assert(!cp::fn<MutableFn, int, int>);
static_assert(cp::fn_mut<ConstFn, int, int>);
static_assert(cp::fn_mut<MutableFn, int, int>);
static_assert(cp::fn_once<OnceFn, int, int>);
static_assert(!cp::fn_mut<OnceFn, int, int>);
static_assert(!cp::fn<WrongReturnFn, int, int>);
static_assert(!cp::fn<ConstFn, int, std::string>);

static_assert(cp::pair_like<std::pair<int, int>, int, int>);
static_assert(cp::pair_like<std::array<int, 2>, int, int>);
static_assert(cp::arithmetic_like<int>);

int main() {}

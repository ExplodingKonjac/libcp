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

static_assert(cp::Fn<ConstFn, int(int)>);
static_assert(!cp::Fn<MutableFn, int(int)>);
static_assert(cp::FnMut<ConstFn, int(int)>);
static_assert(cp::FnMut<MutableFn, int(int)>);
static_assert(cp::FnOnce<OnceFn, int(int)>);
static_assert(!cp::FnMut<OnceFn, int(int)>);
static_assert(!cp::Fn<WrongReturnFn, int(int)>);
static_assert(!cp::Fn<ConstFn, int(std::string)>);

static_assert(cp::PairLike<std::pair<int, int>, int, int>);
static_assert(cp::PairLike<std::array<int, 2>, int, int>);
static_assert(cp::ArithmeticLike<int>);

int main() {}

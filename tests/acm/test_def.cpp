#include <type_traits>

#include "acm/def.hpp"

static_assert(sizeof(acm::i32) == 4);
static_assert(sizeof(acm::i64) == 8);
static_assert(std::is_unsigned_v<acm::u64>);
int main() {}

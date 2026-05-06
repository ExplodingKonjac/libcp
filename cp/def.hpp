#pragma once
#include <cstddef>
#include <cstdint>

namespace cp::inline defs
{

using i8 = std::int8_t;
using u8 = std::uint8_t;
using i16 = std::int16_t;
using u16 = std::uint16_t;
using i32 = std::int32_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;
using u64 = std::uint64_t;
using i128 = __int128;
using u128 = unsigned __int128;
using isize = std::ptrdiff_t;
using usize = std::size_t;
using f32 = float;
using f64 = double;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wliteral-suffix"
#define DEF_I(t) \
    t operator"" t(unsigned long long x) { return x; }
#define DEF_F(t) \
    t operator"" t(long double x) { return x; }
DEF_I(i8) DEF_I(i16) DEF_I(i32) DEF_I(i64) DEF_I(i128) DEF_I(isize)
DEF_I(u8) DEF_I(u16) DEF_I(u32) DEF_I(u64) DEF_I(u128) DEF_I(usize)
DEF_F(f32) DEF_F(f64)
#undef DEF_I
#undef DEF_F
#pragma GCC diagnostic pop
}  // namespace cp::inline defs

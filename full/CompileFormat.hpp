#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <iterator>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace cp
{

using namespace std;

#ifndef CP_FORMAT_STRING
#define CP_FORMAT_STRING
template <size_t N>
struct FixedString: array<char, N> {
    consteval FixedString(const char (&str)[N]) { ranges::copy(str, *this); }
    constexpr auto view() const { return string_view{this->data(), N - 1}; }
};

template <FixedString S>
struct FormatString {
    constexpr operator auto() const { return S.view(); }
};

namespace literals
{

template <FixedString S>
consteval auto operator""_fmt() {
    return FormatString<S>{};
}

}  // namespace literals
#endif

namespace detail
{

struct Literal {
    string_view content;
};

template <typename T>
struct Replacement {
    size_t arg_id;
    formatter<T> formatter;
};

template <FixedString S, typename... Args>
struct CompiledFormatString {
    template <size_t Pos, size_t AutoIndex>
    static consteval auto parseArgIdx() {
        size_t pos = Pos;
        size_t arg_id = 0;
        size_t auto_index = AutoIndex;

        if (fmt[pos] >= '0' && fmt[pos] <= '9') {
            do {
                arg_id = arg_id * 10 + fmt[pos] - '0';
                pos++;
            } while (fmt[pos] >= '0' && fmt[pos] <= '9');
        } else {
            arg_id = auto_index++;
        }
        if (fmt[pos] == ':') return make_tuple(++pos, auto_index, arg_id);
        if (fmt[pos] == '}') return make_tuple(pos, auto_index, arg_id);
        throw format_error("expect ':' or '}'");
    }

    template <size_t Pos, size_t AutoIndex, size_t ArgId>
    static consteval auto parseFmtSpec() {
        using T = tuple_element_t<ArgId, tuple<Args...>>;
        size_t pos = Pos;
        size_t auto_index = AutoIndex;
        formatter<T> formatter{};
        format_parse_context ctx{fmt.substr(Pos), sizeof...(Args) + 1};

        for (size_t _ = 0; _ < auto_index; _++) ctx.next_arg_id();
        pos = formatter.parse(ctx) - fmt.begin();
        if (AutoIndex != 0) auto_index = ctx.next_arg_id();
        return make_tuple(pos, auto_index, formatter);
    }

    template <size_t Pos = 0, size_t AutoIndex = 0>
    static consteval auto compile() {
        constexpr size_t posl = fmt.find('{', Pos);
        constexpr size_t posr = fmt.find('}', Pos);

        if constexpr (posl == posr) {
            return make_tuple(Literal{fmt.substr(Pos)});
        } else if constexpr (posl < posr) {
            if constexpr (posl + 1 < fmt.size() && fmt[posl + 1] == '{') {
                return tuple_cat(
                    make_tuple(Literal{fmt.substr(Pos, posl - Pos + 1)}),
                    compile<posl + 2, AutoIndex>());
            } else {
                constexpr auto r1 = parseArgIdx<posl + 1, AutoIndex>();
                constexpr auto r2 =
                    parseFmtSpec<get<0>(r1), get<1>(r1), get<2>(r1)>();
                if constexpr (get<0>(r2) >= fmt.size() ||
                              fmt[get<0>(r2)] != '}') {
                    throw format_error("unclosed replacement");
                }
                return tuple_cat(
                    make_tuple(Literal{fmt.substr(Pos, posl - Pos)},
                               Replacement{get<2>(r1), get<2>(r2)}),
                    compile<get<0>(r2) + 1, get<1>(r2)>());
            }
        } else {
            if constexpr (posr + 1 < fmt.size() && fmt[posr + 1] == '}') {
                return tuple_cat(
                    make_tuple(Literal{fmt.substr(Pos, posr - Pos + 1)}),
                    compile<posr + 2, AutoIndex>());
            } else {
                throw format_error("unexpected '}'");
            }
        }
    }

    static auto format(auto&& args, auto& ctx) {
        auto out = ctx.out();
        auto f = [&]<size_t I>(integral_constant<size_t, I>) {
            constexpr auto& segment = get<I>(segments);
            if constexpr (is_convertible_v<decltype(segment), Literal>) {
                for (auto c: segment.content) *(out++) = c;
                return out;
            } else {
                decltype(auto) arg = get<segment.arg_id>(args);
                return segment.formatter.format(arg, ctx);
            }
        };
        auto helper = [&]<size_t... Is>(index_sequence<Is...>) {
            return ((out = f(integral_constant<size_t, Is>{})), ...);
        };
        return helper(make_index_sequence<num_segments>{});
    }

    static constexpr auto fmt = S.view();
    static constexpr auto segments = compile<>();
    static constexpr auto num_segments = tuple_size_v<decltype(segments)>;
};

template <typename Compiled, typename... Args>
struct Executor {
    tuple<Args&...> args;
};

template <size_t N>
inline constexpr auto fmt_string = [] {
    array<char, 64> s{};
    size_t pos = 0, sz = N;
    s[pos++] = '}';
    do {
        s[pos++] = '0' + sz % 10;
        sz /= 10;
    } while (sz);
    s[pos++] = '{';
    reverse(s.begin(), s.begin() + pos);
    return s;
}();

}  // namespace detail

template <FixedString S, typename... Args>
inline auto format_to(auto out, FormatString<S>, Args&&... args) {
    using Compiled = detail::CompiledFormatString<S, remove_cvref_t<Args>...>;
    format_to(out, detail::fmt_string<sizeof...(Args)>.data(), args...,
              detail::Executor<Compiled, Args...>{{args...}});
}

template <FixedString S, typename... Args>
inline auto format(FormatString<S>, Args&&... args) {
    string res{};
    format_to(back_inserter(res), FormatString<S>{}, args...);
    return res;
}

}  // namespace cp

template <typename Compiled, typename... Args>
struct std::formatter<cp::detail::Executor<Compiled, Args...>, char> {
    constexpr auto parse(auto& ctx) { return ctx.begin(); }
    constexpr auto format(auto&& arg, auto& ctx) const {
        return Compiled::format(arg.args, ctx);
    }
};

#pragma once
#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <concepts>
#include <cstddef>
#include <format>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#define CP_FASTIO_USE_BUF

namespace cp
{

constexpr size_t IN_BUF_SIZE = 1 << 21;
constexpr size_t OUT_BUF_SIZE = 1 << 21;

#ifndef CP_FORMAT_STRING
#define CP_FORMAT_STRING
template <size_t N>
struct FixedString: std::array<char, N> {
    consteval FixedString(const char (&str)[N]) {
        std::ranges::copy(str, this->data());
    }
    constexpr auto view() const {
        return std::string_view{this->data(), N - 1};
    }
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

class FastIOBase {
protected:
    FILE* _target;
#ifdef CP_FASTIO_USE_BUF
    char* _buf = nullptr;
    size_t _size = 0, _pos = -1;
#endif

public:
    FastIOBase(FILE* target): _target(target) {}
};

class FastInput final: FastIOBase {
private:
#if defined(CP_FASTIO_USE_BUF) && defined(__linux__)
    bool _is_mmap = false;
#endif

    struct BufIterator {
        using value_type = char;
        using difference_type = ptrdiff_t;

        [[gnu::always_inline]] BufIterator& operator++() {
#ifdef CP_FASTIO_USE_BUF
            c = (++t->_pos == t->_size ? t->sync() : t->_buf[t->_pos]);
#else
            c = fgetc(t->_target);
#endif
            return *this;
        }
        char operator*() { return c; }
        void operator++(int) { ++(*this); }
        void skipws() {
            while (c <= 32) ++(*this);
        }
        bool eof() { return c == EOF; }

        FastInput* t = nullptr;
#ifdef CP_FASTIO_USE_BUF
        char c = t ? (~t->_pos ? t->_buf[t->_pos] : t->sync()) : EOF;
#else
        char c = t ? std::fgetc(t->_target) : EOF;
        ~BufIterator() { std::ungetc(c, t->_target); }
#endif
    };

public:
    FastInput(FILE* target): FastIOBase(target) {
#ifdef CP_FASTIO_USE_BUF
#ifdef __linux__
        int fd = fileno(target);
        struct stat sb;
        if (fstat(fd, &sb) == 0 && S_ISREG(sb.st_mode) && sb.st_size > 0) {
            _buf =
                (char*)mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (_buf != MAP_FAILED) {
                _size = sb.st_size;
                _pos = 0;
                _is_mmap = true;
                return;
            }
        }
#endif
        _buf = new char[IN_BUF_SIZE];
        _size = IN_BUF_SIZE;
#endif
    }
#ifdef CP_FASTIO_USE_BUF
    char sync() {
        if (_is_mmap) return EOF;
        size_t read_sz = std::fread(_buf, 1, IN_BUF_SIZE, _target);
        _pos = 0;
        if (read_sz < IN_BUF_SIZE) _buf[read_sz] = EOF;
        return _buf[0];
    }
#endif

    operator bool() { return !eof(); }
    bool eof() { return BufIterator{this}.eof(); }

    template <std::integral T>
    std::optional<T> scan() {
        BufIterator it{this};
        bool neg = false;
        it.skipws();
        if constexpr (std::is_unsigned_v<T>) {
            if (*it == '-') neg = true, it++;
            else if (*it == '+') it++;
        }
        if (!std::isdigit(*it)) return std::nullopt;
        T res = 0;
        do res = res * 10 + (*it ^ 48), it++;
        while (std::isdigit(*it));
        return neg ? -res : res;
    }

    template <std::same_as<std::string> T>
    std::optional<T> scan() {
        BufIterator it{this};
        it.skipws();
        if (it.eof()) return std::nullopt;
        T res{};
        do res.push_back(*it), it++;
        while (std::isgraph(*it));
        return res;
    }

    template <std::floating_point T>
    std::optional<T> scan() {
        BufIterator it{this};
        T res = 0;
        bool neg = false, ok = false;
        it.skipws();
        if (*it == '-') neg = true, it++;
        else if (*it == '+') it++;
        if (std::isdigit(*it)) {
            ok = true;
            do res = res * 10 + (*it ^ 48), it++;
            while (std::isdigit(*it));
        }
        if (*it == '.' && std::isdigit(*(++it))) {
            ok = true;
            T mul = 0.1;
            do res += mul * (*it ^ 48), mul *= 0.1, it++;
            while (std::isdigit(*it));
        }
        return ok ? std::optional(neg ? -res : res) : std::nullopt;
    }

    template <typename... Args>
        requires(sizeof...(Args) > 1)
    std::optional<std::tuple<Args...>> scan() {
        std::tuple<std::optional<Args>...> tmp;
        auto try_scan = [this](auto&... r) {
            return ((r = scan<std::remove_cvref_t<decltype(r.value())>>()) &&
                    ...);
        };
        auto unwrap = [](auto&&... items) {
            return std::tuple(std::move(items).value()...);
        };
        if (std::apply(try_scan, tmp)) return std::apply(unwrap, tmp);
        return std::nullopt;
    }
};

class FastOutput final: FastIOBase {
private:
    static constexpr std::array<char, 200> lut = [] {
        std::array<char, 200> res;
        for (size_t i = 0; i < 100; i++) {
            res[i * 2] = i / 10 + '0';
            res[i * 2 + 1] = i % 10 + '0';
        }
        return res;
    }();

    struct BufIterator {
        using difference_type = ptrdiff_t;

        [[gnu::always_inline]] BufIterator& operator=(char c) {
#ifdef CP_FASTIO_USE_BUF
            t->_buf[t->_pos++] = c;
            if (t->_pos == t->_size) [[unlikely]] {
                t->flush();
            }
#else
            fputc(c, t->_target);
#endif
            return *this;
        }
        BufIterator& operator*() { return *this; }
        BufIterator& operator++() { return *this; }
        BufIterator& operator++(int) { return ++*this; }

        FastOutput* t;
    };

public:
    FastOutput(FILE* target): FastIOBase{target} {
#ifdef CP_FASTIO_USE_BUF
        _buf = new char[OUT_BUF_SIZE];
        _size = OUT_BUF_SIZE;
        _pos = 0;
    }
    ~FastOutput() { flush(); }

    void flush() {
        fwrite(_buf, 1, _pos, _target);
        _pos = 0;
#endif
    }

    template <FixedString S, typename... Args>
    void print(FormatString<S> fmt, Args&&... args) {
        using std::format_to;
        format_to(BufIterator{this}, fmt, args...);
    }

    void print(char c) { BufIterator{this} = c; }

    template <std::integral T>
    void print(T x) {
        using U = std::make_unsigned_t<T>;
        char stk[std::numeric_limits<T>::digits10], *top = stk;
        BufIterator it{this};
        U u = (x < 0 ? print('-'), -x : x);
        while (u >= 100) {
            int pos = u % 100 * 2;
            *(top++) = lut[pos + 1];
            *(top++) = lut[pos];
            u /= 100;
        }
        if (u >= 10) {
            *(top++) = lut[u * 2 + 1];
            *(top++) = lut[u * 2];
        } else {
            *(top++) = u | 48;
        }
        while (top > stk) it = *(--top);
    }

    template <std::floating_point T>
    void print(T x) {
        constexpr size_t S = sizeof(T) > 8 ? 64 : 32;
        char buf[S];
        auto [p, e] =
            std::to_chars(buf, buf + S, x, std::chars_format::general);
        if (e == std::errc{}) {
            BufIterator it{this};
            for (auto i = buf; i != p; i++) it = *i;
        }
    }

    template <std::convertible_to<std::string_view> T>
    void print(T&& x) {
        std::ranges::copy(std::string_view(x), BufIterator{this});
    }

    template <typename First, typename... Args>
        requires(sizeof...(Args) > 0)
    void print(First&& first, Args&&... args) {
        print(first), print(' '), print(args...);
    }

    template <typename... Args>
    void println(Args&&... args) {
        print(args...), print('\n');
    }
};

inline cp::FastInput qin(stdin);
inline cp::FastOutput qout(stdout);

}  // namespace cp

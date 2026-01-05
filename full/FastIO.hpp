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

using namespace std;
constexpr size_t IN_BUF_SIZE = 1 << 21;
constexpr size_t OUT_BUF_SIZE = 1 << 21;

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
        void skipws() {
            while (isspace(c)) ++(*this);
        }
        char operator*() { return c; }
        bool eof() { return c == EOF; }
        void operator++(int) { ++(*this); }

        FastInput* t = nullptr;
#ifdef CP_FASTIO_USE_BUF
        char c = t ? (~t->_pos ? t->_buf[t->_pos] : t->sync()) : EOF;
        ~BufIterator() { ungetc(c, t->_target); }
#else
        char c = t ? fgetc(t->_target) : EOF;
#endif
    };

public:
    FastInput(FILE* target = stdin): FastIOBase(target) {
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
        size_t read_sz = fread(_buf, 1, IN_BUF_SIZE, _target);
        _pos = 0;
        if (read_sz < IN_BUF_SIZE) _buf[read_sz] = EOF;
        return _buf[0];
    }
#endif

    operator bool() { return !eof(); }
    bool eof() { return BufIterator{this}.eof(); }

    template <integral T>
    optional<T> scan() {
        BufIterator it{this};
        bool neg = false;
        T res = 0;
        it.skipws();
        if (*it == '-') neg = true, it++;
        else if (*it == '+') it++;
        if (!isdigit(*it)) return nullopt;
        do {
            res = res * 10 + (*it ^ 48);
            it++;
        } while (isdigit(*it));
        return neg ? -res : res;
    }

    template <same_as<string> T>
    optional<T> scan() {
        BufIterator it{this};
        it.skipws();
        if (it.eof()) return nullopt;
        T res{};
        do {
            res.push_back(*it);
            it++;
        } while (!isspace(*it));
        return res;
    }

    template <floating_point T>
    optional<T> scan() {
        BufIterator it{this};
        bool neg = false, ok = false;
        double res = 0.0;
        it.skipws();
        if (*it == '-') neg = true, it++;
        else if (*it == '+') it++;
        if (isdigit(*it)) {
            ok = true;
            do {
                res = res * 10 + (*it ^ 48);
                it++;
            } while (isdigit(*it));
        }
        if (*it == '.' && isdigit(*(++it))) {
            ok = false;
            double mul = 0.1;
            do {
                res += mul * (*it ^ 48);
                it++;
                mul *= 0.1;
            } while (isdigit(*it));
        }
        return ok ? optional(neg ? -res : res) : nullopt;
    }

    template <typename... Args>
        requires(sizeof...(Args) > 1)
    optional<tuple<Args...>> scan() {
        tuple<optional<Args>...> tmp;
        auto try_scan = [this](auto&... items) {
            return ((items = scan<remove_cvref_t<decltype(items.value())>>()) &&
                    ...);
        };
        auto unwrap = [](auto&&... items) {
            return make_tuple(std::move(items).value()...);
        };
        return apply(try_scan, tmp) ? optional(apply(unwrap, tmp)) : nullopt;
    }
};

class FastOutput final: FastIOBase {
private:
    static constexpr array<char, 200> lut = [] {
        array<char, 200> res;
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
    FastOutput(FILE* target = stdout): FastIOBase{target} {
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
    void print(FormatString<S>, Args&&... args) {
        format_to(BufIterator{this}, S, args...);
    }

    template <FixedString S, typename... Args>
    void println(FormatString<S>, Args&&... args) {
        format_to(BufIterator{this}, S, args...);
        print('\n');
    }

    void print(char c) { BufIterator{this} = c; }

    template <integral T>
    void print(T x) {
        using U = make_unsigned_t<T>;
        char stk[numeric_limits<T>::digits10], *top = stk;
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

    template <floating_point T>
    void print(T x) {
        constexpr size_t S = sizeof(T) > 8 ? 64 : 32;
        char buf[S];
        auto [p, e] = to_chars(buf, buf + S, x, chars_format::general);
        if (e == errc{}) {
            BufIterator it{this};
            for (auto i = buf; i != p; i++) it = *i;
        }
    }

    template <convertible_to<string_view> T>
    void print(T&& x) {
        BufIterator it{this};
        for (auto& c: string_view(x)) it = c;
    }

    template <typename First, typename... Args>
        requires(sizeof...(Args) > 0)
    void print(First&& first, Args&&... args) {
        print(first);
        ((print(' '), print(args)), ...);
    }

    template <typename... Args>
    void println(Args&&... args) {
        print(args...);
        print('\n');
    }
};

}  // namespace cp

#pragma once
#include <algorithm>
#include <cctype>
#include <charconv>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <format>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#endif

// #define CP_FASTIO_ACCELERATE

namespace cp
{

constexpr size_t IN_BUF_SIZE = 1 << 16;
constexpr size_t OUT_BUF_SIZE = 1 << 16;

#ifndef CP_FORMAT_STRING
#define CP_FORMAT_STRING
template <size_t N>
struct FixedString {
    char s[N];
    consteval FixedString(const char (&str)[N]) { std::ranges::copy(str, s); }
    constexpr auto view() const { return std::string_view{s, N - 1}; }
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
    char* _buf = nullptr;
    char* _end = nullptr;
    char* _pos = nullptr;

public:
    FastIOBase(FILE* target): _target(target) {}
};

class FastInput final: FastIOBase {
private:
    struct ReadIterator {
        [[gnu::always_inline]] auto& operator++() {
            return (++_t->_pos == _t->_end ? _t->sync() : void{}), *this;
        }
        void operator++(int) { ++*this; }
        char operator*() const { return *_t->_pos; }
        bool eof() { return _t->_pos == _t->_end; }
        void skipws() {
            while (!eof() && *_t->_pos <= 32) ++*this;
        }
        ReadIterator(FastInput* t): _t(t) {
            if (_t->_pos == _t->_end) _t->sync();
        }

        FastInput* _t = nullptr;
    };

    bool _eof = false;

public:
    FastInput(FILE* target): FastIOBase(target) {
#ifdef __linux__
        int fd = fileno(target);
        struct stat sb;
        if (fstat(fd, &sb) == 0 && S_ISREG(sb.st_mode) && sb.st_size > 0) {
            _pos = _buf =
                (char*)mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (_buf != MAP_FAILED) {
                _end = _buf + sb.st_size;
                _eof = true;
                return;
            }
        }
#endif
        _buf = new char[IN_BUF_SIZE + 1];
        *(_pos = _end = _buf + IN_BUF_SIZE) = EOF;
    }
    void sync() {
        if (_eof) return;
#ifdef CP_FASTIO_ACCELERATE
        _end = _buf + std::fread(_buf, 1, IN_BUF_SIZE, _target);
#else
        std::fgets(_buf, IN_BUF_SIZE, _target);
        _end = _buf + std::strlen(_buf);
#endif
        _eof = std::feof(_target);
        _pos = _buf, *_end = EOF;
    }

    operator bool() { return !eof(); }
    bool eof() { return _pos == _end; }

    template <std::integral T>
    std::optional<T> scan() {
        ReadIterator it(this);
        bool neg = false;
        it.skipws();
        if constexpr (std::is_signed_v<T>) {
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
        ReadIterator it(this);
        it.skipws();
        if (it.eof()) return std::nullopt;
        T res{};
        do res.push_back(*it), it++;
        while (std::isgraph(*it));
        return res;
    }

    template <std::floating_point T>
    std::optional<T> scan() {
        ReadIterator it(this);
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
    auto scan() {
        using RetT = std::optional<std::tuple<Args...>>;
        auto helper = [this](auto& self, auto&&... args) -> RetT {
            constexpr auto I = sizeof...(args);
            if constexpr (I == sizeof...(Args)) {
                return std::tuple(std::forward<decltype(args)>(args)...);
            } else {
                auto val =
                    this->scan<std::tuple_element_t<I, std::tuple<Args...>>>();
                if (!val) return std::nullopt;
                return self(
                    self, std::forward<decltype(args)>(args)...,
                    std::move(val).value()
                );
            }
        };
        return helper(helper);
    }
};

class FastOutput final: FastIOBase {
private:
    struct WriteIterator {
        using difference_type = ptrdiff_t;

        [[gnu::always_inline]] WriteIterator& operator=(char c) {
            if (_t->_pos == _t->_end) _t->flush();
            *_t->_pos++ = c;
#ifndef CP_FASTIO_ACCELERATE
            if (c == '\n') _t->flush();
#endif
            return *this;
        }
        WriteIterator& operator++() { return *this; }
        auto& operator++(int) { return *this; }
        auto& operator*() { return *this; }

        FastOutput* _t;
    };

    void reserve(size_t size) {
        if (_pos + size >= _end) flush();
    }

public:
    FastOutput(FILE* target): FastIOBase{target} {
        _pos = _buf = new char[OUT_BUF_SIZE];
        _end = _buf + OUT_BUF_SIZE;
    }
    ~FastOutput() { flush(); }

    void flush() {
        fwrite(_buf, 1, _pos - _buf, _target);
        fflush(_target);
        _pos = _buf;
    }

    void print(char c) { WriteIterator{this} = c; }

    template <FixedString S, typename... Args>
    void print(FormatString<S> fmt, Args&&... args) {
        using std::format_to;
        format_to(WriteIterator{this}, fmt, std::forward<Args>(args)...);
    }

    template <typename T>
        requires std::integral<T> || std::floating_point<T>
    void print(T x) {
        reserve(64);
        _pos = std::to_chars(_pos, _end, x).ptr;
    }

    template <std::convertible_to<std::string_view> T>
    void print(T&& x) {
        auto s = std::string_view(x);
        reserve(s.size());
        _pos = std::ranges::copy(s, _pos).out;
    }

    template <typename First, typename... Args>
        requires(sizeof...(Args) > 0)
    void print(First&& first, Args&&... args) {
        print(first), print(' '), print(args...);
    }

    template <typename... Args>
    void println(Args&&... args) {
        if constexpr (sizeof...(Args) > 0) print(args...);
        print('\n');
    }

    template <typename... Args>
    void printsp(Args&&... args) {
        if constexpr (sizeof...(Args) > 0) print(args...);
        print(' ');
    }
};

inline cp::FastInput qin(stdin);
inline cp::FastOutput qout(stdout);

}  // namespace cp

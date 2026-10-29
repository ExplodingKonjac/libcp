#pragma once

#include <charconv>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>

#include "def.hpp"

namespace acm
{

struct FastIn {
    static constexpr usize SZ = 1 << 16;

    FILE* f;
    char buf[SZ];
    usize p = 0, n = 0;

    explicit FastIn(FILE* f = stdin): f(f) {}

    int gc() {
        if (p == n) {
            n = fread(buf, 1, SZ, f);
            p = 0;
            if (!n) return EOF;
        }
        return (unsigned char)buf[p++];
    }

    void skip() {
        int c = gc();
        while (c != EOF && c <= ' ') c = gc();
        if (c != EOF) --p;
    }

    template <typename T>
    T read() {
        if constexpr (std::is_same_v<T, std::string>) {
            skip();
            std::string s;
            for (int c = gc(); c != EOF && c > ' '; c = gc())
                s.push_back(char(c));
            return s;
        } else if constexpr (std::is_same_v<T, char>) {
            skip();
            int c = gc();
            return c == EOF ? char(0) : char(c);
        } else if constexpr (std::is_same_v<T, bool>) {
            return read<int>() != 0;
        } else if constexpr (std::is_integral_v<T>) {
            skip();
            int c = gc();
            bool neg = false;
            if constexpr (std::is_signed_v<T>) {
                if (c == '-') neg = true, c = gc();
                else if (c == '+') c = gc();
            }
            using U = std::make_unsigned_t<T>;
            U x = 0;
            while (c >= '0' && c <= '9') {
                x = U(x * 10 + (c - '0'));
                c = gc();
            }
            if (c != EOF) --p;
            if constexpr (std::is_signed_v<T>) return neg ? T(-x) : T(x);
            else return T(x);
        } else if constexpr (std::is_floating_point_v<T>) {
            skip();
            int c = gc();
            bool neg = false;
            if (c == '-') neg = true, c = gc();
            else if (c == '+') c = gc();
            long double x = 0;
            while (c >= '0' && c <= '9') {
                x = x * 10 + (c - '0');
                c = gc();
            }
            if (c == '.') {
                c = gc();
                long double base = 0.1L;
                while (c >= '0' && c <= '9') {
                    x += base * (c - '0');
                    base *= 0.1L;
                    c = gc();
                }
            }
            if (c == 'e' || c == 'E') {
                c = gc();
                bool eneg = false;
                if (c == '-') eneg = true, c = gc();
                else if (c == '+') c = gc();
                int e = 0;
                while (c >= '0' && c <= '9') {
                    e = e * 10 + (c - '0');
                    c = gc();
                }
                long double pw = 1;
                for (int i = 0; i < e; i++) pw *= 10;
                x = eneg ? x / pw : x * pw;
            }
            if (c != EOF) --p;
            return T(neg ? -x : x);
        } else {
            static_assert(sizeof(T) == 0, "unsupported read type");
        }
    }

    template <typename T, typename... U>
    void read(T& x, U&... y) {
        x = read<T>();
        if constexpr (sizeof...(U)) read(y...);
    }

    std::string read_string() { return read<std::string>(); }
};

struct FastOut {
    static constexpr usize SZ = 1 << 16;

    FILE* f;
    char buf[SZ];
    usize p = 0;

    explicit FastOut(FILE* f = stdout): f(f) {}
    FastOut(const FastOut&) = delete;
    FastOut& operator=(const FastOut&) = delete;
    ~FastOut() { flush(); }

    void flush() {
        if (p) {
            fwrite(buf, 1, p, f);
            p = 0;
        }
        fflush(f);
    }

    void put(char c) {
        if (p == SZ) flush();
        buf[p++] = c;
    }

    void write(const char* s, usize n) {
        if (n >= SZ) {
            flush();
            fwrite(s, 1, n, f);
            return;
        }
        if (p + n > SZ) flush();
        memcpy(buf + p, s, n);
        p += n;
    }

    void print(char c) { put(c); }
    void print(const char* s) { write(s, strlen(s)); }
    void print(std::string_view s) { write(s.data(), s.size()); }
    void print(const std::string& s) { write(s.data(), s.size()); }
    void print(bool x) { print(x ? "1" : "0"); }

    template <typename T>
        requires std::is_integral_v<T> && (!std::is_same_v<T, bool>)
    void print(T x) {
        char tmp[64];
        auto r = std::to_chars(tmp, tmp + 64, x);
        write(tmp, usize(r.ptr - tmp));
    }

    template <typename T>
        requires std::is_floating_point_v<T>
    void print(T x) {
        char tmp[128];
        int n;
        if constexpr (std::is_same_v<T, float>)
            n = snprintf(tmp, sizeof(tmp), "%.9g", double(x));
        else if constexpr (std::is_same_v<T, double>)
            n = snprintf(tmp, sizeof(tmp), "%.17g", x);
        else
            n = snprintf(tmp, sizeof(tmp), "%.21Lg", x);
        write(tmp, usize(n));
    }

    template <typename T, typename U, typename... V>
    void print(const T& x, const U& y, const V&... z) {
        print(x);
        put(' ');
        print(y, z...);
    }

    template <typename... T>
    void println(const T&... x) {
        if constexpr (sizeof...(T)) print(x...);
        put('\n');
    }
};

inline FastIn qin;
inline FastOut qout;

}  // namespace acm

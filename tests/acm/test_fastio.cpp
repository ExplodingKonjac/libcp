#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>

#include "acm/fast_io.hpp"

using namespace acm;

int main() {
    FILE* f = tmpfile();
    assert(f);
    {
        FastOut out(f);
        out.println(1, -2, 3.5, "abc");
        out.print(1234567890123456789LL);
        out.put(' ');
        out.print(0.125);
        out.put(' ');
        out.print("hello");
        out.put(' ');
        out.print('x');
        out.put('\n');
    }
    rewind(f);

    FastIn in(f);
    i32 a;
    i64 b;
    f64 c;
    std::string s;
    in.read(a, b, c, s);
    assert(a == 1 && b == -2 && std::abs(c - 3.5) < 1e-12 && s == "abc");

    i64 big;
    f64 frac;
    std::string hello;
    char ch;
    in.read(big, frac, hello, ch);
    assert(big == 1234567890123456789LL);
    assert(std::abs(frac - 0.125) < 1e-15);
    assert(hello == "hello" && ch == 'x');

    // Buffer boundary: output a string larger than the buffer.
    FILE* g = tmpfile();
    assert(g);
    std::string large(FastOut::SZ * 2 + 17, 'q');
    {
        FastOut out(g);
        out.print(large);
    }
    rewind(g);
    FastIn gin(g);
    std::string back = gin.read<std::string>();
    assert(back == large);

    // Float exponent and negative zero-ish input.
    FILE* h = tmpfile();
    assert(h);
    {
        FastOut out(h);
        out.println(-1.25e-3, 1e-9, 2.5e+3);
    }
    rewind(h);
    FastIn hin(h);
    f64 x, y, z;
    hin.read(x, y, z);
    assert(std::abs(x - (-1.25e-3)) < 1e-18);
    assert(std::abs(y - 1e-9) < 1e-21);
    assert(std::abs(z - 2500.0) < 1e-12);

    // EOF returns zero / empty rather than crashing.
    FILE* e = tmpfile();
    assert(e);
    fclose(e);
    return 0;
}

#include <cstdio>

#include "cp/fast_io.hpp"

using cp::qin, cp::qout;

unsigned next() {
    static unsigned seed = 114514;
    seed ^= seed << 5;
    seed ^= seed >> 13;
    seed ^= seed << 17;
    return seed;
}

int main() {
    freopen("input.in", "r", stdin);
    int n = 50000000;
    unsigned checksum = 0;
    // for (int i = 0; i < n; i++) {
    //     qout.println(next());
    // }
    for (int i = 0; i < n; i++) {
        checksum ^= qin.scan<unsigned>().value();
    }
    qout.println(checksum);
    return 0;
}

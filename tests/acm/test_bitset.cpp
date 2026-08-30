#include <bitset>
#include <cassert>
#include <random>

#include "acm/bitset.hpp"

int main() {
    acm::Bitset<130> a;
    std::bitset<130> b;
    std::mt19937 rng(1);
    for (int i = 0; i < 130; i++)
        if (rng() & 1) a.set_bit(i), b.set(i);
    assert(a.count() == b.count());
    a.flip_range(17, 80);
    for (int i = 17; i < 97; i++) b.flip(i);
    for (int i = 0; i < 130; i++) assert(a[i] == b[i]);
    auto c = a << 19;
    b <<= 19;
    for (int i = 0; i < 130; i++) assert(c[i] == b[i]);
    a = c;

    for (int tc = 0; tc < 1000; tc++) {
        int p = rng() % 150, n = rng() % 150, op = rng() % 5;
        if (op < 3) {
            if (op == 0) a.set_range(p, n);
            if (op == 1) a.unset_range(p, n);
            if (op == 2) a.flip_range(p, n);
            for (int i = p; i < std::min(130, p + n); i++) {
                if (op == 0) b.set(i);
                if (op == 1) b.reset(i);
                if (op == 2) b.flip(i);
            }
        } else {
            int s = rng() % 150;
            if (op == 3) a <<= s, b <<= s;
            else a >>= s, b >>= s;
        }
        for (int i = 0; i < 130; i++) assert(a[i] == b[i]);
    }
}

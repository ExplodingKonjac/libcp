#include <cstdint>

struct Line {
    int k, b;

    int operator()(int x) const { return k * x + b; }
};

int n;
Line t[1 << 20];

void update(const Line& ln, int i = 1, int l = 1, int r = n) {
    if (ln(l) >= t[i](l) && ln(r) >= t[i](r)) return;
    if (ln(l) <= t[i](l) && ln(r) <= t[i](r)) t[i] = ln;
    else {
        int mid = (l + r) / 2;
        update(ln, i << 1, l, mid);
        update(ln, i << 1 | 1, mid + 1, r);
    }
}
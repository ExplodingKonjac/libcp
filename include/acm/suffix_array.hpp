#pragma once

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "def.hpp"

namespace acm
{

class SuffixArray {
private:
    std::vector<usize> a, b, h;

public:
    template <typename R>
    explicit SuffixArray(const R& s) {
        usize n = s.size();
        a.resize(n);
        b.resize(n);
        h.assign(n, 0);
        if (!n) return;

        std::iota(a.begin(), a.end(), usize(0));
        std::sort(a.begin(), a.end(), [&](usize x, usize y) {
            return s[x] < s[y];
        });

        usize m = 1;
        for (usize i = 0; i < n; i++) {
            if (i && s[a[i - 1]] != s[a[i]]) m++;
            b[a[i]] = m - 1;
        }

        std::vector<usize> old(n), tmp(n), bk(n);
        for (usize w = 1; m < n; w <<= 1) {
            if (w >= n) break;
            auto eq = [&](usize x, usize y) {
                return old[x] == old[y] &&
                    ((x + w < n && y + w < n && old[x + w] == old[y + w]) ||
                     (x + w >= n && y + w >= n));
            };
            usize tot = 0;
            for (usize i = n - w; i < n; i++) tmp[tot++] = i;
            for (usize i = 0; i < n; i++)
                if (a[i] >= w) tmp[tot++] = a[i] - w;
            bk.assign(m, 0);
            for (usize i = 0; i < n; i++) bk[b[i]]++;
            for (usize i = 1; i < m; i++) bk[i] += bk[i - 1];
            for (usize i = n; i > 0; i--)
                a[--bk[b[tmp[i - 1]]]] = tmp[i - 1];
            old.swap(b);
            m = 1;
            for (usize i = 0; i < n; i++) {
                if (i && !eq(a[i], a[i - 1])) m++;
                b[a[i]] = m - 1;
            }
        }

        for (usize i = 0, j = 0; i < n; i++) {
            if (j) j--;
            if (b[i] == 0) continue;
            usize k = a[b[i] - 1];
            while (i + j < n && k + j < n && s[i + j] == s[k + j]) j++;
            h[b[i]] = j;
        }
    }

    usize size() const { return a.size(); }
    usize sa(usize i) const { return a[i]; }
    usize rk(usize i) const { return b[i]; }
    usize height(usize i) const { return h[i]; }
};

}  // namespace acm

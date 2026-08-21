#pragma once

#include <algorithm>
#include <concepts>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <vector>

#include "cp/def.hpp"

namespace cp
{

// Implements Suffix Array. Construct from a range r[0..n-1], and computes:
//   sa(i): the i-th suffix in ascending order (0-indexed)
//   rk(i): number of suffices LESS than r[i..n-1]
//   hgt(i): LCP of r[sa(i-1)..n-1] and r[sa(i)..n-1]
template <typename Symbol>
class SuffixArray {
private:
    std::vector<usize> _sa, _rk, _hgt;

public:
    template <std::ranges::random_access_range R>
        requires std::convertible_to<std::ranges::range_value_t<R>, Symbol>
    SuffixArray(R&& r) {
        usize n = std::ranges::size(r), m = 1;
        auto s = [&](usize i) { return *(r.begin() + i); };
        _sa.resize(n, 0);
        _rk.resize(n, 0);
        _hgt.resize(n, 0);
        std::ranges::iota(_sa, 0);
        std::ranges::sort(_sa, {}, [&](usize x) { return s(x); });
        for (usize i = 0; i < n; i++) {
            if (i > 0 && s(_sa[i - 1]) != s(_sa[i])) m++;
            _rk[_sa[i]] = m - 1;
        }
        std::vector<usize> old(n), tmp(n), bk(n);
        for (usize w = 1; m < n; w <<= 1) {
            auto eq = [&](usize x, usize y) {
                return old[x] == old[y] &&
                    ((x + w < n && y + w < n && old[x + w] == old[y + w]) ||
                     (x + w >= n && y + w >= n));
            };
            usize tot = 0;
            for (usize i = n - w; i < n; i++) tmp[tot++] = i;
            for (usize i = 0; i < n; i++)
                if (_sa[i] >= w) tmp[tot++] = _sa[i] - w;
            bk.assign(m, 0);
            for (usize i = 0; i < n; i++) bk[_rk[i]]++;
            for (usize i = 1; i < m; i++) bk[i] += bk[i - 1];
            for (usize i = n; i > 0; i--)
                _sa[--bk[_rk[tmp[i - 1]]]] = tmp[i - 1];
            old.swap(_rk);
            m = 1;
            for (usize i = 0; i < n; i++) {
                if (i > 0 && !eq(_sa[i], _sa[i - 1])) m++;
                _rk[_sa[i]] = m - 1;
            }
        }
        for (usize i = 0, j = 0; i < n; i++) {
            if (j) j--;
            if (_rk[i] == 0) continue;
            auto lst = _sa[_rk[i] - 1];
            while (i + j < n && lst + j < n && s(i + j) == s(lst + j)) j++;
            _hgt[_rk[i]] = j;
        }
    }

    usize sa(usize i) const { return _sa[i]; }
    usize rk(usize i) const { return _rk[i]; }
    usize height(usize i) const { return _hgt[i]; }
};

template <std::ranges::random_access_range R>
SuffixArray(R&&)
    -> SuffixArray<std::remove_cvref_t<std::ranges::range_value_t<R>>>;

}  // namespace cp
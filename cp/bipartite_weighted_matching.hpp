#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "cp/def.hpp"

namespace cp
{

template <typename T>
class BipartiteWeightedMatching {
private:
    static constexpr auto inf = [] {
        if constexpr (std::is_integral_v<T>) {
            if constexpr (sizeof(T) < 8) return i64(1) << 60;
            else return i128(1) << 120;
        } else {
            return std::numeric_limits<T>::infinity();
        }
    }();
    using calc_t = std::remove_cv_t<decltype(inf)>;

    usize _n, _m;
    std::vector<std::vector<T>> _w;

public:
    static constexpr usize npos = -1;
    static constexpr T no_edge = std::numeric_limits<T>::lowest();

    BipartiteWeightedMatching(usize n, usize m):
        _n{n}, _m{m}, _w(n, std::vector<T>(m, no_edge)) {}

    usize size_l() const { return _n; }
    usize size_r() const { return _m; }

    void add_edge(usize u, usize v, T w) { _w[u][v] = std::max(_w[u][v], w); }
    void set_edge(usize u, usize v, T w) { _w[u][v] = w; }
    T get_edge(usize u, usize v) const { return _w[u][v]; }

    std::optional<std::pair<T, std::vector<usize>>> max_weighted_matching() {
        if (_n > _m) return std::nullopt;

        std::vector<usize> p(_m + 1), way(_m + 1);
        std::vector<calc_t> lx(_n + 1), ly(_m + 1);

        for (usize i = 1; i <= _n; i++) {
            lx[i] = calc_t(no_edge);
            for (usize j = 1; j <= _m; j++)
                if (_w[i - 1][j - 1] != no_edge)
                    lx[i] = std::max(lx[i], calc_t(_w[i - 1][j - 1]));
            if (lx[i] == no_edge) return std::nullopt;
        }
        std::vector<calc_t> slack;
        std::vector<u8> used;
        for (usize s = 1; s <= _n; s++) {
            p[0] = s;
            slack.assign(_m + 1, inf);
            used.assign(_m + 1, false);
            usize u = 0;
            do {
                usize i = p[u], v = npos;
                calc_t delta = inf;
                used[u] = true;
                for (usize j = 1; j <= _m; ++j) {
                    if (used[j]) continue;
                    if (_w[i - 1][j - 1] != no_edge) {
                        calc_t cur = lx[i] +
                            ly[j] -
                            static_cast<calc_t>(_w[i - 1][j - 1]);
                        if (cur < slack[j]) {
                            slack[j] = cur;
                            way[j] = u;
                        }
                    }
                    if (slack[j] < delta) {
                        delta = slack[j];
                        v = j;
                    }
                }
                if (delta == inf) return std::nullopt;
                for (usize j = 0; j <= _m; ++j) {
                    if (used[j]) {
                        lx[p[j]] -= delta;
                        ly[j] += delta;
                    } else if (j && slack[j] != inf) {
                        slack[j] -= delta;
                    }
                }
                u = v;
            } while (p[u] != 0);
            do {
                usize v = way[u];
                p[u] = p[v];
                u = v;
            } while (u);
        }
        T result{};
        std::vector<usize> match(_n);
        for (usize j = 1; j <= _m; ++j) {
            if (p[j]) {
                usize i = p[j] - 1;
                match[i] = j - 1;
                result += _w[i][j - 1];
            }
        }
        return std::pair{result, std::move(match)};
    }
};

}  // namespace cp

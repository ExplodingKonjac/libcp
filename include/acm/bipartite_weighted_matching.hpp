#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "def.hpp"

namespace acm
{

template <typename T>
class BipartiteWeightedMatching {
    static_assert(std::is_signed_v<T> || std::is_floating_point_v<T>);

    using Calc =
        std::conditional_t<std::is_floating_point_v<T>, long double, i128>;

    static constexpr Calc inf() {
        if constexpr (std::is_floating_point_v<T>)
            return std::numeric_limits<Calc>::infinity();
        else return Calc(1) << 120;
    }
    static constexpr Calc INF = inf();

    usize n, m;
    std::vector<std::vector<T>> w;

public:
    static constexpr usize npos = usize(-1);
    static constexpr T no_edge = std::numeric_limits<T>::lowest();

    BipartiteWeightedMatching(usize n, usize m):
        n(n), m(m), w(n, std::vector<T>(m, no_edge)) {}

    usize size_l() const { return n; }
    usize size_r() const { return m; }

    void add_edge(usize u, usize v, T x) { w[u][v] = std::max(w[u][v], x); }
    void set_edge(usize u, usize v, T x) { w[u][v] = x; }
    T get_edge(usize u, usize v) const { return w[u][v]; }

    std::optional<std::pair<T, std::vector<usize>>> max_weighted_matching() {
        if (n > m) return std::nullopt;

        std::vector<usize> p(m + 1), way(m + 1);
        std::vector<Calc> lx(n + 1), ly(m + 1);
        for (usize i = 1; i <= n; i++) {
            lx[i] = Calc(no_edge);
            for (usize j = 1; j <= m; j++)
                if (w[i - 1][j - 1] != no_edge)
                    lx[i] = std::max(lx[i], Calc(w[i - 1][j - 1]));
            if (lx[i] == Calc(no_edge)) return std::nullopt;
        }

        std::vector<Calc> slack;
        std::vector<u8> used;
        for (usize s = 1; s <= n; s++) {
            p[0] = s;
            slack.assign(m + 1, INF);
            used.assign(m + 1, false);
            usize u = 0;
            do {
                usize i = p[u], v = npos;
                Calc delta = INF;
                used[u] = true;
                for (usize j = 1; j <= m; j++) {
                    if (used[j]) continue;
                    if (w[i - 1][j - 1] != no_edge) {
                        Calc cur =
                            lx[i] + ly[j] - Calc(w[i - 1][j - 1]);
                        if (cur < slack[j]) slack[j] = cur, way[j] = u;
                    }
                    if (slack[j] < delta) delta = slack[j], v = j;
                }
                if (delta == INF) return std::nullopt;
                for (usize j = 0; j <= m; j++) {
                    if (used[j]) lx[p[j]] -= delta, ly[j] += delta;
                    else if (j && slack[j] != INF) slack[j] -= delta;
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
        std::vector<usize> match(n);
        for (usize j = 1; j <= m; j++) {
            if (!p[j]) continue;
            usize i = p[j] - 1;
            match[i] = j - 1;
            result += w[i][j - 1];
        }
        return std::pair<T, std::vector<usize>>{result, std::move(match)};
    }
};

}  // namespace acm

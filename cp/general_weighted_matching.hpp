#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "cp/def.hpp"

namespace cp
{

template <typename T>
    requires(std::signed_integral<T> || std::floating_point<T>)
class GeneralWeightedMatching {
private:
    using calc_t = std::conditional_t<
        std::floating_point<T>,
        long double,
        std::conditional_t<(sizeof(T) < 8), i64, i128>
    >;

    struct Edge {
        usize u{}, v{};
        calc_t w{};
    };

    usize _n;
    std::vector<std::vector<T>> _w;

public:
    static constexpr usize npos = -1;
    static constexpr T no_edge = std::numeric_limits<T>::lowest();

    explicit GeneralWeightedMatching(usize n):
        _n{n}, _w(n, std::vector<T>(n, no_edge)) {}

    usize size() const { return _n; }

    void add_edge(usize u, usize v, T w) {
        _w[u][v] = _w[v][u] = std::max(_w[u][v], w);
    }

    void set_edge(usize u, usize v, T w) { _w[u][v] = _w[v][u] = w; }
    T get_edge(usize u, usize v) const { return _w[u][v]; }

    std::optional<std::pair<T, std::vector<usize>>> max_weighted_matching() {
        if (_n % 2) return std::nullopt;
        if (!_n) return std::pair{T{}, std::vector<usize>{}};

        const usize n = _n;
        usize node_count = n, visit_id = 0;
        calc_t eps{};
        std::vector g(2 * n + 1, std::vector<Edge>(2 * n + 1));
        std::vector<usize> match(2 * n + 1), slack(2 * n + 1);
        std::vector<usize> rep(2 * n + 1), par(2 * n + 1);
        std::vector<usize> visit(2 * n + 1);
        // Alternating-forest state: -1 outside, 0 outer, 1 inner.
        std::vector<int> state(2 * n + 1);
        std::vector<calc_t> label(2 * n + 1);
        std::vector from(2 * n + 1, std::vector<usize>(n + 1));
        std::vector<std::vector<usize>> blossom(2 * n + 1);
        std::vector<usize> queue;
        usize queue_head = 0;

        bool has_edge = false;
        calc_t minimum{}, maximum{};
        for (usize u = 0; u < n; ++u) {
            for (usize v = u + 1; v < n; ++v) {
                if (_w[u][v] == no_edge) continue;
                calc_t w = static_cast<calc_t>(_w[u][v]);
                if (!has_edge) minimum = maximum = w;
                else {
                    minimum = std::min(minimum, w);
                    maximum = std::max(maximum, w);
                }
                has_edge = true;
            }
        }
        calc_t range = has_edge ? maximum - minimum : 0;
        // One more matched edge must dominate every possible weight gain.
        calc_t bonus = (range + 1) * static_cast<calc_t>(n / 2 + 1);
        calc_t largest = 0;
        for (usize u = 1; u <= n; ++u) {
            rep[u] = u;
            from[u][u] = u;
            for (usize v = u + 1; v <= n; ++v) {
                if (_w[u - 1][v - 1] == no_edge) continue;
                calc_t transformed =
                    static_cast<calc_t>(_w[u - 1][v - 1]) - minimum + bonus;
                g[u][v] = {u, v, transformed};
                g[v][u] = {v, u, transformed};
                largest = std::max(largest, transformed);
            }
        }
        std::fill(label.begin() + 1, label.begin() + n + 1, largest);
        if constexpr (std::floating_point<T>) {
            eps = static_cast<calc_t>(std::numeric_limits<T>::epsilon()) *
                64 *
                std::max<calc_t>(1, largest);
        }

        auto delta = [&](const Edge& edge) {
            return label[edge.u] + label[edge.v] - edge.w * 2;
        };
        auto less = [&](calc_t a, calc_t b) {
            if constexpr (std::floating_point<T>) return a < b - eps;
            else return a < b;
        };
        auto zero = [&](calc_t x) {
            if constexpr (std::floating_point<T>) return std::abs(x) <= eps;
            else return x == 0;
        };
        auto normalize = [&](calc_t& x) {
            if constexpr (std::floating_point<T>) {
                if (zero(x)) x = 0;
            }
        };
        auto update_slack = [&](usize u, usize x) {
            if (!slack[x] || less(delta(g[u][x]), delta(g[slack[x]][x])))
                slack[x] = u;
        };
        auto set_slack = [&](usize x) {
            slack[x] = 0;
            for (usize u = 1; u <= n; ++u) {
                if (g[u][x].w > 0 && rep[u] != x && state[rep[u]] == 0)
                    update_slack(u, x);
            }
        };
        auto queue_push = [&](auto&& self, usize x) -> void {
            if (x <= n) queue.push_back(x);
            else
                for (usize v: blossom[x]) self(self, v);
        };
        auto set_representative = [&](auto&& self, usize x, usize b) -> void {
            rep[x] = b;
            if (x > n)
                for (usize v: blossom[x]) self(self, v, b);
        };
        auto rotate_blossom = [&](usize b, usize entry) {
            usize pos = std::find(blossom[b].begin(), blossom[b].end(), entry) -
                blossom[b].begin();
            if (pos % 2) {
                std::reverse(blossom[b].begin() + 1, blossom[b].end());
                return blossom[b].size() - pos;
            }
            return pos;
        };
        auto set_match = [&](auto&& self, usize u, usize v) -> void {
            match[u] = g[u][v].v;
            if (u <= n) return;
            usize entry = from[u][g[u][v].u];
            usize pos = rotate_blossom(u, entry);
            for (usize i = 0; i < pos; ++i)
                self(self, blossom[u][i], blossom[u][i ^ 1]);
            self(self, entry, v);
            std::rotate(
                blossom[u].begin(), blossom[u].begin() + pos, blossom[u].end()
            );
        };
        auto augment = [&](usize u, usize v) {
            while (true) {
                usize next = rep[match[u]];
                set_match(set_match, u, v);
                if (!next) return;
                set_match(set_match, next, rep[par[next]]);
                u = rep[par[next]];
                v = next;
            }
        };
        auto lca = [&](usize u, usize v) {
            ++visit_id;
            for (; u || v; std::swap(u, v)) {
                if (!u) continue;
                if (visit[u] == visit_id) return u;
                visit[u] = visit_id;
                u = rep[match[u]];
                if (u) u = rep[par[u]];
            }
            return usize{0};
        };
        auto add_blossom = [&](usize u, usize base, usize v) {
            usize b = n + 1;
            while (b <= node_count && rep[b]) ++b;
            if (b > node_count) ++node_count;

            label[b] = 0;
            state[b] = 0;
            match[b] = match[base];
            blossom[b].clear();
            blossom[b].push_back(base);
            for (usize x = u, y; x != base; x = rep[par[y]]) {
                blossom[b].push_back(x);
                y = rep[match[x]];
                blossom[b].push_back(y);
                queue_push(queue_push, y);
            }
            std::reverse(blossom[b].begin() + 1, blossom[b].end());
            for (usize x = v, y; x != base; x = rep[par[y]]) {
                blossom[b].push_back(x);
                y = rep[match[x]];
                blossom[b].push_back(y);
                queue_push(queue_push, y);
            }
            set_representative(set_representative, b, b);

            for (usize x = 1; x <= node_count; ++x) {
                g[b][x] = {b, x, 0};
                g[x][b] = {x, b, 0};
            }
            std::fill(from[b].begin(), from[b].end(), 0);
            for (usize x: blossom[b]) {
                for (usize y = 1; y <= node_count; ++y) {
                    if (!g[x][y].w) continue;
                    if (!g[b][y].w || less(delta(g[x][y]), delta(g[b][y]))) {
                        g[b][y] = g[x][y];
                        g[y][b] = g[y][x];
                    }
                }
                for (usize y = 1; y <= n; ++y)
                    if (from[x][y]) from[b][y] = x;
            }
            set_slack(b);
        };
        auto expand_blossom = [&](usize b) {
            for (usize x: blossom[b])
                set_representative(set_representative, x, x);
            usize entry = from[b][g[b][par[b]].u];
            usize pos = rotate_blossom(b, entry);
            for (usize i = 0; i < pos; i += 2) {
                usize x = blossom[b][i], next = blossom[b][i + 1];
                par[x] = g[next][x].u;
                state[x] = 1;
                state[next] = 0;
                slack[x] = 0;
                set_slack(next);
                queue_push(queue_push, next);
            }
            state[entry] = 1;
            par[entry] = par[b];
            for (usize i = pos + 1; i < blossom[b].size(); ++i) {
                usize x = blossom[b][i];
                state[x] = -1;
                set_slack(x);
            }
            rep[b] = 0;
        };
        auto found_edge = [&](const Edge& edge) {
            usize u = rep[edge.u], v = rep[edge.v];
            if (state[v] == -1) {
                par[v] = edge.u;
                state[v] = 1;
                usize next = rep[match[v]];
                slack[v] = slack[next] = 0;
                state[next] = 0;
                queue_push(queue_push, next);
            } else if (state[v] == 0) {
                usize base = lca(u, v);
                if (!base) {
                    augment(u, v);
                    augment(v, u);
                    return true;
                }
                add_blossom(u, base, v);
            }
            return false;
        };
        auto matching_step = [&]() {
            std::fill(state.begin(), state.end(), -1);
            std::fill(slack.begin(), slack.end(), 0);
            queue.clear();
            queue_head = 0;
            for (usize x = 1; x <= node_count; ++x) {
                if (rep[x] == x && !match[x]) {
                    par[x] = 0;
                    state[x] = 0;
                    queue_push(queue_push, x);
                }
            }
            if (queue.empty()) return false;

            while (true) {
                while (queue_head < queue.size()) {
                    usize u = queue[queue_head++];
                    if (state[rep[u]] == 1) continue;
                    for (usize v = 1; v <= n; ++v) {
                        if (!g[u][v].w || rep[u] == rep[v]) continue;
                        if (zero(delta(g[u][v]))) {
                            if (found_edge(g[u][v])) return true;
                        } else {
                            update_slack(u, rep[v]);
                        }
                    }
                }

                calc_t d = std::numeric_limits<calc_t>::max();
                for (usize u = 1; u <= n; ++u)
                    if (state[rep[u]] == 0) d = std::min(d, label[u]);
                for (usize b = n + 1; b <= node_count; ++b)
                    if (rep[b] == b && state[b] == 1)
                        d = std::min(d, label[b] / 2);
                for (usize x = 1; x <= node_count; ++x) {
                    if (rep[x] != x || !slack[x]) continue;
                    calc_t candidate = delta(g[slack[x]][x]);
                    if (state[x] == 0) candidate /= 2;
                    if (state[x] != 1 && less(candidate, d)) d = candidate;
                }
                if (d == std::numeric_limits<calc_t>::max()) return false;
                normalize(d);

                for (usize u = 1; u <= n; ++u) {
                    if (state[rep[u]] == 0) {
                        if (zero(label[u] - d)) return false;
                        label[u] -= d;
                    } else if (state[rep[u]] == 1) {
                        label[u] += d;
                    }
                    normalize(label[u]);
                }
                for (usize b = n + 1; b <= node_count; ++b) {
                    if (rep[b] != b) continue;
                    if (state[b] == 0) label[b] += d * 2;
                    else if (state[b] == 1) label[b] -= d * 2;
                    normalize(label[b]);
                }

                queue.clear();
                queue_head = 0;
                for (usize x = 1; x <= node_count; ++x) {
                    if (rep[x] == x &&
                        slack[x] &&
                        rep[slack[x]] != x &&
                        zero(delta(g[slack[x]][x]))) {
                        if (found_edge(g[slack[x]][x])) return true;
                    }
                }
                for (usize b = n + 1; b <= node_count; ++b)
                    if (rep[b] == b && state[b] == 1 && zero(label[b]))
                        expand_blossom(b);
            }
        };

        while (matching_step());
        std::vector<usize> result_match(n, npos);
        for (usize u = 1; u <= n; ++u) {
            if (match[u] && match[u] <= n) result_match[u - 1] = match[u] - 1;
        }
        T result{};
        for (usize u = 0; u < n; ++u) {
            usize v = result_match[u];
            if (v == npos || result_match[v] != u || _w[u][v] == no_edge)
                return std::nullopt;
            if (u < v) result += _w[u][v];
        }
        return std::pair{result, std::move(result_match)};
    }
};

}  // namespace cp

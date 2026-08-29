#pragma once
#include <ranges>
#include <utility>
#include <vector>

#include "def.hpp"

namespace cp
{

template <typename E = void>
class Graph {
private:
    std::vector<std::vector<std::pair<usize, E>>> _g;
    usize _e_size = 0;

    template <typename R>
    static auto out_view(auto& g, usize u) {
        auto f = [](auto& v) -> std::pair<const usize, R> {
            return {v.first, v.second};
        };
        return g[u] | std::views::transform(f);
    }
    template <typename R>
    static auto edges_view(auto& g) {
        auto f = [](auto&& t) {
            auto& [i, row] = t;
            auto f = [i](auto& v) -> std::tuple<const usize, const usize, R> {
                return {i, v.first, v.second};
            };
            return row | std::views::transform(f);
        };
        return g
            | std::views::enumerate
            | std::views::transform(f)
            | std::views::join;
    }

public:
    Graph(): _g{}, _e_size{0} {}
    Graph(usize n): _g(n), _e_size{0} {}

    usize v_size() const { return _g.size(); }
    usize e_size() const { return _e_size; }

    void add_edge(usize u, usize v, E w) {
        _g[u].emplace_back(v, std::move(w));
        _e_size++;
    }
    auto out(usize u) { return out_view<E&>(_g, u); }
    auto out(usize u) const { return out_view<const E&>(_g, u); }
    auto edges() { return edges_view<E&>(_g); }
    auto edges() const { return edges_view<const E&>(_g); }
};

template <>
class Graph<void> {
private:
    std::vector<std::vector<usize>> _g;
    usize _e_size = 0;

public:
    Graph(): _g{}, _e_size{0} {}
    Graph(usize n): _g(n), _e_size{0} {}

    usize v_size() const { return _g.size(); }
    usize e_size() const { return _e_size; }

    void add_edge(usize u, usize v) {
        _g[u].emplace_back(v);
        _e_size++;
    }
    auto out(usize u) const { return std::views::as_const(_g[u]); }
    auto edges() const {
        auto f = [](auto&& t) {
            auto& [i, row] = t;
            auto f = [i](auto& v) -> std::pair<const usize, const usize> {
                return {i, v};
            };
            return row | std::views::transform(f);
        };
        return _g
            | std::views::enumerate
            | std::views::transform(f)
            | std::views::join;
    }
};

}  // namespace cp

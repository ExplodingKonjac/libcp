#pragma once

#include <array>
#include <concepts>
#include <optional>
#include <ranges>
#include <utility>
#include <vector>

#include "cp/def.hpp"

namespace cp
{

template <typename MapT, typename SymbolT>
concept SamSymbolMap = std::semiregular<MapT> &&
    std::ranges::range<MapT> &&
    std::ranges::range<const MapT> &&
    requires(MapT& m, const MapT& cm, SymbolT s, usize u) {
        { m.try_emplace(s, u) };
        { m.find(s) } -> std::same_as<std::ranges::iterator_t<MapT>>;
        { cm.find(s) } -> std::same_as<std::ranges::iterator_t<const MapT>>;
        { m.find(s) != m.end() } -> std::convertible_to<bool>;
        { cm.find(s) != cm.end() } -> std::convertible_to<bool>;
        { (*m.find(s)).second } -> std::convertible_to<usize&>;
        { (*cm.find(s)).second } -> std::convertible_to<usize>;
    };

template <std::integral Symbol, usize N>
class DenseMap {
    std::array<usize, N> a;

public:
    static constexpr usize npos = usize(-1);

    template <bool Const>
    struct Iter {
        using Map = std::conditional_t<Const, const DenseMap, DenseMap>;
        using Item =
            std::pair<Symbol, std::conditional_t<Const, const usize&, usize&>>;
        using value_type = std::pair<Symbol, usize>;
        using difference_type = isize;

        struct ArrowProxy {
            Item item;
            const Item* operator->() const { return &item; }
        };

        Map* m{};
        usize i{N};

        // clang-format off
        Iter& skip() { while (i < N && m->a[i] == npos) i++; return *this; }
        Item operator*() const { return {Symbol(i), m->a[i]}; }
        ArrowProxy operator->() const { return {**this}; }
        Iter& operator++() { return i++, skip(); }
        Iter operator++(int) { auto x = *this; return ++*this, x; }
        bool operator==(const Iter&) const = default;
        // clang-format on
    };

    using iterator = Iter<false>;
    using const_iterator = Iter<true>;

    DenseMap() { a.fill(npos); }

    iterator begin() { return iterator{this, 0}.skip(); }
    iterator end() { return {this, N}; }
    const_iterator begin() const { return const_iterator{this, 0}.skip(); }
    const_iterator end() const { return {this, N}; }

    iterator find(Symbol s) {
        usize i = usize(s);
        return a[i] == npos ? end() : iterator{this, i};
    }
    const_iterator find(Symbol s) const {
        usize i = usize(s);
        return a[i] == npos ? end() : const_iterator{this, i};
    }
    void try_emplace(Symbol s, usize u) {
        auto& x = a[usize(s)];
        if (x == npos) x = u;
    }
};

template <typename SymbolT, SamSymbolMap<SymbolT> MapT>
class SuffixAutomaton {
private:
    struct State {
        usize link;
        usize max_len;
    };
    usize _cnt, _lst;
    std::vector<State> _states;
    std::vector<MapT> _next;

public:
    static constexpr usize npos = -1;

    SuffixAutomaton():
        _cnt{1}, _lst{0}, _states{State{npos, 0}}, _next{MapT{}} {}

    usize size() const { return _cnt; }

    usize extend(SymbolT c) {
        usize np = _cnt++, p = _lst;
        _states.push_back({0, _states[_lst].max_len + 1});
        _next.emplace_back();
        _lst = np;
        while (p != npos && !(_next[p].find(c) != _next[p].end())) {
            _next[p].try_emplace(c, np);
            p = _states[p].link;
        }
        if (p == npos) {
            _states[np].link = 0;
        } else {
            usize q = (*_next[p].find(c)).second;
            if (_states[p].max_len + 1 == _states[q].max_len) {
                _states[np].link = q;
            } else {
                usize nq = _cnt++;
                _states.push_back({_states[q].link, _states[p].max_len + 1});
                _next.push_back(_next[q]);
                while (p != npos) {
                    auto it = _next[p].find(c);
                    if (!(it != _next[p].end()) || (*it).second != q) break;
                    (*it).second = nq;
                    p = _states[p].link;
                }
                _states[q].link = nq;
                _states[np].link = nq;
            }
        }
        return np;
    }

    usize last() const { return _lst; }

    usize link(usize i) const { return _states[i].link; }

    usize max_len(usize i) const { return _states[i].max_len; }

    const MapT& transitions(usize i) const { return _next[i]; }

    std::optional<usize> transition(usize i, SymbolT c) const {
        auto it = _next[i].find(c);
        if (it != _next[i].end()) return (*it).second;
        return std::nullopt;
    }
};

}  // namespace cp

#pragma once

#include <array>
#include <concepts>
#include <optional>
#include <vector>

#include "def.hpp"

namespace acm
{

template <std::integral Symbol, usize Z>
class SuffixAutomaton {
    struct State {
        usize link;
        usize max_len;
    };

    using Transitions = std::array<usize, Z>;

    static Transitions empty_transitions() {
        Transitions next;
        next.fill(npos);
        return next;
    }

    usize _last = 0;
    std::vector<State> _states{{npos, 0}};
    std::vector<Transitions> _next{empty_transitions()};

public:
    static constexpr usize npos = usize(-1);

    usize size() const { return _states.size(); }
    usize last() const { return _last; }
    usize link(usize state) const { return _states[state].link; }
    usize max_len(usize state) const { return _states[state].max_len; }
    const Transitions& transitions(usize state) const { return _next[state]; }

    std::optional<usize> transition(usize state, Symbol symbol) const {
        usize next = _next[state][usize(symbol)];
        return next == npos ? std::nullopt : std::optional<usize>{next};
    }

    usize extend(Symbol symbol) {
        usize c = usize(symbol), current = _states.size(), p = _last;
        _last = current;
        _states.push_back({0, _states[p].max_len + 1});
        _next.push_back(empty_transitions());

        while (p != npos && _next[p][c] == npos) {
            _next[p][c] = current;
            p = _states[p].link;
        }
        if (p == npos) return _states[current].link = 0, current;

        usize q = _next[p][c];
        if (_states[p].max_len + 1 == _states[q].max_len)
            return _states[current].link = q, current;

        usize clone = _states.size();
        _states.push_back({_states[q].link, _states[p].max_len + 1});
        _next.push_back(_next[q]);
        while (p != npos && _next[p][c] == q) {
            _next[p][c] = clone;
            p = _states[p].link;
        }
        _states[q].link = _states[current].link = clone;
        return current;
    }
};

}  // namespace acm

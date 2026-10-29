#pragma once

#include <array>
#include <optional>
#include <vector>

#include "def.hpp"

namespace acm
{

template <typename Symbol, usize SIGMA>
class SuffixAutomaton {
    static_assert(SIGMA > 0);

    std::vector<std::array<usize, SIGMA>> nxt;
    std::vector<usize> fa, len;
    usize lst = 0;

public:
    static constexpr usize npos = usize(-1);

    SuffixAutomaton() {
        std::array<usize, SIGMA> e;
        e.fill(npos);
        nxt.push_back(e);
        fa.push_back(npos);
        len.push_back(0);
    }

    usize size() const { return len.size(); }
    usize last() const { return lst; }
    usize link(usize i) const { return fa[i]; }
    usize max_len(usize i) const { return len[i]; }
    const std::array<usize, SIGMA>& transitions(usize i) const {
        return nxt[i];
    }

    std::optional<usize> transition(usize i, Symbol c) const {
        usize to = nxt[i][usize(c)];
        return to == npos ? std::nullopt : std::optional<usize>{to};
    }

    usize extend(Symbol c) {
        usize x = usize(c), cur = size(), p = lst;
        lst = cur;
        std::array<usize, SIGMA> e;
        e.fill(npos);
        nxt.push_back(e);
        fa.push_back(0);
        len.push_back(len[p] + 1);

        while (p != npos && nxt[p][x] == npos) {
            nxt[p][x] = cur;
            p = fa[p];
        }
        if (p == npos) {
            fa[cur] = 0;
            return cur;
        }

        usize q = nxt[p][x];
        if (len[p] + 1 == len[q]) {
            fa[cur] = q;
            return cur;
        }

        usize clone = size();
        nxt.push_back(nxt[q]);
        fa.push_back(fa[q]);
        len.push_back(len[p] + 1);
        while (p != npos && nxt[p][x] == q) {
            nxt[p][x] = clone;
            p = fa[p];
        }
        fa[q] = fa[cur] = clone;
        return cur;
    }
};

}  // namespace acm

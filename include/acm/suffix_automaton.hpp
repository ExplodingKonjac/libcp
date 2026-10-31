#pragma once

#include <array>
#include <optional>
#include <vector>

#include "def.hpp"

namespace acm
{

template <typename Symbol, usize Z>
class SuffixAutomaton {
    static_assert(Z > 0);

    std::vector<std::array<usize, Z>> nxt;
    std::vector<usize> fa, len;
    usize lst = 0;

public:
    static constexpr usize npos = usize(-1);

    SuffixAutomaton() {
        std::array<usize, Z> e;
        e.fill(npos);
        nxt.push_back(e);
        fa.push_back(npos);
        len.push_back(0);
    }

    usize size() const { return len.size(); }
    usize last() const { return lst; }
    usize link(usize i) const { return fa[i]; }
    usize max_len(usize i) const { return len[i]; }
    const std::array<usize, Z>& transitions(usize i) const { return nxt[i]; }

    std::optional<usize> transition(usize i, Symbol c) const {
        usize to = nxt[i][usize(c)];
        return to == npos ? std::nullopt : std::optional<usize>{to};
    }

    usize extend(Symbol c) {
        usize x = usize(c), cur = size(), p = lst;
        std::array<usize, Z> e;
        e.fill(npos);
        nxt.push_back(e);
        fa.push_back(0);
        len.push_back(len[p] + 1);
        lst = cur;
        for (; p != npos && nxt[p][x] == npos; p = fa[p]) nxt[p][x] = cur;
        if (p == npos) {
            fa[cur] = 0;
        } else {
            usize q = nxt[p][x];
            if (len[p] + 1 == len[q]) {
                fa[cur] = q;
            } else {
                usize nq = size();
                nxt.push_back(nxt[q]);
                fa.push_back(fa[q]);
                len.push_back(len[p] + 1);
                for (; p != npos && nxt[p][x] == q; p = fa[p]) nxt[p][x] = nq;
                fa[q] = nq;
                fa[cur] = nq;
            }
        }
        return cur;
    }
};

}  // namespace acm

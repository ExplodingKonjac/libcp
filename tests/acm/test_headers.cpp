#include <cassert>
#include <string>
#include <type_traits>
#include <vector>

#include "acm/all.hpp"

int main() {
    static_assert(std::is_same_v<acm::i32, std::int32_t>);
    static_assert(std::is_same_v<acm::i64, std::int64_t>);
    static_assert(std::is_same_v<acm::u64, std::uint64_t>);
    static_assert(std::is_same_v<acm::usize, std::size_t>);

    acm::SModint<998244353> m = 1;
    (void)m;
    acm::SuffixArray sa(std::string("a"));
    (void)sa;
    acm::SuffixAutomaton<char, 26> sam;
    sam.extend(0);
    acm::BipartiteMatching bm(1, 1);
    (void)bm;
    acm::GeneralMatching gm(1);
    (void)gm;
    acm::MaxFlow<acm::i32> mf(1);
    (void)mf;
    acm::MinCostFlow<acm::i32, acm::i64> mcf(1);
    (void)mcf;
    acm::MinCostCirculation<acm::i32, acm::i64> mcc(1);
    (void)mcc;
    acm::FPoly<998244353> f{1, 2};
    (void)f;
    acm::Vec2<acm::i64> v{0, 0};
    (void)v;
    acm::Polygon<acm::i64> poly;
    (void)poly;
    return 0;
}

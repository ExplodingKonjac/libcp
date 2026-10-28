#include <cassert>
#include <iostream>
#include <map>
#include <string>

#include "cp/suffix_automaton.hpp"

using TreeMap = std::map<char, cp::usize>;
using DenseMap = cp::DenseMap<char, 128>;

template <typename Map>
using Sam = cp::SuffixAutomaton<char, Map>;

static_assert(cp::sam_symbol_map<TreeMap, char>);
static_assert(cp::sam_symbol_map<DenseMap, char>);

template <typename Map>
bool contains(const Sam<Map>& sam, const std::string& pattern) {
    cp::usize state = 0;
    for (char c: pattern) {
        auto next = sam.transition(state, c);
        if (!next) return false;
        state = *next;
    }
    return true;
}

template <typename Map>
void assert_structure(const Sam<Map>& sam) {
    assert(sam.size() >= 1);
    assert(sam.link(0) == Sam<Map>::npos);
    assert(sam.max_len(0) == 0);

    for (cp::usize i = 0; i < sam.size(); ++i) {
        auto link = sam.link(i);
        if (link != Sam<Map>::npos) {
            assert(link < sam.size());
            assert(sam.max_len(link) < sam.max_len(i));
        }
        for (auto [c, next]: sam.transitions(i)) {
            assert(next < sam.size());
            assert(sam.transition(i, c) == next);
        }
    }
}

template <typename Map>
void test_text(const std::string& text) {
    Sam<Map> sam;
    for (cp::usize i = 0; i < text.size(); ++i) {
        auto state = sam.extend(text[i]);
        assert(state == sam.last());
        assert(sam.max_len(state) == i + 1);
    }

    assert(sam.size() <= 2 * text.size() + 1);
    assert_structure(sam);

    for (cp::usize length = 0; length <= text.size() + 1; ++length) {
        for (cp::usize mask = 0; mask < (cp::usize{1} << length); ++mask) {
            std::string pattern(length, 'a');
            for (cp::usize i = 0; i < length; ++i) {
                if (mask & (cp::usize{1} << i)) pattern[i] = 'b';
            }
            bool expected = text.find(pattern) != std::string::npos;
            assert(contains(sam, pattern) == expected);
        }
    }
}

template <typename Map>
void test_empty() {
    std::cout << "test_empty... ";
    Sam<Map> sam;
    assert(sam.size() == 1);
    assert(sam.last() == 0);
    assert(!sam.transition(0, 'a'));
    assert(sam.transitions(0).begin() == sam.transitions(0).end());
    assert_structure(sam);
    std::cout << "OK\n";
}

template <typename Map>
void test_exhaustive_binary_texts() {
    std::cout << "test_exhaustive_binary_texts... ";
    for (cp::usize length = 0; length <= 6; ++length) {
        for (cp::usize mask = 0; mask < (cp::usize{1} << length); ++mask) {
            std::string text(length, 'a');
            for (cp::usize i = 0; i < length; ++i) {
                if (mask & (cp::usize{1} << i)) text[i] = 'b';
            }
            test_text<Map>(text);
        }
    }
    std::cout << "OK\n";
}

template <typename Map>
void test_clone_case() {
    std::cout << "test_clone_case... ";
    Sam<Map> sam;
    for (char c: std::string{"abcbc"}) sam.extend(c);
    assert(sam.size() > 6);
    assert_structure(sam);
    assert(contains(sam, "cbc"));
    assert(!contains(sam, "ccb"));
    std::cout << "OK\n";
}

int main() {
    test_empty<TreeMap>();
    test_empty<DenseMap>();
    test_exhaustive_binary_texts<TreeMap>();
    test_exhaustive_binary_texts<DenseMap>();
    test_clone_case<TreeMap>();
    test_clone_case<DenseMap>();
    std::cout << "All tests passed!\n";
}

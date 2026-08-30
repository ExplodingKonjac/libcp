#include <cassert>
#include <iostream>
#include <string>

#include "acm/suffix_automaton.hpp"

using Sam = acm::SuffixAutomaton<char, 128>;

bool contains(const Sam& sam, const std::string& pattern) {
    acm::usize state = 0;
    for (char c: pattern) {
        auto next = sam.transition(state, c);
        if (!next) return false;
        state = *next;
    }
    return true;
}

void assert_structure(const Sam& sam) {
    assert(sam.size() >= 1);
    assert(sam.link(0) == Sam::npos);
    assert(sam.max_len(0) == 0);
    for (acm::usize state = 0; state < sam.size(); state++) {
        auto link = sam.link(state);
        if (link != Sam::npos) {
            assert(link < sam.size());
            assert(sam.max_len(link) < sam.max_len(state));
        }
        const auto& next = sam.transitions(state);
        for (acm::usize c = 0; c < next.size(); c++) {
            if (next[c] == Sam::npos) continue;
            assert(next[c] < sam.size());
            assert(sam.transition(state, char(c)) == next[c]);
        }
    }
}

void test_text(const std::string& text) {
    Sam sam;
    for (acm::usize i = 0; i < text.size(); i++) {
        auto state = sam.extend(text[i]);
        assert(state == sam.last());
        assert(sam.max_len(state) == i + 1);
    }
    assert(sam.size() <= 2 * text.size() + 1);
    assert_structure(sam);
    for (acm::usize length = 0; length <= text.size() + 1; length++) {
        for (acm::usize mask = 0; mask < (acm::usize{1} << length); mask++) {
            std::string pattern(length, 'a');
            for (acm::usize i = 0; i < length; i++)
                if (mask & (acm::usize{1} << i)) pattern[i] = 'b';
            assert(contains(sam, pattern) ==
                   (text.find(pattern) != std::string::npos));
        }
    }
}

int main() {
    Sam empty;
    assert(empty.size() == 1 && empty.last() == 0);
    assert(!empty.transition(0, 'a'));
    for (auto next: empty.transitions(0)) assert(next == Sam::npos);
    for (acm::usize length = 0; length <= 6; length++) {
        for (acm::usize mask = 0; mask < (acm::usize{1} << length); mask++) {
            std::string text(length, 'a');
            for (acm::usize i = 0; i < length; i++)
                if (mask & (acm::usize{1} << i)) text[i] = 'b';
            test_text(text);
        }
    }
    Sam clone_case;
    for (char c: std::string{"abcbc"}) clone_case.extend(c);
    assert(clone_case.size() > 6);
    assert_structure(clone_case);
    assert(contains(clone_case, "cbc"));
    assert(!contains(clone_case, "ccb"));
    std::cout << "All tests passed!\n";
}

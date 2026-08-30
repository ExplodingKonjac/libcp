#include <cassert>
#include <random>
#include <unordered_map>
#include "acm/hash_map.hpp"

int main() {
    acm::FlatHashMap<int, int> a;
    std::unordered_map<int, int> b;
    std::mt19937 rng(2);
    for (int i = 0; i < 5000; i++) {
        int k = rng() % 200;
        if (rng() & 1) a[k]++, b[k]++; else assert(a.erase(k) == (b.erase(k) != 0));
        assert(a.size() == b.size());
        for (auto [x, y]: b) assert(a.contains(x) && *a.get(x) == y);
    }
}

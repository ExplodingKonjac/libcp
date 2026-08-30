#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "def.hpp"

namespace acm
{

template <
    typename Key,
    typename Mapped,
    typename Hash = std::hash<Key>,
    typename Eq = std::equal_to<Key>
>
class FlatHashMap {
    struct Slot {
        std::optional<std::pair<Key, Mapped>> value;
        bool deleted = false;
    };
    std::vector<Slot> tab;
    usize used = 0;
    Hash hash{};
    Eq eq{};

    usize locate(const Key& key) const {
        if (tab.empty()) return 0;
        usize i = hash(key) & (tab.size() - 1), first_deleted = tab.size();
        for (usize step = 0; step < tab.size(); step++) {
            if (tab[i].value && eq(tab[i].value->first, key)) return i;
            if (!tab[i].value && tab[i].deleted && first_deleted == tab.size())
                first_deleted = i;
            if (!tab[i].value && !tab[i].deleted)
                return first_deleted == tab.size() ? i : first_deleted;
            i = (i + 1) & (tab.size() - 1);
        }
        return first_deleted;
    }
    void grow() {
        auto old = std::move(tab);
        tab.assign(old.empty() ? 8 : old.size() * 2, {});
        used = 0;
        for (auto& s: old)
            if (s.value) insert(std::move(*s.value));
    }

public:
    template <bool C>
    class Iter {
        using Map = std::conditional_t<C, const FlatHashMap, FlatHashMap>;
        Map* map = nullptr;
        usize i = 0;
        void skip() {
            while (map && i < map->tab.size() && !map->tab[i].value) i++;
        }
        friend class FlatHashMap;
        Iter(Map* m, usize p): map(m), i(p) { skip(); }

    public:
        Iter() = default;
        auto& operator*() const { return *map->tab[i].value; }
        auto* operator->() const { return &*map->tab[i].value; }
        Iter& operator++() {
            i++;
            skip();
            return *this;
        }
        bool operator==(const Iter&) const = default;
    };
    using iterator = Iter<false>;
    using const_iterator = Iter<true>;

    iterator begin() { return {this, 0}; }
    iterator end() { return {this, tab.size()}; }
    const_iterator begin() const { return {this, 0}; }
    const_iterator end() const { return {this, tab.size()}; }
    usize size() const { return used; }
    bool empty() const { return !used; }
    iterator find(const Key& key) {
        usize i = locate(key);
        return i < tab.size() && tab[i].value ? iterator(this, i) : end();
    }
    const_iterator find(const Key& key) const {
        usize i = locate(key);
        return i < tab.size() && tab[i].value ? const_iterator(this, i) : end();
    }
    bool contains(const Key& key) const { return find(key) != end(); }
    Mapped* get(const Key& key) {
        auto i = find(key);
        return i == end() ? nullptr : &i->second;
    }
    const Mapped* get(const Key& key) const {
        auto i = find(key);
        return i == end() ? nullptr : &i->second;
    }
    std::pair<iterator, bool> insert(std::pair<Key, Mapped> value) {
        if ((used + 1) * 10 >= tab.size() * 7) grow();
        usize i = locate(value.first);
        if (tab[i].value) return {iterator(this, i), false};
        tab[i].value.emplace(std::move(value));
        tab[i].deleted = false;
        used++;
        return {iterator(this, i), true};
    }
    template <typename... A>
    std::pair<iterator, bool> try_emplace(Key key, A&&... args) {
        return insert({std::move(key), Mapped(std::forward<A>(args)...)});
    }
    Mapped& operator[](Key key) {
        return try_emplace(std::move(key)).first->second;
    }
    bool erase(const Key& key) {
        usize i = locate(key);
        if (i >= tab.size() || !tab[i].value) return false;
        tab[i].value.reset();
        tab[i].deleted = true;
        --used;
        return true;
    }
    void clear() {
        tab.clear();
        used = 0;
    }
};

}  // namespace acm

#pragma once

#include <emmintrin.h>

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "def.hpp"

namespace cp
{

namespace detail
{

enum SlotType : u8 {
    Empty = 0b10000000,
    Deleted = 0b10000001,
    Sentinel = 0b10000010,
};

alignas(16) inline const u8 default_ctrl[16]{
    Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty,
    Empty, Empty, Empty, Empty, Empty, Empty, Empty, Sentinel,
};

#pragma GCC push_options
#pragma GCC target("sse2")
template <
    typename Key, typename Mapped, typename Hash = std::hash<Key>,
    typename Eq = std::equal_to<Key>,
    typename Alloc = std::allocator<std::pair<const Key, Mapped>>>
    requires requires(Key k, Hash hash, Eq eq) {
        { hash(k) } -> std::convertible_to<usize>;
        { eq(k, k) } -> std::convertible_to<bool>;
    }
class FlatHashMap {
private:
    using Slot = std::pair<const Key, Mapped>;
    struct alignas(Slot) AlignedBlock {};

    using SlotAlloc = std::allocator_traits<Alloc>::template rebind_alloc<Slot>;
    using SlotAllocTraits = std::allocator_traits<SlotAlloc>;

    using BlockAlloc =
        std::allocator_traits<Alloc>::template rebind_alloc<AlignedBlock>;
    using BlockAllocTraits = std::allocator_traits<BlockAlloc>;

    template <bool is_const>
    class Iterator {
        friend class FlatHashMap;

    private:
        using CtrlPtr = std::conditional_t<is_const, const u8*, u8*>;
        using SlotPtr = std::conditional_t<is_const, const Slot*, Slot*>;
        CtrlPtr _ctrl;
        SlotPtr _data;

        Iterator(CtrlPtr ctrl, SlotPtr data) noexcept:
            _ctrl(ctrl), _data(data) {
            while (*_ctrl == Empty || *_ctrl == Deleted) {
                _ctrl++;
                _data++;
            }
        }

    public:
        using value_type = Slot;
        using reference =
            std::conditional_t<is_const, const value_type&, value_type&>;
        using pointer =
            std::conditional_t<is_const, const value_type*, value_type*>;

        Iterator() = default;
        reference operator*() const noexcept {
            return reinterpret_cast<reference>(*_data);
        }
        pointer operator->() const noexcept {
            return reinterpret_cast<pointer>(_data);
        }
        Iterator& operator++() noexcept {
            return *this = Iterator{_ctrl + 1, _data + 1};
        }
        Iterator operator++(int) noexcept {
            auto tmp = *this;
            ++*this;
            return tmp;
        }
        bool operator==(const Iterator& other) const noexcept {
            return _ctrl == other._ctrl;
        }
        bool operator!=(const Iterator& other) const noexcept {
            return _ctrl != other._ctrl;
        }
    };

    struct HashValue {
        u8 h;
        usize p;
    };

    static usize block_num(usize cap) {
        constexpr usize B = alignof(Slot);
        return (cap * sizeof(Slot) + cap + 16 + B - 1) / B;
    }

    HashValue do_hash(const Key& key) const {
        auto raw_hash = _hash(key);
        return HashValue{u8(raw_hash & 0x7F), (raw_hash >> 7) & _capacity};
    }

    void rehash(usize new_cap) {
        auto old_data = _data;
        auto old_ctrl = _ctrl;
        auto old_cap = _capacity;

        auto raw = BlockAllocTraits::allocate(_block_alloc, block_num(new_cap));
        auto ptr = reinterpret_cast<std::byte*>(raw);
        _capacity = new_cap;
        _size = 0;
        _growth_left = _capacity * 7 / 8;
        _data = reinterpret_cast<Slot*>(ptr);
        _ctrl = reinterpret_cast<u8*>(ptr + sizeof(Slot) * new_cap);
        std::memset(_ctrl, Empty, new_cap + 16);
        _ctrl[new_cap] = Sentinel;

        // move old data
        if (old_data) {
            for (usize i = 0; i < old_cap; i += 16) {
                auto v = _mm_loadu_si128((const __m128i*)(old_ctrl + i));
                for (u16 m = ~_mm_movemask_epi8(v); m > 0; m &= (m - 1)) {
                    auto old_idx = i + std::countr_zero(m);
                    auto& slot = old_data[old_idx];
                    auto hsh = do_hash(slot.first);
                    auto [_, new_idx] = lookup<true, true>(hsh, slot.first);
                    emplace_at(new_idx, hsh.h, std::move(slot));
                    SlotAllocTraits::destroy(_slot_alloc, &slot);
                }
            }
            auto old_block_num = block_num(old_cap);
            BlockAllocTraits::deallocate(
                _block_alloc, reinterpret_cast<AlignedBlock*>(old_data),
                old_block_num
            );
        }
    }

    template <bool need_insert_idx, bool on_init>
    std::pair<bool, usize> lookup(HashValue hsh, const Key& k) const {
        constexpr usize npos = -1;
        auto v_h = _mm_set1_epi8(hsh.h);
        auto v_e = _mm_set1_epi8(Empty);
        auto v_d = _mm_set1_epi8(Deleted);
        usize insert_idx = npos;
        for (usize i = hsh.p;; i = (i + 16) & _capacity) {
            auto v_ctrl = _mm_loadu_si128((const __m128i*)(_ctrl + i));
            if constexpr (!on_init) {
                // check match place
                u16 m_h = _mm_movemask_epi8(_mm_cmpeq_epi8(v_h, v_ctrl));
                for (; m_h != 0; m_h &= (m_h - 1)) {
                    auto offset = std::countr_zero(m_h);
                    usize idx = (i + offset) & _capacity;
                    if (_eq(_data[idx].first, k)) return {true, idx};
                }
                // check available insert index
                if (insert_idx == npos) {
                    u16 m_d = _mm_movemask_epi8(_mm_cmpeq_epi8(v_d, v_ctrl));
                    if (m_d != 0) {
                        auto offset = std::countr_zero(m_d);
                        insert_idx = (i + offset) & _capacity;
                    }
                }
            }
            // check empty index
            u16 m_e = _mm_movemask_epi8(_mm_cmpeq_epi8(v_e, v_ctrl));
            if (m_e != 0) {
                if constexpr (need_insert_idx) {
                    if (insert_idx != npos) return {false, insert_idx};
                    auto offset = std::countr_zero(m_e);
                    return {false, (i + offset) & _capacity};
                } else {
                    return {false, npos};
                }
            }
        }
    }

    template <typename... Args>
    void emplace_at(usize idx, u8 h, Args&&... args) {
        SlotAllocTraits::construct(
            _slot_alloc, _data + idx, std::forward<Args>(args)...
        );
        if (_ctrl[idx] == Empty) _growth_left--;
        _size++;
        _ctrl[idx] = h;
        if (idx < 15) _ctrl[_capacity + 1 + idx] = h;
    }

    Slot erase_at(usize idx) {
        auto res = std::move(_data[idx]);
        SlotAllocTraits::destroy(_slot_alloc, _data + idx);
        _size--;
        _ctrl[idx] = Deleted;
        if (idx < 15) _ctrl[_capacity + 1 + idx] = Deleted;
        return res;
    }

    void try_rehash() {
        if (_capacity == 0) {
            rehash(15);
        } else if (_growth_left == 0) {
            rehash(std::bit_ceil(_size * 2 + 1) - 1);
        }
    }

    usize _size{0};
    usize _capacity{0};
    usize _growth_left{0};
    Slot* _data{nullptr};
    u8* _ctrl{const_cast<u8*>(default_ctrl)};

    [[no_unique_address]] mutable Hash _hash;
    [[no_unique_address]] mutable Eq _eq;
    [[no_unique_address]] BlockAlloc _block_alloc;
    [[no_unique_address]] SlotAlloc _slot_alloc;

public:
    using key_type = Key;
    using mapped_type = Mapped;
    using value_type = Slot;
    using size_type = usize;
    using allocator_type = Alloc;
    using iterator = Iterator<false>;
    using const_iterator = Iterator<true>;

    FlatHashMap(Hash hash = {}, Eq eq = {}, const Alloc& alloc = {}) noexcept:
        _hash(std::move(hash)),
        _eq(std::move(eq)),
        _block_alloc(alloc),
        _slot_alloc(alloc) {}
    FlatHashMap(FlatHashMap&& other) noexcept { swap(other); }
    FlatHashMap(const FlatHashMap& other) = delete;
    auto& operator=(FlatHashMap&& other) noexcept {
        swap(other);
        return *this;
    }
    auto& operator=(const FlatHashMap& other) = delete;

    ~FlatHashMap() {
        if (_data) {
            for (usize i = 0; i < _capacity; i += 16) {
                auto v = _mm_loadu_si128((const __m128i*)(_ctrl + i));
                for (u16 m = ~_mm_movemask_epi8(v); m > 0; m &= (m - 1)) {
                    auto idx = i + std::countr_zero(m);
                    SlotAllocTraits::destroy(_slot_alloc, _data + idx);
                }
            }
            auto n = block_num(_capacity);
            BlockAllocTraits::deallocate(
                _block_alloc, reinterpret_cast<AlignedBlock*>(_data), n
            );
        }
    }

    void swap(FlatHashMap& other) noexcept {
        std::swap(_size, other._size);
        std::swap(_capacity, other._capacity);
        std::swap(_growth_left, other._growth_left);
        std::swap(_data, other._data);
        std::swap(_ctrl, other._ctrl);
        std::swap(_hash, other._hash);
        std::swap(_eq, other._eq);
        std::swap(_block_alloc, other._block_alloc);
        std::swap(_slot_alloc, other._slot_alloc);
    }

    auto begin() noexcept { return iterator(_ctrl, _data); }
    auto end() noexcept {
        return iterator(_ctrl + _capacity, _data + _capacity);
    }
    auto begin() const noexcept { return const_iterator(_ctrl, _data); }
    auto end() const noexcept {
        return const_iterator(_ctrl + _capacity, _data + _capacity);
    }

    mapped_type* get(const key_type& k) noexcept {
        auto [fl, idx] = lookup<false, false>(do_hash(k), k);
        return fl ? &_data[idx].second : nullptr;
    }
    const mapped_type* get(const key_type& k) const noexcept {
        auto [fl, idx] = lookup<false, false>(do_hash(k), k);
        return fl ? &_data[idx].second : nullptr;
    }

    iterator find(const key_type& k) noexcept {
        auto [fl, idx] = lookup<false, false>(do_hash(k), k);
        if (!fl) return end();
        return iterator(_ctrl + idx, _data + idx);
    }
    const_iterator find(const key_type& k) const noexcept {
        auto [fl, idx] = lookup<false, false>(do_hash(k), k);
        if (!fl) return end();
        return const_iterator(_ctrl + idx, _data + idx);
    }

    template <typename P>
    std::optional<iterator> insert(P&& value) {
        return emplace(std::forward<P>(value));
    }
    template <typename... Args>
        requires std::constructible_from<value_type, Args&&...>
    std::optional<iterator> emplace(Args&&... args) {
        Slot value(std::forward<Args>(args)...);
        try_rehash();
        auto hsh = do_hash(value.first);
        auto [fl, idx] = lookup<true, false>(hsh, value.first);
        if (fl) return std::nullopt;
        emplace_at(idx, hsh.h, std::move(value));
        return iterator(_ctrl + idx, _data + idx);
    }

    template <typename K, typename... Args>
        requires std::constructible_from<Key, K&&> &&
                 std::constructible_from<Mapped, Args&&...>
    iterator try_emplace(K&& k, Args&&... args) {
        Key key(std::forward<K>(k));
        try_rehash();
        auto hsh = do_hash(key);
        auto [fl, idx] = lookup<true, false>(hsh, key);
        if (!fl)
            emplace_at(
                idx, hsh.h, std::piecewise_construct,
                std::forward_as_tuple(std::move(key)),
                std::forward_as_tuple(std::forward<Args>(args)...)
            );
        return iterator(_ctrl + idx, _data + idx);
    }

    template <typename K, typename Fn>
        requires std::constructible_from<Key, K&&> && requires(Fn f) {
            { f() } -> std::convertible_to<Mapped>;
        }
    iterator try_insert_with(K&& k, Fn f) {
        Key key(std::forward<K>(k));
        try_rehash();
        auto hsh = do_hash(key);
        auto [fl, idx] = lookup<true, false>(hsh, key);
        if (!fl) emplace_at(idx, hsh.h, std::move(key), f());
        return iterator(_ctrl + idx, _data + idx);
    }

    bool erase(const key_type& k) {
        auto hsh = do_hash(k);
        auto [fl, idx] = lookup<false, false>(hsh, k);
        if (!fl) return false;
        erase_at(idx);
        return true;
    }
    void erase(iterator it) { erase_at(it._ctrl - _ctrl); }

    bool empty() const noexcept { return _size == 0; }
    usize size() const noexcept { return _size; }
};
#pragma GCC pop_options

}  // namespace detail

using detail::FlatHashMap;

}  // namespace cp

#pragma once
#include <immintrin.h>

#include <climits>
#include <compare>
#include <concepts>
#include <cstring>
#include <type_traits>
#include <utility>

#include "def.hpp"

#pragma GCC push_options
#pragma GCC target("avx2")

namespace cp
{

namespace details
{

template <typename E>
struct BitsetNot {
    E&& e_;
    __m256i block_at(usize i) const {
        return _mm256_xor_si256(e_.block_at(i), _mm256_set1_epi64x(-1));
    }
};

template <typename XE, typename YE>
struct BitsetAnd {
    XE&& xe_;
    YE&& ye_;
    __m256i block_at(usize i) const {
        return _mm256_and_si256(ye_.block_at(i), xe_.block_at(i));
    }
};

template <typename XE, typename YE>
struct BitsetOr {
    XE&& xe_;
    YE&& ye_;
    __m256i block_at(usize i) const {
        return _mm256_or_si256(ye_.block_at(i), xe_.block_at(i));
    }
};

template <typename XE, typename YE>
struct BitsetXor {
    XE&& xe_;
    YE&& ye_;
    __m256i block_at(usize i) const {
        return _mm256_xor_si256(ye_.block_at(i), xe_.block_at(i));
    }
};

template <typename XE, typename YE>
struct BitsetAnt {
    XE&& xe_;
    YE&& ye_;
    __m256i block_at(usize i) const {
        return _mm256_andnot_si256(ye_.block_at(i), xe_.block_at(i));
    }
};

}  // namespace details

template <typename E>
concept BitsetExpr =
    requires(const std::remove_reference_t<E>& e, usize i) { e.block_at(i); };

template <BitsetExpr E>
inline auto operator~(E&& e) {
    return details::BitsetNot<E>{std::forward<E>(e)};
}

template <BitsetExpr XE, BitsetExpr YE>
inline auto operator&(XE&& xe, YE&& ye) {
    return details::BitsetAnd<XE, YE>{std::forward<XE>(xe),
                                      std::forward<YE>(ye)};
}

template <BitsetExpr XE, BitsetExpr YE>
inline auto operator|(XE&& xe, YE&& ye) {
    return details::BitsetOr<XE, YE>{std::forward<XE>(xe),
                                     std::forward<YE>(ye)};
}

template <BitsetExpr XE, BitsetExpr YE>
inline auto operator^(XE&& xe, YE&& ye) {
    return details::BitsetXor<XE, YE>{std::forward<XE>(xe),
                                      std::forward<YE>(ye)};
}

template <BitsetExpr XE, BitsetExpr YE>
inline auto operator-(XE&& xe, YE&& ye) {
    return details::BitsetAnt<XE, YE>{std::forward<XE>(xe),
                                      std::forward<YE>(ye)};
}

template <usize SIZE>
class Bitset {
protected:
    typedef u64 Word;
    typedef __m256i Block;

    static const usize WORD_SIZE = sizeof(Word) * CHAR_BIT;
    static const usize BLOCK_SIZE = sizeof(Block) / sizeof(Word);
    static const usize WORD_COUNT = (SIZE + WORD_SIZE - 1) / WORD_SIZE;
    static const usize BLOCK_COUNT = (WORD_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;

    alignas(Block) Word data_[BLOCK_COUNT * BLOCK_SIZE]{};

public:
    Bitset() = default;
    ~Bitset() = default;

    Word word_at(usize position) const { return data_[position]; }

    Block block_at(usize position) const {
        return _mm256_load_si256((const Block*)&data_[position * BLOCK_SIZE]);
    }

protected:
    void word_set(usize position, Word value) { data_[position] = value; }

    void block_set(usize position, Block value) {
        _mm256_store_si256((Block*)&data_[position * BLOCK_SIZE], value);
    }

    void trim() {
        if (SIZE % WORD_SIZE)
            data_[WORD_COUNT - 1] &= (1ULL << (SIZE % WORD_SIZE)) - 1;
        if (BLOCK_COUNT * BLOCK_SIZE > WORD_COUNT)
            memset(
                data_ + WORD_COUNT, 0,
                (BLOCK_COUNT * BLOCK_SIZE - WORD_COUNT) * sizeof(Word)
            );
    }

    void normalize() {
        if (SIZE % WORD_SIZE)
            data_[WORD_COUNT - 1] &= (1ULL << (SIZE % WORD_SIZE)) - 1;
        if (BLOCK_COUNT * BLOCK_SIZE > WORD_COUNT)
            memset(
                data_ + WORD_COUNT, 0,
                (BLOCK_COUNT * BLOCK_SIZE - WORD_COUNT) * sizeof(Word)
            );
    }

public:
    Bitset(const Bitset&) = default;
    Bitset(Bitset&&) = default;

    Bitset(BitsetExpr auto&& e) {
        for (usize i = 0; i != BLOCK_COUNT; ++i) block_set(i, e.block_at(i));
        trim();
    }

    Bitset& operator=(const Bitset&) = default;
    Bitset& operator=(Bitset&&) = default;

    Bitset& operator=(BitsetExpr auto&& e) {
        for (usize i = 0; i != BLOCK_COUNT; ++i) block_set(i, e.block_at(i));
        trim();
        return *this;
    }

    bool operator[](usize position) const {
        if (position >= SIZE) return false;
        return data_[position / WORD_SIZE] >> position % WORD_SIZE & 1;
    }

    bool operator()(usize position) const {
        if (position >= SIZE) return false;
        return data_[position / WORD_SIZE] >> position % WORD_SIZE & 1;
    }

    void set_bit(usize position) {
        if (position >= SIZE) return;
        data_[position / WORD_SIZE] |= 1ULL << position % WORD_SIZE;
    }

    void unset_bit(usize position) {
        if (position >= SIZE) return;
        data_[position / WORD_SIZE] &= ~(1ULL << position % WORD_SIZE);
    }

    void flip_bit(usize position) {
        if (position >= SIZE) return;
        data_[position / WORD_SIZE] ^= 1ULL << position % WORD_SIZE;
    }

    usize size() const { return SIZE; }

    usize length() const { return SIZE; }

    usize count() const {
        const Block MASK = _mm256_set1_epi8(0x0F);
        const Block TABLE = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2,
            3, 1, 2, 2, 3, 2, 3, 3, 4
        );
        Block result = _mm256_setzero_si256();
        for (usize i = 0; i != BLOCK_COUNT; ++i) {
            Block block = block_at(i);
            Block low =
                _mm256_shuffle_epi8(TABLE, _mm256_and_si256(block, MASK));
            Block high = _mm256_shuffle_epi8(
                TABLE, _mm256_and_si256(_mm256_srli_epi16(block, 4), MASK)
            );
            result = _mm256_add_epi64(
                result,
                _mm256_sad_epu8(
                    _mm256_add_epi8(low, high), _mm256_setzero_si256()
                )
            );
        }
        Word parts[BLOCK_SIZE];
        _mm256_storeu_si256((Block*)parts, result);
        return usize(parts[0] + parts[1] + parts[2] + parts[3]);
    }

    usize popcnt() const { return count(); }

    bool none() const {
        const usize FULL_BLOCKS = SIZE / (sizeof(Block) * CHAR_BIT);
        for (usize i = 0; i != FULL_BLOCKS; ++i) {
            Block mask = _mm256_set1_epi8(-1), block = block_at(i);
            if (!_mm256_testz_si256(block, mask)) return false;
        }
        if (usize remaining = SIZE % (sizeof(Block) * CHAR_BIT)) {
            Word mask_array[BLOCK_SIZE] = {};
            for (usize i = 0; i != BLOCK_SIZE; ++i) {
                if (remaining >= WORD_SIZE) {
                    mask_array[i] = ~0ULL;
                    remaining -= WORD_SIZE;
                } else {
                    mask_array[i] = (1ULL << remaining) - 1;
                    break;
                }
            }
            Block mask = _mm256_loadu_si256((const Block*)mask_array),
                  block = block_at(FULL_BLOCKS);
            if (!_mm256_testz_si256(block, mask)) return false;
        }
        return true;
    }

    bool any() const {
        const usize FULL_BLOCKS = SIZE / (sizeof(Block) * CHAR_BIT);
        for (usize i = 0; i != FULL_BLOCKS; ++i) {
            Block mask = _mm256_set1_epi8(-1), block = block_at(i);
            if (!_mm256_testz_si256(block, mask)) return true;
        }
        if (usize remaining = SIZE % (sizeof(Block) * CHAR_BIT)) {
            Word mask_array[BLOCK_SIZE] = {};
            for (usize i = 0; i != BLOCK_SIZE; ++i) {
                if (remaining >= WORD_SIZE) {
                    mask_array[i] = ~0ULL;
                    remaining -= WORD_SIZE;
                } else {
                    mask_array[i] = (1ULL << remaining) - 1;
                    break;
                }
            }
            Block mask = _mm256_loadu_si256((const Block*)mask_array),
                  block = block_at(FULL_BLOCKS);
            if (!_mm256_testz_si256(block, mask)) return true;
        }
        return false;
    }

    bool all() const {
        const usize FULL_BLOCKS = SIZE / (sizeof(Block) * CHAR_BIT);
        for (usize i = 0; i != FULL_BLOCKS; ++i) {
            Block mask = _mm256_set1_epi8(-1), block = block_at(i);
            if (!_mm256_testc_si256(block, mask)) return false;
        }
        if (usize remaining = SIZE % (sizeof(Block) * CHAR_BIT)) {
            Word mask_array[BLOCK_SIZE] = {};
            for (usize i = 0; i != BLOCK_SIZE; ++i) {
                if (remaining >= WORD_SIZE) {
                    mask_array[i] = ~0ULL;
                    remaining -= WORD_SIZE;
                } else {
                    mask_array[i] = (1ULL << remaining) - 1;
                    break;
                }
            }
            Block mask = _mm256_loadu_si256((const Block*)mask_array),
                  block = block_at(FULL_BLOCKS);
            if (!_mm256_testc_si256(block, mask)) return false;
        }
        return true;
    }

    void set_all() {
        const Block value = _mm256_set1_epi64x(-1);
        for (usize i = 0; i != BLOCK_COUNT; ++i) block_set(i, value);
        trim();
    }

    void unset_all() {
        const Block value = _mm256_setzero_si256();
        for (usize i = 0; i != BLOCK_COUNT; ++i) block_set(i, value);
        trim();
    }

    void flip_all() {
        const Block value = _mm256_set1_epi64x(-1);
        for (usize i = 0; i != BLOCK_COUNT; ++i)
            block_set(i, _mm256_xor_si256(block_at(i), value));
        trim();
    }

    Bitset& operator&=(BitsetExpr auto&& e) {
        for (usize i = 0; i != BLOCK_COUNT; ++i)
            block_set(i, _mm256_and_si256(block_at(i), e.block_at(i)));
        trim();
        return *this;
    }

    Bitset& operator|=(BitsetExpr auto&& e) {
        for (usize i = 0; i != BLOCK_COUNT; ++i)
            block_set(i, _mm256_or_si256(block_at(i), e.block_at(i)));
        trim();
        return *this;
    }

    Bitset& operator^=(BitsetExpr auto&& e) {
        for (usize i = 0; i != BLOCK_COUNT; ++i)
            block_set(i, _mm256_xor_si256(block_at(i), e.block_at(i)));
        trim();
        return *this;
    }

    Bitset& operator-=(BitsetExpr auto&& e) {
        for (usize i = 0; i != BLOCK_COUNT; ++i)
            block_set(i, _mm256_andnot_si256(e.block_at(i), block_at(i)));
        trim();
        return *this;
    }

    friend bool operator==(const Bitset& A, const Bitset& B) {
        for (usize i = 0; i != BLOCK_COUNT; ++i) {
            Block bxa = _mm256_xor_si256(B.block_at(i), A.block_at(i));
            if (!_mm256_testz_si256(bxa, bxa)) return false;
        }
        return true;
    }

    friend bool operator!=(const Bitset& A, const Bitset& B) {
        return !(A == B);
    }

    friend std::partial_ordering operator<=>(const Bitset& A, const Bitset& B) {
        bool a_extra = false, b_extra = false;
        for (usize i = 0; i != BLOCK_COUNT; ++i) {
            Block a = A.block_at(i);
            Block b = B.block_at(i);
            a_extra |= !_mm256_testc_si256(b, a);
            b_extra |= !_mm256_testc_si256(a, b);
            if (a_extra && b_extra) return std::partial_ordering::unordered;
        }
        return a_extra ? std::partial_ordering::greater
            : b_extra  ? std::partial_ordering::less
                       : std::partial_ordering::equivalent;
    }

    usize find_first_set(usize position) const {
        if (position >= SIZE) return usize(-1);
        usize word_index = position / WORD_SIZE,
              bit_in_word = position % WORD_SIZE;
        usize block_index = word_index / BLOCK_SIZE,
              word_in_block = word_index % BLOCK_SIZE;
        if (Word word = data_[word_index] & ~((1ULL << bit_in_word) - 1)) {
            usize result =
                word_index * WORD_SIZE + usize(__builtin_ctzll(word));
            return result < SIZE ? result : usize(-1);
        }
        while (++word_in_block != BLOCK_SIZE) {
            usize current = block_index * BLOCK_SIZE + word_in_block;
            if (current >= WORD_COUNT) break;
            if (Word word = data_[current]) {
                usize result = current * WORD_SIZE + __builtin_ctzll(word);
                return result < SIZE ? result : usize(-1);
            }
        }
        while (++block_index != BLOCK_COUNT) {
            Block mask = _mm256_set1_epi64x(-1), block = block_at(block_index);
            if (_mm256_testz_si256(block, mask)) continue;
            for (usize i = 0; i != BLOCK_SIZE; ++i) {
                usize current = block_index * BLOCK_SIZE + i;
                if (current >= WORD_COUNT) break;
                if (Word word = data_[current]) {
                    usize result = current * WORD_SIZE + __builtin_ctzll(word);
                    return result < SIZE ? result : usize(-1);
                }
            }
        }
        return usize(-1);
    }

    usize find_first_unset(usize position) const {
        if (position >= SIZE) return usize(-1);
        usize word_index = position / WORD_SIZE,
              bit_in_word = position % WORD_SIZE;
        usize block_index = word_index / BLOCK_SIZE,
              word_in_block = word_index % BLOCK_SIZE;
        if (Word word = ~data_[word_index] & ~((1ULL << bit_in_word) - 1)) {
            usize result =
                word_index * WORD_SIZE + usize(__builtin_ctzll(word));
            return result < SIZE ? result : usize(-1);
        }
        while (++word_in_block != BLOCK_SIZE) {
            usize current = block_index * BLOCK_SIZE + word_in_block;
            if (current >= WORD_COUNT) break;
            if (Word word = ~data_[current]) {
                usize result = current * WORD_SIZE + __builtin_ctzll(word);
                return result < SIZE ? result : usize(-1);
            }
        }
        while (++block_index != BLOCK_COUNT) {
            Block mask = _mm256_set1_epi64x(-1), block = block_at(block_index);
            if (_mm256_testc_si256(block, mask)) continue;
            for (usize i = 0; i != BLOCK_SIZE; ++i) {
                usize current = block_index * BLOCK_SIZE + i;
                if (current >= WORD_COUNT) break;
                if (Word word = ~data_[current]) {
                    usize result = current * WORD_SIZE + __builtin_ctzll(word);
                    return result < SIZE ? result : usize(-1);
                }
            }
        }
        return usize(-1);
    }

    void set_range(usize position, usize length) {
        if (position + length > SIZE || !length) return;
        usize word_index = position / WORD_SIZE,
              bit_in_word = position % WORD_SIZE;
        if (bit_in_word) {
            usize head_length = WORD_SIZE - bit_in_word < length
                ? WORD_SIZE - bit_in_word
                : length;
            data_[word_index++] |= ((1ULL << head_length) - 1) << bit_in_word;
            length -= head_length;
        }
        for (
            const Block value = _mm256_set1_epi64x(-1);
            length >= sizeof(Block) * CHAR_BIT;
        ) {
            _mm256_storeu_si256((Block*)&data_[word_index], value);
            length -= sizeof(Block) * CHAR_BIT;
            word_index += BLOCK_SIZE;
        }
        while (length >= WORD_SIZE) {
            data_[word_index++] = -1ULL;
            length -= WORD_SIZE;
        }
        data_[word_index] |= (1ULL << length) - 1;
    }

    void unset_range(usize position, usize length) {
        if (position + length > SIZE || !length) return;
        usize word_index = position / WORD_SIZE,
              bit_in_word = position % WORD_SIZE;
        if (bit_in_word) {
            usize head_length = WORD_SIZE - bit_in_word < length
                ? WORD_SIZE - bit_in_word
                : length;
            data_[word_index++] &=
                ~(((1ULL << head_length) - 1) << bit_in_word);
            length -= head_length;
        }
        for (
            const Block value = _mm256_setzero_si256();
            length >= sizeof(Block) * CHAR_BIT;
        ) {
            _mm256_storeu_si256((Block*)&data_[word_index], value);
            length -= sizeof(Block) * CHAR_BIT;
            word_index += BLOCK_SIZE;
        }
        while (length >= WORD_SIZE) {
            data_[word_index++] = 0ULL;
            length -= WORD_SIZE;
        }
        data_[word_index] &= ~((1ULL << length) - 1);
    }

    void flip_range(usize position, usize length) {
        if (position + length > SIZE || !length) return;
        usize word_index = position / WORD_SIZE,
              bit_in_word = position % WORD_SIZE;
        if (bit_in_word) {
            usize head_length = WORD_SIZE - bit_in_word < length
                ? WORD_SIZE - bit_in_word
                : length;
            data_[word_index++] ^= ((1ULL << head_length) - 1) << bit_in_word;
            length -= head_length;
        }
        for (
            const Block value = _mm256_set1_epi64x(-1);
            length >= sizeof(Block) * CHAR_BIT;
        ) {
            _mm256_storeu_si256(
                (Block*)&data_[word_index],
                _mm256_xor_si256(
                    _mm256_loadu_si256((const Block*)&data_[word_index]), value
                )
            );
            length -= sizeof(Block) * CHAR_BIT;
            word_index += BLOCK_SIZE;
        }
        while (length >= WORD_SIZE) {
            data_[word_index++] ^= -1ULL;
            length -= WORD_SIZE;
        }
        data_[word_index] ^= (1ULL << length) - 1;
    }

    Bitset& operator<<=(usize step) {
        if (step >= SIZE) return unset_all(), *this;
        usize word_shift = step / WORD_SIZE, bit_shift = step % WORD_SIZE;
        if (!bit_shift) {
            memmove(
                data_ + word_shift, data_,
                (WORD_COUNT - word_shift) * sizeof(Word)
            );
            memset(data_, 0, word_shift * sizeof(Word));
        } else {
            usize remaining = WORD_COUNT - word_shift - 1;
            Word *destination = data_ + WORD_COUNT,
                 *source = destination - word_shift;
            __m128i l_shift = _mm_cvtsi32_si128(int(bit_shift)),
                    r_shift = _mm_cvtsi32_si128(int(WORD_SIZE - bit_shift));
            while (remaining >= BLOCK_SIZE) {
                destination -= BLOCK_SIZE, source -= BLOCK_SIZE;
                Block low = _mm256_srl_epi64(
                    _mm256_loadu_si256((const Block*)&source[-1]), r_shift
                );
                Block high = _mm256_sll_epi64(
                    _mm256_loadu_si256((const Block*)&source[0]), l_shift
                );
                _mm256_storeu_si256(
                    (Block*)destination, _mm256_or_si256(low, high)
                );
                remaining -= BLOCK_SIZE;
            }
            while (remaining) {
                --destination, --source;
                *destination = (source[0] << bit_shift)
                    | (source[-1] >> (WORD_SIZE - bit_shift));
                --remaining;
            }
            *--destination = *--source << bit_shift;
            memset(data_, 0, word_shift * sizeof(Word));
        }
        trim();
        return *this;
    }

    Bitset& operator>>=(usize step) {
        if (step >= SIZE) return unset_all(), *this;
        usize word_shift = step / WORD_SIZE, bit_shift = step % WORD_SIZE;
        if (!bit_shift) {
            memmove(
                data_, data_ + word_shift,
                (WORD_COUNT - word_shift) * sizeof(Word)
            );
            memset(
                data_ + WORD_COUNT - word_shift, 0, word_shift * sizeof(Word)
            );
        } else {
            usize remaining = WORD_COUNT - word_shift - 1;
            Word *destination = data_, *source = data_ + word_shift;
            __m128i r_shift = _mm_cvtsi32_si128(int(bit_shift)),
                    l_shift = _mm_cvtsi32_si128(int(WORD_SIZE - bit_shift));
            while (remaining >= BLOCK_SIZE) {
                Block low = _mm256_srl_epi64(
                    _mm256_loadu_si256((const Block*)&source[0]), r_shift
                );
                Block high = _mm256_sll_epi64(
                    _mm256_loadu_si256((const Block*)&source[1]), l_shift
                );
                _mm256_storeu_si256(
                    (Block*)destination, _mm256_or_si256(low, high)
                );
                destination += BLOCK_SIZE, source += BLOCK_SIZE;
                remaining -= BLOCK_SIZE;
            }
            while (remaining) {
                *destination = (source[0] >> bit_shift)
                    | (source[1] << (WORD_SIZE - bit_shift));
                ++destination, ++source;
                --remaining;
            }
            *destination = *source >> bit_shift;
            memset(
                data_ + WORD_COUNT - word_shift, 0, word_shift * sizeof(Word)
            );
        }
        trim();
        return *this;
    }

    Bitset operator<<(usize step) const {
        Bitset result;
        if (step >= SIZE) return result;
        usize word_shift = step / WORD_SIZE, bit_shift = step % WORD_SIZE;
        if (!bit_shift) {
            memcpy(
                result.data_ + word_shift, data_,
                (WORD_COUNT - word_shift) * sizeof(Word)
            );
        } else {
            usize remaining = WORD_COUNT - word_shift - 1;
            Word* destination = result.data_ + WORD_COUNT;
            const Word* source = data_ + WORD_COUNT - word_shift;
            __m128i l_shift = _mm_cvtsi32_si128(int(bit_shift)),
                    r_shift = _mm_cvtsi32_si128(int(WORD_SIZE - bit_shift));
            while (remaining >= BLOCK_SIZE) {
                destination -= BLOCK_SIZE, source -= BLOCK_SIZE;
                Block low = _mm256_srl_epi64(
                    _mm256_loadu_si256((const Block*)&source[-1]), r_shift
                );
                Block high = _mm256_sll_epi64(
                    _mm256_loadu_si256((const Block*)&source[0]), l_shift
                );
                _mm256_storeu_si256(
                    (Block*)destination, _mm256_or_si256(low, high)
                );
                remaining -= BLOCK_SIZE;
            }
            while (remaining) {
                --destination, --source;
                *destination = (source[0] << bit_shift)
                    | (source[-1] >> (WORD_SIZE - bit_shift));
                --remaining;
            }
            *--destination = *--source << bit_shift;
        }
        result.trim();
        return result;
    }

    Bitset operator>>(usize step) const {
        Bitset result;
        if (step >= SIZE) return result;
        usize word_shift = step / WORD_SIZE, bit_shift = step % WORD_SIZE;
        if (!bit_shift) {
            memcpy(
                result.data_, data_ + word_shift,
                (WORD_COUNT - word_shift) * sizeof(Word)
            );
        } else {
            usize remaining = WORD_COUNT - word_shift - 1;
            Word* destination = result.data_;
            const Word* source = data_ + word_shift;
            __m128i r_shift = _mm_cvtsi32_si128(int(bit_shift)),
                    l_shift = _mm_cvtsi32_si128(int(WORD_SIZE - bit_shift));
            while (remaining >= BLOCK_SIZE) {
                Block low = _mm256_srl_epi64(
                    _mm256_loadu_si256((const Block*)&source[0]), r_shift
                );
                Block high = _mm256_sll_epi64(
                    _mm256_loadu_si256((const Block*)&source[1]), l_shift
                );
                _mm256_storeu_si256(
                    (Block*)destination, _mm256_or_si256(low, high)
                );
                destination += BLOCK_SIZE, source += BLOCK_SIZE;
                remaining -= BLOCK_SIZE;
            }
            while (remaining) {
                *destination = (source[0] >> bit_shift)
                    | (source[1] << (WORD_SIZE - bit_shift));
                ++destination, ++source;
                --remaining;
            }
            *destination = *source >> bit_shift;
        }
        result.trim();
        return result;
    }
};

}  // namespace cp
#pragma GCC pop_options

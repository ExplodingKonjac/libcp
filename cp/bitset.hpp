#pragma once
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#pragma GCC push_options
#pragma GCC target("avx2")

namespace cp
{

template <typename E>
struct Expression {
    inline const E& operator()() const {
        return static_cast<const E&>(*this);
    }
};

template <typename E>
struct BitsetFlip : Expression<BitsetFlip<E>> {
    const E& mE;

    BitsetFlip(const E& e) : mE{e} {}

    inline __m256i blockAt(size_t i) const {
        return _mm256_xor_si256(mE.blockAt(i), _mm256_set1_epi64x(-1));
    }
};

template <typename E>
struct BitsetNot : Expression<BitsetNot<E>> {
    const E& mE;

    BitsetNot(const E& e) : mE{e} {}

    inline __m256i blockAt(size_t i) const {
        return _mm256_xor_si256(mE.blockAt(i), _mm256_set1_epi64x(-1));
    }
};

template <typename XE, typename YE>
struct BitsetAnd : Expression<BitsetAnd<XE, YE>> {
    const XE& mXE;
    const YE& mYE;

    BitsetAnd(const XE& xE, const YE& yE) : mXE{xE}, mYE{yE} {}

    inline __m256i blockAt(size_t i) const {
        return _mm256_and_si256(mYE.blockAt(i), mXE.blockAt(i));
    }
};

template <typename XE, typename YE>
struct BitsetOr : Expression<BitsetOr<XE, YE>> {
    const XE& mXE;
    const YE& mYE;

    BitsetOr(const XE& xE, const YE& yE) : mXE{xE}, mYE{yE} {}

    inline __m256i blockAt(size_t i) const {
        return _mm256_or_si256(mYE.blockAt(i), mXE.blockAt(i));
    }
};

template <typename XE, typename YE>
struct BitsetXor : Expression<BitsetXor<XE, YE>> {
    const XE& mXE;
    const YE& mYE;

    BitsetXor(const XE& xE, const YE& yE) : mXE{xE}, mYE{yE} {}

    inline __m256i blockAt(size_t i) const {
        return _mm256_xor_si256(mYE.blockAt(i), mXE.blockAt(i));
    }
};

template <typename XE, typename YE>
struct BitsetAnt : Expression<BitsetAnt<XE, YE>> {
    const XE& mXE;
    const YE& mYE;

    BitsetAnt(const XE& xE, const YE& yE) : mXE{xE}, mYE{yE} {}

    inline __m256i blockAt(size_t i) const {
        return _mm256_andnot_si256(mYE.blockAt(i), mXE.blockAt(i));
    }
};

template <typename E>
inline BitsetFlip<E> operator~(const Expression<E>& e) {
    return BitsetFlip<E>(e());
}

template <typename E>
inline BitsetNot<E> operator!(const Expression<E>& e) {
    return BitsetNot<E>(e());
}

template <typename XE, typename YE>
inline BitsetAnd<XE, YE> operator&(const Expression<XE>& xE, const Expression<YE>& yE) {
    return BitsetAnd<XE, YE>(xE(), yE());
}

template <typename XE, typename YE>
inline BitsetOr<XE, YE> operator|(const Expression<XE>& xE, const Expression<YE>& yE) {
    return BitsetOr<XE, YE>(xE(), yE());
}

template <typename XE, typename YE>
inline BitsetXor<XE, YE> operator^(const Expression<XE>& xE, const Expression<YE>& yE) {
    return BitsetXor<XE, YE>(xE(), yE());
}

template <typename XE, typename YE>
inline BitsetAnt<XE, YE> operator-(const Expression<XE>& xE, const Expression<YE>& yE) {
    return BitsetAnt<XE, YE>(xE(), yE());
}

template <size_t kSize>
class Bitset : public Expression<Bitset<kSize>> {
protected:
    typedef uint64_t Word;
    typedef __m256i Block;

    static const size_t kWordSize = sizeof(Word) * CHAR_BIT;
    static const size_t kBlockSize = sizeof(Block) / sizeof(Word);
    static const size_t kWordCount = (kSize + kWordSize - 1) / kWordSize;
    static const size_t kBlockCount = (kWordCount + kBlockSize - 1) / kBlockSize;

    alignas(Block) Word mData[kBlockCount * kBlockSize]{};

public:
    Bitset() = default;
    ~Bitset() = default;

    inline Word wordAt(size_t position) const {
        return mData[position];
    }

    inline Block blockAt(size_t position) const {
        return _mm256_load_si256((const Block*)&mData[position * kBlockSize]);
    }

protected:
    inline void wordSet(size_t position, Word value) {
        mData[position] = value;
    }

    inline void blockSet(size_t position, Block value) {
        _mm256_store_si256((Block*)&mData[position * kBlockSize], value);
    }

    void trim() {
        if (kSize % kWordSize) mData[kWordCount - 1] &= (1ULL << (kSize % kWordSize)) - 1;
        if (kBlockCount * kBlockSize > kWordCount) memset(mData + kWordCount, 0, (kBlockCount * kBlockSize - kWordCount) * sizeof(Word));
    }

    void normalize() {
        if (kSize % kWordSize) mData[kWordCount - 1] &= (1ULL << (kSize % kWordSize)) - 1;
        if (kBlockCount * kBlockSize > kWordCount) memset(mData + kWordCount, 0, (kBlockCount * kBlockSize - kWordCount) * sizeof(Word));
    }

public:
    Bitset(const Bitset&) = default;
    Bitset(Bitset&&) = default;

    template <typename E>
    Bitset(const Expression<E>& e) {
        for (size_t i = 0; i != kBlockCount; ++i)
            blockSet(i, e().blockAt(i));
        trim();
    }

    Bitset& operator=(const Bitset&) = default;
    Bitset& operator=(Bitset&&) = default;

    template <typename E>
    Bitset& operator=(const Expression<E>& e) {
        for (size_t i = 0; i != kBlockCount; ++i)
            blockSet(i, e().blockAt(i));
        trim();
        return *this;
    }

    bool operator[](size_t position) const {
        if (position >= kSize) return false;
        return mData[position / kWordSize] >> position % kWordSize & 1;
    }

    bool operator()(size_t position) const {
        if (position >= kSize) return false;
        return mData[position / kWordSize] >> position % kWordSize & 1;
    }

    void set(size_t position) {
        if (position >= kSize) return;
        mData[position / kWordSize] |= 1ULL << position % kWordSize;
    }

    void unset(size_t position) {
        if (position >= kSize) return;
        mData[position / kWordSize] &= ~(1ULL << position % kWordSize);
    }

    void flip(size_t position) {
        if (position >= kSize) return;
        mData[position / kWordSize] ^= 1ULL << position % kWordSize;
    }

    size_t size() const {
        return kSize;
    }

    size_t length() const {
        return kSize;
    }

    size_t count() const {
        const Block mask = _mm256_set1_epi8(0x0F);
        const Block table = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
        );
        Block result = _mm256_setzero_si256();
        for (size_t i = 0; i != kBlockCount; ++i) {
            Block block = blockAt(i);
            Block low = _mm256_shuffle_epi8(table, _mm256_and_si256(block, mask));
            Block high = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(block, 4), mask));
            result = _mm256_add_epi64(result, _mm256_sad_epu8(_mm256_add_epi8(low, high), _mm256_setzero_si256()));
        }
        Word parts[kBlockSize];
        _mm256_storeu_si256((Block*)parts, result);
        return size_t(parts[0] + parts[1] + parts[2] + parts[3]);
    }

    size_t popcnt() const {
        const Block mask = _mm256_set1_epi8(0x0F);
        const Block table = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
        );
        Block result = _mm256_setzero_si256();
        for (size_t i = 0; i != kBlockCount; ++i) {
            Block block = blockAt(i);
            Block low = _mm256_shuffle_epi8(table, _mm256_and_si256(block, mask));
            Block high = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(block, 4), mask));
            result = _mm256_add_epi64(result, _mm256_sad_epu8(_mm256_add_epi8(low, high), _mm256_setzero_si256()));
        }
        Word parts[kBlockSize];
        _mm256_storeu_si256((Block*)parts, result);
        return size_t(parts[0] + parts[1] + parts[2] + parts[3]);
    }

    bool none() const {
        const size_t fullBlocks = kSize / (sizeof(Block) * CHAR_BIT);
        for (size_t i = 0; i != fullBlocks; ++i) {
            Block mask = _mm256_set1_epi8(-1), block = blockAt(i);
            if (!_mm256_testz_si256(block, mask)) return false;
        }
        if (size_t remaining = kSize % (sizeof(Block) * CHAR_BIT)) {
            Word maskArray[kBlockSize] = {};
            for (size_t i = 0; i != kBlockSize; ++i) {
                if (remaining >= kWordSize) {
                    maskArray[i] = ~0ULL;
                    remaining -= kWordSize;
                } else {
                    maskArray[i] = (1ULL << remaining) - 1;
                    break;
                }
            }
            Block mask = _mm256_loadu_si256((const Block*)maskArray), block = blockAt(fullBlocks);
            if (!_mm256_testz_si256(block, mask)) return false;
        }
        return true;
    }

    bool any() const {
        const size_t fullBlocks = kSize / (sizeof(Block) * CHAR_BIT);
        for (size_t i = 0; i != fullBlocks; ++i) {
            Block mask = _mm256_set1_epi8(-1), block = blockAt(i);
            if (!_mm256_testz_si256(block, mask)) return true;
        }
        if (size_t remaining = kSize % (sizeof(Block) * CHAR_BIT)) {
            Word maskArray[kBlockSize] = {};
            for (size_t i = 0; i != kBlockSize; ++i) {
                if (remaining >= kWordSize) {
                    maskArray[i] = ~0ULL;
                    remaining -= kWordSize;
                } else {
                    maskArray[i] = (1ULL << remaining) - 1;
                    break;
                }
            }
            Block mask = _mm256_loadu_si256((const Block*)maskArray), block = blockAt(fullBlocks);
            if (!_mm256_testz_si256(block, mask)) return true;
        }
        return false;
    }

    bool all() const {
        const size_t fullBlocks = kSize / (sizeof(Block) * CHAR_BIT);
        for (size_t i = 0; i != fullBlocks; ++i) {
            Block mask = _mm256_set1_epi8(-1), block = blockAt(i);
            if (!_mm256_testc_si256(block, mask)) return false;
        }
        if (size_t remaining = kSize % (sizeof(Block) * CHAR_BIT)) {
            Word maskArray[kBlockSize] = {};
            for (size_t i = 0; i != kBlockSize; ++i) {
                if (remaining >= kWordSize) {
                    maskArray[i] = ~0ULL;
                    remaining -= kWordSize;
                } else {
                    maskArray[i] = (1ULL << remaining) - 1;
                    break;
                }
            }
            Block mask = _mm256_loadu_si256((const Block*)maskArray), block = blockAt(fullBlocks);
            if (!_mm256_testc_si256(block, mask)) return false;
        }
        return true;
    }

    void set() {
        const Block value = _mm256_set1_epi64x(-1);
        for (size_t i = 0; i != kBlockCount; ++i)
            blockSet(i, value);
        trim();
    }

    void unset() {
        const Block value = _mm256_setzero_si256();
        for (size_t i = 0; i != kBlockCount; ++i)
            blockSet(i, value);
        trim();
    }

    void flip() {
        const Block value = _mm256_set1_epi64x(-1);
        for (size_t i = 0; i != kBlockCount; ++i)
            blockSet(i, _mm256_xor_si256(blockAt(i), value));
        trim();
    }

    template <typename E>
    Bitset& operator&=(const Expression<E>& e) {
        for (size_t i = 0; i != kBlockCount; ++i)
            blockSet(i, _mm256_and_si256(blockAt(i), e().blockAt(i)));
        trim();
        return *this;
    }

    template <typename E>
    Bitset& operator|=(const Expression<E>& e) {
        for (size_t i = 0; i != kBlockCount; ++i)
            blockSet(i, _mm256_or_si256(blockAt(i), e().blockAt(i)));
        trim();
        return *this;
    }

    template <typename E>
    Bitset& operator^=(const Expression<E>& e) {
        for (size_t i = 0; i != kBlockCount; ++i)
            blockSet(i, _mm256_xor_si256(blockAt(i), e().blockAt(i)));
        trim();
        return *this;
    }

    template <typename E>
    Bitset& operator-=(const Expression<E>& e) {
        for (size_t i = 0; i != kBlockCount; ++i)
            blockSet(i, _mm256_andnot_si256(e().blockAt(i), blockAt(i)));
        trim();
        return *this;
    }

    friend bool operator==(const Bitset& A, const Bitset& B) {
        for (size_t i = 0; i != kBlockCount; ++i) {
            Block bxa = _mm256_xor_si256(B.blockAt(i), A.blockAt(i));
            if (!_mm256_testz_si256(bxa, bxa)) return false;
        }
        return true;
    }

    friend bool operator!=(const Bitset& A, const Bitset& B) {
        for (size_t i = 0; i != kBlockCount; ++i) {
            Block bxa = _mm256_xor_si256(B.blockAt(i), A.blockAt(i));
            if (!_mm256_testz_si256(bxa, bxa)) return true;
        }
        return false;
    }

    friend bool operator<=(const Bitset& A, const Bitset& B) {
        for (size_t i = 0; i != kBlockCount; ++i)
            if (!_mm256_testc_si256(B.blockAt(i), A.blockAt(i)))
                return false;
        return true;
    }

    friend bool operator>=(const Bitset& A, const Bitset& B) {
        for (size_t i = 0; i != kBlockCount; ++i)
            if (!_mm256_testc_si256(A.blockAt(i), B.blockAt(i)))
                return false;
        return true;
    }

    friend bool operator<(const Bitset& A, const Bitset& B) {
        bool different = false;
        for (size_t i = 0; i != kBlockCount; ++i) {
            Block a = A.blockAt(i), b = B.blockAt(i);
            if (!_mm256_testc_si256(b, a)) return false;
            different |= !_mm256_testc_si256(a, b);
        }
        return different;
    }

    friend bool operator>(const Bitset& A, const Bitset& B) {
        bool different = false;
        for (size_t i = 0; i != kBlockCount; ++i) {
            Block a = A.blockAt(i), b = B.blockAt(i);
            if (!_mm256_testc_si256(a, b)) return false;
            different |= !_mm256_testc_si256(b, a);
        }
        return different;
    }

    size_t findFirstSet(size_t position) const {
        if (position >= kSize) return size_t(-1);
        size_t wordIndex = position / kWordSize, bitInWord = position % kWordSize;
        size_t blockIndex = wordIndex / kBlockSize, wordInBlock = wordIndex % kBlockSize;
        if (Word word = mData[wordIndex] & ~((1ULL << bitInWord) - 1)) {
            size_t result = wordIndex * kWordSize + size_t(__builtin_ctzll(word));
            return result < kSize ? result : size_t(-1);
        }
        while (++wordInBlock != kBlockSize) {
            size_t current = blockIndex * kBlockSize + wordInBlock;
            if (current >= kWordCount) break;
            if (Word word = mData[current]) {
                size_t result = current * kWordSize + __builtin_ctzll(word);
                return result < kSize ? result : size_t(-1);
            }
        }
        while (++blockIndex != kBlockCount) {
            Block mask = _mm256_set1_epi64x(-1), block = blockAt(blockIndex);
            if (_mm256_testz_si256(block, mask)) continue;
            for (size_t i = 0; i != kBlockSize; ++i) {
                size_t current = blockIndex * kBlockSize + i;
                if (current >= kWordCount) break;
                if (Word word = mData[current]) {
                    size_t result = current * kWordSize + __builtin_ctzll(word);
                    return result < kSize ? result : size_t(-1);
                }
            }
        }
        return size_t(-1);
    }

    size_t findFirstUnset(size_t position) const {
        if (position >= kSize) return size_t(-1);
        size_t wordIndex = position / kWordSize, bitInWord = position % kWordSize;
        size_t blockIndex = wordIndex / kBlockSize, wordInBlock = wordIndex % kBlockSize;
        if (Word word = ~mData[wordIndex] & ~((1ULL << bitInWord) - 1)) {
            size_t result = wordIndex * kWordSize + size_t(__builtin_ctzll(word));
            return result < kSize ? result : size_t(-1);
        }
        while (++wordInBlock != kBlockSize) {
            size_t current = blockIndex * kBlockSize + wordInBlock;
            if (current >= kWordCount) break;
            if (Word word = ~mData[current]) {
                size_t result = current * kWordSize + __builtin_ctzll(word);
                return result < kSize ? result : size_t(-1);
            }
        }
        while (++blockIndex != kBlockCount) {
            Block mask = _mm256_set1_epi64x(-1), block = blockAt(blockIndex);
            if (_mm256_testc_si256(block, mask)) continue;
            for (size_t i = 0; i != kBlockSize; ++i) {
                size_t current = blockIndex * kBlockSize + i;
                if (current >= kWordCount) break;
                if (Word word = ~mData[current]) {
                    size_t result = current * kWordSize + __builtin_ctzll(word);
                    return result < kSize ? result : size_t(-1);
                }
            }
        }
        return size_t(-1);
    }

    void set(size_t position, size_t length) {
        if (position + length > kSize || !length) return;
        size_t wordIndex = position / kWordSize, bitInWord = position % kWordSize;
        if (bitInWord) {
            size_t headLength = kWordSize - bitInWord < length ? kWordSize - bitInWord : length;
            mData[wordIndex++] |= ((1ULL << headLength) - 1) << bitInWord;
            length -= headLength;
        }
        for (const Block value = _mm256_set1_epi64x(-1); length >= sizeof(Block) * CHAR_BIT; ) {
            _mm256_storeu_si256((Block*)&mData[wordIndex], value);
            length -= sizeof(Block) * CHAR_BIT;
            wordIndex += kBlockSize;
        }
        while (length >= kWordSize) {
            mData[wordIndex++] = -1ULL;
            length -= kWordSize;
        }
        mData[wordIndex] |= (1ULL << length) - 1;
    }

    void unset(size_t position, size_t length) {
        if (position + length > kSize || !length) return;
        size_t wordIndex = position / kWordSize, bitInWord = position % kWordSize;
        if (bitInWord) {
            size_t headLength = kWordSize - bitInWord < length ? kWordSize - bitInWord : length;
            mData[wordIndex++] &= ~(((1ULL << headLength) - 1) << bitInWord);
            length -= headLength;
        }
        for (const Block value = _mm256_setzero_si256(); length >= sizeof(Block) * CHAR_BIT; ) {
            _mm256_storeu_si256((Block*)&mData[wordIndex], value);
            length -= sizeof(Block) * CHAR_BIT;
            wordIndex += kBlockSize;
        }
        while (length >= kWordSize) {
            mData[wordIndex++] = 0ULL;
            length -= kWordSize;
        }
        mData[wordIndex] &= ~((1ULL << length) - 1);
    }

    void flip(size_t position, size_t length) {
        if (position + length > kSize || !length) return;
        size_t wordIndex = position / kWordSize, bitInWord = position % kWordSize;
        if (bitInWord) {
            size_t headLength = kWordSize - bitInWord < length ? kWordSize - bitInWord : length;
            mData[wordIndex++] ^= ((1ULL << headLength) - 1) << bitInWord;
            length -= headLength;
        }
        for (const Block value = _mm256_set1_epi64x(-1); length >= sizeof(Block) * CHAR_BIT; ) {
            _mm256_storeu_si256((Block*)&mData[wordIndex], _mm256_xor_si256(_mm256_loadu_si256((const Block*)&mData[wordIndex]), value));
            length -= sizeof(Block) * CHAR_BIT;
            wordIndex += kBlockSize;
        }
        while (length >= kWordSize) {
            mData[wordIndex++] ^= -1ULL;
            length -= kWordSize;
        }
        mData[wordIndex] ^= (1ULL << length) - 1;
    }

    Bitset& operator<<=(size_t step) {
        if (step >= kSize) return unset(), *this;
        size_t wordShift = step / kWordSize, bitShift = step % kWordSize;
        if (!bitShift) {
            memmove(mData + wordShift, mData, (kWordCount - wordShift) * sizeof(Word));
            memset(mData, 0, wordShift * sizeof(Word));
        } else {
            size_t remaining = kWordCount - wordShift - 1;
            Word *destination = mData + kWordCount, *source = destination - wordShift;
            __m128i lShift = _mm_cvtsi32_si128(int(bitShift)), rShift = _mm_cvtsi32_si128(int(kWordSize - bitShift));
            while (remaining >= kBlockSize) {
                destination -= kBlockSize, source -= kBlockSize;
                Block low = _mm256_srl_epi64(_mm256_loadu_si256((const Block*)&source[-1]), rShift);
                Block high = _mm256_sll_epi64(_mm256_loadu_si256((const Block*)&source[0]), lShift);
                _mm256_storeu_si256((Block*)destination, _mm256_or_si256(low, high));
                remaining -= kBlockSize;
            }
            while (remaining) {
                --destination, --source;
                *destination = (source[0] << bitShift) | (source[-1] >> (kWordSize - bitShift));
                --remaining;
            }
            *--destination = *--source << bitShift;
            memset(mData, 0, wordShift * sizeof(Word));
        }
        trim();
        return *this;
    }

    Bitset& operator>>=(size_t step) {
        if (step >= kSize) return unset(), *this;
        size_t wordShift = step / kWordSize, bitShift = step % kWordSize;
        if (!bitShift) {
            memmove(mData, mData + wordShift, (kWordCount - wordShift) * sizeof(Word));
            memset(mData + kWordCount - wordShift, 0, wordShift * sizeof(Word));
        } else {
            size_t remaining = kWordCount - wordShift - 1;
            Word *destination = mData, *source = mData + wordShift;
            __m128i rShift = _mm_cvtsi32_si128(int(bitShift)), lShift = _mm_cvtsi32_si128(int(kWordSize - bitShift));
            while (remaining >= kBlockSize) {
                Block low = _mm256_srl_epi64(_mm256_loadu_si256((const Block*)&source[0]), rShift);
                Block high = _mm256_sll_epi64(_mm256_loadu_si256((const Block*)&source[1]), lShift);
                _mm256_storeu_si256((Block*)destination, _mm256_or_si256(low, high));
                destination += kBlockSize, source += kBlockSize;
                remaining -= kBlockSize;
            }
            while (remaining) {
                *destination = (source[0] >> bitShift) | (source[1] << (kWordSize - bitShift));
                ++destination, ++source;
                --remaining;
            }
            *destination = *source >> bitShift;
            memset(mData + kWordCount - wordShift, 0, wordShift * sizeof(Word));
        }
        trim();
        return *this;
    }

    Bitset operator<<(size_t step) const {
        Bitset result;
        if (step >= kSize) return result;
        size_t wordShift = step / kWordSize, bitShift = step % kWordSize;
        if (!bitShift) {
            memcpy(result.mData + wordShift, mData, (kWordCount - wordShift) * sizeof(Word));
        } else {
            size_t remaining = kWordCount - wordShift - 1;
            Word* destination = result.mData + kWordCount;
            const Word* source = mData + kWordCount - wordShift;
            __m128i lShift = _mm_cvtsi32_si128(int(bitShift)), rShift = _mm_cvtsi32_si128(int(kWordSize - bitShift));
            while (remaining >= kBlockSize) {
                destination -= kBlockSize, source -= kBlockSize;
                Block low = _mm256_srl_epi64(_mm256_loadu_si256((const Block*)&source[-1]), rShift);
                Block high = _mm256_sll_epi64(_mm256_loadu_si256((const Block*)&source[0]), lShift);
                _mm256_storeu_si256((Block*)destination, _mm256_or_si256(low, high));
                remaining -= kBlockSize;
            }
            while (remaining) {
                --destination, --source;
                *destination = (source[0] << bitShift) | (source[-1] >> (kWordSize - bitShift));
                --remaining;
            }
            *--destination = *--source << bitShift;
        }
        result.trim();
        return result;
    }

    Bitset operator>>(size_t step) const {
        Bitset result;
        if (step >= kSize) return result;
        size_t wordShift = step / kWordSize, bitShift = step % kWordSize;
        if (!bitShift) {
            memcpy(result.mData, mData + wordShift, (kWordCount - wordShift) * sizeof(Word));
        } else {
            size_t remaining = kWordCount - wordShift - 1;
            Word* destination = result.mData;
            const Word* source = mData + wordShift;
            __m128i rShift = _mm_cvtsi32_si128(int(bitShift)), lShift = _mm_cvtsi32_si128(int(kWordSize - bitShift));
            while (remaining >= kBlockSize) {
                Block low = _mm256_srl_epi64(_mm256_loadu_si256((const Block*)&source[0]), rShift);
                Block high = _mm256_sll_epi64(_mm256_loadu_si256((const Block*)&source[1]), lShift);
                _mm256_storeu_si256((Block*)destination, _mm256_or_si256(low, high));
                destination += kBlockSize, source += kBlockSize;
                remaining -= kBlockSize;
            }
            while (remaining) {
                *destination = (source[0] >> bitShift) | (source[1] << (kWordSize - bitShift));
                ++destination, ++source;
                --remaining;
            }
            *destination = *source >> bitShift;
        }
        result.trim();
        return result;
    }
};

} // namespace cp
#pragma GCC pop_options

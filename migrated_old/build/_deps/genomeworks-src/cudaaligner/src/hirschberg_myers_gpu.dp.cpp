/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "hirschberg_myers_gpu.dp.hpp"
#include <cassert>
#include "batched_device_matrices.dp.hpp"
#include <claraparabricks/genomeworks/cudaaligner/aligner.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/mathutils.hpp>
#include <claraparabricks/genomeworks/utils/limits.cuh>
#include <cstring>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

namespace hirschbergmyers
{

constexpr int32_t warp_size = 32;
constexpr int32_t word_size = sizeof(WordType) * CHAR_BIT;

inline WordType warp_leftshift_sync(uint32_t warp_mask, WordType v,
                                    sycl::nd_item<3> item_ct1)
{
    assert(((warp_mask >> threadIdx.x) & 1u) == 1);
const WordType x = sycl::shift_group_right(item_ct1.get_sub_group(), v >> (word_size - 1), 1);    v <<= 1;
    if (item_ct1.get_local_id(2) != 0)
        v |= x;
    return v;
}

inline WordType warp_add_sync(uint32_t warp_mask, WordType a, WordType b,
                              sycl::nd_item<3> item_ct1)
{
    assert(((warp_mask >> threadIdx.x) & 1u) == 1);
    static_assert(sizeof(WordType) == 4, "This function assumes WordType to have 4 bytes.");
    static_assert(CHAR_BIT == 8, "This function assumes a char width of 8 bit.");
    const uint64_t ax = a;
    const uint64_t bx = b;
    uint64_t r        = ax + bx;
    uint32_t carry    = static_cast<uint32_t>(r >> 32);
    if (warp_mask == 1)
    {
        return static_cast<WordType>(r);
    }
    r &= 0xffff'ffffull;
    // TODO: I think due to the structure of the Myer blocks,
    // a carry cannot propagate over more than a single block.
    // I.e. a single carry propagation without the loop should be sufficient.
    while (sycl::any_of_group(
        item_ct1.get_sub_group(),
        (warp_mask & (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
            carry))
    {
uint32_t x = sycl::shift_group_right(item_ct1.get_sub_group(), carry, 1);        if (item_ct1.get_local_id(2) != 0)
            r += x;
        carry = static_cast<uint32_t>(r >> 32);
        r &= 0xffff'ffffull;
    }
    return static_cast<WordType>(r);
}

int32_t myers_advance_block(uint32_t warp_mask, WordType highest_bit, WordType eq, WordType& pv, WordType& mv, int32_t carry_in,
                            sycl::nd_item<3> item_ct1)
{
    assert((pv & mv) == WordType(0));

    // Stage 1
    WordType xv = eq | mv;
    if (carry_in < 0)
        eq |= WordType(1);
    WordType xh = warp_add_sync(warp_mask, eq & pv, pv, item_ct1);
    xh          = (xh ^ pv) | eq;
    WordType ph = mv | (~(xh | pv));
    WordType mh = pv & xh;

    int32_t carry_out = ((ph & highest_bit) == WordType(0) ? 0 : 1) - ((mh & highest_bit) == WordType(0) ? 0 : 1);

    ph = warp_leftshift_sync(warp_mask, ph, item_ct1);
    mh = warp_leftshift_sync(warp_mask, mh, item_ct1);

    if (carry_in < 0)
        mh |= WordType(1);

    if (carry_in > 0)
        ph |= WordType(1);

    // Stage 2
    pv = mh | (~(xv | ph));
    mv = ph & xv;

    return carry_out;
}

inline int32_t get_myers_score(int32_t i, int32_t j, device_matrix_view<WordType> const& pv, device_matrix_view<WordType> const& mv, device_matrix_view<int32_t> const& score, WordType last_entry_mask)
{
    assert(i > 0); // row 0 is implicit, NW matrix is shifted by i -> i-1
    const int32_t word_idx = (i - 1) / word_size;
    const int32_t bit_idx  = (i - 1) % word_size;
    int32_t s              = score(word_idx, j);
    WordType mask          = (~WordType(1)) << bit_idx;
    if (word_idx == score.num_rows() - 1)
        mask &= last_entry_mask;
    s -= sycl::popcount(mask & pv(word_idx, j));
    s += sycl::popcount(mask & mv(word_idx, j));
    return s;
}

int32_t append_myers_backtrace(int8_t* path, device_matrix_view<WordType> const& pv, device_matrix_view<WordType> const& mv, device_matrix_view<int32_t> const& score, int32_t query_size)
{
    assert(threadIdx.x == 0);
    using nw_score_t = int32_t;
    assert(pv.num_rows() == score.num_rows());
    assert(mv.num_rows() == score.num_rows());
    assert(pv.num_cols() == score.num_cols());
    assert(mv.num_cols() == score.num_cols());
    assert(score.num_rows() == ceiling_divide(query_size, word_size));
    int32_t i = query_size;
    int32_t j = score.num_cols() - 1;

    const WordType last_entry_mask = query_size % word_size != 0 ? (WordType(1) << (query_size % word_size)) - 1 : ~WordType(0);

    nw_score_t myscore = score((i - 1) / word_size, j);
    int32_t pos        = 0;
    while (i > 0 && j > 0)
    {
        int8_t r               = 0;
        nw_score_t const above = i == 1 ? j : get_myers_score(i - 1, j, pv, mv, score, last_entry_mask);
        nw_score_t const diag  = i == 1 ? j - 1 : get_myers_score(i - 1, j - 1, pv, mv, score, last_entry_mask);
        nw_score_t const left  = get_myers_score(i, j - 1, pv, mv, score, last_entry_mask);
        if (left + 1 == myscore)
        {
            r       = static_cast<int8_t>(AlignmentState::insertion);
            myscore = left;
            --j;
        }
        else if (above + 1 == myscore)
        {
            r       = static_cast<int8_t>(AlignmentState::deletion);
            myscore = above;
            --i;
        }
        else
        {
            r       = (diag == myscore ? static_cast<int8_t>(AlignmentState::match) : static_cast<int8_t>(AlignmentState::mismatch));
            myscore = diag;
            --i;
            --j;
        }
        path[pos] = r;
        ++pos;
    }
    while (i > 0)
    {
        path[pos] = static_cast<int8_t>(AlignmentState::deletion);
        ++pos;
        --i;
    }
    while (j > 0)
    {
        path[pos] = static_cast<int8_t>(AlignmentState::insertion);
        ++pos;
        --j;
    }
    return pos;
}

inline void hirschberg_myers_fill_path_warp(int8_t*& path, int32_t* path_length, int32_t n, int8_t value,
                                            sycl::nd_item<3> item_ct1)
{
    // TODO parallelize
    if (item_ct1.get_local_id(2) == 0)
    {
        int8_t const* const path_end = path + n;
        while (path != path_end)
        {
            *path = value;
            ++path;
        }
        *path_length += n;
    }
}

WordType myers_generate_query_pattern(char x, char const* query, int32_t query_size, int32_t offset)
{
    // Sets a 1 bit at the position of every matching character
    assert(offset < query_size);
    const int32_t max_i = sycl::min((int)(query_size - offset), (int)word_size);
    WordType r          = 0;
    for (int32_t i = 0; i < max_i; ++i)
    {
        if (x == query[i + offset])
            r = r | (WordType(1) << i);
    }
    return r;
}

WordType myers_generate_query_pattern_reverse(char x, char const* query, int32_t query_size, int32_t offset)
{
    // Sets a 1 bit at the position of every matching character
    assert(offset < query_size);
    const int32_t max_i = sycl::min((int)(query_size - offset), (int)word_size);
    WordType r          = 0;
    // TODO make this a forward loop
    for (int32_t i = 0; i < max_i; ++i)
    {
        if (x == query[query_size - 1 - (i + offset)])
            r = r | (WordType(1) << i);
    }
    return r;
}

void myers_preprocess(device_matrix_view<WordType>& query_pattern, char const* query, int32_t query_size,
                      sycl::nd_item<3> item_ct1)
{
    const int32_t n_words = ceiling_divide(query_size, word_size);
    for (int32_t idx = item_ct1.get_local_id(2); idx < n_words;
         idx += warp_size)
    {
        // TODO query load is inefficient
        query_pattern(idx, 0) = myers_generate_query_pattern('A', query, query_size, idx * word_size);
        query_pattern(idx, 1) = myers_generate_query_pattern('C', query, query_size, idx * word_size);
        query_pattern(idx, 2) = myers_generate_query_pattern('T', query, query_size, idx * word_size);
        query_pattern(idx, 3) = myers_generate_query_pattern('G', query, query_size, idx * word_size);
        query_pattern(idx, 4) = myers_generate_query_pattern_reverse('A', query, query_size, idx * word_size);
        query_pattern(idx, 5) = myers_generate_query_pattern_reverse('C', query, query_size, idx * word_size);
        query_pattern(idx, 6) = myers_generate_query_pattern_reverse('T', query, query_size, idx * word_size);
        query_pattern(idx, 7) = myers_generate_query_pattern_reverse('G', query, query_size, idx * word_size);
    }
}

inline WordType get_query_pattern(device_matrix_view<WordType>& query_patterns, int32_t idx, int32_t query_begin_offset, char x, bool reverse)
{
    static_assert(std::is_unsigned<WordType>::value, "WordType has to be an unsigned type for well-defined >> operations.");
    const int32_t char_idx = [](char x) -> int32_t {
        int32_t r = x;
        return (r >> 1) & 0x3;
    }(x) + (reverse ? 4 : 0);

    // 4-bit word example:
    // query_patterns contains character match bit patterns "XXXX" for the full query string.
    // we want the bit pattern "yyyy" for a view of on the query string starting at eg. character 11:
    //       4    3    2     1      0 (pattern index)
    //    XXXX XXXX XXXX [XXXX] [XXXX]
    //     YYY Yyyy y
    //         1    0 (idx)
    //
    // query_begin_offset = 11
    // => idx_offset = 11/4 = 2, shift = 11%4 = 3

    const int32_t idx_offset = query_begin_offset / word_size;
    const int32_t shift      = query_begin_offset % word_size;

    WordType r = query_patterns(idx + idx_offset, char_idx);
    if (shift != 0)
    {
        r >>= shift;
        if (idx + idx_offset + 1 < query_patterns.num_rows())
        {
            r |= query_patterns(idx + idx_offset + 1, char_idx) << (word_size - shift);
        }
    }
    return r;
}

void
myers_compute_scores(
    device_matrix_view<WordType>& pv,
    device_matrix_view<WordType>& mv,
    device_matrix_view<int32_t>& score,
    device_matrix_view<WordType>& query_patterns,
    char const* target_begin,
    char const* target_end,
    char const* query_begin,
    char const* query_end,
    int32_t const pattern_idx_offset,
    bool full_score_matrix,
    bool reverse,
    sycl::nd_item<3> item_ct1)
{
    assert(warpSize == warp_size);
    assert(threadIdx.x < warp_size);
    assert(blockIdx.x == 0);

    assert(query_end - query_begin > 0);
    assert(target_begin < target_end);

    const int32_t n_words     = ceiling_divide<int32_t>(query_end - query_begin, word_size);
    const int32_t target_size = target_end - target_begin;

    assert(pv.num_rows() == n_words);
    assert(mv.num_rows() == n_words);
    assert(score.num_rows() == full_score_matrix ? n_words : target_size + 1);
    assert(pv.num_cols() == full_score_matrix ? target_size + 1 : 1);
    assert(mv.num_cols() == full_score_matrix ? target_size + 1 : 1);
    assert(score.num_cols() == full_score_matrix ? target_size + 1 : 2);

    {
        for (int32_t idx = item_ct1.get_local_id(2); idx < n_words;
             idx += warp_size)
        {
            pv(idx, 0) = ~WordType(0);
            mv(idx, 0) = 0;
        }

        const int32_t query_size = query_end - query_begin;
        if (full_score_matrix)
        {
            for (int32_t idx = item_ct1.get_local_id(2); idx < n_words;
                 idx += warp_size)
                score(idx, 0) =
                    sycl::min((int)((idx + 1) * word_size), (int)query_size);
        }
        else
        {
            if (item_ct1.get_local_id(2) == 0)
                score(0, reverse ? 1 : 0) = query_size;
        }
    }

    const int32_t n_warp_iterations = ceiling_divide(n_words, warp_size) * warp_size;
    item_ct1.barrier();
    for (int32_t t = 1; t <= target_size; ++t)
    {
        int32_t warp_carry = 0;
        if (item_ct1.get_local_id(2) == 0)
        {
            warp_carry = 1; // for global alignment the (implicit) first row has to be 0,1,2,3,... -> carry 1
        }
        for (int32_t idx = item_ct1.get_local_id(2); idx < n_warp_iterations;
             idx += warp_size)
        {
            if (idx < n_words)
            {
                const uint32_t warp_mask = idx / warp_size < n_words / warp_size ? 0xffff'ffffu : (1u << (n_words % warp_size)) - 1;

                WordType pv_local = pv(idx, full_score_matrix ? t - 1 : 0);
                WordType mv_local = mv(idx, full_score_matrix ? t - 1 : 0);
                // TODO these might be cached or only computed for the specific t at hand.
                const WordType highest_bit = WordType(1) << (idx == (n_words - 1) ? (query_end - query_begin) - (n_words - 1) * word_size - 1 : word_size - 1);

                const WordType eq = get_query_pattern(query_patterns, idx, pattern_idx_offset, target_begin[reverse ? target_size - t : t - 1], reverse);

                warp_carry =
                    myers_advance_block(warp_mask, highest_bit, eq, pv_local,
                                        mv_local, warp_carry, item_ct1);
                if (full_score_matrix)
                {
                    score(idx, t) = score(idx, t - 1) + warp_carry;
                }
                else
                {
                    if (idx + 1 == n_words)
                    {
                        score(t, reverse ? 1 : 0) = score(t - 1, reverse ? 1 : 0) + warp_carry;
                    }
                }
                if (item_ct1.get_local_id(2) == 0)
                {
                    warp_carry = 0;
                }
                if (warp_mask == 0xffff'ffffu &&
                    (item_ct1.get_local_id(2) == 31 ||
                     item_ct1.get_local_id(2) == 0))
                {
warp_carry = sycl::shift_group_left( item_ct1.get_sub_group(), warp_carry, warp_size - 1);                }
                if (item_ct1.get_local_id(2) != 0)
                {
                    warp_carry = 0;
                }
                pv(idx, full_score_matrix ? t : 0) = pv_local;
                mv(idx, full_score_matrix ? t : 0) = mv_local;
            }
            item_ct1.barrier();
        }
    }
}

void hirschberg_myers_compute_path(
    int8_t*& path,
    int32_t* path_length,
    batched_device_matrices<WordType>::device_interface* pvi,
    batched_device_matrices<WordType>::device_interface* mvi,
    batched_device_matrices<int32_t>::device_interface* scorei,
    device_matrix_view<WordType>& query_patterns,
    char const* target_begin,
    char const* target_end,
    char const* query_begin,
    char const* query_end,
    char const* query_begin_absolute,
    int32_t alignment_idx,
    sycl::nd_item<3> item_ct1)
{
    assert(query_begin < query_end);
    const int32_t n_words             = ceiling_divide<int32_t>(query_end - query_begin, word_size);
    device_matrix_view<int32_t> score = scorei->get_matrix_view(alignment_idx, n_words, target_end - target_begin + 1);
    device_matrix_view<WordType> pv   = pvi->get_matrix_view(alignment_idx, n_words, target_end - target_begin + 1);
    device_matrix_view<WordType> mv   = mvi->get_matrix_view(alignment_idx, n_words, target_end - target_begin + 1);
    myers_compute_scores(
        pv, mv, score, query_patterns, target_begin, target_end, query_begin,
        query_end, query_begin - query_begin_absolute, true, false, item_ct1);
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0)
    {
        int32_t len = append_myers_backtrace(path, pv, mv, score, query_end - query_begin);
        path += len;
        *path_length += len;
    }
}

const char* hirschberg_myers_compute_target_mid_warp(
    batched_device_matrices<WordType>::device_interface* pvi,
    batched_device_matrices<WordType>::device_interface* mvi,
    batched_device_matrices<int32_t>::device_interface* scorei,
    device_matrix_view<WordType>& query_patterns,
    char const* target_begin,
    char const* target_end,
    char const* query_begin,
    char const* query_mid,
    char const* query_end,
    char const* query_begin_absolute,
    char const* query_end_absolute,
    int32_t alignment_idx,
    sycl::nd_item<3> item_ct1)
{
    assert(query_begin <= query_mid);
    assert(query_mid < query_end);
    assert(target_begin < target_end);

    device_matrix_view<int32_t> score = scorei->get_matrix_view(alignment_idx, target_end - target_begin + 1, 2);

    if (query_begin < query_mid)
    {
        const int32_t n_words           = ceiling_divide<int32_t>(query_mid - query_begin, word_size);
        device_matrix_view<WordType> pv = pvi->get_matrix_view(alignment_idx, n_words, 2);
        device_matrix_view<WordType> mv = mvi->get_matrix_view(alignment_idx, n_words, 2);
        myers_compute_scores(pv, mv, score, query_patterns, target_begin,
                             target_end, query_begin, query_mid,
                             query_begin - query_begin_absolute, false, false,
                             item_ct1);
    }
    else
    {
        const int32_t target_size = (target_end - target_begin);
        for (int32_t t = item_ct1.get_local_id(2); t <= target_size;
             t += warp_size)
        {
            score(t, 0) = t;
        }
        item_ct1.barrier();
    }

    {
        const int32_t n_words           = ceiling_divide<int32_t>(query_end - query_mid, word_size);
        device_matrix_view<WordType> pv = pvi->get_matrix_view(alignment_idx, n_words, 2);
        device_matrix_view<WordType> mv = mvi->get_matrix_view(alignment_idx, n_words, 2);
        myers_compute_scores(
            pv, mv, score, query_patterns, target_begin, target_end, query_mid,
            query_end, query_end_absolute - query_end, false, true, item_ct1);
    }

    const int32_t target_size = (target_end - target_begin);
    int32_t midpoint          = 0;
    nw_score_t cur_min        = numeric_limits<nw_score_t>::max();
    for (int32_t t = item_ct1.get_local_id(2); t <= target_size; t += warp_size)
    {
        nw_score_t sum = score(t, 0) + score(target_size - t, 1);
        if (sum < cur_min)
        {
            cur_min  = sum;
            midpoint = t;
        }
    }
#pragma unroll
    for (int32_t i = 16; i > 0; i >>= 1)
    {
        const int32_t mv = sycl::shift_group_left(item_ct1.get_sub_group(), cur_min, i);
const int32_t mp = sycl::shift_group_left(item_ct1.get_sub_group(), midpoint, i);        if (mv < cur_min)
        {
            cur_min  = mv;
            midpoint = mp;
        }
    }
sycl::select_from_group(item_ct1.get_sub_group(), midpoint, 0);    return target_begin + midpoint;
}

void hirschberg_myers_single_char_warp(int8_t*& path, int32_t* path_length, char query_char, char const* target_begin, char const* target_end,
                                       sycl::nd_item<3> item_ct1)
{
    // TODO parallelize
    if (item_ct1.get_local_id(2) == 0)
    {
        char const* t = target_end - 1;
        while (t >= target_begin)
        {
            if (*t == query_char)
            {
                *path = static_cast<int8_t>(AlignmentState::match);
                ++path;
                --t;
                break;
            }
            *path = static_cast<int8_t>(AlignmentState::insertion);
            ++path;
            --t;
        }
        if (*(path - 1) != static_cast<int8_t>(AlignmentState::match))
        {
            *(path - 1) = static_cast<int8_t>(AlignmentState::mismatch);
        }
        while (t >= target_begin)
        {
            *path = static_cast<int8_t>(AlignmentState::insertion);
            ++path;
            --t;
        }
        *path_length += target_end - target_begin;
    }
}

template <typename T>
class warp_shared_stack
{
public:
    warp_shared_stack(T* buffer_begin, T* buffer_end)
        : buffer_begin_(buffer_begin)
        , cur_end_(buffer_begin)
        , buffer_end_(buffer_end)
    {
        assert(buffer_begin_ < buffer_end_);
    }

    bool inline push(T const& t, sycl::nd_item<3> item_ct1,
                     const sycl::stream &stream_ct1, unsigned warp_mask = 0xffff'ffffu)
    {
        if (buffer_end_ - cur_end_ >= 1)
        {
            item_ct1.barrier();
            if (item_ct1.get_local_id(2) == 0)
            {
                *cur_end_ = t;
            }
            item_ct1.barrier();
            ++cur_end_;
            return true;
        }
        else
        {
            if (item_ct1.get_local_id(2) == 0)
            {
                stream_ct1 << "ERROR: stack full!";
            }
            return false;
        }
    }

    inline void pop()
    {
        assert(cur_end_ > buffer_begin_);
        if (cur_end_ - 1 >= buffer_begin_)
            --cur_end_;
    }

    inline T back() const
    {
        assert(cur_end_ - 1 >= buffer_begin_);
        return *(cur_end_ - 1);
    }

    inline bool empty() const
    {
        return buffer_begin_ == cur_end_;
    }

private:
    T* buffer_begin_;
    T* cur_end_;
    T* buffer_end_;
};

void hirschberg_myers(
    query_target_range* stack_buffer_begin,
    query_target_range* stack_buffer_end,
    int8_t*& path,
    int32_t* path_length,
    int32_t full_myers_threshold,
    batched_device_matrices<WordType>::device_interface* pvi,
    batched_device_matrices<WordType>::device_interface* mvi,
    batched_device_matrices<int32_t>::device_interface* scorei,
    device_matrix_view<WordType>& query_patterns,
    char const* target_begin_absolute,
    char const* target_end_absolute,
    char const* query_begin_absolute,
    char const* query_end_absolute,
    int32_t alignment_idx,
    sycl::nd_item<3> item_ct1,
    const sycl::stream &stream_ct1)
{
    assert(blockDim.x == warp_size);
    assert(blockDim.z == 1);
    assert(query_begin_absolute <= query_end_absolute);
    assert(target_begin_absolute <= target_end_absolute);

    warp_shared_stack<query_target_range> stack(stack_buffer_begin, stack_buffer_end);
    stack.push({query_begin_absolute, query_end_absolute, target_begin_absolute,
                target_end_absolute},
               item_ct1, stream_ct1);

    assert(pvi->get_max_elements_per_matrix(alignment_idx) == mvi->get_max_elements_per_matrix(alignment_idx));
    assert(scorei->get_max_elements_per_matrix(alignment_idx) >= pvi->get_max_elements_per_matrix(alignment_idx));

    bool success   = true;
    int32_t length = 0;
    while (success && !stack.empty())
    {
        query_target_range e = stack.back();
        stack.pop();
        assert(e.query_begin <= e.query_end);
        assert(e.target_begin <= e.target_end);
        if (e.target_begin == e.target_end)
        {
            hirschberg_myers_fill_path_warp(
                path, &length, e.query_end - e.query_begin,
                static_cast<int8_t>(AlignmentState::deletion), item_ct1);
        }
        else if (e.query_begin == e.query_end)
        {
            hirschberg_myers_fill_path_warp(
                path, &length, e.target_end - e.target_begin,
                static_cast<int8_t>(AlignmentState::insertion), item_ct1);
        }
        else if (e.query_begin + 1 == e.query_end)
        {
            hirschberg_myers_single_char_warp(path, &length, *e.query_begin,
                                              e.target_begin, e.target_end,
                                              item_ct1);
        }
        else
        {
            if (e.query_end - e.query_begin < full_myers_threshold && e.query_end != e.query_begin)
            {
                const int32_t n_words = ceiling_divide<int32_t>(e.query_end - e.query_begin, word_size);
                if ((e.target_end - e.target_begin + 1) * n_words <= pvi->get_max_elements_per_matrix(alignment_idx))
                {
                    hirschberg_myers_compute_path(
                        path, &length, pvi, mvi, scorei, query_patterns,
                        e.target_begin, e.target_end, e.query_begin,
                        e.query_end, query_begin_absolute, alignment_idx,
                        item_ct1);
                    continue;
                }
            }

            const char* query_mid  = e.query_begin + (e.query_end - e.query_begin) / 2;
            const char *target_mid = hirschberg_myers_compute_target_mid_warp(
                pvi, mvi, scorei, query_patterns, e.target_begin, e.target_end,
                e.query_begin, query_mid, e.query_end, query_begin_absolute,
                query_end_absolute, alignment_idx, item_ct1);
            success = success && stack.push({e.query_begin, query_mid,
                                             e.target_begin, target_mid},
                                            item_ct1, stream_ct1);
            success = success && stack.push({query_mid, e.query_end, target_mid,
                                             e.target_end},
                                            item_ct1, stream_ct1);
        }
    }
    if (!success)
        length = 0;
    if (item_ct1.get_local_id(2) == 0)
        *path_length = length;
}

void hirschberg_myers_compute_alignment(
    query_target_range* stack_buffer_base,
    int32_t stack_buffer_size_per_alignment,
    int32_t full_myers_threshold,
    int8_t* paths_base,
    int32_t* path_lengths,
    int32_t max_path_length,
    batched_device_matrices<WordType>::device_interface* pvi,
    batched_device_matrices<WordType>::device_interface* mvi,
    batched_device_matrices<int32_t>::device_interface* scorei,
    batched_device_matrices<WordType>::device_interface* query_patternsi,
    char const* sequences_d, int32_t const* sequence_lengths_d,
    int32_t max_sequence_length,
    int32_t n_alignments, sycl::nd_item<3> item_ct1,
    const sycl::stream &stream_ct1)
{
    assert(blockDim.x == warp_size);
    assert(blockDim.z == 1);

    const int32_t alignment_idx = item_ct1.get_group(0);
    if (alignment_idx >= n_alignments)
        return;

    const char* const query_begin               = sequences_d + 2 * alignment_idx * max_sequence_length;
    const char* const target_begin              = sequences_d + (2 * alignment_idx + 1) * max_sequence_length;
    const char* const query_end                 = query_begin + sequence_lengths_d[2 * alignment_idx];
    const char* const target_end                = target_begin + sequence_lengths_d[2 * alignment_idx + 1];
    int8_t* path                                = paths_base + alignment_idx * max_path_length;
    query_target_range* stack_buffer_begin      = stack_buffer_base + alignment_idx * stack_buffer_size_per_alignment;
    device_matrix_view<WordType> query_patterns = query_patternsi->get_matrix_view(alignment_idx, ceiling_divide<int32_t>(query_end - query_begin, word_size), 8);
    myers_preprocess(query_patterns, query_begin, query_end - query_begin,
                     item_ct1);
    hirschberg_myers(stack_buffer_begin,
                     stack_buffer_begin + stack_buffer_size_per_alignment, path,
                     path_lengths + alignment_idx, full_myers_threshold, pvi,
                     mvi, scorei, query_patterns, target_begin, target_end,
                     query_begin, query_end, alignment_idx, item_ct1,
                     stream_ct1);
}

} // namespace hirschbergmyers

void hirschberg_myers_gpu(
    device_buffer<hirschbergmyers::query_target_range> &stack_buffer,
    int32_t stack_buffer_size_per_alignment, int8_t *paths_d,
    int32_t *path_lengths_d, int32_t max_path_length, char const *sequences_d,
    int32_t const *sequence_lengths_d, int32_t max_sequence_length,
    int32_t n_alignments,
    batched_device_matrices<hirschbergmyers::WordType> &pv,
    batched_device_matrices<hirschbergmyers::WordType> &mv,
    batched_device_matrices<int32_t> &score,
    batched_device_matrices<hirschbergmyers::WordType> &query_patterns,
    int32_t switch_to_myers_threshold, sycl::queue *stream)
{
    using hirschbergmyers::warp_size;

    const sycl::range<3> threads(1, 1, warp_size);
    const sycl::range<3> blocks(
        ceiling_divide<int32_t>(n_alignments, threads[0]), 1, 1);
    /*
    DPCT1049:111: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    stream->submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        auto stack_buffer_data_ct0 = stack_buffer.data();
        auto pv_get_device_interface_ct6 = pv.get_device_interface();
        auto mv_get_device_interface_ct7 = mv.get_device_interface();
        auto score_get_device_interface_ct8 = score.get_device_interface();
        auto query_patterns_get_device_interface_ct9 =
            query_patterns.get_device_interface();

        cgh.parallel_for(
            sycl::nd_range<3>(blocks * threads, threads), [=
        ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                hirschbergmyers::hirschberg_myers_compute_alignment(
                    stack_buffer_data_ct0, stack_buffer_size_per_alignment,
                    switch_to_myers_threshold, paths_d, path_lengths_d,
                    max_path_length, pv_get_device_interface_ct6,
                    mv_get_device_interface_ct7, score_get_device_interface_ct8,
                    query_patterns_get_device_interface_ct9, sequences_d,
                    sequence_lengths_d, max_sequence_length, n_alignments,
                    item_ct1, stream_ct1);
            });
    });
    /*
    DPCT1010:112: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    GW_CU_CHECK_ERR(0);
}

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks

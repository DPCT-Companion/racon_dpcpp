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
#include "ukkonen_gpu.dp.hpp"
#include "batched_device_matrices.dp.hpp"
#include <claraparabricks/genomeworks/cudaaligner/cudaaligner.hpp>
#include <claraparabricks/genomeworks/utils/limits.cuh>

#include <limits>
#include <cstdint>
#include <algorithm>
#include <cassert>

#define GW_UKKONEN_MAX_THREADS_PER_BLOCK 1024

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

namespace kernels
{

template <typename T>
T min3(T t1, T t2, T t3)
{
    return sycl::min(t1, sycl::min(t2, t3));
}

std::tuple<int, int> to_matrix_indices(int k, int l, int p)
{
    int const j = k - (p + l) / 2 + l;
    int const i = l - j;
    return std::make_tuple(i, j);
}

std::tuple<int, int> to_band_indices(int i, int j, int p)
{
    int const k = (j - i + p) / 2;
    int const l = (j + i);
    return std::make_tuple(k, l);
}

#ifndef NDEBUG
__launch_bounds__(GW_UKKONEN_MAX_THREADS_PER_BLOCK) // Workaround for a register allocation problem when compiled with -g
#endif
    void ukkonen_backtrace_kernel(int8_t* paths_base, int32_t* lengths, int32_t max_path_length, batched_device_matrices<nw_score_t>::device_interface* s, int32_t const* sequence_lengths_d, int32_t n_alignments, int32_t p,
                                  sycl::nd_item<3> item_ct1)
{
    // Using scoring schema from cudaaligner.hpp
    // Match = 0
    // Mismatch = 1
    // Insertion = 2
    // Deletion = 3

    using thrust::swap;
    using thrust::tie;
    int32_t const id =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
        item_ct1.get_local_id(2);

    if (id >= n_alignments)
        return;

    GW_CONSTEXPR nw_score_t max = numeric_limits<nw_score_t>::max() - 1;

    int32_t m        = sequence_lengths_d[2 * id] + 1;
    int32_t n        = sequence_lengths_d[2 * id + 1] + 1;
    int8_t insertion = static_cast<int8_t>(AlignmentState::insertion);
    int8_t deletion  = static_cast<int8_t>(AlignmentState::deletion);
    if (m > n)
    {
        /*
        DPCT1007:122: Migration of this CUDA API is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        swap(n, m);
        /*
        DPCT1007:123: Migration of this CUDA API is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        swap(insertion, deletion);
    }
    int8_t* path = paths_base + id * static_cast<ptrdiff_t>(max_path_length);
    assert(p >= 0);
    assert(n >= m);
    int32_t const bw                      = (1 + n - m + 2 * p + 1) / 2;
    device_matrix_view<nw_score_t> scores = s->get_matrix_view(id, bw, n + m);

    int32_t i = m - 1;
    int32_t j = n - 1;

    nw_score_t myscore = [scores, i, j, p] {
        int k, l;
        /*
        DPCT1007:124: Migration of this CUDA API is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        tie(k, l) = to_band_indices(i, j, p);
        return scores(k, l);
    }();
    int32_t pos = 0;
    while (i > 0 && j > 0)
    {
        int8_t r = 0;
        int k, l;
        /*
        DPCT1007:125: Migration of this CUDA API is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        tie(k, l) = to_band_indices(i - 1, j, p);
        nw_score_t const above = k < 0 || k >= scores.num_rows() || l < 0 || l >= scores.num_cols() ? max : scores(k, l);
        /*
        DPCT1007:126: Migration of this CUDA API is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        tie(k, l) = to_band_indices(i - 1, j - 1, p);
        nw_score_t const diag  = k < 0 || k >= scores.num_rows() || l < 0 || l >= scores.num_cols() ? max : scores(k, l);
        /*
        DPCT1007:127: Migration of this CUDA API is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        tie(k, l) = to_band_indices(i, j - 1, p);
        nw_score_t const left  = k < 0 || k >= scores.num_rows() || l < 0 || l >= scores.num_cols() ? max : scores(k, l);

        if (left + 1 == myscore)
        {
            r       = insertion;
            myscore = left;
            --j;
        }
        else if (above + 1 == myscore)
        {
            r       = deletion;
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
        path[pos] = deletion;
        ++pos;
        --i;
    }
    while (j > 0)
    {
        path[pos] = insertion;
        ++pos;
        --j;
    }
    lengths[id] = pos;
}

void ukkonen_compute_score_matrix_odd(device_matrix_view<nw_score_t>& scores, int32_t kmax, int32_t k, int32_t m, int32_t n, char const* query, char const* target, int32_t max_target_query_length, int32_t p, int32_t l,
                                      sycl::nd_item<3> item_ct1)
{
    GW_CONSTEXPR nw_score_t max = numeric_limits<nw_score_t>::max() - 1;
    while (k < kmax)
    {
        int32_t const lmin = sycl::abs(2 * k + 1 - p);
        int32_t const lmax =
            2 * k + 1 <= p
                ? 2 * (m - p + 2 * k + 1) + lmin
                : (2 * sycl::min((int)m, (int)(n - (2 * k + 1) + p)) + lmin);
        if (lmin + 1 <= l && l < lmax)
        {
            int32_t const j        = k - (p + l) / 2 + l;
            int32_t const i        = l - j;
            nw_score_t const diag  = l - 2 < 0 ? max : scores(k, l - 2) + (query[i - 1] == target[j - 1] ? 0 : 1);
            nw_score_t const left  = l - 1 < 0 ? max : scores(k, l - 1) + 1;
            nw_score_t const above = l - 1 < 0 || k + 1 >= scores.num_rows() ? max : scores(k + 1, l - 1) + 1;
            scores(k, l)           = min3(diag, left, above);
        }
        k += item_ct1.get_local_range().get(2);
    }
}

void ukkonen_compute_score_matrix_even(device_matrix_view<nw_score_t>& scores, int32_t kmax, int32_t k, int32_t m, int32_t n, char const* query, char const* target, int32_t max_target_query_length, int32_t p, int32_t l,
                                       sycl::nd_item<3> item_ct1)
{
    GW_CONSTEXPR nw_score_t max = numeric_limits<nw_score_t>::max() - 1;
    while (k < kmax)
    {
        int32_t const lmin = sycl::abs(2 * k - p);
        int32_t const lmax =
            2 * k <= p ? 2 * (m - p + 2 * k) + lmin
                       : (2 * sycl::min((int)m, (int)(n - 2 * k + p)) + lmin);
        if (lmin + 1 <= l && l < lmax)
        {
            int32_t const j        = k - (p + l) / 2 + l;
            int32_t const i        = l - j;
            nw_score_t const left  = k - 1 < 0 || l - 1 < 0 ? max : scores(k - 1, l - 1) + 1;
            nw_score_t const diag  = l - 2 < 0 ? max : scores(k, l - 2) + (query[i - 1] == target[j - 1] ? 0 : 1);
            nw_score_t const above = l - 1 < 0 ? max : scores(k, l - 1) + 1;
            scores(k, l)           = min3(left, diag, above);
        }
        k += item_ct1.get_local_range().get(2);
    }
}

void ukkonen_init_score_matrix(device_matrix_view<nw_score_t>& scores, int32_t k, int32_t p,
                               sycl::nd_item<3> item_ct1)
{
    GW_CONSTEXPR nw_score_t max = numeric_limits<nw_score_t>::max() - 1;
    while (k < scores.num_rows())
    {
        for (int32_t l = 0; l < scores.num_cols(); ++l)
        {
            nw_score_t init_value = max;
            int32_t i, j;
            /*
            DPCT1007:128: Migration of this CUDA API is not supported by the
            Intel(R) DPC++ Compatibility Tool.
            */
            thrust::tie(i, j) = to_matrix_indices(k, l, p);

            if (i == 0)
                init_value = j;
            else if (j == 0)
                init_value = i;

            scores(k, l) = init_value;
        }
        k += item_ct1.get_local_range().get(2);
    }
}

#ifndef NDEBUG
__launch_bounds__(GW_UKKONEN_MAX_THREADS_PER_BLOCK) // Workaround for a register allocation problem when compiled with -g
#endif
    void ukkonen_compute_score_matrix(batched_device_matrices<nw_score_t>::device_interface* s, char const* sequences_d, int32_t const* sequence_lengths_d, int32_t max_target_query_length, int32_t p, int32_t max_cols,
                                      sycl::nd_item<3> item_ct1)
{
    using thrust::swap;
    int32_t const k =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
        item_ct1.get_local_id(2);
    int32_t const id =
        item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
        item_ct1.get_local_id(1);

    int32_t m          = sequence_lengths_d[2 * id] + 1;
    int32_t n          = sequence_lengths_d[2 * id + 1] + 1;
    char const* query  = sequences_d + (2 * id) * max_target_query_length;
    char const* target = sequences_d + (2 * id + 1) * max_target_query_length;
    if (m > n)
    {
        /*
        DPCT1007:130: Migration of this CUDA API is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        swap(n, m);
        /*
        DPCT1007:131: Migration of this CUDA API is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        swap(query, target);
    }
    assert(p >= 0);
    int32_t const bw        = (1 + n - m + 2 * p + 1) / 2;
    int32_t const kmax_odd  = (n - m + 2 * p - 1) / 2 + 1;
    int32_t const kmax_even = (n - m + 2 * p) / 2 + 1;

    device_matrix_view<nw_score_t> scores = s->get_matrix_view(id, bw, n + m);
    ukkonen_init_score_matrix(scores, k, p, item_ct1);
    /*
    DPCT1065:129: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (p % 2 == 0)
    {
        for (int lx = 0; lx < 2 * max_cols; ++lx)
        {
            ukkonen_compute_score_matrix_even(scores, kmax_even, k, m, n, query,
                                              target, max_target_query_length,
                                              p, 2 * lx, item_ct1);
            /*
            DPCT1065:132: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            ukkonen_compute_score_matrix_odd(scores, kmax_odd, k, m, n, query,
                                             target, max_target_query_length, p,
                                             2 * lx + 1, item_ct1);
            /*
            DPCT1065:133: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
    }
    else
    {
        for (int lx = 0; lx < 2 * max_cols; ++lx)
        {
            ukkonen_compute_score_matrix_odd(scores, kmax_odd, k, m, n, query,
                                             target, max_target_query_length, p,
                                             2 * lx, item_ct1);
            /*
            DPCT1065:134: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            ukkonen_compute_score_matrix_even(scores, kmax_even, k, m, n, query,
                                              target, max_target_query_length,
                                              p, 2 * lx + 1, item_ct1);
            /*
            DPCT1065:135: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
    }
}

} // namespace kernels

sycl::range<3> calc_blocks(sycl::range<3> const &n_threads,
                           sycl::range<3> const &blocksize)
{
    sycl::range<3> r(1, 1, 1);
    r[2] = (n_threads[2] + blocksize[2] - 1) / blocksize[2];
    r[1] = (n_threads[1] + blocksize[1] - 1) / blocksize[1];
    r[0] = (n_threads[0] + blocksize[0] - 1) / blocksize[0];
    return r;
}

constexpr int32_t calc_good_blockdim(int32_t n)
{
    constexpr int32_t warpsize = 32;
    int32_t i                  = n + (warpsize - n % warpsize);
    return i > GW_UKKONEN_MAX_THREADS_PER_BLOCK ? GW_UKKONEN_MAX_THREADS_PER_BLOCK : i;
}

void ukkonen_compute_score_matrix_gpu(
    batched_device_matrices<nw_score_t> &score_matrices,
    char const *sequences_d, int32_t const *sequence_lengths_d,
    int32_t max_length_difference, int32_t max_target_query_length,
    int32_t n_alignments, int32_t p, sycl::queue *stream)
{
    using kernels::ukkonen_compute_score_matrix;
    assert(p >= 0);
    assert(max_length_difference >= 0);
    assert(max_target_query_length >= 0);

    int32_t const max_bw   = (1 + max_length_difference + 2 * p + 1) / 2;
    int32_t const max_cols = 2 * (max_target_query_length + 1);

    // Transform to diagonal coordinates
    // (i,j) -> (k=(j-i+p)/2, l=(j+i))
    sycl::range<3> const compute_blockdims(1, 1, calc_good_blockdim(max_bw));
    sycl::range<3> const blocks = sycl::range<3>(1, n_alignments, 1);

    /*
    DPCT1049:136: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    stream->submit([&](sycl::handler &cgh) {
        auto score_matrices_get_device_interface_ct0 =
            score_matrices.get_device_interface();

        cgh.parallel_for(
            sycl::nd_range<3>(blocks * compute_blockdims, compute_blockdims),
            [=](sycl::nd_item<3> item_ct1) {
                kernels::ukkonen_compute_score_matrix(
                    score_matrices_get_device_interface_ct0, sequences_d,
                    sequence_lengths_d, max_target_query_length, p, max_cols,
                    item_ct1);
            });
    });
    /*
    DPCT1010:137: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    GW_CU_CHECK_ERR(0);
}

void ukkonen_backtrace_gpu(int8_t *paths_d, int32_t *path_lengths_d,
                           int32_t max_path_length,
                           batched_device_matrices<nw_score_t> &scores,
                           int32_t const *sequence_lengths_d,
                           int32_t n_alignments, int32_t p, sycl::queue *stream)
{
    stream->submit([&](sycl::handler &cgh) {
        auto scores_get_device_interface_ct3 = scores.get_device_interface();

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_alignments),
                                           sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             kernels::ukkonen_backtrace_kernel(
                                 paths_d, path_lengths_d, max_path_length,
                                 scores_get_device_interface_ct3,
                                 sequence_lengths_d, n_alignments, p, item_ct1);
                         });
    });
    /*
    DPCT1010:138: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    GW_CU_CHECK_ERR(0);
}

void ukkonen_gpu(int8_t *paths_d, int32_t *path_lengths_d,
                 int32_t max_path_length, char const *sequences_d,
                 int32_t const *sequence_lengths_d,
                 int32_t max_length_difference, int32_t max_target_query_length,
                 int32_t n_alignments,
                 batched_device_matrices<nw_score_t> *score_matrices,
                 int32_t ukkonen_p, sycl::queue *stream)
{
    if (score_matrices == nullptr)
        return;

    ukkonen_compute_score_matrix_gpu(*score_matrices, sequences_d, sequence_lengths_d, max_length_difference, max_target_query_length, n_alignments, ukkonen_p, stream);
    ukkonen_backtrace_gpu(paths_d, path_lengths_d, max_path_length, *score_matrices, sequence_lengths_d, n_alignments, ukkonen_p, stream);
}

size_t ukkonen_max_score_matrix_size(int32_t max_query_length, int32_t max_target_length, int32_t max_length_difference, int32_t max_p)
{
    assert(max_target_length >= 0);
    assert(max_query_length >= 0);
    assert(max_p >= 0);
    assert(max_length_difference >= 0);
    size_t const max_target_query_length = std::max(max_target_length, max_query_length);
    size_t const n                       = max_target_query_length + 1;
    size_t const m                       = max_target_query_length + 1;
    size_t const bw                      = (1 + max_length_difference + 2 * max_p + 1) / 2;
    return bw * (n + m);
}

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks

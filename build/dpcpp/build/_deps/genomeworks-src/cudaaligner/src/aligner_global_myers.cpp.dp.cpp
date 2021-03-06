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
#include "aligner_global_myers.hpp"
#include "myers_gpu.dp.hpp"
#include "batched_device_matrices.dp.hpp"

#include <claraparabricks/genomeworks/utils/mathutils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

struct AlignerGlobalMyers::Workspace
{
    Workspace(int32_t max_alignments, int32_t max_n_words,
              int32_t max_target_length, DefaultDeviceAllocator allocator,
              sycl::queue *stream)
        : pvs(max_alignments, max_n_words * (max_target_length + 1), allocator,
              stream),
          mvs(max_alignments, max_n_words * (max_target_length + 1), allocator,
              stream),
          scores(max_alignments, max_n_words * (max_target_length + 1),
                 allocator, stream),
          query_patterns(max_alignments, max_n_words * 4, allocator, stream)
    {
    }
    batched_device_matrices<myers::WordType> pvs;
    batched_device_matrices<myers::WordType> mvs;
    batched_device_matrices<int32_t> scores;
    batched_device_matrices<myers::WordType> query_patterns;
};

AlignerGlobalMyers::AlignerGlobalMyers(int32_t max_query_length,
                                       int32_t max_target_length,
                                       int32_t max_alignments,
                                       DefaultDeviceAllocator allocator,
                                       sycl::queue *stream, int32_t device_id)
    : AlignerGlobal(max_query_length, max_target_length, max_alignments,
                    allocator, stream, device_id),
      workspace_()
{
    scoped_device_switch dev(device_id);
    workspace_ = std::make_unique<Workspace>(max_alignments, ceiling_divide<int32_t>(max_query_length, CHAR_BIT * sizeof(myers::WordType)), max_target_length, allocator, stream);
}

AlignerGlobalMyers::~AlignerGlobalMyers()
{
    // Keep empty destructor to keep Workspace type incomplete in the .hpp file.
}

void AlignerGlobalMyers::run_alignment(
    int8_t *results_d, int32_t *result_lengths_d, int32_t max_result_length,
    const char *sequences_d, int32_t *sequence_lengths_d,
    int32_t *sequence_lengths_h, int32_t max_sequence_length,
    int32_t num_alignments, sycl::queue *stream)
{
    static_cast<void>(sequence_lengths_h);
    myers_gpu(results_d, result_lengths_d, max_result_length,
              sequences_d, sequence_lengths_d, max_sequence_length, num_alignments,
              workspace_->pvs, workspace_->mvs, workspace_->scores, workspace_->query_patterns,
              stream);
}

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks

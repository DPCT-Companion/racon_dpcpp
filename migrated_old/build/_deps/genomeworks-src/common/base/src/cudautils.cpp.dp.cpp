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
#include <claraparabricks/genomeworks/utils/cudautils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudautils
{

std::size_t find_largest_contiguous_device_memory_section() try {
    // find the largest block of contiguous memory
    size_t free;
    size_t total;
    /*
    DPCT1072:148: DPC++ currently does not support getting the available memory
    on the current device. You may need to adjust the code.
    */
    total =dpct::get_current_device().get_device_info().get_global_mem_size();
    const size_t memory_decrement = free / 100;              // decrease requested memory one by one percent
    size_t size_to_try            = free - memory_decrement; // do not go for all memory
    while (true)
    {
        void* dummy_ptr    = nullptr;
        int status = (dummy_ptr = (void *)sycl::malloc_device( size_to_try, dpct::get_default_queue()),0);        // if it was able to allocate memory free the memory and return the size
        if (status == 0)
        {
            sycl::free(dummy_ptr, dpct::get_default_queue());
            return size_to_try;
        }

        if (status == 2)
        {
            // if it was not possible to allocate the memory because there was not enough of it
            // try allocating less memory in next iteration
            if (size_to_try > memory_decrement)
            {
                size_to_try -= memory_decrement;
            }
            else
            { // a very small amount of memory left, report an error
                GW_CU_CHECK_ERR(2);
                return 0;
            }
        }
        else
        {
            // if cudaMalloc failed because of error other than cudaErrorMemoryAllocation process the error
            GW_CU_CHECK_ERR(status);
            return 0;
        }
    }

    // this point should actually never be reached (loop either finds memory or causes an error)
    assert(false);
    GW_CU_CHECK_ERR(2);
    return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
} // namespace cudautils

} // namespace genomeworks

} // namespace claraparabricks

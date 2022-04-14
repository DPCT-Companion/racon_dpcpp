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

#pragma once
/// \file
/// \defgroup cudautils Internal CUDA utilities package

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <claraparabricks/genomeworks/gw_config.hpp>
#include <claraparabricks/genomeworks/logging/logging.hpp>

#include <cassert>
#include <string>

#ifdef GW_PROFILING
#include <nvToolsExt.h>
#endif // GW_PROFILING

/// \ingroup cudautils
/// \{

/// \ingroup cudautils
/// \def GW_CU_CHECK_ERR
/// \brief Log on CUDA error in enclosed expression
#define GW_CU_CHECK_ERR(ans)                                                            \
    {                                                                                   \
        claraparabricks::genomeworks::cudautils::gpu_assert((ans), __FILE__, __LINE__); \
    }
// ^^^^ GW_CU_CHECK_ERR currently has the same implementation as GW_CU_ABORT_ON_ERR.
//      The idea is that in the future GW_CU_CHECK_ERR could have a "softer" error reporting (= not calling std::abort)

/// \ingroup cudautils
/// \def GW_CU_ABORT_ON_ERR
/// \brief Log on CUDA error in enclosed expression and termine in release mode, fail assertion in debug mode
#define GW_CU_ABORT_ON_ERR(ans)                                                         \
    {                                                                                   \
        claraparabricks::genomeworks::cudautils::gpu_assert((ans), __FILE__, __LINE__); \
    }

/// \}

namespace claraparabricks
{

namespace genomeworks
{

namespace cudautils
{

/// gpu_assert
/// Logs and/or exits on cuda error
/// \ingroup cudautils
/// \param code The CUDA status code of the function being asserted
/// \param file Filename of the calling function
/// \param line File line number of the calling function
inline void gpu_assert(int code, const char *file, int line)
{
#ifdef GW_DEVICE_SYNCHRONIZE
    // This device synchronize forces the most recent CUDA call to fully
    // complete, increasing the chance of catching the CUDA error near the
    // offending function. Only run if existing code is success to avoid
    // potentially overwriting previous error code.
    if (code == cudaSuccess)
    {
        code = cudaDeviceSynchronize();
    }
#endif

    /*
    DPCT1000:4: Error handling if-stmt was detected but could not be rewritten.
    */
    if (code != 0)
    {
        std::string err =
            "GPU Error:: " +
            /*
            DPCT1009:5: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            std::string(
                "cudaGetErrorString not supported" /*cudaGetErrorString(code)*/) +
            " " + std::string(file) + " " + std::to_string(line);
        GW_LOG_ERROR("{}\n", err);
        // In Debug mode, this assert will cause a debugger trap
        // which is beneficial when debugging errors.
        /*
        DPCT1001:3: The statement could not be removed.
        */
        /*
        DPCT1001:43: The statement could not be removed.
        */
        /*
        DPCT1001:70: The statement could not be removed.
        */
        /*
        DPCT1001:91: The statement could not be removed.
        */
        /*
        DPCT1001:139: The statement could not be removed.
        */
        assert(false);
        std::abort();
    }
}

/// align
/// Alignment of memory chunks in cudapoa. Must be a power of two
/// \tparam IntType type of data to align
/// \tparam boundary Boundary to align to (NOTE: must be power of 2)
/// \param value Input value that is to be aligned
/// \return Value aligned to boundary
template <typename IntType, int32_t boundary>
__dpct_inline__ IntType align(const IntType &value)
{
    static_assert((boundary & (boundary - 1)) == 0, "Boundary for align must be power of 2");
    return (value + boundary - 1) & ~(boundary - 1);
}

/// Copy and return value from a device memory location with implicit synchronization
template <typename Type>
Type get_value_from_device(const Type *d_ptr, sycl::queue *stream = 0)
{
    Type val;
    /*
    DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    /*
    DPCT1003:44: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:71: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:92: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:140: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    GW_CU_CHECK_ERR((stream->memcpy(&val, d_ptr, sizeof(Type)), 0));
    /*
    DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    /*
    DPCT1003:45: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:72: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:93: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:141: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    GW_CU_CHECK_ERR((stream->wait(), 0));
    return val;
}

/// Copies value from 'src' address to 'dst' address asynchronously.
template <typename Type>
void set_device_value_async(Type *dst, const Type *src, sycl::queue *stream)
{
    /*
    DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    /*
    DPCT1003:46: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:73: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:94: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:142: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    GW_CU_CHECK_ERR((stream->memcpy(dst, src, sizeof(Type)), 0));
}

/// Copies the value 'src' to 'dst' address.
template <typename Type>
void set_device_value(Type* dst, const Type& src)
{
    /*
    DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    /*
    DPCT1003:47: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:74: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:95: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:143: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    GW_CU_CHECK_ERR(
        (dpct::get_default_queue().memcpy(dst, &src, sizeof(Type)).wait(), 0));
}

/// Copies elements from the range [src, src + n) to the range [dst, dst + n) asynchronously.
template <typename Type>
void device_copy_n(const Type *src, size_t n, Type *dst, sycl::queue *stream)
{
    /*
    DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:48: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:75: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:96: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:144: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    GW_CU_CHECK_ERR((stream->memcpy(dst, src, n * sizeof(Type)), 0));
}

/// Copies elements from the range [src, src + n) to the range [dst, dst + n).
template <typename Type>
void device_copy_n(const Type* src, size_t n, Type* dst)
{
    /*
    DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:49: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:76: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:97: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1003:145: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    GW_CU_CHECK_ERR(
        (dpct::get_default_queue().memcpy(dst, src, n * sizeof(Type)).wait(),
         0));
}

/// @brief finds largest section of contiguous memory on device
/// @return number of bytes
std::size_t find_largest_contiguous_device_memory_section();

#ifdef GW_PROFILING
/// \ingroup cudautils
/// \def GW_NVTX_RANGE
/// \brief starts an NVTX range for profiling which stops automatically at the end of the scope
/// \param varname an arbitrary variable name for the nvtx_range object, which doesn't conflict with other variables in the scope
/// \param label the label/name of the NVTX range
#define GW_NVTX_RANGE(varname, label) ::claraparabricks::genomeworks::cudautils::nvtx_range varname(label)
/// nvtx_range
/// implementation of GW_NVTX_RANGE
class nvtx_range
{
public:
    explicit nvtx_range(char const* name)
    {
        nvtxRangePush(name);
    }

    ~nvtx_range()
    {
        nvtxRangePop();
    }
};
#else
/// \ingroup cudautils
/// \def GW_NVTX_RANGE
/// \brief Dummy implementation for GW_NVTX_RANGE macro
/// \param varname Unused variable
/// \param label Unused variable
#define GW_NVTX_RANGE(varname, label)
#endif // GW_PROFILING

} // namespace cudautils

/// \brief A class to switch the CUDA device for the current scope using RAII
///
/// This class takes a CUDA device during construction,
/// switches to the given device using cudaSetDevice,
/// and switches back to the CUDA device which was current before the switch on destruction.
class scoped_device_switch
{
public:
    /// \brief Constructor
    ///
    /// \param device_id ID of CUDA device to switch to while class is in scope
    explicit scoped_device_switch(int32_t device_id)
    {
        GW_CU_CHECK_ERR(device_id_before_ =
                            dpct::dev_mgr::instance().current_device_id());
        /*
        DPCT1003:12: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        /*
        DPCT1003:50: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        /*
        DPCT1003:77: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        /*
        DPCT1003:98: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        /*
        DPCT1003:146: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        GW_CU_CHECK_ERR(
            (dpct::dev_mgr::instance().select_device(device_id), 0));
    }

    /// \brief Destructor switches back to original device ID
    ~scoped_device_switch()
    {
        dpct::dev_mgr::instance().select_device(device_id_before_);
    }

    scoped_device_switch()                            = delete;
    scoped_device_switch(scoped_device_switch const&) = delete;
    scoped_device_switch& operator=(scoped_device_switch const&) = delete;

private:
    int32_t device_id_before_;
};

} // namespace genomeworks

} // namespace claraparabricks

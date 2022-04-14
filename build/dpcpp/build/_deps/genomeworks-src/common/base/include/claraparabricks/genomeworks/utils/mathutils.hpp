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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cassert>
#include <cstdint>
#include <type_traits>
#ifndef DPCT_COMPATIBILITY_TEMP
#include <algorithm>
#endif

namespace claraparabricks
{

namespace genomeworks
{

template <typename Integer>
constexpr inline Integer ceiling_divide(Integer i, Integer j)
{
    static_assert(std::is_integral<Integer>::value, "Arguments have to be integer types.");
    assert(i >= 0);
    assert(j > 0);
    return (i + j - 1) / j;
}

template <typename T>
inline T min3(T t1, T t2, T t3)
{

    return t1<(t2<t3?t2:t3)?t1:(t2<t3?t2:t3);
}

/// @brief Calculates floor of log2() of the given integer number
///
/// int_floor_log2(1) = 0
/// int_floor_log2(2) = 1
/// int_floor_log2(3) = 1
/// int_floor_log2(8) = 3
/// int_floor_log2(11) = 3
///
/// @param val
/// @tparam T type of val
template <typename T>
std::int32_t int_floor_log2(T val)
{
    static_assert(std::is_integral<T>::value, "Expected an integer");

    assert(val > 0);

    std::int32_t power = 0;
    // keep dividing by 2 until value is 1
    while (val != 1)
    {
        // divide by two, i.e. move by one log value
        val >>= 1;
        ++power;
    }

    return power;
}

} // namespace genomeworks

} // namespace claraparabricks

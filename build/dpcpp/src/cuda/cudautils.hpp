// Implementation file for CUDA POA utilities.

#pragma once

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>

namespace racon {

void cudaCheckError(std::string &msg)
{
    /*
    DPCT1010:81: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    int error = 0;
}

} // namespace racon

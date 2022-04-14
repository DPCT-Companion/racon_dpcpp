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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>
#include <dpct/dpl_utils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

template<typename T> class pinned_allocator;

template<>
  class pinned_allocator<void>
{
  public:
    typedef void           value_type;
    typedef void       *   pointer;
    typedef const void *   const_pointer;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    // convert a pinned_allocator<void> to pinned_allocator<U>
    template<typename U>
      struct rebind
    {
      typedef pinned_allocator<U> other;
    }; // end rebind
}; // end pinned_allocator


template<typename T>
  class pinned_allocator
{
  public:
    //! \{
    typedef T              value_type;
    typedef T*             pointer;
    typedef const T*       const_pointer;
    typedef T&             reference;
    typedef const T&       const_reference;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;
    //! \}

    // convert a pinned_allocator<T> to pinned_allocator<U>
    template<typename U>
      struct rebind
    {
      typedef pinned_allocator<U> other;
    }; // end rebind

    /*! \p pinned_allocator's null constructor does nothing.
     */
    
    inline pinned_allocator() {}

    /*! \p pinned_allocator's null destructor does nothing.
     */
    
    inline ~pinned_allocator() {}

    /*! \p pinned_allocator's copy constructor does nothing.
     */
    
    inline pinned_allocator(pinned_allocator const &) {}

    /*! This version of \p pinned_allocator's copy constructor
     *  is templated on the \c value_type of the \p pinned_allocator
     *  to copy from.  It is provided merely for convenience; it
     *  does nothing.
     */
    template<typename U>
    
    inline pinned_allocator(pinned_allocator<U> const &) {}

    /*! This method returns the address of a \c reference of
     *  interest.
     *
     *  \p r The \c reference of interest.
     *  \return \c r's address.
     */
    
    inline pointer address(reference r) { return &r; }

    /*! This method returns the address of a \c const_reference
     *  of interest.
     *
     *  \p r The \c const_reference of interest.
     *  \return \c r's address.
     */
    
    inline const_pointer address(const_reference r) { return &r; }

    /*! This method allocates storage for objects in pinned host
     *  memory.
     *
     *  \p cnt The number of objects to allocate.
     *  \return a \c pointer to the newly allocated objects.
     *  \note This method does not invoke \p value_type's constructor.
     *        It is the responsibility of the caller to initialize the
     *        objects at the returned \c pointer.
     */

    inline pointer allocate(size_type cnt, const_pointer = 0) try {
      if(cnt > this->max_size())
      {
        throw std::bad_alloc();
      } // end if

      pointer result(0);
      *(reinterpret_cast<void **>(&result)) = (void *)sycl::malloc_host(
               cnt * sizeof(value_type), dpct::get_default_queue());
      dpct::get_default_queue().memset(result, 0, cnt * sizeof(value_type)).wait();
      return result;
    }
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
} // end allocate()

    /*! This method deallocates pinned host memory previously allocated
     *  with this \c pinned_allocator.
     *
     *  \p p A \c pointer to the previously allocated memory.
     *  \p cnt The number of objects previously allocated at
     *         \p p.
     *  \note This method does not invoke \p value_type's destructor.
     *        It is the responsibility of the caller to destroy
     *        the objects stored at \p p.
     */
    
    inline void deallocate(pointer p, size_type /*cnt*/)
    {
      sycl::free(p, dpct::get_default_queue());
    } // end deallocate()

    /*! This method returns the maximum size of the \c cnt parameter
     *  accepted by the \p allocate() method.
     *
     *  \return The maximum number of objects that may be allocated
     *          by a single call to \p allocate().
     */
    inline size_type max_size() const
    {
      return (std::numeric_limits<size_type>::max)() / sizeof(T);
    } // end max_size()

    /*! This method tests this \p pinned_allocator for equality to
     *  another.
     *
     *  \param x The other \p pinned_allocator of interest.
     *  \return This method always returns \c true.
     */
    
    inline bool operator==(pinned_allocator const& x) const { return true; }

    /*! This method tests this \p pinned_allocator for inequality
     *  to another.
     *
     *  \param x The other \p pinned_allocator of interest.
     *  \return This method always returns \c false.
     */
    
    inline bool operator!=(pinned_allocator const &x) const { return !operator==(x); }
}; // end pinned_allocator





/// \brief An vector using pinned host memory for fast asynchronous transfers to the GPU
///
/// It is a std::vector with a special allocator for pinned host memory.
/// Please see C++ documentation for std::vector.
/// \tparam T The object's type
template <typename T>
using pinned_host_vector = std::vector<T, claraparabricks::genomeworks::pinned_allocator<T>>;

} // namespace genomeworks
} // namespace claraparabricks

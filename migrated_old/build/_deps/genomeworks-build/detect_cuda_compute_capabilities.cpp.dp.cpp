#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
int main() try {
  int count = 0;
  if (0 != (count = dpct::dev_mgr::instance().device_count(), 0)) return -1;   if (count == 0) return -1;
  for (int device = 0; device < count; ++device)
  {
    dpct::device_info prop;
    if (0 ==
        (dpct::dev_mgr::instance().get_device(device).get_device_info(prop), 0))
      /*
      DPCT1005:2: The SYCL device version is different from CUDA Compute
      Compatibility. You may need to rewrite this code.
      */
      std::printf("%d.%d ", prop.get_major_version(), prop.get_minor_version());   }
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

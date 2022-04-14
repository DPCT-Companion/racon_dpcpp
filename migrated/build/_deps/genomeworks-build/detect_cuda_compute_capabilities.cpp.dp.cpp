#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <iostream>
int main() {
  int count = 0;
  if (0 != (count = dpct::dev_mgr::instance().device_count(), 0)) return -1;   if (count == 0) return -1;
  std::printf("%d devices\n", count);
  for (int device = 0; device < count; ++device)
  {
    dpct::device_info prop;
    try {
      std::cout << dpct::dev_mgr::instance().get_device(device).get_device_info().get_global_mem_size() << std::endl;
      dpct::dev_mgr::instance().get_device(device).get_device_info(prop);
      std::printf("%s ", prop.get_name());
      std::printf("%d.%d ", prop.get_major_version(), prop.get_minor_version());
      std::printf("%zu \n", prop.get_global_mem_size());  }
    catch (...) {continue;}

  }
  return 0;
}

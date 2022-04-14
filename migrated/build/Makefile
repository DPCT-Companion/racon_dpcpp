.PHONY: default_target
default_target: all


CC := dpcpp
LD := $(CC)
AR := ar rc

LIB := $(CURDIR)/lib
BIN := $(CURDIR)/bin

ALL_MAIN_LIBS := $(LIB)/libracon.a $(LIB)/libedlib.a $(LIB)/libspoa.a $(LIB)/libcudapoa.a $(LIB)/libgwio.a $(LIB)/libcudaaligner.a $(LIB)/libgwbase.a

# genomeworks common base
GWBASE_DIR := $(CURDIR)/_deps/genomeworks-build/common/base/CMakeFiles/gwbase.dir/src
GWBASE_FLAGS := -g -DGW_ENABLE_CACHING_ALLOCATOR -fopenmp -Og -DNDEBUG -std=c++20 -Wno-tautological-constant-compare -DPSTL_USE_PARALLEL_POLICIES=0
GWBASE_LIB := -I/opt/intel/oneapi/intelpython/python3.9/include -I$(CURDIR)/_deps/genomeworks-src/common/base/include -I$(CURDIR)/_deps/genomeworks-src/3rdparty/spdlog/include
GWBASE_SRC := $(CURDIR)/_deps/genomeworks-src/common/base/src


# genomeworks common io
GWIO_DIR := $(CURDIR)/_deps/genomeworks-build/cudapoa/CMakeFiles/cudapoa.dir/src
GWIO_FLAGS := -g -DGW_ENABLE_CACHING_ALLOCATOR -fopenmp -Og -DNDEBUG -Wno-tautological-constant-compare -DPSTL_USE_PARALLEL_POLICIES=0
GWIO_LIB := -I/opt/intel/oneapi/intelpython/python3.9/include -I$(CURDIR)/_deps/genomeworks-src/3rdparty/kseqpp/src -I$(CURDIR)/_deps/genomeworks-src/common/io/include -I$(CURDIR)/_deps/genomeworks-src/common/base/include -I$(CURDIR)/_deps/genomeworks-src/3rdparty/spdlog/include
GWIO_SRC := $(CURDIR)/_deps/genomeworks-src/common/io/src


# genomeworks cudaaligner
CUDAALIGNER_DIR := $(CURDIR)/_deps/genomeworks-build/cudaaligner/CMakeFiles/cudaaligner.dir/src
CUDAALIGNER_FLAGS := -g -Wno-tautological-constant-compare -DPSTL_USE_PARALLEL_POLICIES=0 -m64 -DGW_ENABLE_CACHING_ALLOCATOR -fopenmp -Og -DNDEBUG -Wno-pedantic -std=c++20 -DNVCC
CUDAALIGNER_LIB := -I/opt/intel/oneapi/intelpython/python3.9/include -I$(CURDIR)/_deps/genomeworks-src/cudaaligner/include -I$(CURDIR)/_deps/genomeworks-src/common/base/include -I$(CURDIR)/_deps/genomeworks-src/3rdparty/spdlog/include -isystem $(CURDIR)/_deps/genomeworks-src/3rdparty/cub
CUDAALIGNER_SRC := $(CURDIR)/_deps/genomeworks-src/cudaaligner/src


# genomeworks cudapoa
CUDAPOA_DIR := $(CURDIR)/_deps/genomeworks-build/cudapoa/CMakeFiles/cudapoa.dir
CUDAPOA_FLAGS := -g -Wno-tautological-constant-compare -DPSTL_USE_PARALLEL_POLICIES=0 -DGW_ENABLE_CACHING_ALLOCATOR -fopenmp -Og -DNDEBUG -m64 -Wno-pedantic -std=c++20 -DNVCC -D__CUDACC__=1
CUDAPOA_LIB := -I/opt/intel/oneapi/intelpython/python3.9/include -I$(CURDIR)/_deps/genomeworks-src/cudapoa/include -I$(CURDIR)/_deps/genomeworks-src/common/base/include -I$(CURDIR)/_deps/genomeworks-src/3rdparty/spdlog/include -I$(CURDIR)/_deps/genomeworks-src/common/io/include
CUDAPOA_SRC := $(CURDIR)/_deps/genomeworks-src/cudapoa/src
CUDAPOA_VER_SRC := $(CURDIR)/_deps/genomeworks-build/cudapoa


# edlib
EDLIB_DIR := $(CURDIR)/_deps/edlib-build/CMakeFiles/edlib.dir/edlib/src
EDLIB_FLAGS := -g -DDLIB_BUILD -Wold-style-cast -Wshadow -Og -DNDEBUG -std=c++20 -Wno-tautological-constant-compare -DPSTL_USE_PARALLEL_POLICIES=0
EDLIB_LIB := -I$(CURDIR)/_deps/edlib-src/edlib/include
EDLIB_SRC := $(CURDIR)/_deps/edlib-src/edlib/src


# spoa
SPOA_DIR := $(CURDIR)/_deps/spoa-build/CMakeFiles/spoa.dir/src
SPOA_FLAGS := -g -Wno-aligned-new -Wno-ignored-attributes -Og -DNDEBUG -std=c++20 -Wno-tautological-constant-compare -DPSTL_USE_PARALLEL_POLICIES=0
SPOA_LIB := -I$(CURDIR)/_deps/spoa-src/include
SPOA_SRC := $(CURDIR)/_deps/spoa-src/src


# racon
RACON_MAIN_DIR := $(CURDIR)/CMakeFiles/racon_exe.dir/src
RACON_DIR := $(CURDIR)/CMakeFiles/racon.dir/src
RACON_FLAGS := -g '-DVERSION="1.5.0"' -DCUDA_ENABLED -DGW_ENABLE_CACHING_ALLOCATOR -Og -DNDEBUG -pthread -std=c++20 -Wno-tautological-constant-compare -DPSTL_USE_PARALLEL_POLICIES=0
RACON_LIB := -I$(CURDIR)/../src -I$(CURDIR)/_deps/bioparser-src/include -I$(CURDIR)/_deps/edlib-src/edlib/include -I$(CURDIR)/_deps/spoa-src/include -I$(CURDIR)/_deps/thread_pool-src/include -I$(CURDIR)/_deps/genomeworks-src/cudapoa/include -I$(CURDIR)/_deps/genomeworks-src/common/base/include -I$(CURDIR)/_deps/genomeworks-src/3rdparty/spdlog/include -I$(CURDIR)/_deps/genomeworks-src/common/io/include -I$(CURDIR)/_deps/genomeworks-src/cudaaligner/include -isystem $(CURDIR)/_deps/genomeworks-src/3rdparty/cub
RACON_SRC := $(CURDIR)/../src


# racon test
RACON_TEST_DIR := $(CURDIR)/CMakeFiles/racon_test.dir/test
RACON_TEST_SRC := $(CURDIR)/../test
RACON_TEST_FLAGS := -g -DCUDA_ENABLED -DGW_ENABLE_CACHING_ALLOCATOR '-DTEST_DATA="$(RACON_TEST_SRC)/data/"' -Og -DNDEBUG -pthread -std=c++20 -Wno-tautological-constant-compare -DPSTL_USE_PARALLEL_POLICIES=0
RACON_TEST_LIB := -I$(CURDIR)/../src -I$(CURDIR)/_deps/bioparser-src/include -I$(CURDIR)/_deps/edlib-src/edlib/include -I$(CURDIR)/_deps/spoa-src/include -I$(CURDIR)/_deps/thread_pool-src/include -I$(CURDIR)/_deps/genomeworks-src/cudapoa/include -I$(CURDIR)/_deps/genomeworks-src/common/base/include -I$(CURDIR)/_deps/genomeworks-src/3rdparty/spdlog/include -I$(CURDIR)/_deps/genomeworks-src/common/io/include -I$(CURDIR)/_deps/genomeworks-src/cudaaligner/include -isystem $(CURDIR)/_deps/genomeworks-src/3rdparty/cub -isystem $(CURDIR)/_deps/googletest-src/googletest/include -isystem $(CURDIR)/_deps/googletest-src/googletest


# gtest
GTEST_DIR := $(CURDIR)/_deps/googletest-build/googletest/CMakeFiles/gtest.dir/src
GTEST_FLAGS := -g -Og -DNDEBUG -Wshadow -DGTEST_HAS_PTHREAD=1 -fexceptions -Wno-unused-parameter -Wno-missing-field-initializers -pthread -std=c++20 -Wno-tautological-constant-compare -DPSTL_USE_PARALLEL_POLICIES=0
GTEST_LIB := -I$(CURDIR)/_deps/googletest-src/googletest/include -I$(CURDIR)/_deps/googletest-src/googletest
GTEST_SRC := $(CURDIR)/_deps/googletest-src/googletest/src

GTEST_MAIN_DIR := $(CURDIR)/_deps/googletest-build/googletest/CMakeFiles/gtest_main.dir/src
GTEST_MAIN_LIB := -isystem $(CURDIR)/_deps/googletest-src/googletest/include -isystem $(CURDIR)/_deps/googletest-src/googletest


.PHONY: all
all: $(BIN)/racon
all: $(BIN)/racon_test


# genomeworks common base
$(LIB)/libgwbase.a: $(patsubst $(GWBASE_SRC)/%.cpp,$(GWBASE_DIR)/%.o,$(wildcard $(GWBASE_SRC)/*.cpp))
	$(AR) $@ $^

$(GWBASE_DIR)/%.o: $(GWBASE_SRC)/%.cpp
	$(CC) -c $(GWBASE_FLAGS) $(GWBASE_LIB) -o $@ $<


# genomeworks common io
$(LIB)/libgwio.a: $(patsubst $(GWIO_SRC)/%.cpp,$(GWIO_DIR)/%.o,$(wildcard $(GWIO_SRC)/*.cpp))
	$(AR) $@ $^

$(GWIO_DIR)/%.o: $(GWIO_SRC)/%.cpp
	$(CC) -c $(GWIO_FLAGS) $(GWIO_LIB) -o $@ $<


# genomeworks cudaaligner
$(LIB)/libcudaaligner.a: $(patsubst $(CUDAALIGNER_SRC)/%.cpp,$(CUDAALIGNER_DIR)/%.o,$(wildcard $(CUDAALIGNER_SRC)/*.cpp))
	$(AR) $@ $^

$(CUDAALIGNER_DIR)/%.o: $(CUDAALIGNER_SRC)/%.cpp
	$(CC) -c $(CUDAALIGNER_FLAGS) $(CUDAALIGNER_LIB) -o $@ $<


# genomeworks cudapoa
$(LIB)/libcudapoa.a: $(patsubst $(CUDAPOA_VER_SRC)/%.cpp,$(CUDAPOA_DIR)/%.o,$(wildcard $(CUDAPOA_VER_SRC)/*.cpp)) $(patsubst $(CUDAPOA_SRC)/%.cpp,$(CUDAPOA_DIR)/src/%.o,$(wildcard $(CUDAPOA_SRC)/*.cpp))
	$(AR) $@ $^

$(CUDAPOA_DIR)/%.o: $(CUDAPOA_VER_SRC)/%.cpp
	$(CC) -c $(CUDAPOA_FLAGS) $(CUDAPOA_LIB) -o $@ $<

$(CUDAPOA_DIR)/src/%.o: $(CUDAPOA_SRC)/%.cpp
	$(CC) -c $(CUDAPOA_FLAGS) $(CUDAPOA_LIB) -o $@ $<


# edlib
$(LIB)/libedlib.a: $(patsubst $(EDLIB_SRC)/%.cpp,$(EDLIB_DIR)/%.o,$(wildcard $(EDLIB_SRC)/*.cpp))
	$(AR) $@ $^

$(EDLIB_DIR)/%.o: $(EDLIB_SRC)/%.cpp
	$(CC) -c $(EDLIB_FLAGS) $(EDLIB_LIB) -o $@ $<


# spoa
$(LIB)/libspoa.a: $(patsubst $(SPOA_SRC)/%.cpp,$(SPOA_DIR)/%.o,$(wildcard $(SPOA_SRC)/*.cpp))
	$(AR) $@ $^

$(SPOA_DIR)/%.o: $(SPOA_SRC)/%.cpp
	$(CC) -c $(SPOA_FLAGS) $(SPOA_LIB) -o $@ $<


# gtest
$(LIB)/libgtest.a: $(GTEST_DIR)/gtest-all.o
	$(AR) $@ $^

$(LIB)/libgtest_main.a: $(GTEST_MAIN_DIR)/gtest_main.o
	$(AR) $@ $^

$(GTEST_DIR)/gtest-all.o: $(GTEST_SRC)/gtest-all.cc
	$(CC) -c $(GTEST_FLAGS) $(GTEST_LIB) -o $@ $<

$(GTEST_MAIN_DIR)/gtest_main.o: $(GTEST_SRC)/gtest_main.cc
	$(CC) -c $(GTEST_FLAGS) $(GTEST_MAIN_LIB) -o $@ $<


# racon
$(LIB)/libracon.a: $(patsubst $(RACON_SRC)/%.cpp,$(RACON_DIR)/%.o,$(filter-out $(RACON_SRC)/main.cpp,$(wildcard $(RACON_SRC)/*.cpp))) $(patsubst $(RACON_SRC)/cuda/%.cpp,$(RACON_DIR)/cuda/%.o,$(wildcard $(RACON_SRC)/cuda/*.cpp))
	$(AR) $@ $^

$(RACON_DIR)/%.o: $(RACON_SRC)/%.cpp
	@if [ $(notdir $<) != main.cpp ]; \
	then $(CC) -c $(RACON_FLAGS) $(RACON_LIB) -o $@ $<; \
	fi

$(RACON_DIR)/cuda/%.o: $(RACON_SRC)/cuda/%.cpp
	$(CC) -c $(RACON_FLAGS) $(RACON_LIB) -o $@ $<


# racon_test
$(RACON_TEST_DIR)/racon_test.o: $(RACON_TEST_SRC)/racon_test.cpp
	$(CC) -c $(RACON_TEST_FLAGS) $(RACON_TEST_LIB) -o $@ $<

$(BIN)/racon_test: $(RACON_TEST_DIR)/racon_test.o $(ALL_MAIN_LIBS) $(LIB)/libgtest.a $(LIB)/libgtest_main.a
	$(CC) -o $@ $^ /usr/lib/x86_64-linux-gnu/libz.so /usr/lib/x86_64-linux-gnu/librt.so -lpthread


# racon main
$(RACON_MAIN_DIR)/main.o: $(RACON_SRC)/main.cpp
	$(CC) -c $(RACON_FLAGS) $(RACON_LIB) -o $@ $< 

$(BIN)/racon: $(RACON_MAIN_DIR)/main.o $(ALL_MAIN_LIBS)
	$(CC) -o $@ $^ /usr/lib/x86_64-linux-gnu/libz.so /usr/lib/x86_64-linux-gnu/librt.so -lpthread
	

.PHONY: demo
demo: $(CURDIR)/_deps/genomeworks-build/CMakeFiles/CMakeTmp/demo

.PHONY: run_demo
run_demo: demo
	$(CURDIR)/_deps/genomeworks-build/CMakeFiles/CMakeTmp/demo

$(CURDIR)/_deps/genomeworks-build/CMakeFiles/CMakeTmp/demo: $(CURDIR)/_deps/genomeworks-build/detect_cuda_compute_capabilities.cpp.dp.cpp
	$(CC) -o $@ $<


.PHONY: clean
clean:
	-rm $(GWBASE_DIR)/*.o
	-rm $(CUDAALIGNER_DIR)/*.o
	-rm $(CUDAPOA_DIR)/*.o
	-rm $(CUDAPOA_DIR)/src/*.o
	-rm $(EDLIB_DIR)/*.o
	-rm $(SPOA_DIR)/*.o
	-rm $(GTEST_DIR)/*.o
	-rm $(GTEST_MAIN_DIR)/*.o
	-rm $(RACON_DIR)/*.o
	-rm $(RACON_DIR)/cuda/*.o
	-rm $(RACON_MAIN_DIR)/*.o
	-rm $(RACON_TEST_DIR)/*.o
	-rm $(LIB)/*.a
	-rm $(BIN)/*
	-rm $(CURDIR)/_deps/genomeworks-build/CMakeFiles/CMakeTmp/demo

.PHONY: poaclean
poaclean:
	-rm $(CUDAPOA_DIR)/src/*.o
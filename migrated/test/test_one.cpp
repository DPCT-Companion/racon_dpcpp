
#include "sequence.hpp"
#include "polisher.hpp"

#include "edlib.h"
#include "bioparser/fasta_parser.hpp"
#include "gtest/gtest.h"

uint32_t calculateEditDistance(const std::string& query, const std::string& target) {

    EdlibAlignResult result = edlibAlign(query.c_str(), query.size(), target.c_str(),
        target.size(), edlibDefaultAlignConfig());

    uint32_t edit_distance = result.editDistance;
    edlibFreeAlignResult(result);

    return edit_distance;
}

std::unique_ptr<racon::Polisher> SetUp(const std::string& sequences_path, const std::string& overlaps_path,
    const std::string& target_path, racon::PolisherType type,
    uint32_t window_length, double quality_threshold, double error_threshold,
    int8_t match, int8_t mismatch, int8_t gap, uint32_t cuda_batches = 0,
    bool cuda_banded_alignment = false, uint32_t cudaaligner_batches = 0) {

    return racon::createPolisher(sequences_path, overlaps_path, target_path,
        type, window_length, quality_threshold, error_threshold, true, match,
        mismatch, gap, 4, cuda_batches, cuda_banded_alignment, cudaaligner_batches);
}

//     void TearDown() {}

//     void initialize() {
//         polisher->initialize();
//     }

//     void polish(std::vector<std::unique_ptr<racon::Sequence>>& dst,
//         bool drop_unpolished_sequences) {

//         return polisher->polish(dst, drop_unpolished_sequences);
//     }

//     std::unique_ptr<racon::Polisher> polisher;
// };

int main(int argc, char **argv) {
    auto polisher = SetUp(std::string(TEST_DATA) + "sample_reads.fastq.gz", std::string(TEST_DATA) +
        "sample_overlaps.paf.gz", std::string(TEST_DATA) + "sample_layout.fasta.gz",
        racon::PolisherType::kC, 500, 10, 0.3, 5, -4, -8, 1);

    polisher->initialize();

    std::vector<std::unique_ptr<racon::Sequence>> polished_sequences;
    polisher->polish(polished_sequences, true);
    std::cout << polished_sequences.size() << std::endl;

    std::cout << "before" << std::endl;
    polished_sequences[0]->create_reverse_complement();
    std::cout << "after" << std::endl;

    auto parser = bioparser::Parser<racon::Sequence>::Create<bioparser::FastaParser>(
        std::string(TEST_DATA) + "sample_reference.fasta.gz");
    auto reference = parser->Parse(-1);
    std::cout << reference.size() << std::endl;

    std::cout << calculateEditDistance(
        polished_sequences[0]->reverse_complement(),
        reference[0]->data()) << std::endl;  // CPU 1312
    return 0;
}


/*
dpcpp -c -DCUDA_ENABLED -DGW_ENABLE_CACHING_ALLOCATOR '-DTEST_DATA="/home/tianchen/racon/test/data/"' -I/home/tianchen/racon/migrated/src -I/home/tianchen/racon/migrated/build/_deps/bioparser-src/include -I/home/tianchen/racon/migrated/build/_deps/edlib-src/edlib/include -I/home/tianchen/racon/migrated/build/_deps/spoa-src/include -I/home/tianchen/racon/migrated/build/_deps/thread_pool-src/include -I/home/tianchen/racon/migrated/build/_deps/genomeworks-src/cudapoa/include -I/home/tianchen/racon/migrated/build/_deps/genomeworks-src/common/base/include -I/home/tianchen/racon/migrated/build/_deps/genomeworks-src/3rdparty/spdlog/include -I/home/tianchen/racon/migrated/build/_deps/genomeworks-src/common/io/include -I/home/tianchen/racon/migrated/build/_deps/genomeworks-src/cudaaligner/include -isystem /home/tianchen/racon/build/_deps/genomeworks-src/3rdparty/cub -isystem /home/tianchen/racon/build/_deps/googletest-src/googletest/include -isystem /home/tianchen/racon/build/_deps/googletest-src/googletest -O3 -DNDEBUG -pthread -std=c++20 -Wno-tautological-constant-compare -DPSTL_USE_PARALLEL_POLICIES=0 -o /home/tianchen/racon/migrated/build/CMakeFiles/racon_test.dir/test/test_one.o /home/tianchen/racon/migrated/test/test_one.cpp && dpcpp -o /home/tianchen/racon/migrated/build/bin/test_one /home/tianchen/racon/migrated/build/CMakeFiles/racon_test.dir/test/test_one.o /home/tianchen/racon/migrated/build/lib/libracon.a /home/tianchen/racon/migrated/build/lib/libgtest_main.a /usr/lib/x86_64-linux-gnu/libz.so /home/tianchen/racon/migrated/build/lib/libedlib.a /home/tianchen/racon/migrated/build/lib/libspoa.a /home/tianchen/racon/migrated/build/lib/libcudapoa.a /home/tianchen/racon/migrated/build/lib/libgwio.a /home/tianchen/racon/migrated/build/lib/libcudaaligner.a /home/tianchen/racon/migrated/build/lib/libgwbase.a /usr/lib/x86_64-linux-gnu/librt.so /home/tianchen/racon/migrated/build/lib/libgtest.a -lpthread

*/
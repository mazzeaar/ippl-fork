#include "OrthoTree/balancing/algo10.hpp"

#include <gtest/gtest.h>

#include "OrthoTree/OrthoTree.h"

using namespace ippl;

TEST(CompleteSubtreeTest, TestTest) {
    static constexpr size_t Dim = 2;
    const size_t max_depth      = 3;
    const size_t max_particles  = 100;
    const size_t n_particles    = 10;

    BoundingBox<Dim> bounds({0.0, 0.0}, {1.0, 1.0});
    OrthoTree<Dim> tree(max_depth, max_particles, bounds);
    tree.setVisualisation(true);

    morton_code octant_N = 1;
    std::vector<morton_code> descs;
    descs.push_back(18);
    Kokkos::View<morton_code*> partial_desc(descs.data(), descs.size());
    auto res = tree.algo10(octant_N, partial_desc);
    tree.octants_to_file(res);
    EXPECT_TRUE(true);
}

int main(int argc, char** argv) {
    // Initialize MPI and IPPL
    ippl::initialize(argc, argv, MPI_COMM_WORLD);

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int result = RUN_ALL_TESTS();

    // Finalize IPPL and MPI
    ippl::finalize();

    return result;
}
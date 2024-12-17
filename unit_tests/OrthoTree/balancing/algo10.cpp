#include <gtest/gtest.h>

#include "OrthoTree/OrthoTree.h"

using namespace ippl;

TEST(CompleteSubtreeTest, TestTest) {
    static constexpr size_t Dim = 2;
    const size_t max_depth      = 6;
    const size_t max_particles  = 100;
    const size_t n_particles    = 10;

    BoundingBox<Dim> bounds({0.0, 0.0}, {1.0, 1.0});
    OrthoTree<Dim> tree(max_depth, max_particles, bounds);
    tree.setVisualisation(true);

    Morton<Dim> morton_helper(max_depth);

    morton_code octant_N = 1;

    std::vector<morton_code> descs;
    descs.push_back(morton_helper.get_deepest_first_descendant(octant_N));
    descs.push_back(morton_helper.get_deepest_last_descendant(octant_N));

    /*
    // different inputs
    descs.push_back(
        morton_helper.get_deepest_first_descendant(morton_helper.get_last_child(octant_N)));
    descs.push_back(
        morton_helper.get_deepest_last_descendant(morton_helper.get_first_child(octant_N)));

    // technically invalid
    descs.push_back(morton_helper.get_deepest_first_descendant(octant_N) + ((340) * 8));
    descs.push_back(morton_helper.get_deepest_first_descendant(octant_N) + ((680) * 8));
    */

    Kokkos::View<morton_code*> partial_desc(descs.data(), descs.size());

    auto res = tree.algo10(octant_N, partial_desc);

    EXPECT_EQ(res.size(), 28);

    /*
    // output to test
    Kokkos::resize(res, res.size() + 1);
    res[res.size() - 1] = 0;
    tree.octants_to_file(res);
    */
}

TEST(CompleteSubtreeTest, EmptyPartialDescendants) {
    /**
     * If we pass in an octant containing no octants that need to be rebalanced we expect the result
     * to be empty.
     */

    static constexpr size_t Dim = 2;
    const size_t max_depth      = 3;
    const size_t max_particles  = 100;
    const size_t n_particles    = 10;

    BoundingBox<Dim> bounds({0.0, 0.0}, {1.0, 1.0});
    OrthoTree<Dim> tree(max_depth, max_particles, bounds);
    tree.setVisualisation(true);

    morton_code octant_N = 1;
    std::vector<morton_code> descs;
    Kokkos::View<morton_code*> partial_desc(descs.data(), descs.size());

    auto res = tree.algo10(octant_N, partial_desc);

    EXPECT_EQ(res.size(), 0);
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
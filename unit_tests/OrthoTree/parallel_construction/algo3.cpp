#include<gtest/gtest.h>
#include "OrthoTree/OrthoTree.h"

using namespace ippl;

TEST(CompleteOctree, GeneralCase) {
    // Testing the example in the paper, Fig. 3.2
    constexpr size_t Dim = 2;
    size_t max_depth = 4;
    OrthoTree<Dim> tree(max_depth, 2, BoundingBox<Dim>(real_coordinate_template<Dim>{0, 0}, real_coordinate_template<Dim>{1, 1}));
    Morton<Dim> morton(max_depth);

    Kokkos::View<morton_code*> input_octants("input_octants", 4);
    input_octants(0) = morton.encode({2, 2}, 4);
    input_octants(1) = morton.encode({1, 13}, 4);
    input_octants(2) = morton.encode({8, 11}, 4);
    input_octants(3) = morton.encode({12, 12}, 2);

    size_t num_elements = 31;
    Kokkos::View<morton_code*> expected("expected", num_elements);
    // Inserting in Z order (technically И ordering)
    expected(0)  = morton.encode({0, 0}, 3);
    expected(1)  = morton.encode({2, 0}, 3);
    expected(2)  = morton.encode({0, 2}, 3);
    expected(3)  = morton.encode({2, 2}, 4);
    expected(4)  = morton.encode({3, 2}, 4);
    expected(5)  = morton.encode({2, 3}, 4);
    expected(6)  = morton.encode({3, 3}, 4);
    expected(7)  = morton.encode({4, 0}, 2);
    expected(8)  = morton.encode({0, 4}, 2);
    expected(9)  = morton.encode({4, 4}, 2);
    expected(10) = morton.encode({8, 0}, 1); // left half done
    expected(11) = morton.encode({0, 8}, 2);
    expected(12) = morton.encode({4, 8}, 2);
    expected(13) = morton.encode({0, 12}, 4);
    expected(14) = morton.encode({1, 12}, 4);
    expected(15) = morton.encode({0, 13}, 4);
    expected(16) = morton.encode({1, 13}, 4);
    expected(17) = morton.encode({2, 12}, 3);
    expected(18) = morton.encode({0, 14}, 3);
    expected(19) = morton.encode({2, 14}, 3);
    expected(20) = morton.encode({4, 12}, 2); // top right corner done (wrt. dimension 1)
    expected(21) = morton.encode({8, 8}, 3);
    expected(22) = morton.encode({10, 8}, 3);
    expected(23) = morton.encode({8, 10}, 4);
    expected(24) = morton.encode({9, 10}, 4);
    expected(25) = morton.encode({8, 11}, 4);
    expected(26) = morton.encode({9, 11}, 4);
    expected(27) = morton.encode({10, 10}, 3);
    expected(28) = morton.encode({12, 8}, 2);
    expected(29) = morton.encode({8, 12}, 2);
    expected(30) = morton.encode({12, 12}, 2); // bottom right corner done

    std::sort(expected.data(), expected.data() + num_elements); // should already be sorted, nonetheless

    Kokkos::View<morton_code*> complete_region = tree.complete_tree(input_octants);

    ASSERT_EQ(expected.size(), complete_region.size()) << "Sizes dont match!";

    for (size_t i = 0; i < num_elements; ++i) {
        const auto expected_octant = expected(i);
        const auto actual_octant   = complete_region(i);
        EXPECT_EQ(actual_octant, expected_octant)
            << "expected=" << expected_octant << ", actual=" << actual_octant;
    }
}

// special cases (overlapping octants; neighboring octants) should work because complete_region handles this already

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

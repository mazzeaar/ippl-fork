#include<gtest/gtest.h>
#include<OrthoTree/OrthoTree.h>

#include<algorithm> 
#include<bitset>
#include<vector>
#include<iostream>

using namespace ippl;

TEST(CompleteRegion, CompleteSimpleQuad) {
    constexpr size_t Dim = 2;
    size_t max_depth = 3;
    OrthoTree<Dim> tree(max_depth, 2, BoundingBox<Dim>(real_coordinate_template<Dim>{0, 0}, real_coordinate_template<Dim>{1, 1}));
    Morton<Dim> morton(max_depth);

    morton_code code_a = morton.encode({0, 0}, 2);
    morton_code code_b = morton.encode({6, 6}, 2);

    size_t num_elements = 8;
    Kokkos::View<morton_code*> expected("expected", num_elements);
    expected(0) = morton.encode({2, 0}, 2);
    expected(1) = morton.encode({0, 2}, 2);
    expected(2) = morton.encode({2, 2}, 2);
    expected(3) = morton.encode({4, 4}, 2);
    expected(4) = morton.encode({4, 6}, 2);
    expected(5) = morton.encode({6, 4}, 2);
    expected(6) = morton.encode({4, 0}, 1);
    expected(7) = morton.encode({0, 4}, 1);
    std::sort(expected.data(), expected.data() + expected.size());

    Kokkos::View<morton_code*> complete_region = tree.complete_region(code_a, code_b);

    ASSERT_EQ(expected.size(), complete_region.size()) << "Sizes dont match!";

    for (size_t i = 0; i < expected.size(); ++i) {
        const auto expected_octant = expected(i);
        const auto actual_octant   = complete_region(i);
        EXPECT_EQ(actual_octant, expected_octant)
            << "expected=" << expected_octant << ", actual=" << actual_octant;
    }
}

TEST(CompleteRegion, NextToEachOther) {
    constexpr size_t Dim = 2;
    size_t max_depth = 3;
    OrthoTree<Dim> tree(max_depth, 2, BoundingBox<Dim>(real_coordinate_template<Dim>{0, 0}, real_coordinate_template<Dim>{1, 1}));
    Morton<Dim> morton(max_depth);

    morton_code code_a = morton.encode({0, 0}, 3);
    morton_code code_b = morton.encode({1, 0}, 3);

    Kokkos::View<morton_code*> complete_region = tree.complete_region(code_a, code_b);

    ASSERT_EQ(0, complete_region.size()) 
        << "Complete region isn't empty! Size: " << complete_region.size();
}

TEST(CompleteRegion, OneParentOfAnother) {
    constexpr size_t Dim = 2;
    size_t max_depth = 3;
    OrthoTree<Dim> tree(max_depth, 2, BoundingBox<Dim>(real_coordinate_template<Dim>{0, 0}, real_coordinate_template<Dim>{1, 1}));
    Morton<Dim> morton(max_depth);

    morton_code code_a = morton.encode({0, 0}, 1);
    morton_code code_b = morton.encode({0, 2}, 2);

    Kokkos::View<morton_code*> complete_region = tree.complete_region(code_a, code_b);

    ASSERT_EQ(0, complete_region.size()) 
        << "Complete region ! Size: " << complete_region.size();
}

// this is required to test the orthotree, as it depends on ippl
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
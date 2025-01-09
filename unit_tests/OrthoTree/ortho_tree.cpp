//
// unit test of morton_codes.h

#include "gtest/gtest.h"

#include "OrthoTree/OrthoTree.h"

#include <algorithm>
#include <cstdint>
#include <bitset>
#include <vector>
#include <array>

using namespace ippl;

TEST(OrthoTreeTest, BuildSimpleQuadTree)
{
    /*
    static constexpr size_t Dim = 2;
    ippl::OrthoTree<Dim> tree_2d(5, 1, ippl::BoundingBox<Dim>({ 0.0, 0.0}, { 1.0, 1.0}));
    typedef ippl::ParticleSpatialLayout<double, Dim> playout_type;
    playout_type PLayout;

    OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> particles(PLayout);

    // Example coordinate list for particles
    std::vector<ippl::Vector<double,Dim>> coordinates{{0,0},{0.1,0.6}};

    particles.create(coordinates.size());

    for(size_t i = 0; i < coordinates.size(); i++){
        particles.R(i) = coordinates.at(i);
    }

    ippl::vector_t<morton_code> tree_codes = tree_2d.build_tree_topdown_sequential(0,particles);

    Morton<Dim> morton_helper(5);
    ippl::vector_t<morton_code> expected = morton_helper.get_children(0);
    std::sort(tree_codes.begin(),tree_codes.end());
    EXPECT_EQ(tree_codes, expected);
    */
}

TEST(OrthoTreeTest, IsBalancedTest) {
    constexpr size_t Dim = 2;
    size_t max_depth = 3;
    OrthoTree<Dim> tree(max_depth, 2, BoundingBox<Dim>(real_coordinate_template<Dim>{0, 0}, real_coordinate_template<Dim>{1, 1}));
    Morton<Dim> morton(max_depth);

    Kokkos::View<morton_code*> tree_view("tree_view", 10);
    tree_view(0) = morton.encode({0, 0}, 2);
    tree_view(1) = morton.encode({2, 0}, 2);
    tree_view(2) = morton.encode({0, 2}, 2);
    tree_view(3) = morton.encode({2, 2}, 3);
    tree_view(4) = morton.encode({2, 3}, 3);
    tree_view(5) = morton.encode({3, 2}, 3);
    tree_view(6) = morton.encode({3, 3}, 3);
    tree_view(7) = morton.encode({0, 4}, 1);
    tree_view(8) = morton.encode({4, 0}, 1);
    tree_view(9) = morton.encode({4, 4}, 1);

    EXPECT_FALSE( tree.is_balanced(tree_view) );


    Kokkos::View<morton_code*> tree_view2("tree_view2", 7);
    tree_view2(0) = morton.encode({0, 0}, 2);
    tree_view2(1) = morton.encode({2, 0}, 2);
    tree_view2(2) = morton.encode({0, 2}, 2);
    tree_view2(3) = morton.encode({2, 2}, 2);
    tree_view2(4) = morton.encode({0, 4}, 1);
    tree_view2(5) = morton.encode({4, 0}, 1);
    tree_view2(6) = morton.encode({4, 4}, 1);

    EXPECT_TRUE( tree.is_balanced(tree_view2) );
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

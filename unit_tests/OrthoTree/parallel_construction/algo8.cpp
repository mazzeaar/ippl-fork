#include<gtest/gtest.h>
#include "OrthoTree/OrthoTree.h"

using namespace ippl;

// TEST(LineariseTestOct, LineariseOctantsTest) {
//     static constexpr size_t Dim = 3;
//     const size_t max_depth = 3;
//     Morton<Dim> morton(max_depth);
//     OrthoTree<Dim> tree(max_depth, 10, BoundingBox<Dim>({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}));
//     Kokkos::vector<morton_code> octs;
//     octs.push_back(morton.encode({0, 0, 0}, 2));
//     octs.push_back(morton.encode({0, 4, 0}, 1));
//     octs.push_back(morton.encode({0, 6, 6}, 2));
//     Kokkos::vector<morton_code> expected = {};

//     size_t n_parents = octs.size();
//     for(size_t i = 0; i < n_parents; ++i)
//     {
//         vector_t<morton_code> children = morton.get_children(octs[i]);
//         for(size_t j = 0; j < 8; ++j)
//         {
//             octs.push_back(children[j]);
//             expected.push_back(children[j]);
//         }
//     }

//     std::cerr << "I MAANGED TO GET TILL HERE\n";

//     // octants need to be sorted
//     std::sort(expected.begin(), expected.end());
//     std::sort(octs.begin(), octs.end());

//     Kokkos::vector<morton_code> linearised = tree.linearise_octants(octs);
//     EXPECT_EQ(linearised.size(), expected.size());
//     for (size_t i = 0; i < linearised.size(); ++i) {
//         EXPECT_EQ(linearised[i], expected[i]);
//     }
// }

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
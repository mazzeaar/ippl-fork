#include<gtest/gtest.h>
#include<OrthoTree/OrthoTree.h>

#include<algorithm> 
#include<bitset>
#include <string>
#include<vector>
#include<iostream>

using namespace ippl;

TEST(BlockPartition, BlockPartitionTest) {
    constexpr size_t Dim = 2;
    size_t max_depth = 4;
    size_t max_particles = 2;
    OrthoTree<Dim> orthotree(max_depth, max_particles, BoundingBox<Dim>(real_coordinate_template<Dim>{0.0, 0.0}, real_coordinate_template<Dim>{1.0, 1.0}));
    Morton<Dim> morton(max_depth);

    Kokkos::View<morton_code*> tree("tree", 8);
    if(Comm->rank() == 0){
        tree(0) = morton.encode({0, 0}, 2);
        tree(1) = morton.encode({4, 0}, 2);
        tree(2) = morton.encode({0, 4}, 2);
        tree(3) = morton.encode({4, 4}, 3);
        tree(4) = morton.encode({6, 4}, 3);
        tree(5) = morton.encode({4, 6}, 4);
        tree(6) = morton.encode({5, 6}, 4);
        tree(7) = morton.encode({4, 7}, 4);
        orthotree.getAidList()->resize(2);
        orthotree.getAidList()->setOctant(452, 0);
        orthotree.getAidList()->setID(1, 0);
        orthotree.getAidList()->setOctant(452, 1);
        orthotree.getAidList()->setID(2, 1);
    } else if(Comm->rank() == 1){
        tree(0) = morton.encode({5, 7}, 4);
        tree(1) = morton.encode({6, 6}, 3);
        tree(2) = morton.encode({8, 0}, 2);
        tree(3) = morton.encode({12, 0}, 2);
        tree(4) = morton.encode({8, 4}, 2);
        tree(5) = morton.encode({12, 4}, 3);
        tree(6) = morton.encode({14, 4}, 3);
        tree(7) = morton.encode({12, 6}, 3);
        orthotree.getAidList()->resize(1);
        orthotree.getAidList()->setOctant(484, 0);
        orthotree.getAidList()->setID(3, 0);
    } else if(Comm->rank() == 2){
        tree(0) = morton.encode({14, 6}, 4);
        tree(1) = morton.encode({15, 6}, 4);
        tree(2) = morton.encode({14, 7}, 4);
        tree(3) = morton.encode({15, 7}, 4);
        tree(4) = morton.encode({0, 8}, 1);
        tree(5) = morton.encode({8, 8}, 2);
        tree(6) = morton.encode({12, 8}, 2);
        tree(7) = morton.encode({8, 12}, 3);
        orthotree.getAidList()->resize(3);
        orthotree.getAidList()->setOctant(996, 0);
        orthotree.getAidList()->setID(4, 0);
        orthotree.getAidList()->setOctant(996, 1);
        orthotree.getAidList()->setID(5, 1);
        orthotree.getAidList()->setOctant(996, 2);
        orthotree.getAidList()->setID(6, 2);
    } else if(Comm->rank() == 3){
        Kokkos::resize(tree, 7);
        tree(0) = morton.encode({10, 12}, 3);
        tree(1) = morton.encode({8, 14}, 3);
        tree(2) = morton.encode({10, 14}, 3);
        tree(3) = morton.encode({12, 12}, 3);
        tree(4) = morton.encode({14, 12}, 3);
        tree(5) = morton.encode({12, 14}, 3);
        tree(6) = morton.encode({14, 14}, 3);
        orthotree.getAidList()->resize(6);
        orthotree.getAidList()->setOctant(1827, 0);
        orthotree.getAidList()->setID(7, 0);
        orthotree.getAidList()->setOctant(1827, 1);
        orthotree.getAidList()->setID(8, 1);
        orthotree.getAidList()->setOctant(1891, 2);
        orthotree.getAidList()->setID(9, 2);
        orthotree.getAidList()->setOctant(1923, 3);
        orthotree.getAidList()->setID(10, 3);
        orthotree.getAidList()->setOctant(1955, 4);
        orthotree.getAidList()->setID(11, 4);
        orthotree.getAidList()->setOctant(2019, 5);
        orthotree.getAidList()->setID(12, 5);
    }

    //Kokkos::View<morton_code*> partition = orthotree.block_partition(tree(0), tree(tree.size()-1));
    Kokkos::View<morton_code*> partition = orthotree.algo4_11(tree);

    std::string log_str = "Rank " + std::to_string(Comm->rank()) + ": partition = {";
    for (size_t i = 0; i < tree.size(); i++) {
        log_str += std::to_string(partition(i));
        if(i != tree.size()-1) log_str += ", ";
    }
    log_str += "}\n";
    std::cerr << log_str;
    if(Comm->rank() == 0){
        EXPECT_EQ(3, partition.size());
        EXPECT_EQ(2, partition(0));
        EXPECT_EQ(130, partition(1));
        EXPECT_EQ(258, partition(2));
    } else if(Comm->rank() == 1){
        EXPECT_EQ(4, partition.size());
        EXPECT_EQ(386, partition(0));
        EXPECT_EQ(514, partition(1));
        EXPECT_EQ(642, partition(2));
        EXPECT_EQ(770, partition(3));
    } else if(Comm->rank() == 2){
        EXPECT_EQ(4, partition.size());
        EXPECT_EQ(898, partition(0));
        EXPECT_EQ(1025, partition(1));
        EXPECT_EQ(1538, partition(2));
        EXPECT_EQ(1666, partition(3));
    } else if(Comm->rank() == 3){
        EXPECT_EQ(2, partition.size());
        EXPECT_EQ(1922, partition(0));
        EXPECT_EQ(1794, partition(1));
    }
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

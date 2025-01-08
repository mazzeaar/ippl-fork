#include <Kokkos_Core.hpp>

#include <Kokkos_Sort.hpp>
#include <ctime>
#include <ostream>
#include <random>
#include <string>

#include "OrthoTree/OrthoTreeTypes.h"

#include "Communicate/Communicator.h"
#include "OrthoTree/OrthoTree.h"
#include "gtest/gtest.h"
#include "test_helper.h"

using namespace ippl;

// ================================================================
//                          TESTS START
// ================================================================

// sanity check, tests are designed to run (and pass) with 4 ranks
TEST(ParallelConstruction4, AssertWorldSize) {
    ASSERT_EQ(Comm->size(), 4);
}

TEST(ParallelConstruction4, ActualTest) {
    runTests();
    /*
    const size_t num_particles = 6;
    const size_t max_depth = 4;
    const size_t max_particles = 3;
    const double min_bound = 0.0;
    const double max_bound = 1.0;
    static constexpr size_t Dim = 2;

    typedef ippl::ParticleSpatialLayout<double, Dim> playout_type;
    typedef ippl::OrthoTreeParticle<playout_type> bunch_type;

    playout_type PLayout;
    bunch_type bunch(PLayout);

    bunch.create(num_particles);

    typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();
    if(Comm->rank() == 0){
        R_host(0) = ippl::Vector<double, Dim>{0.170, 0.080};
        R_host(1) = ippl::Vector<double, Dim>{0.171, 0.080};
        R_host(2) = ippl::Vector<double, Dim>{0.172, 0.080};
        R_host(3) = ippl::Vector<double, Dim>{0.173, 0.080};
        R_host(4) = ippl::Vector<double, Dim>{0.440, 0.440};
        R_host(5) = ippl::Vector<double, Dim>{0.441, 0.440};
    } else if(Comm->rank() == 1){
        R_host(0) = ippl::Vector<double, Dim>{0.442, 0.440};
        R_host(1) = ippl::Vector<double, Dim>{0.443, 0.440};
        R_host(2) = ippl::Vector<double, Dim>{0.960, 0.080};
        R_host(3) = ippl::Vector<double, Dim>{0.961, 0.080};
        R_host(4) = ippl::Vector<double, Dim>{0.962, 0.080};
        R_host(5) = ippl::Vector<double, Dim>{0.963, 0.080};
    } else if(Comm->rank() == 2){
        R_host(0) = ippl::Vector<double, Dim>{0.800, 0.300};
        R_host(1) = ippl::Vector<double, Dim>{0.801, 0.300};
        R_host(2) = ippl::Vector<double, Dim>{0.960, 0.300};
        R_host(3) = ippl::Vector<double, Dim>{0.961, 0.300};
        R_host(4) = ippl::Vector<double, Dim>{0.800, 0.700};
        R_host(5) = ippl::Vector<double, Dim>{0.801, 0.700};
    } else if(Comm->rank() == 3){
        R_host(0) = ippl::Vector<double, Dim>{0.802, 0.700};
        R_host(1) = ippl::Vector<double, Dim>{0.803, 0.700};
        R_host(2) = ippl::Vector<double, Dim>{0.960, 0.960};
        R_host(3) = ippl::Vector<double, Dim>{0.961, 0.960};
        R_host(4) = ippl::Vector<double, Dim>{0.962, 0.960};
        R_host(5) = ippl::Vector<double, Dim>{0.963, 0.960};
    }

    Kokkos::deep_copy(bunch.R.getView(), R_host);
    bunch.update();

    BoundingBox<Dim> root_bounds({min_bound, min_bound}, {max_bound, max_bound});
    OrthoTree<Dim> tree(max_depth, max_particles, root_bounds);
    Kokkos::View<morton_code*> parallel_tree =  tree.build_tree(bunch);

    std::string log_str = "Rank " + std::to_string(Comm->rank()) + ": parallel_tree = {";
    for(size_t i = 0; i < parallel_tree.size(); i++){
        log_str += std::to_string(parallel_tree(i));
        if(i != parallel_tree.size()-1) log_str += ", ";
    }
    log_str += "}\n";
    std::cerr << log_str;
    */

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

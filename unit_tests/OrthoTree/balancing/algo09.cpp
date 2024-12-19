#include <gtest/gtest.h>
#include <random>

#include "OrthoTree/OrthoTree.h"

using namespace ippl;

template <size_t Dim>
auto generateParticles(size_t num_particles_per_proc, const double min_bound, double max_bound,
                       uint64_t seed) {
    typedef ippl::ParticleSpatialLayout<double, Dim> playout_type;
    typedef ippl::OrthoTreeParticle<playout_type> bunch_type;

    playout_type PLayout;
    bunch_type bunch(PLayout);

    const size_t num_particles = num_particles_per_proc * Comm->size();

    bunch.create(num_particles_per_proc);

    typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();
    const double bounds_size                                       = max_bound - min_bound;

    std::mt19937_64 eng(seed);
    std::uniform_real_distribution<double> unif(min_bound, max_bound);

    // center of the bounding box
    const double center_x = bounds_size / 2;
    const double center_y = bounds_size / 2;
    const double center_z = (Dim == 3) ? bounds_size / 2 : 0.0;

    double armCount     = 2.0;   // number of spiral arms
    double armTightness = -0.1;  // tightness of the spiral arms

    // max distance from the center
    double max_distance = 0.9 * bounds_size / 2;

    for (unsigned i = 0; i < num_particles; ++i) {
        double angle    = unif(eng) * 2.0 * M_PI;
        double distance = unif(eng) * max_distance;

        double totalArmAngle = 5.0;

        for (int j = 0; j < armCount; ++j) {
            double armAngle = armTightness * angle + (j + 1) * distance / max_distance * 2.0 * M_PI;
            double x        = center_x + distance * cos(armAngle + totalArmAngle);
            double y        = center_y + distance * sin(armAngle + totalArmAngle);
            double z        = (Dim == 3) ? center_z + distance * sin(angle) : 0.0;

            if constexpr (Dim == 2) {
                bunch.R(i) = {x, y};
            } else if constexpr (Dim == 3) {
                bunch.R(i) = {x, y, z};
            }

            totalArmAngle += armAngle;
        }
    }

    Kokkos::deep_copy(bunch.R.getView(), R_host);
    bunch.update();
    return bunch;
}

TEST(RipplePropagation, TestTest) {
    constexpr size_t Dim = 2;
    size_t max_depth = 3;
    OrthoTree<Dim> tree(max_depth, 2, BoundingBox<Dim>(real_coordinate_template<Dim>{0, 0}, real_coordinate_template<Dim>{1, 1}));
    Morton<Dim> morton(max_depth);

    morton_code root = 0;

    Kokkos::View<morton_code*> tree_view("tree_view", 10);
    tree_view(0) = morton.encode({0, 0}, 2);
    tree_view(1) = morton.encode({2, 0}, 2);
    tree_view(2) = morton.encode({0, 2}, 2);
    tree_view(3) = morton.encode({2, 2}, 3);
    tree_view(4) = morton.encode({2, 3}, 3);
    tree_view(5) = morton.encode({3, 2}, 3);
    tree_view(6) = morton.encode({3, 3}, 3);
    tree_view(7) = morton.encode({4, 0}, 1);
    tree_view(8) = morton.encode({0, 4}, 1);
    tree_view(9) = morton.encode({4, 4}, 1);

    auto balanced_tree = tree.algo9(tree_view);

    Kokkos::View<morton_code*> expected("expected", 19);
    for (size_t i = 0; i < 7; ++i) {
        expected(i) = tree_view(i);
    }
    expected(7) = morton.encode({0, 4}, 2);
    expected(8) = morton.encode({2, 4}, 2);
    expected(9) = morton.encode({4, 4}, 2);
    expected(10) = morton.encode({4, 2}, 2);
    expected(11) = morton.encode({4, 0}, 2);
    expected(12) = morton.encode({0, 6}, 2);
    expected(13) = morton.encode({2, 6}, 2);
    expected(14) = morton.encode({4, 6}, 2);
    expected(15) = morton.encode({6, 6}, 2);
    expected(16) = morton.encode({6, 4}, 2);
    expected(17) = morton.encode({6, 2}, 2);
    expected(18) = morton.encode({6, 0}, 2);
    
    std::sort(expected.data(), expected.data() + expected.size());
    
    EXPECT_EQ(expected.size(), balanced_tree.size()) << "Sizes dont match!";
    for (int i = 0; i < std::min(expected.size(), balanced_tree.size()); ++i) {
        EXPECT_EQ(balanced_tree(i), expected(i))
          << "expected=" << morton.decode(expected(i)) 
          << ", actual=" << morton.decode(balanced_tree(i));
    }
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

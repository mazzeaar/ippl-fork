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

TEST(BalancingTest, TestTest) {
    static constexpr size_t Dim = 2;
    const size_t max_depth      = 10;
    const size_t max_particles  = 50;
    const size_t n_particles    = 20000;

    BoundingBox<Dim> bounds({0.0, 0.0}, {1.0, 1.0});
    OrthoTree<Dim> tree(max_depth, max_particles, bounds);

    Morton<Dim> morton_helper(max_depth);

    auto particles  = generateParticles<Dim>(n_particles, 0.0, 1.0, 100);
    auto built_tree = tree.build_tree(particles);
    tree.setVisualisation(true);
    std::cerr << "HERE!!!!!" << std::endl;
    // auto res = tree.algo11(built_tree);
    auto res = tree.algo7(0, built_tree);
    // EXPECT_EQ(res.size(), 28);

    // output to test
    Kokkos::resize(res, res.size() + 1);
    res[res.size() - 1] = 0;

    tree.particles_to_file(particles);
    tree.octants_to_file(res);
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
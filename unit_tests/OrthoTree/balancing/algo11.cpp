#include <gtest/gtest.h>
#include <random>
#include <string>

#include "OrthoTree/OrthoTree.h"

using namespace ippl;

/*
template <size_t Dim>
auto generateParticles(size_t num_particles_per_proc, const double min_bound, double max_bound) {
    typedef ippl::ParticleSpatialLayout<double, Dim> playout_type;
    typedef ippl::OrthoTreeParticle<playout_type> bunch_type;

    playout_type PLayout;
    bunch_type bunch(PLayout);

    const size_t num_particles = num_particles_per_proc * Comm->size();

    bunch.create(num_particles_per_proc);

    typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();
    const double bounds_size                                       = max_bound - min_bound;

    std::mt19937_64 eng;
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
*/

template <size_t Dim>
auto initializeRandom(size_t num_particles, double min_bounds, double max_bounds) {
    static_assert((Dim == 2 || Dim == 3) && "We only specialise for 2D and 3D!");

    typedef ippl::ParticleSpatialLayout<double, Dim> playout_type;
    typedef ippl::OrthoTreeParticle<playout_type> bunch_type;

    playout_type PLayout;
    bunch_type bunch(PLayout);

    bunch.create(num_particles);

    typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();

    std::mt19937_64 eng;
    std::uniform_real_distribution<double> unif(min_bounds, max_bounds);

    for (unsigned int i = 0; i < num_particles * Comm->size(); ++i) {
        if constexpr (Dim == 2) {
            R_host(i) = ippl::Vector<double, Dim>{unif(eng), unif(eng)};
        } else if constexpr (Dim == 3) {
            R_host(i) = ippl::Vector<double, Dim>{unif(eng), unif(eng), unif(eng)};
        } else {
            std::cerr << "We only specialise for 2D and 3D!" << std::endl;
            exit(1);
        }
    }

    Kokkos::deep_copy(bunch.R.getView(), R_host);
    bunch.update();
    return bunch;
}

TEST(BalancingTest, TestTest) {
    /*
    static constexpr size_t Dim = 2;
    const size_t max_depth      = 10;
    const size_t max_particles  = 2;
    const size_t n_particles    = 5000;

    BoundingBox<Dim> bounds({0.0, 0.0}, {1.0, 1.0});
    OrthoTree<Dim> tree(max_depth, max_particles, bounds);

    Morton<Dim> morton_helper(max_depth);

    // auto particles = generateParticles<Dim>(n_particles, 0.0, 1.0);
    auto particles  = initializeRandom<Dim>(n_particles, 0.0, 1.0);
    auto built_tree = tree.build_tree(particles);

    std::cerr << "STARTING ALGO11" << std::endl;
    tree.setVisualisation(true);

    auto res = tree.algo11(built_tree);

    //   EXPECT_EQ(res.size(), 28);

    // output to test
    Kokkos::resize(res, res.size() + 1);
    res[res.size() - 1] = 0;

    tree.particles_to_file(particles);
    tree.octants_to_file(res);
    /*/
     constexpr size_t Dim = 2;
     size_t max_depth = 4;
     OrthoTree<Dim> tree(max_depth, 2, BoundingBox<Dim>(real_coordinate_template<Dim>{0, 0},
    real_coordinate_template<Dim>{1, 1})); Morton<Dim> morton(max_depth);

     /*
     Kokkos::View<morton_code*> tree_view("tree_view", 3);

     if(Comm->rank() == 0){
        Kokkos::resize(tree_view, 3);
        tree_view(0) = morton.encode({0, 0}, 2);
        tree_view(1) = morton.encode({2, 0}, 2);
        tree_view(2) = morton.encode({0, 2}, 2);
     } else if(Comm->rank() == 1){
        Kokkos::resize(tree_view, 3);
        tree_view(0) = morton.encode({2, 2}, 3);
        tree_view(1) = morton.encode({2, 3}, 3);
        tree_view(2) = morton.encode({3, 2}, 3);
        
     } else if(Comm->rank() == 2){
        Kokkos::resize(tree_view, 2);
        tree_view(0) = morton.encode({3, 3}, 3);
        tree_view(1) = morton.encode({4, 0}, 1);
        
     } else{
        Kokkos::resize(tree_view, 2);
        tree_view(0) = morton.encode({0, 4}, 1);
        tree_view(1) = morton.encode({4, 4}, 1);

     }
     */
    /*
     Kokkos::View<morton_code*> tree_view("tree_view", 10);
    
    tree_view(0) = morton.encode({0, 0}, 2);
    tree_view(1) = morton.encode({2, 0}, 2);
    tree_view(2) = morton.encode({0, 2}, 2);
    tree_view(3) = morton.encode({2, 2}, 3);
    tree_view(4) = morton.encode({3, 2}, 3);
    tree_view(5) = morton.encode({2, 3}, 3);
    tree_view(6) = morton.encode({3, 3}, 3);
    tree_view(7) = morton.encode({4, 0}, 1);
    tree_view(8) = morton.encode({0, 4}, 1);
    tree_view(9) = morton.encode({4, 4}, 1);

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
    */
     Kokkos::View<morton_code*> tree_view("tree_view", 12);

     if(Comm->rank() == 0){
        Kokkos::resize(tree_view, 12);
        tree_view(0) = morton.encode({0, 0}, 3);
        tree_view(1) = morton.encode({2, 0}, 4);
        tree_view(2) = morton.encode({3, 0}, 4);
        tree_view(3) = morton.encode({2, 1}, 4);
        tree_view(4) = morton.encode({3, 1}, 4);
        tree_view(5) = morton.encode({0, 2}, 3);
        tree_view(6) = morton.encode({2, 2}, 3);
        tree_view(7) = morton.encode({4, 0}, 2);
        tree_view(8) = morton.encode({0, 4}, 2);
        tree_view(9) = morton.encode({4, 4}, 3);
        tree_view(10) = morton.encode({6, 4}, 3);
        tree_view(11) = morton.encode({4, 6}, 3);
     } else if(Comm->rank() == 1){
        Kokkos::resize(tree_view, 12);
        tree_view(0) = morton.encode({6, 6}, 4);
        tree_view(1) = morton.encode({7, 6}, 4);
        tree_view(2) = morton.encode({6, 7}, 4);
        tree_view(3) = morton.encode({7, 7}, 4);
        tree_view(4) = morton.encode({8, 0}, 2);
        tree_view(5) = morton.encode({12, 0}, 3);
        tree_view(6) = morton.encode({14, 0}, 4);
        tree_view(7) = morton.encode({15, 0}, 4);
        tree_view(8) = morton.encode({14, 1}, 4);
        tree_view(9) = morton.encode({15, 1}, 4);
        tree_view(10) = morton.encode({12, 2}, 3);
        tree_view(11) = morton.encode({14, 2}, 3);
        
     } else if(Comm->rank() == 2){
        Kokkos::resize(tree_view, 11);
        tree_view(0) = morton.encode({8, 4}, 2);
        tree_view(1) = morton.encode({12, 4}, 3);
        tree_view(2) = morton.encode({14, 4}, 3);
        tree_view(3) = morton.encode({12, 6}, 3);
        tree_view(4) = morton.encode({14, 6}, 3);
        tree_view(5) = morton.encode({0, 8}, 1);
        tree_view(6) = morton.encode({8, 8}, 2);
        tree_view(7) = morton.encode({12, 8}, 3);
        tree_view(8) = morton.encode({14, 8}, 3);
        tree_view(9) = morton.encode({12, 10}, 4);
        tree_view(10) = morton.encode({13, 10}, 4);
        
     } else{
        Kokkos::resize(tree_view, 11);
        tree_view(0) = morton.encode({12, 11}, 4);
        tree_view(1) = morton.encode({13, 11}, 4);
        tree_view(2) = morton.encode({14, 10}, 3);
        tree_view(3) = morton.encode({8, 12}, 2);
        tree_view(4) = morton.encode({12, 12}, 3);
        tree_view(5) = morton.encode({14, 12}, 3);
        tree_view(6) = morton.encode({12, 14}, 3);
        tree_view(7) = morton.encode({14, 14}, 4);
        tree_view(8) = morton.encode({15, 14}, 4);
        tree_view(9) = morton.encode({14, 15}, 4);
        tree_view(10) = morton.encode({15, 15}, 4);

     }

     std::sort(tree_view.data(), tree_view.data() + tree_view.size());
     auto balanced_tree = tree_view;
    try {
        balanced_tree = tree.algo11(tree_view);

    } catch (IpplException& e) {
        std::cout << "IpplException: " << e.what() << std::endl;
        throw e;
    } catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        throw e;
    }

    /*
   auto size_str = "On rank " + std::to_string(Comm->rank()) + ", balanced_tree.size(): " +
   std::to_string(balanced_tree.size()); std::cerr << size_str << std::endl; for (int i = 0; i <
   balanced_tree.size(); ++i) { auto output = "On rank " + std::to_string(Comm->rank()) + ",
   balanced_tree(" + std::to_string(i) + "): " + std::to_string(balanced_tree(i)); std::cerr <<
   output << std::endl;
    }
    *
    EXPECT_EQ(expected.size(), balanced_tree.size()) << "Sizes dont match!";
    for (int i = 0; i < std::min(expected.size(), balanced_tree.size()); ++i) {
        EXPECT_EQ(balanced_tree(i), expected(i))
          << "expected=" << morton.decode(expected(i))
          << ", actual=" << morton.decode(balanced_tree(i));
    }

   if(Comm->rank() == 0){
       EXPECT_EQ(3, balanced_tree.size()) << "Sizes dont match!";
       EXPECT_EQ(2, balanced_tree(0)) << "expected = " << 2 << ", actual = " << balanced_tree(0);
       EXPECT_EQ(18, balanced_tree(1)) << "expected = " << 18 << ", actual = " << balanced_tree(1);
       EXPECT_EQ(34, balanced_tree(2)) << "expected = " << 34 << ", actual = " << balanced_tree(2);
   } else if(Comm->rank() == 1){
       EXPECT_EQ(3, balanced_tree.size()) << "Sizes dont match!";
       EXPECT_EQ(51, balanced_tree(0)) << "expected = " << 51 << ", actual = " << balanced_tree(0);
       EXPECT_EQ(55, balanced_tree(1)) << "expected = " << 55 << ", actual = " << balanced_tree(1);
       EXPECT_EQ(59, balanced_tree(2)) << "expected = " << 59 << ", actual = " << balanced_tree(2);
   } else if(Comm->rank() == 2){
       EXPECT_EQ(5, balanced_tree.size()) << "Sizes dont match!";
       EXPECT_EQ(63, balanced_tree(0)) << "expected = " << 63 << ", actual = " << balanced_tree(0);
       EXPECT_EQ(66, balanced_tree(1)) << "expected = " << 66 << ", actual = " << balanced_tree(1);
       EXPECT_EQ(82, balanced_tree(2)) << "expected = " << 82 << ", actual = " << balanced_tree(2);
       EXPECT_EQ(98, balanced_tree(3)) << "expected = " << 98 << ", actual = " << balanced_tree(3);
       EXPECT_EQ(114, balanced_tree(4)) << "expected = " << 114 << ", actual = " <<
   balanced_tree(4); } else if(Comm->rank() == 3){ EXPECT_EQ(8, balanced_tree.size()) << "Sizes dont
   match!"; EXPECT_EQ(130, balanced_tree(0)) << "expected = " << 130 << ", actual = " <<
   balanced_tree(0); EXPECT_EQ(146, balanced_tree(1)) << "expected = " << 146 << ", actual = " <<
   balanced_tree(1); EXPECT_EQ(162, balanced_tree(2)) << "expected = " << 162 << ", actual = " <<
   balanced_tree(2); EXPECT_EQ(178, balanced_tree(3)) << "expected = " << 178 << ", actual = " <<
   balanced_tree(3); EXPECT_EQ(194, balanced_tree(4)) << "expected = " << 194 << ", actual = " <<
   balanced_tree(4); EXPECT_EQ(210, balanced_tree(5)) << "expected = " << 210 << ", actual = " <<
   balanced_tree(5); EXPECT_EQ(226, balanced_tree(6)) << "expected = " << 226 << ", actual = " <<
   balanced_tree(6); EXPECT_EQ(242, balanced_tree(7)) << "expected = " << 242 << ", actual = " <<
   balanced_tree(7);
   }
   */
    if(Comm->rank() == 0){
        EXPECT_EQ(12, balanced_tree.size());
        EXPECT_EQ(morton.encode({0, 0}, 3), balanced_tree(0));
        EXPECT_EQ(morton.encode({2, 0}, 4), balanced_tree(1));
        EXPECT_EQ(morton.encode({3, 0}, 4), balanced_tree(2));
        EXPECT_EQ(morton.encode({2, 1}, 4), balanced_tree(3));
        EXPECT_EQ(morton.encode({3, 1}, 4), balanced_tree(4));
        EXPECT_EQ(morton.encode({0, 2}, 3), balanced_tree(5));
        EXPECT_EQ(morton.encode({2, 2}, 3), balanced_tree(6));
        EXPECT_EQ(morton.encode({4, 0}, 3), balanced_tree(7));
        EXPECT_EQ(morton.encode({6, 0}, 3), balanced_tree(8));
        EXPECT_EQ(morton.encode({4, 2}, 3), balanced_tree(9));
        EXPECT_EQ(morton.encode({6, 2}, 3), balanced_tree(10));
        EXPECT_EQ(morton.encode({0, 4}, 2), balanced_tree(11));
    } else if(Comm->rank() == 1){
        EXPECT_EQ(15, balanced_tree.size());
        EXPECT_EQ(morton.encode({4, 4}, 3), balanced_tree(0));
        EXPECT_EQ(morton.encode({6, 4}, 3), balanced_tree(1));
        EXPECT_EQ(morton.encode({4, 6}, 3), balanced_tree(2));
        EXPECT_EQ(morton.encode({6, 6}, 4), balanced_tree(3));
        EXPECT_EQ(morton.encode({7, 6}, 4), balanced_tree(4));
        EXPECT_EQ(morton.encode({6, 7}, 4), balanced_tree(5));
        EXPECT_EQ(morton.encode({7, 7}, 4), balanced_tree(6));
        EXPECT_EQ(morton.encode({8, 0}, 2), balanced_tree(7));
        EXPECT_EQ(morton.encode({12, 0}, 3), balanced_tree(8));
        EXPECT_EQ(morton.encode({14, 0}, 4), balanced_tree(9));
        EXPECT_EQ(morton.encode({15, 0}, 4), balanced_tree(10));
        EXPECT_EQ(morton.encode({14, 1}, 4), balanced_tree(11));
        EXPECT_EQ(morton.encode({15, 1}, 4), balanced_tree(12));
        EXPECT_EQ(morton.encode({12, 2}, 3), balanced_tree(13));
        EXPECT_EQ(morton.encode({14, 2}, 3), balanced_tree(14));
    } else if(Comm->rank() == 2){
        EXPECT_EQ(19, balanced_tree.size());
        EXPECT_EQ(morton.encode({8, 4}, 3), balanced_tree(0));
        EXPECT_EQ(morton.encode({10, 4}, 3), balanced_tree(1));
        EXPECT_EQ(morton.encode({8, 6}, 3), balanced_tree(2));
        EXPECT_EQ(morton.encode({10, 6}, 3), balanced_tree(3));
        EXPECT_EQ(morton.encode({12, 4}, 3), balanced_tree(4));
        EXPECT_EQ(morton.encode({14, 4}, 3), balanced_tree(5));
        EXPECT_EQ(morton.encode({12, 6}, 3), balanced_tree(6));
        EXPECT_EQ(morton.encode({14, 6}, 3), balanced_tree(7));
        EXPECT_EQ(morton.encode({0, 8}, 2), balanced_tree(8));
        EXPECT_EQ(morton.encode({4, 8}, 3), balanced_tree(9));
        EXPECT_EQ(morton.encode({6, 8}, 3), balanced_tree(10));
        EXPECT_EQ(morton.encode({4, 10}, 3), balanced_tree(11));
        EXPECT_EQ(morton.encode({6, 10}, 3), balanced_tree(12));
        EXPECT_EQ(morton.encode({0, 12}, 2), balanced_tree(13));
        EXPECT_EQ(morton.encode({4, 12}, 2), balanced_tree(14));
        EXPECT_EQ(morton.encode({8, 8}, 3), balanced_tree(15));
        EXPECT_EQ(morton.encode({10, 8}, 3), balanced_tree(16));
        EXPECT_EQ(morton.encode({8, 10}, 3), balanced_tree(17));
        EXPECT_EQ(morton.encode({10, 10}, 3), balanced_tree(18));
    } else if(Comm->rank() == 3){
        EXPECT_EQ(18, balanced_tree.size());
        EXPECT_EQ(morton.encode({12, 8}, 3), balanced_tree(0));
        EXPECT_EQ(morton.encode({14, 8}, 3), balanced_tree(1));
        EXPECT_EQ(morton.encode({12, 10}, 4), balanced_tree(2));
        EXPECT_EQ(morton.encode({13, 10}, 4), balanced_tree(3));
        EXPECT_EQ(morton.encode({12, 11}, 4), balanced_tree(4));
        EXPECT_EQ(morton.encode({13, 11}, 4), balanced_tree(5));
        EXPECT_EQ(morton.encode({14, 10}, 3), balanced_tree(6));
        EXPECT_EQ(morton.encode({8, 12}, 3), balanced_tree(7));
        EXPECT_EQ(morton.encode({10, 12}, 3), balanced_tree(8));
        EXPECT_EQ(morton.encode({8, 14}, 3), balanced_tree(9));
        EXPECT_EQ(morton.encode({10, 14}, 3), balanced_tree(10));
        EXPECT_EQ(morton.encode({12, 12}, 3), balanced_tree(11));
        EXPECT_EQ(morton.encode({14, 12}, 3), balanced_tree(12));
        EXPECT_EQ(morton.encode({12, 14}, 3), balanced_tree(13));
        EXPECT_EQ(morton.encode({14, 14}, 4), balanced_tree(14));
        EXPECT_EQ(morton.encode({15, 14}, 4), balanced_tree(15));
        EXPECT_EQ(morton.encode({14, 15}, 4), balanced_tree(16));
        EXPECT_EQ(morton.encode({15, 15}, 4), balanced_tree(17));
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

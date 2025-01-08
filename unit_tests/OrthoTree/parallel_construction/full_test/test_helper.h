/**
 * This file is a bit messy, sry.
 *
 * Whats going on:
 * At the bottom there is a test function which gets called multiple times, each time with a
 * different amount of ranks. The function executes a number of configurations and compares the
 * parallel tree to the sequentially generated tree each time.
 */

#pragma once

#include <Kokkos_Core.hpp>

#include <Kokkos_Sort.hpp>
#include <exception>
#include <gtest/gtest.h>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>

#include "OrthoTree/OrthoTreeTypes.h"

#include "Communicate/Communicator.h"
#include "OrthoTree/OrthoTree.h"
#include "gtest/gtest.h"

using namespace ippl;

/**
 * @brief Helper to generate a command that lets us replicate the failed test exactly.
 */
template <size_t Dim>
std::string generateReplicationCommand(int world_size, uint64_t seed, size_t num_particles_per_proc,
                                       size_t max_depth, size_t max_particles_per_octant,
                                       double min_bounds, double max_bounds) {
    std::ostringstream command;
    command << "\n./visualise.sh " << world_size << " -dim=" << Dim << " -seed=" << seed
            << " -num_particles=" << num_particles_per_proc << " -max_depth=" << max_depth
            << " -max_particles=" << max_particles_per_octant << " -min_bounds=" << min_bounds
            << " -max_bounds=" << max_bounds << " -dist=random\n";
    return command.str();
}

/**
 * @brief Function to generate particles that are randomly distributed in a square
 */
template <size_t Dim>
auto generateParticles(size_t num_particles_per_proc, const double min_bound, double max_bound,
                       uint64_t seed) {
    typedef ippl::ParticleSpatialLayout<double, Dim> playout_type;
    typedef ippl::OrthoTreeParticle<playout_type> bunch_type;

    playout_type PLayout;
    bunch_type bunch(PLayout);

    std::mt19937_64 eng(seed);
    std::uniform_real_distribution<double> unif(min_bound, max_bound);

    bunch.create(num_particles_per_proc);

    typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();
    for (unsigned int i = 0; i < num_particles_per_proc; ++i) {
        if constexpr (Dim == 3) {
            R_host(i) = ippl::Vector<double, Dim>{unif(eng), unif(eng), unif(eng)};
        } else if constexpr (Dim == 2) {
            R_host(i) = ippl::Vector<double, Dim>{unif(eng), unif(eng)};
        }
    }

    Kokkos::deep_copy(bunch.R.getView(), R_host);
    bunch.update();
    return bunch;
}

Kokkos::View<ippl::morton_code*> gatherTreeOnRootRank(Kokkos::View<ippl::morton_code*>& tree) {
    const size_t world_rank = static_cast<size_t>(Comm->rank());
    const size_t world_size = static_cast<size_t>(Comm->size());

    // gather sizes from all ranks on rank 0
    size_t local_size = tree.extent(0);
    std::vector<size_t> local_sizes(world_size);
    Comm->gather(&local_size, local_sizes.data(), 1, 0);

    if (world_rank == 0) {
        const size_t total_size = std::accumulate(local_sizes.begin(), local_sizes.end(), 0);
        Kokkos::resize<morton_code*>(tree, total_size);

        size_t offset = local_sizes[0];
        for (size_t rank = 1; rank < world_size; ++rank) {
            mpi::Status data_status;
            Comm->recv(tree.data() + offset, local_sizes[rank], rank, 0, data_status);
            offset += local_sizes[rank];
        }
    } else {
        Comm->send(*tree.data(), local_size, 0, 0);
    }

    return tree;
}

template <size_t Dim>
std::string executeTestRun(BoundingBox<Dim>& root_bounds, OrthoTree<Dim>& tree,
                           const auto& particles) {
    // disable annoying logs, we dont need them in tests
    tree.setVisualisation(false);
    tree.setPrintStats(false);
    tree.setLogOutput(false);

    Kokkos::View<ippl::morton_code*> parallel_tree   = tree.build_tree(particles);
    Kokkos::View<ippl::morton_code*> sequential_tree;

    // gather sizes from all ranks on rank 0
    parallel_tree = gatherTreeOnRootRank(parallel_tree);

    ippl::Morton<Dim> morton_helper(tree.getMaxDepth());

    const size_t world_rank = Comm->rank();
    std::ostringstream oss;
    if (world_rank == 0) {
       sequential_tree = tree.build_tree_naive(particles);
        // check that the sizes match
        const size_t total_size    = parallel_tree.extent(0);
        const size_t expected_size = sequential_tree.extent(0);
        if (total_size != expected_size) {
            oss << "Size missmatch: expected: " << expected_size << ", got: " << total_size
                << std::endl;
        }

        // check that all octants match
        for (size_t i = 0; i < total_size; ++i) {
            morton_code par_octant = parallel_tree[i];
            morton_code seq_octant = sequential_tree[i];
            if (par_octant != seq_octant) {
                oss << "octants dont match at index=" << i << " par=" << morton_helper.decode(par_octant)
                    << " | " << morton_helper.get_depth(par_octant) 
                    << " seq=" << morton_helper.decode(seq_octant)
                    << " | " << morton_helper.get_depth(seq_octant) << std::endl;

            }
        }
    }

    // we return a string instead of using gtest directly, this way we get clearer error messages
    // and i get less headaches writing this monstrosity
    return oss.str();
}

template <size_t Dim>
void runTest(double min_bounds, double max_bounds, size_t max_particles, size_t max_depth,
             size_t seed, size_t num_particles, const auto& particles) {
    // message to copy past and replicate the test run if something fails
    auto replicate_msg = generateReplicationCommand<Dim>(
        Comm->size(), seed, num_particles, max_depth, max_particles, min_bounds, max_bounds);

    // we need this to synchronise the ranks in case one fails, else everything deadlocks
    // 1 for success, 0 for failure
    int test_passed = 1;
    try {
        BoundingBox<Dim> root_bounds({min_bounds, min_bounds}, {max_bounds, max_bounds});
        OrthoTree<Dim> tree(max_depth, max_particles, root_bounds);

        std::string result = executeTestRun<Dim>(root_bounds, tree, particles);

        if (Comm->rank() == 0) {
            if (!result.empty()) {
                test_passed = 0;
                std::cerr << "Error: " << result << "\nReplicate with: " << replicate_msg
                          << std::endl;
            }
        }

    }
    // those catch blocks are not really necessary anymore but i will keep them in. its usefull
    // to debug when a test should fail
    catch (const std::exception& e) {
        test_passed = 0;
        if (Comm->rank() == 0) {
            std::cerr << "Standard exception occurred: " << e.what()
                      << "\nReplicate with: " << replicate_msg << std::endl;
        }
    } catch (const IpplException& e) {
        test_passed = 0;
        if (Comm->rank() == 0) {
            std::cerr << "Ippl exception occurred: " << e.what()
                      << "\nReplicate with: " << replicate_msg << std::endl;
        }
    } catch (...) {
        test_passed = 0;
        if (Comm->rank() == 0) {
            std::cerr << "Unknown exception occurred. "
                      << "Replicate with: " << replicate_msg << std::endl;
        }
    }

    if (test_passed == 0) {
        // this way all ranks abort if a test fails
        Comm->abort();
    }
}

/**
 * @brief We call this function from multiple test files, each running with a different number
 * of ranks and each one running all those tests.
 */
void runTests() {
    /**
     * As of now we have (3 * 5 * 4 * 4) = 240 * 3 (world sizes) = 720 tests
     */
    struct TestParameters {
        std::vector<double> min_bounds_v = {-1.0, 0.0};
        double max_bounds                = 1.0;

        std::vector<size_t> num_particles_v       = {500, 1000, 2000, 5000, 10000, 50000};
        std::vector<size_t> max_depth_v           = {3, 4, 6, 8};
        std::vector<size_t> max_particles_v = {20, 10, 5, 2, 1};
    } test_data;

    static constexpr size_t Dim2 = 2;
    static constexpr size_t Dim3 = 3;

    const size_t seed       = std::random_device{}();
    const double max_bounds = test_data.max_bounds;

    for (size_t num_particles : test_data.num_particles_v) {
        for (double min_bounds : test_data.min_bounds_v) {
            auto particles2D = generateParticles<Dim2>(num_particles, min_bounds, max_bounds, seed);
            auto particles3D = generateParticles<Dim3>(num_particles, min_bounds, max_bounds, seed);

            for (size_t max_depth : test_data.max_depth_v) {
                for (double max_particles_ratio : test_data.max_particles_v) {
                    const size_t max_particles = max_particles_ratio;

                    runTest<Dim2>(min_bounds, max_bounds, max_particles, max_depth, seed,
                                  num_particles, particles2D);

                    runTest<Dim3>(min_bounds, max_bounds, max_particles, max_depth, seed,
                                  num_particles, particles3D);
                }
            }
        }
    }
}

#undef COMMAND_TO_REPLICATE_RUN
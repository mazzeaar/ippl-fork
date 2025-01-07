#include <Particle/ParticleSpatialLayout.h>
#include <ostream>
#include <random>

#include "OrthoTree/OrthoTreeTypes.h"

#include "Communicate/Communicator.h"
#include "OrthoTree/helpers/AidList.h"
#include "gtest/gtest.h"

using namespace ippl;

template <class PLayout = ippl::ParticleSpatialLayout<double, 3>>
class TestParticle : public ippl::ParticleBase<PLayout> {
public:
    TestParticle() noexcept = default;
    TestParticle(PLayout& L)
        : ippl::ParticleBase<PLayout>(L) {}
};

template <size_t Dim>
auto getParticles(size_t num_particles_per_proc, const double min_bound, double max_bound) {
    typedef ippl::ParticleSpatialLayout<double, Dim> playout_type;
    typedef ippl::OrthoTreeParticle<playout_type> bunch_type;

    playout_type PLayout;
    bunch_type bunch(PLayout);

    std::mt19937_64 eng;
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

template <size_t Dim>
auto getUngatheredParticles(size_t num_particles_per_proc, const double min_bound,
                            double max_bound) {
    typedef ippl::ParticleSpatialLayout<double, Dim> playout_type;
    typedef ippl::OrthoTreeParticle<playout_type> bunch_type;

    playout_type PLayout;
    bunch_type bunch(PLayout);

    std::mt19937_64 eng;
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
    return bunch;
}

/**
 * @brief This distributes the particles to the 4 corners of our 2d domain
 */
template <typename Particles>
Particles distributeToCorners2D(Particles& particles, double min_bounds, double max_bounds) {
    static constexpr size_t Dim = 2;
    if (Comm->rank() == 0) {
        for (int i = 0; i < Comm->size(); ++i) {
            ippl::Vector<double, Dim> pos;
            switch (i) {
                case 0: {
                    pos = {min_bounds, min_bounds};
                } break;
                case 1: {
                    pos = {min_bounds, max_bounds};
                } break;
                case 2: {
                    pos = {max_bounds, min_bounds};
                } break;
                case 3: {
                    pos = {max_bounds, max_bounds};
                } break;
                default:
                    break;  // no need to handle, we assert world_size in another test.
            }
            const size_t n_particles_per_proc = particles.getTotalNum() / Comm->size();
            for (size_t j = 0; j < n_particles_per_proc; ++j) {
                size_t start_idx           = ((size_t)i * n_particles_per_proc);
                particles.R(start_idx + j) = pos;
            }
        }
    }

    return particles;
}

// ================================================================
//                          TESTS START
// ================================================================

// sanity check, tests are designed to run (and pass) with 4 ranks
TEST(AidListTest, AssertWorldSize) {
    ASSERT_EQ(Comm->size(), 4);
}

TEST(AidListTest, ChecksIfGatheredCorrectly) {
    // #### SETUP ####
    static constexpr size_t Dim = 3;
    const size_t max_depth      = 5;
    AidList<Dim> aid_list(max_depth);
    // #### SETUP DONE ####

    auto gathered_particles   = getParticles<Dim>(100, 0.0, 1.0);
    auto ungathered_particles = getUngatheredParticles<Dim>(100, 0.0, 1.0);

    if (Comm->rank() == 0) {
        EXPECT_FALSE(aid_list.is_gathered(ungathered_particles));
        EXPECT_TRUE(aid_list.is_gathered(gathered_particles));
    } else {
        EXPECT_FALSE(aid_list.is_gathered(ungathered_particles));
        EXPECT_FALSE(aid_list.is_gathered(gathered_particles));
    }
}

TEST(AidListTest, ConstructsSortedTest) {
    auto is_sorted = [](const auto& container) {
        return std::is_sorted(container.data(), container.data() + container.size());
    };

    // #### SETUP ####
    static constexpr size_t Dim       = 3;
    const size_t max_depth            = 5;
    const double min_bounds           = 0.0;
    const double max_bounds           = 1.0;
    const size_t n_particles_per_proc = 100;

    BoundingBox<Dim> root_bounds({min_bounds, min_bounds, min_bounds},
                                 {max_bounds, max_bounds, max_bounds});

    auto gathered_particles = getParticles<Dim>(n_particles_per_proc, min_bounds, max_bounds);

    AidList<Dim> working_aid_list(max_depth);
    working_aid_list.initialize(root_bounds, gathered_particles);

    // #### SETUP DONE ####

    EXPECT_TRUE(is_sorted(working_aid_list.getOctants()))
        << "AidList is not sorted on rank " << Comm->rank();
}

TEST(AidListTest, ConstructorTest) {
    auto is_sorted = [](const auto& container) {
        return std::is_sorted(container.data(), container.data() + container.size());
    };

    auto sort = [](auto& container) {
        std::sort(container.data(), container.data() + container.size());
    };

    auto get_span = [](auto& container) {
        return std::span(container.data(), container.size());
    };

    auto init_vec_from_view = [get_span](auto& source, auto& target) {
        auto source_span = get_span(source);
        std::copy(source_span.begin(), source_span.end(), target.begin());
    };

    // #### SETUP ####
    static constexpr size_t Dim       = 3;
    const size_t max_depth            = 5;
    const double min_bounds           = 0.0;
    const double max_bounds           = 1.0;
    const size_t n_particles_per_proc = 100;

    BoundingBox<Dim> root_bounds({min_bounds, min_bounds, min_bounds},
                                 {max_bounds, max_bounds, max_bounds});

    auto gathered_particles = getParticles<Dim>(n_particles_per_proc, min_bounds, max_bounds);

    AidList<Dim> working_aid_list(max_depth);
    // #### SETUP DONE ####

    working_aid_list.initialize(root_bounds, gathered_particles);

    std::vector<morton_code> all_octants;
    std::vector<size_t> all_particle_ids;

    // #### gather the AidList on rank 0 ####
    {
        int total_num_octants = 0;
        int local_size        = working_aid_list.size();
        Comm->allreduce(local_size, total_num_octants, 1, std::plus<>());

        // the sum of the number of octants on each rank should be equal to the total number of
        // octants
        ASSERT_EQ(total_num_octants, gathered_particles.getTotalNum());

        std::cout << "actualy Rank " << Comm->rank() << " has " << local_size << " octants"
                  << std::endl;

        std::vector<int> sizes(Comm->size(), -1);
        std::vector<int> displs(Comm->size(), -1);

        Comm->gather(&local_size, sizes.data(), 1);
        // print the sizes
        if (Comm->rank() == 0) {
            for (int i = 0; i < Comm->size(); ++i) {
                std::cout << "Rank " << i << " has " << sizes[i] << " octants" << std::endl;
            }
        }

        if (Comm->rank() == 0) {
            displs[0] = 0;
            for (int i = 1; i < Comm->size(); ++i) {
                displs[i] = displs[i - 1] + sizes[i - 1];
                std::cout << "Displacement " << i << ": " << displs[i] << std::endl;
            }
        }

        all_octants      = std::vector<morton_code>(total_num_octants, -1);
        all_particle_ids = std::vector<size_t>(total_num_octants, -1);

        Comm->gatherv(working_aid_list.getOctants().data(), all_octants.data(), local_size,
                      sizes.data(), displs.data(), 0);
        Comm->gatherv(working_aid_list.getParticleIDs().data(), all_particle_ids.data(), local_size,
                      sizes.data(), displs.data(), 0);
    }

    // #### validate the gathered AidList ####
    if (Comm->rank() == 0) {
        ASSERT_TRUE(is_sorted(all_octants)) << "AidList is not sorted on rank " << Comm->rank();

        std::vector<morton_code> all_octants_naive;
        std::vector<size_t> all_particle_ids_naive;

        // initialise the naive implementation
        {
            AidList<Dim> aid_list(max_depth);
            aid_list.initialize_from_rank(max_depth, root_bounds, gathered_particles);
            aid_list.sort_local_aidlist();

            all_octants_naive.resize(aid_list.size());
            all_particle_ids_naive.resize(aid_list.size());

            init_vec_from_view(aid_list.getOctants(), all_octants_naive);
            init_vec_from_view(aid_list.getParticleIDs(), all_particle_ids_naive);

            EXPECT_EQ(all_octants_naive, all_octants);
        }

        {  // whats the idea behind this? xD

            // same code below, just rewritten
            size_t start_index = 0;
            for (size_t i = 1; i <= all_octants.size(); ++i) {
                // check if the current oct group has ended (new octant) or if we reached the end of
                // the list
                if (i == all_octants.size() || all_octants[i] != all_octants[start_index]) {
                    auto sort_subrange = [&](auto& container) {
                        auto cur_subrange = std::ranges::subrange(container.begin() + start_index,
                                                                  container.begin() + i);
                        std::ranges::sort(cur_subrange);
                    };

                    sort_subrange(all_particle_ids);
                    sort_subrange(all_particle_ids_naive);

                    start_index = i;
                }
            }

            EXPECT_EQ(all_particle_ids_naive, all_particle_ids);

            /*
                // check if the particle ids are equal for different octants
                auto octant_o  = all_octants[0];
                size_t start_i = 0;
                size_t count   = 0;

                for (size_t i = 1; i <= all_octants.size(); ++i) {
                    if (i < all_octants.size() && all_octants[i] == octant_o) {
                        count++;
                    } else {
                        // check if the particle ids are sorted
                        count++;
                        std::sort(all_particle_ids.begin() + start_i,
                                all_particle_ids.begin() + start_i + count);
                        std::sort(all_particle_ids_naive.begin() + start_i,
                                all_particle_ids_naive.begin() + start_i + count);
                        start_i = i;
                        count   = 0;
                        if (i < all_octants.size())
                            octant_o = all_octants[i];
                    }
                }

                EXPECT_EQ(all_particle_ids_naive, all_particle_ids);
            */
        }

        // #### all ids should appear equally many times ####
        {
            std::unordered_map<size_t, int> pid_map;

            for (size_t i = 0; i < all_particle_ids.size(); ++i) {
                ++pid_map[all_particle_ids[i]];
                --pid_map[all_particle_ids_naive[i]];
            }

            // all counts should be zero
            EXPECT_TRUE(std::ranges::all_of(pid_map, [](const auto& elem) {
                return elem.second == 0;
            })) << "Wrong!";
        }
    }
}

TEST(AidListTest, CorrectConstructionTest2D) {
    std::cout << "CorrectConstructionTest2d started on rank " << Comm->rank() << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Barrier passed on rank " << Comm->rank() << std::endl;
    // #### SETUP ####
    static constexpr size_t Dim       = 2;
    const size_t max_depth            = 1;
    const double min_bounds           = 0.0;
    const double max_bounds           = 1.0;
    const size_t n_particles_per_proc = 100;

    auto particles = getParticles<Dim>(n_particles_per_proc, min_bounds, max_bounds);
    particles      = distributeToCorners2D(particles, min_bounds, max_bounds);

    AidList<Dim> aid_list(max_depth);
    BoundingBox<Dim> root_bounds({min_bounds, min_bounds}, {max_bounds, max_bounds});

    aid_list.initialize(root_bounds, particles);

    // Gather the octants and particle ids

    // #### SETUP DONE ####

    EXPECT_TRUE(std::is_sorted(aid_list.getOctants().data(),
                               aid_list.getOctants().data() + aid_list.size()))
        << "AidList is not sorted on rank " << Comm->rank();
    std::cout << "CorrectConstructionTest2d complteded on rank " << Comm->rank() << std::endl;
}

TEST(AidListTest, NumParticlesInOctantTest) {
    // #### SETUP ####
    static constexpr size_t Dim       = 2;
    const size_t max_depth            = 1;
    const double min_bounds           = 0.0;
    const double max_bounds           = 1.0;
    const size_t n_particles_per_proc = 100;
    const size_t total_num_particles  = n_particles_per_proc * Comm->size();

    auto particles = getParticles<Dim>(n_particles_per_proc, min_bounds, max_bounds);
    particles      = distributeToCorners2D(particles, min_bounds, max_bounds);

    AidList<Dim> aid_list(max_depth);
    BoundingBox<Dim> root_bounds({min_bounds, min_bounds}, {max_bounds, max_bounds});
    Morton<Dim> morton_helper(max_depth);

    aid_list.initialize(root_bounds, particles);
    // #### SETUP DONE ####

    // #### TEST HELPERS ####
    struct TestStruct {
        morton_code octant;
        size_t expected_lower_bound;
        size_t expected_upper_bound_excl;
        size_t expected_upper_bound_incl;
        size_t expected_total_particles;
    };

    auto run_test = [&](TestStruct& test) {
        EXPECT_EQ(aid_list.getLowerBoundIndex(test.octant), test.expected_lower_bound)
            << "Failed lower bound test for octant: " << test.octant;

        EXPECT_EQ(aid_list.getUpperBoundIndexExclusive(test.octant), test.expected_upper_bound_excl)
            << "Failed exclusive upper bound test for octant: " << test.octant;

        EXPECT_EQ(aid_list.getUpperBoundIndexInclusive(test.octant), test.expected_upper_bound_incl)
            << "Failed inclusive upper bound test for octant: " << test.octant;

        EXPECT_EQ(aid_list.getNumParticlesInOctant(test.octant), test.expected_total_particles)
            << "Failed particle count test for octant: " << test.octant;
    };

    std::vector<TestStruct> tests_to_run = {// root node
                                            {.octant                    = 0b000,
                                             .expected_lower_bound      = 0,
                                             .expected_upper_bound_excl = 0,
                                             .expected_upper_bound_incl = 0,
                                             .expected_total_particles  = total_num_particles},

                                            // top-left octant
                                            {.octant                    = 0b001,
                                             .expected_lower_bound      = 0,
                                             .expected_upper_bound_excl = 100,
                                             .expected_upper_bound_incl = 99,
                                             .expected_total_particles  = n_particles_per_proc},

                                            // top-middle octant
                                            {.octant                    = 0b011,
                                             .expected_lower_bound      = 100,
                                             .expected_upper_bound_excl = 200,
                                             .expected_upper_bound_incl = 199,
                                             .expected_total_particles  = n_particles_per_proc},

                                            // left-middle octant
                                            {.octant                    = 0b101,
                                             .expected_lower_bound      = 200,
                                             .expected_upper_bound_excl = 300,
                                             .expected_upper_bound_incl = 299,
                                             .expected_total_particles  = n_particles_per_proc},

                                            // middle-middle octant
                                            {.octant                    = 0b111,
                                             .expected_lower_bound      = 300,
                                             .expected_upper_bound_excl = 400,
                                             .expected_upper_bound_incl = 399,
                                             .expected_total_particles  = n_particles_per_proc}};

    // #### TEST HELPERS DONE ####
    /*
        if (Comm->rank() == 0) {
            // we should only have 4 different octants

            ASSERT_EQ(aid_list.size(), total_num_particles);

            for (auto test_data : tests_to_run) {
                run_test(test_data);
            }
        }
    */
    // barier
    MPI_Barrier(MPI_COMM_WORLD);

    Kokkos::View<morton_code*> octants("octants", 1);
    octants(0) = tests_to_run[Comm->rank() + 1].octant;
    aid_list.getNumParticlesInOctantsParallel(octants);
}

/*
TEST(AidListTest, GetReqOctantsTest) {
    // #### SETUP ####
    static constexpr size_t Dim       = 2;
    const size_t max_depth            = 1;
    const double min_bounds           = 0.0;
    const double max_bounds           = 1.0;
    const size_t n_particles_per_proc = 100;

    auto particles = getParticles<Dim>(n_particles_per_proc, min_bounds, max_bounds);
    particles      = distributeToCorners2D(particles, min_bounds, max_bounds);

    AidList<Dim> aid_list(max_depth);
    BoundingBox<Dim> root_bounds({min_bounds, min_bounds}, {max_bounds, max_bounds});

    aid_list.initialize(root_bounds, particles);
    // #### SETUP DONE ####

    // sanity check
    if (Comm->rank() == 0) {
        std::map<morton_code, size_t> octant_counter;

        for (size_t i = 0; i < (size_t)Comm->size() * n_particles_per_proc; ++i) {
            octant_counter[aid_list.getOctant(i)]++;
        }

        // we should only have 4 different octants
        ASSERT_EQ(octant_counter.size(), 4);
    }

    auto [min_octant, max_octant] = aid_list.getMinReqOctants();

    ASSERT_EQ(min_octant, max_octant)
        << "Rank: " << Comm->rank() << " got min: " << min_octant << ", max: " << max_octant;

    morton_code base_octant = 1;
    morton_code step_size   = 2;

    morton_code expected_octant = base_octant + (Comm->rank() * step_size);
    EXPECT_EQ(min_octant, expected_octant)
        << "Rank " << Comm->rank() << " expected: " << expected_octant
        << ", but got: " << min_octant;
}

/**
 * @brief If this test fails, but only once then it was random chance, everything is ok.
 * You could just turn up the tolerance though:)
 */
/*
TEST(AidListTest, InitialiseForRank) {
    // #### SETUP ####
    static constexpr size_t Dim       = 2;
    const size_t max_depth            = 4;
    const double min_bounds           = 0.0;
    const double max_bounds           = 1.0;
    const size_t n_particles_per_proc = 100;
    const size_t total_particles      = Comm->size() * n_particles_per_proc;

    const size_t tolerance = 15;

    auto particles = getParticles<Dim>(n_particles_per_proc, min_bounds, max_bounds);

    AidList<Dim> aid_list(max_depth);
    BoundingBox<Dim> root_bounds({min_bounds, min_bounds}, {max_bounds, max_bounds});

    aid_list.initialize(root_bounds, particles);
    // #### SETUP DONE ####

    auto [min_octant, max_octant] = aid_list.getMinReqOctants();

    aid_list.innitFromOctants(min_octant, max_octant);

    if (Comm->rank() == 0) {
        EXPECT_NEAR(aid_list.size(), total_particles, tolerance)
            << "Rank 0 has wrong aid_list_size! Expected roughly: " << total_particles
            << ", got: " << aid_list.size();
        EXPECT_TRUE(std::is_sorted(aid_list.getOctants().data(),
                                   aid_list.getOctants().data() + aid_list.size()))
            << "Octants are not sorted on rank " << Comm->rank();
    } else {
        EXPECT_NEAR(aid_list.size(), n_particles_per_proc, tolerance)
            << "Rank " << Comm->rank()
            << " has wrong aid_list_size! Expected roughly: " << n_particles_per_proc
            << ", got: " << aid_list.size();
    }

    EXPECT_TRUE(std::is_sorted(aid_list.getOctants().data(),
                               aid_list.getOctants().data() + aid_list.size()))
        << "Octants are not sorted on rank " << Comm->rank();
}
*/

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

#include <string>
#include <utility>

#include "../OrthoTree.h"

/*
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {

    /**
     * @brief get weights for all octants in G equal to the total number of descendants in
     * the global F view across all processors
     */
    template <size_t Dim>
    std::pair<Kokkos::View<size_t*>, Kokkos::View<morton_code*>> getWeights(
        Morton<Dim> morton_helper, Kokkos::View<morton_code*> G,
        Kokkos::View<morton_code*> F_view) {
        Kokkos::View<size_t*> weights("weights", G.size());
        Kokkos::deep_copy(weights, int(0));

        const size_t world_size = Comm->size();
        const size_t world_rank = Comm->rank();

        Kokkos::View<morton_code*> borders("borders", world_size - 1);

        // communicate sizes
        Kokkos::View<size_t*> window_sizes("window_sizes", world_size);
        size_t F_view_size = F_view.size();
        Comm->allgather(&F_view_size, window_sizes.data(), 1);

        mpi::rma::Window<mpi::rma::Active> Glob_view;
        {  // initialise window
            auto F_span = std::span(F_view.data(), F_view.data() + F_view.size());
            Glob_view.create(*Comm, F_span.begin(), F_span.end());
            Glob_view.fence(0);
        }

        const size_t data_to_send_base_size = 100;
        Kokkos::View<morton_code*> data_to_send("octants_to_send", data_to_send_base_size);

        // scan the window of each rank
        for (size_t source_rank = 0; source_rank < world_size; ++source_rank) {
            if (window_sizes(source_rank) == 0)
                continue;
            // load data
            Kokkos::View<morton_code*> source_data(
                "source_data", std::max(window_sizes(source_rank), F_view.size()));
            // TODO make this better
            for (size_t i = 0; i < F_view.size(); i++) {
                source_data(i) = F_view(i);
            }
            if (world_rank != source_rank) {
                auto source_span =
                    std::span(source_data.data(), source_data.data() + source_data.size());
                Glob_view.fence(0);
                Glob_view.get(source_span.begin(), source_span.begin() + window_sizes(source_rank),
                              source_rank, 0);
                Glob_view.fence(0);
                Kokkos::resize(source_data, window_sizes(source_rank));
            }

            if (source_rank != world_size - 1)
                borders(source_rank) = source_data(source_data.size() - 1) + 1;

            for (size_t i = 0; i < G.size(); i++) {
                const auto lower_bound_it = std::lower_bound(
                    source_data.data(), source_data.data() + source_data.extent(0), G(i),
                    [](const morton_code& octants_entry, const morton_code& target) {
                        return octants_entry < target;
                    });

                auto lower_bound_idx = static_cast<size_t>(lower_bound_it - source_data.data());

                const auto upper_bound_it = std::upper_bound(
                    source_data.data(), source_data.data() + source_data.extent(0),
                    morton_helper.get_deepest_last_descendant(G(i)),
                    [](const morton_code& target, const morton_code& octants_entry) {
                        return target < octants_entry;
                    });

                auto upper_bound_idx = static_cast<size_t>(upper_bound_it - source_data.data());
                weights(i) += upper_bound_idx - lower_bound_idx;
            }
        }

        return std::make_pair(weights, borders);
    }

    template <size_t Dim>
    Kokkos::View<morton_code*> updateFview(Morton<Dim> morton_helper,
                                           Kokkos::View<morton_code*> F_view,
                                           Kokkos::View<morton_code*> bucket_borders,
                                           Kokkos::View<morton_code*> octants) {
        size_t world_size = Comm->size();
        size_t world_rank = Comm->rank();
        // holds min/max octant from each rank
        Kokkos::View<size_t*> ranges("ranges", 2 * world_size);
        // min/max indices of data we will send to other ranks
        Kokkos::View<size_t*> send_indices("send_indices", 2 * world_size);
        // min/max indices of octants we receive from each rank
        Kokkos::View<size_t*> recv_indices("recv_indices", 2 * world_size);

        // number of octants this rank will receive
        size_t new_size_after_exchange = 0;
        morton_code min_octant         = octants(0);
        morton_code max_octant =
            morton_helper.get_deepest_last_descendant(octants(octants.size() - 1));

        /**
         * Populate the ranges view with the min/max octant for each rank.
         */
        {
            auto ranges_span = std::span(ranges.data(), ranges.size());
            Kokkos::deep_copy(ranges, 0);

            mpi::rma::Window<mpi::rma::Active> range_window;
            range_window.create(*Comm, ranges_span.begin(), ranges_span.end());
            range_window.fence(0);

            const morton_code dld_root     = morton_helper.get_deepest_last_descendant(0);
            morton_code lower_bound_octant = 0;
            morton_code upper_bound_octant = 0;
            for (size_t i = 0; i < world_size; ++i) {
                upper_bound_octant = dld_root + 1;
                if (i < world_size - 1) {
                    upper_bound_octant = bucket_borders(i);
                }

                if (i > 0) {
                    lower_bound_octant = bucket_borders(i - 1);
                }

                // skip processor if no interesting octants are there
                if (upper_bound_octant < min_octant || lower_bound_octant >= max_octant) {
                    continue;
                }

                morton_code lower_range = std::max(min_octant, lower_bound_octant);
                morton_code upper_range = std::min(max_octant, upper_bound_octant);

                if (i == world_size - 1)
                    upper_range += 1;

                // no need to send to ourselves
                if (i == world_rank) {
                    ranges(2 * i)     = lower_range;
                    ranges(2 * i + 1) = upper_range;
                    continue;
                }

                // find the range of octants that are in the current bucket
                range_window.put(lower_range, i, 2 * world_rank);
                range_window.put(upper_range, i, 2 * world_rank + 1);
            }

            range_window.fence(0);
        }

        /**
         * Calculate the amount of octants we send and receive to/from each rank.
         * - populate: send_indices
         * - populate: recv_indices
         * - calcualte: new_size_after_exchange = total number of octants/P_ids this rank will
         * receive
         */
        {
            auto recv_indices_span = std::span(recv_indices.data(), recv_indices.size());
            Kokkos::deep_copy(recv_indices, 0);

            mpi::rma::Window<mpi::rma::Active> idx_window;
            idx_window.create(*Comm, recv_indices_span.begin(), recv_indices_span.end());
            idx_window.fence(0);

            for (unsigned rank = 0; rank < world_size; ++rank) {
                /*
                 * Skip ranks where min_octant == max_octant
                 * this works since ranks whose octants are for example way
                 * bigger than the ones here will have set min_octant = max_octant = 0
                 */
                if (ranges(2 * rank) == ranges(2 * rank + 1)) {
                    continue;
                }
                auto lower_bound_it = std::lower_bound(
                    F_view.data(), F_view.data() + F_view.extent(0), ranges(2 * rank),
                    [](const morton_code& octants_entry, const morton_code& target) {
                        return octants_entry < target;
                    });

                send_indices(2 * rank) = static_cast<size_t>(lower_bound_it - F_view.data());
                lower_bound_it         = std::lower_bound(
                    F_view.data(), F_view.data() + F_view.extent(0), ranges(2 * rank + 1),
                    [](const morton_code& octants_entry, const morton_code& target) {
                        return octants_entry < target;
                    });
                send_indices(2 * rank + 1) = static_cast<size_t>(lower_bound_it - F_view.data());

                // no need to communicate with ourselves
                if (rank == world_rank) {
                    recv_indices(2 * rank)     = send_indices(2 * rank);
                    recv_indices(2 * rank + 1) = send_indices(2 * rank + 1);
                    continue;
                }

                auto indices_a = send_indices(2 * rank);
                auto indices_b = send_indices(2 * rank + 1);
                idx_window.put(indices_a, rank, 2 * world_rank);
                idx_window.put(indices_b, rank, 2 * world_rank + 1);
            }

            idx_window.fence(0);

            Kokkos::parallel_reduce(
                "newFview::compute new size", world_size,
                KOKKOS_LAMBDA(const size_t i, size_t& local_new_size) {
                    local_new_size += recv_indices(2 * i + 1) - recv_indices(2 * i);
                },
                new_size_after_exchange);
        }

        /**
         * Exchange the octants between ranks.
         */
        {
            Kokkos::View<morton_code*> new_octants("newFview::new_octants",
                                                   new_size_after_exchange);

            auto new_octants_span = std::span(new_octants.data(), new_octants.size());

            auto new_octants_start_it = new_octants_span.begin();

            auto octants_span = std::span(F_view.data(), F_view.size());

            mpi::rma::Window<mpi::rma::Active> octants_window;

            octants_window.create(*Comm, octants_span.begin(), octants_span.end());

            octants_window.fence(0);

            size_t last_insert_idx = 0;
            for (unsigned rank = 0; rank < world_size; ++rank) {
                if (recv_indices(2 * rank) == recv_indices(2 * rank + 1)) {
                    continue;
                }

                size_t recv_size = recv_indices(2 * rank + 1) - recv_indices(2 * rank);
                assert(recv_size > 0);

                // get the iterators inbetween which the new octants from this
                // rank will be inserted
                auto start_it_octants = new_octants_start_it + last_insert_idx;
                auto end_it_octants   = start_it_octants + recv_size;

                static_assert(std::contiguous_iterator<decltype(start_it_octants)>,
                              "Iterator does not satisfy contiguous_iterator");

                // we don't need to communicate with ourselves
                // This piece of spaghetti code is the Kokkos compatible way
                // I found to copy subarrays from one view to another
                // improvements are welcome :)
                if (rank == world_rank) {
                    last_insert_idx += recv_size;
                    auto source_index_pair =
                        std::make_pair(recv_indices(2 * rank), recv_indices(2 * rank + 1));
                    auto source_subview = Kokkos::subview(F_view, source_index_pair);

                    auto dest_index_pair =
                        std::make_pair(last_insert_idx - recv_size, last_insert_idx);
                    auto dest_subview = Kokkos::subview(new_octants, dest_index_pair);

                    Kokkos::deep_copy(dest_subview, source_subview);

                    continue;
                }

                last_insert_idx += recv_size;
                octants_window.get(start_it_octants, end_it_octants, rank, recv_indices(2 * rank));
            }

            octants_window.fence(0);

            F_view = new_octants;
        }
        return F_view;
    }

    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::block_partition(morton_code min_octant,
                                                               morton_code max_octant) {
        // logger.on(true);
        // logger.setOutputLevel(1);
        // logger << level1 << "starting algo4" << endl;

        IpplTimings::TimerRef blockPartitionTimer = IpplTimings::getTimer("block_partition");
        IpplTimings::startTimer(blockPartitionTimer);

        auto local_morton_helper     = morton_helper;
        Kokkos::View<morton_code*> T = complete_region(min_octant, max_octant);

        // find the lowest level (smallest depth)
        size_t lowest_level = max_depth_m;
        Kokkos::parallel_reduce(
            "algo4::FindLowestLevel", T.size(),
            KOKKOS_LAMBDA(const size_t i, size_t& min_depth) {
                size_t depth = local_morton_helper.get_depth(T(i));
                if (depth < min_depth) {
                    min_depth = depth;
                }
            },
            Kokkos::Min<size_t>(lowest_level));

        // count the number of elements at the lowest level
        size_t C_size;
        Kokkos::parallel_reduce(
            "algo4::CountAtLowestLevel", T.size(),
            KOKKOS_LAMBDA(const size_t i, size_t& count) {
                if (local_morton_helper.get_depth(T(i)) == lowest_level) {
                    count++;
                }
            },
            C_size);

        Kokkos::View<morton_code*> C("algo4::C_view", C_size);

        // populate C_view
        Kokkos::parallel_scan(
            "algo4::PopulateC", T.size(), KOKKOS_LAMBDA(const size_t i, size_t& index, bool final) {
                if (local_morton_helper.get_depth(T(i)) == lowest_level) {
                    if (final) {
                        C(index) = T(i);
                    }
                    index++;
                }
            });

        if (C_size == 0) {
            Kokkos::resize(C, 2);
            C(0) = min_octant;
            C(1) = max_octant;
        }

        if (aid_list_m.size() == 0) {
            throw std::runtime_error("No particles on rank algo4");
        }
        Kokkos::View<morton_code*> G = complete_tree(C);

        Kokkos::View<size_t*> weights      = this->aid_list_m.getNumParticlesInOctantsParallel(G);
        Kokkos::View<morton_code*> octants = partition(G, weights);

        morton_code min_step   = morton_helper.get_step_size(max_depth_m);
        morton_code max_parent = *(octants.data() + octants.size() - 1);

        morton_code new_min_octant = morton_helper.get_deepest_first_descendant(octants[0]);
        morton_code new_max_octant =
            morton_helper.get_deepest_last_descendant(max_parent) + min_step;

        IpplTimings::TimerRef innitfromoctants = IpplTimings::getTimer("innitfromoctants");
        IpplTimings::startTimer(innitfromoctants);

        this->aid_list_m.innitFromOctants(new_min_octant, new_max_octant);
        n_particles = this->aid_list_m.size();
        IpplTimings::stopTimer(innitfromoctants);

        IpplTimings::stopTimer(blockPartitionTimer);
        return octants;
    }

    template <size_t Dim>
    std::pair<Kokkos::View<morton_code*>, Kokkos::View<morton_code*>> OrthoTree<Dim>::algo4_11(
        Kokkos::View<morton_code*> F_view) {
        assert(F_view.size() > 0 && "Size missmatch");
        const auto local_morton_helper = morton_helper;

        const morton_code min_oct    = F_view(0);
        const morton_code max_oct    = F_view(F_view.size() - 1);
        Kokkos::View<morton_code*> T = complete_region(min_oct, max_oct);

        // the lowest level is actually the 'highest' (closest to root) node in our tree
        size_t lowest_level;
        Kokkos::parallel_reduce(
            T.size(),
            KOKKOS_LAMBDA(const size_t i, size_t& min_depth) {
                size_t depth = local_morton_helper.get_depth(T(i));
                if (depth < min_depth) {
                    min_depth = depth;
                }
            },
            Kokkos::Min<size_t>(lowest_level));

        // count the number of elements at the lowest level
        size_t C_size;
        Kokkos::parallel_reduce(
            T.size(),
            KOKKOS_LAMBDA(const size_t i, size_t& count) {
                if (local_morton_helper.get_depth(T(i)) == lowest_level) {
                    count++;
                }
            },
            C_size);

        Kokkos::View<morton_code*> C("C_view", C_size);

        // populate C_view
        Kokkos::parallel_scan(
            T.size(), KOKKOS_LAMBDA(const size_t i, size_t& index, bool final) {
                if (local_morton_helper.get_depth(T(i)) == lowest_level) {
                    if (final) {
                        C(index) = T(i);
                    }
                    index++;
                }
            });

        if (C_size == 0) {
            Kokkos::resize(C, 2);
            C(0) = min_oct;
            C(1) = max_oct;
        }

        Kokkos::View<morton_code*> G = complete_tree(C);

        auto [weights_view, borders] = getWeights(this->morton_helper, G, F_view);

        Kokkos::View<morton_code*> octants = partition(G, weights_view);

        Kokkos::View<morton_code*> new_F_view =
            updateFview(this->morton_helper, F_view, borders, octants);

        return std::make_pair(octants, new_F_view);
    }
}  // namespace ippl

#include <Kokkos_Core.hpp>

#include "../OrthoTree.h"
namespace ippl {
    /*
    TODO:
    - IMPLEMENT THIS FUNCTION
    - WRITE TESTS FOR IT
    - THINK HARD IF THE GIVEN SIGNATURE MAKES SENSE
    - UNCOMMENT THE INCLUSION OF THIS FILE IN ORTHO_TREE.HPP (BOTTOM)
    */

    // ================================
    // ================================
    // DEVELOPMENT HELPERS
    // TODO: delete those
    // ================================
    // ================================

    /**
     * @warning This function is just for development purposes, this way we know if stuff is sorted
     * or not. this has to be removed somehow before merging this branch
     */
    void sort_if_necessary(Kokkos::View<morton_code*> check_view, std::string_view view_name,
                           std::string_view func_name) {
        const std::string_view COLOR_RED   = "\e[0;31m";
        const std::string_view COLOR_RESET = "\e[0;31m";

        if (!std::is_sorted(check_view.data(), check_view.data() + check_view.size())) {
            std::cerr << COLOR_RED << "{view: " << view_name << "}" << " IS NOT SORTED IN "
                      << "{func: " << func_name << "}"
                      << " THE PROBLEM IS PROBABLY THE PARALLEL FOR TO PUPULATE THE VIEW!"
                      << COLOR_RESET << std::endl;
            std::sort(check_view.data(), check_view.data() + check_view.size());
        }
    }

    /**
     * @brief As of now morton_helper.get_neighbours returns a std::vector (aliased as vector_t),
     * this is no bueno. This helper enables us to write algo11 as if neighbours is already
     * implemented to return views
     */
    Kokkos::View<morton_code*> get_neighbour_view(const auto& morton_helper,
                                                  const morton_code octant) {
        /**
         * this 'vector_t' is on purpose, we will get
         * compilation errors when we change the
         * return type to view
         *
         * just delete this whole function and replace its call with a direct call to
         * 'get_neighbors'
         */
        vector_t<morton_code> neighbour_vec =
            morton_helper.get_neighbors(octant, morton_helper.get_depth(octant));

        Kokkos::View<morton_code*> neighbour_view("neighbour_view", neighbour_vec.size());
        Kokkos::parallel_for(
            "CopyStdVectorToKokkosView", neighbour_vec.size(),
            KOKKOS_LAMBDA(const size_t i) { neighbour_view(i) = neighbour_vec[i]; });

        return neighbour_view;
    }

    // ================================
    // ================================
    // VIEW HELPERS
    // just to reduce code duplication
    // ================================
    // ================================

    /**
     * @brief Takes an arbitrary amount of views and concatenates them into one view. The views are
     * concatenated in the same order they are passed into the function
     */
    template <typename... ViewTypes>
    Kokkos::View<morton_code*> concatenateViews(ViewTypes... views) {
        size_t total_size = (views.extent(0) + ...);

        Kokkos::View<morton_code*> result("concatenated_view", total_size);

        size_t offset  = 0;
        auto copy_view = [&](auto& view) {
            Kokkos::parallel_for(
                "CopyViewData", view.extent(0),
                KOKKOS_LAMBDA(const size_t i) { result(i + offset) = view(i); });
            offset += view.extent(0);
        };

        (copy_view(views), ...);
        return result;
    }

    /**
     * @brief Counts the number of octants in the given view that fullfill the predicate
     */
    template <typename Predicate>
    size_t count_octants(Kokkos::View<morton_code*> count_view, Predicate predicate) {
        size_t count = 0;

        Kokkos::parallel_reduce(
            "CountValidOctants", count_view.extent(0),
            KOKKOS_LAMBDA(const size_t i, size_t& local_count) {
                // no branching
                local_count += (predicate(count_view(i)) == true);
            },
            count);

        return count;
    }

    /**
     * @brief Filters octants based on the given predicate
     * @warning out_view will NOT be sorted!
     *
     * by @mathieu TODO: Why not just initialize the G_View with a fairly high number of memory, add
     * more if necessary and fit to size in the end by counting when you insert an element that way
     * you dont have to find out which octants to insert twice TODO
     *
     * answer by @aaron:
     * 1.   i have now extracted the filtering steps into one function, this way we can
     *      just apply this optimisation in here.
     * 2.   Question: we should measure if (allocation + resizing) is faster than two filter passes
     *      (the filter is fully Kokkos::parallel'ed)
     */
    template <typename Predicate>
    Kokkos::View<morton_code*> filter_octants(Kokkos::View<morton_code*> in_view,
                                              Predicate predicate) {
        const size_t n_pass = count_octants(in_view, predicate);

        Kokkos::View<morton_code*> out_view("filtered_octants", n_pass);
        Kokkos::View<size_t> index("index");
        Kokkos::deep_copy(index, size_t(0));

        Kokkos::parallel_for(
            "filter_octants", in_view.extent(0), KOKKOS_LAMBDA(const size_t i) {
                const morton_code val = in_view(i);
                if (predicate(val)) {
                    const size_t my_index = Kokkos::atomic_fetch_add(&index(), size_t(1));
                    out_view(my_index)    = val;
                }
            });

        return out_view;
    }

    /**
     * @brief Returns true if any of the octants in the count_view fullfill the predicate
     */
    template <typename Predicate>
    bool any_of(Kokkos::View<morton_code*> count_view, Predicate predicate) {
        return (count_octants(count_view, predicate) != 0);
    }

    // ================================
    // ================================
    // ALGO11 HELPERS
    // just to reduce code duplication and make the function algo11 more readable
    // ================================
    // ================================

    Kokkos::View<morton_code*> initialise_D_view(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> C_view) {
        auto should_insert = KOKKOS_LAMBDA(morton_code check_octant) {
            Kokkos::View<morton_code*> neighbour_view =
                get_neighbour_view(morton_helper, check_octant);

            return any_of(
                neighbour_view, KOKKOS_LAMBDA(const morton_code neighbour_octant) {
                    return any_of(
                        B_view, KOKKOS_LAMBDA(const morton_code octant_B) {
                            const bool overlaps_neighbour =
                                morton_helper.does_overlap(neighbour_octant, octant_B);

                            const bool overlaps_check_octant =
                                morton_helper.is_ancestor(check_octant, octant_B);

                            return overlaps_neighbour && !overlaps_check_octant;
                        });
                });
        };

        // D = intra-proc boundaries
        Kokkos::View<morton_code*> D_view = filter_octants(C_view, should_insert);
        sort_if_necessary(D_view, "D_view", __func__);
        return D_view;
    }

    Kokkos::View<morton_code*> initialise_G_view(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> F_view) {
        auto should_insert = KOKKOS_LAMBDA(morton_code octant_to_check) {
            Kokkos::View<morton_code*> neighbour_view =
                get_neighbour_view(morton_helper, octant_to_check);

            return any_of(
                neighbour_view, KOKKOS_LAMBDA(const morton_code octant_Z) {
                    return any_of(
                        B_view, KOKKOS_LAMBDA(const morton_code octant_B) {
                            const bool overlaps_neighbour =
                                morton_helper.does_overlap(neighbour_octant, octant_B);

                            return !overlaps_neighbour;
                        });
                });
        };

        // D = inter-proc boundaries
        Kokkos::View<morton_code*> G_view = filter_octants(F_view, should_insert);
        sort_if_necessary(G_view, "G_view", __func__);
        return G_view;
    }

    Kokkos::View<morton_code*> initialise_R_view(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> H_view,
                                                 Kokkos::View<morton_code*> F_view) {
        auto should_insert = KOKKOS_LAMBDA(morton_code octant_to_insert) {
            Kokkos::View<morton_code*> neighbour_view =
                get_neighbour_view(morton_helper, octant_to_insert);

            return any_of(
                neighbour_view, KOKKOS_LAMBDA(const morton_code octant_Z) {
                    return any_of(
                        B_view, KOKKOS_LAMBDA(const morton_code octant_B) {
                            return morton_helper.does_overlap(octant_to_insert, octant_B);
                        });
                });
        };

        Kokkos::View<morton_code*> H_filtered = filter_octants(H_view, should_insert);
        Kokkos::View<morton_code*> F_filtered = filter_octants(F_view, should_insert);

        Kokkos::View<morton_code*> R_view = concatenateViews(H_filtered, F_filtered);
        sort_if_necessary(R_view, "R_view", __func__);
        return R_view;
    }

    Kokkos::View<morton_code* [2]> exchange_min_max_octants(Kokkos::View<morton_code*> B_view) {
        const morton_code local_min = B_view(0);
        const morton_code local_max = B_view(B_view.extent(0) - 1);

        morton_code local_data[2] = {local_min, local_max};

        std::vector<morton_code> gathered_data_buff(Comm->size() * 2);

        Comm->allgather(local_data, gathered_data_buff.data(), 2);

        Kokkos::View<morton_code* [2]> min_max_view("min_max_view", Comm->size());
        Kokkos::parallel_for(
            "FillMinMaxView", Comm->size(), KOKKOS_LAMBDA(const size_t rank) {
                min_max_view(rank, 0) = gathered_data_buff[rank * 2];      // min oct
                min_max_view(rank, 1) = gathered_data_buff[rank * 2 + 1];  // max oct
            });

        return min_max_view;
    }

    std::pair<Kokkos::View<morton_code*>, Kokkos::View<size_t*>> exchange_B_glob_with_offsets(
        Kokkos::View<morton_code*> B_view) {
        // chatgpt function, im too cooked for this

        const size_t world_size = Comm->size();
        const size_t world_rank = Comm->rank();

        // Step 1: Gather sizes of B_view across ranks
        size_t local_size = B_view.extent(0);
        Kokkos::View<size_t*> sizes("sizes", world_size);

        // Gather sizes from all ranks
        Comm->allgather(&local_size, sizes.data(), 1);

        // Step 2: Compute rank_offsets using prefix sum
        Kokkos::View<size_t*> rank_offsets("rank_offsets", world_size + 1);
        Kokkos::parallel_scan(
            "ComputeOffsets", world_size,
            KOKKOS_LAMBDA(const size_t i, size_t& partial_sum, const bool final) {
                if (final)
                    rank_offsets(i) = partial_sum;
                partial_sum += sizes(i);
                if (final && i == world_size - 1)
                    rank_offsets(world_size) = partial_sum;
            });

        size_t total_size = rank_offsets(world_size - 1);

        // Step 3: Allocate B_glob
        Kokkos::View<morton_code*> B_glob("B_glob", total_size);

        // Step 4: Copy local data to host space
        Kokkos::View<morton_code*, Kokkos::HostSpace> local_data("local_data", local_size);
        Kokkos::deep_copy(local_data, B_view);

        // Prepare recvcounts and displacements for allgatherv
        std::vector<int> recvcounts(world_size);
        std::vector<int> displacements(world_size);
        for (size_t i = 0; i < world_size; ++i) {
            recvcounts[i]    = static_cast<int>(sizes(i));
            displacements[i] = static_cast<int>(rank_offsets(i));
        }

        // Step 5: Use Comm::allgatherv to exchange data
        Comm->allgatherv(local_data.data(), local_size, B_glob.data(), recvcounts.data(),
                         displacements.data());

        return {B_glob, rank_offsets};
    }

    Kokkos::View<morton_code*> communicate_octants(
        Kokkos::View<std::vector<morton_code>*> data_to_send) {
        const size_t world_size = Comm->size();
        const size_t world_rank = Comm->rank();

        // Step 1: Determine send and receive counts
        std::vector<int> send_counts(world_size, 0);
        std::vector<int> recv_counts(world_size, 0);

        for (size_t i = 0; i < world_size; ++i) {
            send_counts[i] = data_to_send(i).size();  // Count of data to send to each rank
        }

        // Exchange send counts to get recv_counts
        Comm->alltoall(send_counts.data(), recv_counts.data(), 1);

        // Step 2: Flatten send buffer and prepare displacements
        std::vector<int> send_displs(world_size, 0);
        std::vector<int> recv_displs(world_size, 0);
        std::vector<morton_code> send_buffer;

        for (size_t i = 0; i < world_size; ++i) {
            send_displs[i] = send_buffer.size();
            send_buffer.insert(send_buffer.end(), data_to_send(i).begin(), data_to_send(i).end());
        }

        int total_recv_size = 0;
        for (size_t i = 0; i < world_size; ++i) {
            recv_displs[i] = total_recv_size;
            total_recv_size += recv_counts[i];
        }

        // Step 3: Allocate receive buffer and perform Alltoallv
        std::vector<morton_code> recv_buffer(total_recv_size);

        Comm->alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(),
                        recv_buffer.data(), recv_counts.data(), recv_displs.data());

        // Step 4: Copy received data into a Kokkos::View
        Kokkos::View<morton_code*> T_view("T_view", total_recv_size);
        Kokkos::parallel_for(
            "CopyRecvToView", total_recv_size,
            KOKKOS_LAMBDA(const int i) { T_view(i) = recv_buffer[i]; });

        std::sort(T_view.data(), T_view.data() + T_view.size());
        return T_view;
    }

    /*
    Input:      L_view:     A distributed sorted complete linear octree
    Output:     R_view:     A distributed complete balanced linear octree

    1.  B_view = block_partition(L_view)
    2.  C_view = BalanceSubtree(B_view, L_view)         (algo7)
    3.  D_view = intra-processor boundary octants
    4.  S_view = ripple(D_view)                         (algo9)
    5.  F_view = linearise(C ∪ S)                       (algo8)
    6.  G_view = inter-processor boundary octants

    */

    // INPUT HAS TO BE SORTED
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::algo11(Kokkos::View<morton_code*> L_view) {
        // B = algo4
        const morton_code min_oct = L_view(0);
        const morton_code max_oct = L_view(L_view.size() - 1);
        std::cerr << "Min_oct: " << min_oct << " Max_oct: " << max_oct << endl;

        Kokkos::View<morton_code*> B_view = block_partition(min_oct, max_oct);
        std::cerr << "SURVIVED BLOCK_PARTITION" << std::endl;

        // this has to be sequential, else we have to sort C_view at the end
        Kokkos::View<morton_code*> C_view("C_view", 0);
        std::for_each(B_view.data(), B_view.data() + B_view.size(),
                      [&, this](const morton_code octant_B) {
                          auto morton_helper_copy = this->morton_helper;
                          auto should_copy        = KOKKOS_LAMBDA(const morton_code octant_L) {
                              return morton_helper_copy.is_descendant(octant_L, octant_B);
                          };

                          const size_t count = count_octants(L_view, should_copy);

                          Kokkos::View<morton_code*> Temp_view("Temp_view", count);

                          size_t index = 0;
                          // this has to be sequential as well
                          std::for_each(L_view.data(), L_view.data() + L_view.size(),
                                        [&](const morton_code octant_L) {
                                            if (should_copy(octant_L)) {
                                                Temp_view(index) = octant_L;
                                                ++index;
                                            }
                                        });

                          auto algo7_view = algo7(octant_B, Temp_view);
                          concatenateViews(C_view, algo7_view);
                      });

        // D = intra proc boundaries
        Kokkos::View<morton_code*> D_view = initialise_D_view(this->morton_helper, B_view, C_view);

        // ripple propagation
        // D_view must be sorted here TODO possible bug
        auto S_view = algo9(D_view);
        auto F_view = linearise_octants(concatenateViews(S_view, C_view));

        // G = inter proc boundaries
        auto G_view = initialise_G_view(this->morton_helper, B_view, F_view);

        // copy octants that go to ranks in here
        Kokkos::View<std::vector<morton_code>*> data_to_send1("send_data", Comm->size());

        auto [B_glob_view, B_glob_rank_offsets] = exchange_B_glob_with_offsets(B_view);

        for (const morton_code octant_G : std::span(G_view.data(), G_view.data() + G_view.size())) {
            // using a set is probably worth it here,
            // has the inner loop is (probably
            // relatively large)
            const auto insulation_layer_data = morton_helper.get_insulation_layer(octant_G);
            auto insulation_span =
                std::span(insulation_layer_data.data(),
                          insulation_layer_data.data() + insulation_layer_data.size());
            std::unordered_set<morton_code> i_layer(insulation_span.begin(), insulation_span.end());

            for (size_t rank = 0; rank < Comm->size(); ++rank) {
                if (rank == Comm->rank()) {
                    // no need to send own data
                    continue;
                }

                // I don't think this works, the condition is just gonna be false TODO
                for (size_t i = B_glob_rank_offsets(rank);
                     (i < B_glob_view.size()) && (i < B_glob_rank_offsets(rank)); ++i) {
                    const morton_code octant_B_glob = B_glob_view(i);

                    if (i_layer.count(octant_B_glob) == 0) {
                        continue;
                    }

                    data_to_send1(rank).push_back(octant_B_glob);
                }
            }
        }

        Kokkos::View<morton_code*> T_view = communicate_octants(data_to_send1);

        // Couldn't you just check if search_octant is in data_to_send1[i] ? TODO
        auto get_rank = [&B_glob_view, B_glob_rank_offsets](const morton_code search_octant) {
            for (size_t rank = 0; rank < Comm->size(); ++rank) {
                for (size_t i = B_glob_rank_offsets(rank);
                     (i < B_glob_view.size()) && (i < B_glob_rank_offsets(rank)); ++i) {
                    const morton_code octant_B_glob = B_glob_view(i);

                    if (octant_B_glob == search_octant) {
                        return rank;
                    }
                }
            }
            return size_t(-1);
        };

        Kokkos::View<std::vector<morton_code>*> data_to_send2("send_data", Comm->size());
        for (const morton_code octant_G : std::span(G_view.data(), G_view.data() + G_view.size())) {
            for (const morton_code octant_T :
                 std::span(T_view.data(), T_view.data() + T_view.size())) {
                auto is_in_insulation_layer = [&](const morton_code octant,
                                                  const morton_code octant_to_insulate) -> bool {
                    for (const morton_code i_octant :
                         morton_helper.get_insulation_layer(octant_to_insulate)) {
                        if (i_octant == octant) {
                            return true;
                        }
                    }
                    return false;
                };

                if (!is_in_insulation_layer(octant_G, octant_T)) {
                    continue;
                }

                size_t target_rank = get_rank(octant_T);

                if (std::find(data_to_send1(target_rank).begin(), data_to_send1(target_rank).end(),
                              octant_G)
                    != data_to_send1(target_rank).end()) {
                    continue;
                }

                data_to_send2(target_rank).push_back(octant_G);
            }
        }

        Kokkos::View<morton_code*> K_view = communicate_octants(data_to_send2);

        auto H_view = algo9(concatenateViews(G_view, T_view, K_view));

        auto R_view = initialise_R_view(this->morton_helper, B_view, H_view, F_view);
        R_view = linearise_octants(R_view);

        return R_view;
    }

}  // namespace ippl

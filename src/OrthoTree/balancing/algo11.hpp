#include <Kokkos_Core.hpp>

#include <cassert>
#include <cstddef>
#include <string>

#include "../OrthoTree.h"
namespace ippl {

    // ================================
    // ================================
    // DEVELOPMENT HELPERS
    // ================================
    // ================================

    /**
     * @brief the morton_helper function get_neighbors only returns neighbors of the same level as the input octant,
     * but we also need the neighbors at all the finer levels
     */
    Kokkos::View<morton_code*> get_neighbour_view(const auto& morton_helper,
                                                  const morton_code octant, const size_t max_depth) {
        //TODO change morton codes to views
        // Kokkos::View<morton_code> neighbour_vec = morton_helper.get_neighbors(octant, morton_helper.get_depth(octant));
        vector_t<morton_code> neighbour_vec =
            morton_helper.get_neighbors(octant, morton_helper.get_depth(octant));
        Kokkos::View<morton_code*> neighbour_view("neighbour_view", 0);
        for(morton_code neighbour: neighbour_vec){
            vector_t<morton_code> children;
            if(morton_helper.get_depth(neighbour) < max_depth){
                children = morton_helper.get_children(neighbour);
            }
            children.push_back(neighbour);
            const size_t offset = neighbour_view.size();
            Kokkos::resize(neighbour_view, offset + children.size());
            Kokkos::parallel_for(
                "CopyStdVectorToKokkosView", children.size(),
                KOKKOS_LAMBDA(const size_t i) { neighbour_view(offset + i) = children[i]; });

        }

        return neighbour_view;
    }

    // ================================
    // ================================
    // VIEW HELPERS
    // just to reduce code duplication
    // ================================
    // ================================

    /**
     * @brief removes duplicates from a view
     */
    Kokkos::View<morton_code*> removeDuplicates(Kokkos::View<morton_code*> input){
        Kokkos::View<morton_code*> output("output", input.size());
        if(input.size() != 0){
            size_t unique_count = 0;
            Kokkos::parallel_reduce(
                "algo3::CountUniqueElements", input.size() - 1,
                KOKKOS_LAMBDA(const size_t i, size_t& local_count) {
                    local_count += static_cast<size_t>(input(i) != input(i + 1));
                },
                unique_count);

            const size_t out_size = unique_count + 1;

            Kokkos::resize(output, out_size);

            size_t index = 0;
            Kokkos::parallel_scan(
                    "algo3::PopulateUniqueElements", input.size() - 1, 
                    KOKKOS_LAMBDA(const size_t i, size_t& index, bool final) {
                        if (input(i) != input(i+1)) {
                            if (final) {
                                output(index) = input(i);
                            }
                            ++index;
                        }
                    });

            std::string log_str = "Rank " + std::to_string(Comm->rank()) + ": out_size: " + std::to_string(out_size) + "\n";
            std::cerr << log_str;
            Kokkos::parallel_for(
                "algo3::AddLastElement", 1, KOKKOS_LAMBDA(const int) {
                    output(output.size() - 1) = input(input.extent(0) - 1);
                });
        }
        return output;
    }

    /**
     * @brief Takes two sorted views as input and concatenates them into a sorted output view
     */
    Kokkos::View<morton_code*> concatenateViews(Kokkos::View<morton_code*> view_one, Kokkos::View<morton_code*> view_two) {
        size_t total_size = view_one.extent(0) + view_two.extent(0);

        Kokkos::View<morton_code*> result("concatenated_view", total_size);
        std::merge(view_one.data(), view_one.data() + view_one.extent(0), view_two.data(), view_two.data() + view_two.extent(0), result.data());

        result = removeDuplicates(result);

        return result;
    }

    /**
     * @brief Counts the number of octants in the given view that fullfill the predicate
     */
    template <typename Predicate>
    size_t count_octants(const Kokkos::View<morton_code*> count_view, Predicate predicate) {
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
     */
    template <typename Predicate>
    Kokkos::View<morton_code*> filter_octants(const Kokkos::View<morton_code*> in_view,
                                              Predicate predicate) {
        size_t n_pass = 0;

        Kokkos::parallel_reduce(
            "filter_octants", in_view.extent(0),
            KOKKOS_LAMBDA(const size_t i, size_t& count) {
                const morton_code val = in_view(i);
                if (predicate(val)) {
                    count++;
                }
            },
            n_pass);

        Kokkos::View<morton_code*> out_view("out_view", n_pass);

        Kokkos::parallel_scan(
            in_view.extent(0), KOKKOS_LAMBDA(const size_t i, size_t& index, bool final) {
                const morton_code val = in_view(i);
                if (predicate(val)) {
                    if (final) {
                        out_view(index) = in_view(i);
                    }
                    index++;
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

    /**
     * @brief This function initializes the D_view of algorithm11, this view
     * will contain all intra-processor boundary octants
     */
    Kokkos::View<morton_code*> initialise_D_view(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> C_view,
                                                 const size_t max_depth) {
        auto should_insert = KOKKOS_LAMBDA(morton_code check_octant) {
            const Kokkos::View<morton_code*> neighbour_view =
                get_neighbour_view(morton_helper, check_octant, max_depth);

            for (size_t i = 0; i < neighbour_view.size(); ++i) {
                const morton_code neighbour_oct = neighbour_view[i];
                for (size_t j = 0; j < B_view.size(); ++j) {
                    const morton_code B_oct = B_view[j];

                    const bool overlaps_neighbour =
                        morton_helper.does_overlap(neighbour_oct, B_oct);

                    const bool overlaps_check_octant =
                        check_octant == B_oct || morton_helper.is_ancestor(check_octant, B_oct);

                    if (overlaps_neighbour && !overlaps_check_octant) {
                        return true;
                    }
                }
            }
            return false;
        };

        Kokkos::View<morton_code*> D_view = filter_octants(C_view, should_insert);
        return D_view;
    }

    /**
     * @brief This function initializes the G_view for algorithm 11, this view will
     * contain all inter-processor boundary octants
     */
    Kokkos::View<morton_code*> initialise_G_view(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> F_view,
                                                 const size_t max_depth) {
        auto should_insert = KOKKOS_LAMBDA(morton_code octant_to_check) {
            Kokkos::View<morton_code*> neighbour_view =
                get_neighbour_view(morton_helper, octant_to_check, max_depth);

            for (size_t i = 0; i < neighbour_view.size(); ++i) {
                const morton_code neighbour_oct = neighbour_view[i];
                for (size_t j = 0; j < B_view.size(); ++j) {
                    const morton_code B_oct = B_view[i];

                    const bool overlaps_neighbour =
                        morton_helper.does_overlap(neighbour_oct, B_oct);
                    if (!overlaps_neighbour) {
                        return true;
                    }
                }
            }

            return false;
        };

        Kokkos::View<morton_code*> G_view = filter_octants(F_view, should_insert);
        return G_view;
    }

    /**
     * @brief This function initializes the R_view of algorithm 11, this view will
     * contain the balanced tree on the given processor
     */
    Kokkos::View<morton_code*> initialise_R_view(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> H_view,
                                                 Kokkos::View<morton_code*> F_view) {
        auto should_insert = KOKKOS_LAMBDA(morton_code octant_to_insert) {
            for (size_t i = 0; i < B_view.size(); i++) {
                auto b_oct = B_view(i);
                if (b_oct == octant_to_insert || morton_helper.is_ancestor(octant_to_insert, b_oct))
                    return true;
            }
            return false;
        };

        Kokkos::View<morton_code*> H_filtered = filter_octants(H_view, should_insert);
        Kokkos::View<morton_code*> F_filtered = filter_octants(F_view, should_insert);
        Kokkos::View<morton_code*> R_view     = concatenateViews(H_filtered, F_filtered);
        return R_view;
    }

    /**
     * @brief this function returns two views that give the number of octants to send to other ranks
     * and the octants that have to be sent to those ranks for the first stage communication
     * @return data_to_send: the octants that have to be sent to the different ranks
     * @return offsets: a prefix sum that gives the index ranges in data_to_send for the different ranks
     * as difference of offsets(target_rank) - offsets(target_rank - 1)
     */
    auto inter_proc_boundaries(const auto& morton_helper, Kokkos::View<morton_code*> G_view,
                               Kokkos::View<morton_code*> B_view) {

        const size_t world_size = Comm->size();
        const size_t world_rank = Comm->rank();

        // communicate sizes
        Kokkos::View<size_t*> window_sizes("window_sizes", world_size);
        size_t B_view_size = B_view.size();
        Comm->allgather(&B_view_size, window_sizes.data(), 1);

        mpi::rma::Window<mpi::rma::Active> Glob_view;
        {  // initialise window
            auto B_span = std::span(B_view.data(), B_view.data() + B_view.size());
            Glob_view.create(*Comm, B_span.begin(), B_span.end());
            Glob_view.fence(0);
        }

        size_t send_idx                     = 0;
        const size_t data_to_send_base_size = 100;
        Kokkos::View<morton_code*> data_to_send("octants_to_send", data_to_send_base_size);
        Kokkos::View<size_t*> offsets("offsets", world_size);
        Kokkos::deep_copy(offsets, int(0));

        size_t offset = 0;
        // scan the window of each rank
        for (size_t source_rank = 0; source_rank < world_size; ++source_rank) {
            if (source_rank == world_rank || (window_sizes(source_rank) == 0)) {
                offsets(source_rank) = offset;
                continue;
            }
            // load data
            Kokkos::View<morton_code*> source_data(
                "source_data", std::max(window_sizes(source_rank), B_view.size()));
            // TODO make this better
            for (size_t i = 0; i < B_view.size(); i++) {
                source_data(i) = B_view(i);
            }
            {
                auto source_span =
                    std::span(source_data.data(), source_data.data() + source_data.size());
                Glob_view.fence(0);
                Glob_view.get(source_span.begin(), source_span.begin() + window_sizes(source_rank),
                              source_rank, 0);
                Glob_view.fence(0);
                Kokkos::resize(source_data, window_sizes(source_rank));
            }

            {  // do stuff with data
                auto contains = [&](const auto& i_layer, const morton_code search_code) -> bool {
                    return std::any_of(i_layer.data(), i_layer.data() + i_layer.size(),
                                       [&](const morton_code i_oct) {
                                           return (i_oct == search_code)
                                                  || morton_helper.is_descendant(i_oct,
                                                                                 search_code)
                                                        || morton_helper.is_descendant(search_code, i_oct);
                                       });
                };

                // naive for now, im lazy
                bool new_insert = true;
                for (size_t i = 0; i < G_view.size(); ++i) {
                    new_insert = true;
                    const morton_code G_oct     = G_view(i);
                     auto insulation_layer = morton_helper.get_insulation_layer(G_oct);

                    for (size_t j = 0; j < source_data.size(); ++j) {
                        const morton_code Glob_oct = source_data(j);
                        if (!contains(insulation_layer, Glob_oct)) {
                            continue;
                        }

                        if (send_idx == data_to_send.size()) {
                            Kokkos::resize(data_to_send,
                                           data_to_send.size() + data_to_send_base_size);
                        }
                        if(!new_insert){
                            if(data_to_send(send_idx-1) != G_oct){
                                data_to_send(send_idx) = G_oct;
                                ++offset;
                                ++send_idx;
                            }
                        } else{
                            new_insert = false;
                            data_to_send(send_idx) = G_oct;
                            ++offset;
                            ++send_idx;
                        }
                    }
                }

                offsets(source_rank) = offset;
            }
        }

        Glob_view.fence(0);
        Kokkos::resize(data_to_send, send_idx);
        return std::make_pair(data_to_send, offsets);
    }

    /**
     * @brief this function returns two views that give the number of octants to send to other ranks
     * and the octants that have to be sent to those ranks for the second stage communication
     * @return data_to_send: the octants that have to be sent to the different ranks
     * @return offsets: a prefix sum that gives the index ranges in data_to_send for the different ranks
     * as difference of offsets(target_rank) - offsets(target_rank - 1)
     */
    auto K_inter_proc_boundaries(const auto& morton_helper, Kokkos::View<morton_code*> G_view,
                               Kokkos::View<morton_code*> T_view, Kokkos::View<morton_code*> old_overlapping_octants, Kokkos::View<size_t*> old_overlap_offsets, Kokkos::View<size_t*> T_recv_sizes) {

        const size_t world_size = Comm->size();
        const size_t world_rank = Comm->rank();

        size_t send_idx                     = 0;
        const size_t data_to_send_base_size = 100;
        Kokkos::View<morton_code*> data_to_send("octants_to_send", data_to_send_base_size);
        Kokkos::View<size_t*> offsets("offsets", world_size);
        Kokkos::deep_copy(offsets, int(0));
        for(size_t source_rank = 0; source_rank < world_size; source_rank++){
            size_t offset = 0;
            if (source_rank == world_rank) {
                offsets(source_rank) = offset;
                continue;
            }

            {  // do stuff with data

                auto contains = [&](const auto& i_layer, const morton_code search_code) -> bool {
                    return std::any_of(i_layer.data(), i_layer.data() + i_layer.size(),
                                       [&](const morton_code i_oct) {
                                           return (i_oct == search_code)
                                                  || morton_helper.is_descendant(search_code,
                                                                                 i_oct);
                                       });
                };

            auto already_sent =
                KOKKOS_LAMBDA(const morton_code search_oct, const size_t target_rank) {
                const size_t start   = target_rank == 0? 0 : old_overlap_offsets(target_rank-1);
                const size_t end     = old_overlap_offsets(target_rank);
                const auto sent_octs = std::span(old_overlapping_octants.data() + start, end - start);
                return std::find(sent_octs.begin(), sent_octs.end(), search_oct) != sent_octs.end();
            };

                bool new_insert = true;
                for (size_t i = 0; i < G_view.size(); ++i) {
                    new_insert = true;
                    const morton_code G_oct     = G_view(i);

                    size_t T_start = source_rank == 0? 0 : T_recv_sizes(source_rank-1);
                    size_t T_end = T_recv_sizes(source_rank);

                    for (size_t j = T_start; j < T_end; j++) {
                        const morton_code T_oct = T_view(j);
                        auto insulation_layer = morton_helper.get_insulation_layer(T_oct);
                        size_t rank_t = 0;

                        if (!contains(insulation_layer, G_oct)) {
                            continue;
                        }

                        if(already_sent(G_oct, source_rank)){
                            continue;
                        }

                        if (send_idx == data_to_send.size()) {
                            Kokkos::resize(data_to_send,
                                           data_to_send.size() + data_to_send_base_size);
                        }
                        if(!new_insert){
                            if(data_to_send(send_idx-1) != G_oct){
                                data_to_send(send_idx) = G_oct;
                                ++offset;
                                ++send_idx;
                            }
                        } else{
                            new_insert = false;
                            data_to_send(send_idx) = G_oct;
                            ++offset;
                            ++send_idx;
                        }
                    }
                }

                offsets(source_rank) = offset;
            }
        }

        Kokkos::resize(data_to_send, send_idx);
        return std::make_pair(data_to_send, offsets);
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
        std::cout << std::to_string(Comm->rank()) + ": starting algo11\n";

        IpplTimings::TimerRef algo11Timer = IpplTimings::getTimer("algo11");
        IpplTimings::startTimer(algo11Timer);

        IpplTimings::TimerRef algo11_B_view_Timer = IpplTimings::getTimer("algo11_B_view");
        IpplTimings::startTimer(algo11_B_view_Timer);

        // Kokkos::View<morton_code*> B_view = block_partition(L_view(0), L_view(L_view.size() - 1));
        Kokkos::View<morton_code*> B_view = L_view; // algo4_11(L_view);

        IpplTimings::stopTimer(algo11_B_view_Timer);

        IpplTimings::TimerRef algo11_C_view_Timer = IpplTimings::getTimer("algo11_C_view");
        IpplTimings::startTimer(algo11_C_view_Timer);
        // this has to be sequential, else we have to sort C_view at the end
        Kokkos::View<morton_code*> C_view("C_view", 0);
        std::for_each(B_view.data(), B_view.data() + B_view.size(),
                      [&, this](const morton_code octant_B) {
                          auto morton_helper_copy = this->morton_helper;
                          auto should_copy        = KOKKOS_LAMBDA(const morton_code octant_L) {
                              return octant_L == octant_B
                                     || morton_helper_copy.is_descendant(octant_L, octant_B);
                          };

                          const size_t count = count_octants(L_view, should_copy);
                          if (count == 0) {
                              std::cerr << "why is count == 0?" << endl;
                              return;
                          }

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
                          C_view = concatenateViews(C_view, algo7_view);
                      });

        IpplTimings::stopTimer(algo11_C_view_Timer);

        IpplTimings::TimerRef algo11_D_view_Timer = IpplTimings::getTimer("algo11_D_view");
        IpplTimings::startTimer(algo11_D_view_Timer);

        Kokkos::View<morton_code*> D_view = initialise_D_view(this->morton_helper, B_view, C_view, max_depth_m);

        IpplTimings::stopTimer(algo11_D_view_Timer);

        IpplTimings::TimerRef algo11_S_view_Timer = IpplTimings::getTimer("algo11_S_view");
        IpplTimings::startTimer(algo11_S_view_Timer);

        // ripple propagation
        auto S_view = algo9(D_view);

        IpplTimings::stopTimer(algo11_S_view_Timer);

        IpplTimings::TimerRef algo11_F_view_Timer = IpplTimings::getTimer("algo11_F_view");
        IpplTimings::startTimer(algo11_F_view_Timer);

        auto concatenated_S_C = concatenateViews(S_view, C_view);
        auto F_view = linearise_octants(concatenated_S_C);

        IpplTimings::stopTimer(algo11_S_view_Timer);

        IpplTimings::TimerRef algo11_G_view_Timer = IpplTimings::getTimer("algo11_G_view");
        IpplTimings::startTimer(algo11_G_view_Timer);

        auto G_view = initialise_G_view(this->morton_helper, B_view, F_view, max_depth_m);

        IpplTimings::stopTimer(algo11_G_view_Timer);

        std::cout << std::to_string(Comm->rank()) + ": Starting with T_view\n";

        IpplTimings::TimerRef algo11_T_view_Timer = IpplTimings::getTimer("algo11_T_view");
        IpplTimings::startTimer(algo11_T_view_Timer);

        // T_view
        auto [overlapping_octants, overlap_offsets] =
            inter_proc_boundaries(morton_helper, G_view, B_view);

        /**
         * Main idea:
         * - each rank puts the amount of octants it will send into the corresponding size_window
         * (recv_sizes)
         * - using this information each rank can initialise its T_view and the other ranks can look
         * up the insertion offsets
         */

        mpi::rma::Window<mpi::rma::Active> size_window;
        mpi::rma::Window<mpi::rma::Active> T_window;
        Kokkos::View<size_t*> recv_sizes("recv_sizes", world_size);
        Kokkos::deep_copy(recv_sizes, 0);
        size_t total_recv_size = 0;

        /**
         * Put the sizes that each rank will receive into its window
         */
        {
            auto size_span = std::span(recv_sizes.data(), recv_sizes.size());
            size_window.create(*Comm, size_span.begin(), size_span.end());
            size_window.fence(0);

            size_t size_buff = 0;
            for (size_t target_rank = 0; target_rank < world_size; ++target_rank) {
                if (target_rank == world_rank) {
                    continue;
                }
                size_buff = overlap_offsets(target_rank) - (target_rank == 0? 0 : overlap_offsets(target_rank-1));

                size_window.put(size_buff, target_rank, world_rank);
                size_window.fence(0);
            }

            Kokkos::parallel_scan(
                recv_sizes.extent(0), KOKKOS_LAMBDA(const size_t i, size_t& update, bool final) {
                    const size_t val = recv_sizes(i);
                    update += val;
                    if (final) {
                        recv_sizes(i) = update;
                    }
                });

            total_recv_size = recv_sizes(recv_sizes.size() - 1);
        }
        Kokkos::View<morton_code*> T_view("T_view", total_recv_size + overlapping_octants.size());
        for (size_t i = total_recv_size; i < T_view.size(); i++) {
            T_view(i) = overlapping_octants(i - total_recv_size);
        }

        /**
         * Put necessary data into each ranks window
         */
        {
            auto overlapping_octants_span =
                std::span(overlapping_octants.data(), overlapping_octants.size());
            auto T_span = std::span(T_view.data(), T_view.size());

            T_window.create(*Comm, T_span.begin(), T_span.end());
            T_window.fence(0);

            size_t start      = 0;
            size_t end        = 0;
            size_t target_idx = 0;
            for (size_t target_rank = 0; target_rank < world_size; ++target_rank) {
                start = end;
                end += overlap_offsets(target_rank) - (target_rank == 0? 0 : overlap_offsets(target_rank-1));
                if (target_rank == world_rank) {
                    continue;
                }
                if(end - start == 0){
                    size_window.fence(0);
                    T_window.fence(0);
                    continue;
                }

                auto start_iter = T_span.begin() + total_recv_size + start;
                auto end_iter   = T_span.begin() + total_recv_size + end;

                if (world_rank != 0) {
                    size_window.get(&target_idx, target_rank, world_rank-1);
                }
                size_window.fence(0);
                if (start != end) {
                    T_window.put(start_iter, end_iter, target_rank, target_idx);
                }
                T_window.fence(0);
            }
            Kokkos::resize(T_view, total_recv_size);
        }

        IpplTimings::stopTimer(algo11_T_view_Timer);

        /**
         * Each rank should now have all the octants it needs
         */
        std::cout << std::to_string(Comm->rank()) + ": Starting with K_view\n";

        IpplTimings::TimerRef algo11_K_view_Timer = IpplTimings::getTimer("algo11_K_view");
        IpplTimings::startTimer(algo11_K_view_Timer);

        // K_view
        auto [K_overlapping_octants, K_overlap_offsets] =
            K_inter_proc_boundaries(morton_helper, G_view, T_view, overlapping_octants, overlap_offsets, recv_sizes);

        /**
         * Main idea:
         * - each rank puts the amount of octants it will send into the corresponding size_window
         * (recv_sizes)
         * - using this information each rank can initialise its T_view and the other ranks can look
         * up the insertion offsets
         */

        mpi::rma::Window<mpi::rma::Active> K_size_window;
        mpi::rma::Window<mpi::rma::Active> K_window;
        Kokkos::View<size_t*> K_recv_sizes("K_recv_sizes", world_size);
        Kokkos::deep_copy(K_recv_sizes, 0);
        size_t K_total_recv_size = 0;

        /**
         * Put the sizes that each rank will receive into its window
         */
        {
            auto K_size_span = std::span(K_recv_sizes.data(), K_recv_sizes.size());
            K_size_window.create(*Comm, K_size_span.begin(), K_size_span.end());
            K_size_window.fence(0);

            size_t size_buff = 0;
            for (size_t target_rank = 0; target_rank < world_size; ++target_rank) {
                if (target_rank == world_rank) {
                    continue;
                }

                size_buff = K_overlap_offsets(target_rank);

                K_size_window.put(size_buff, target_rank, world_rank);
                K_size_window.fence(0);
            }

            Kokkos::parallel_scan(
                K_recv_sizes.extent(0), KOKKOS_LAMBDA(const size_t i, size_t& update, bool final) {
                    const size_t val = K_recv_sizes(i);
                    update += val;
                    if (final) {
                        K_recv_sizes(i) = update;
                    }
                });

            K_total_recv_size = K_recv_sizes(K_recv_sizes.size() - 1);
        }
        Kokkos::View<morton_code*> K_view("K_view", K_total_recv_size + K_overlapping_octants.size());
        for (size_t i = K_total_recv_size; i < K_view.size(); i++) {
            K_view(i) = K_overlapping_octants(i - K_total_recv_size);
        }

        /**
         * Put necessary data into each ranks window
         */
        {
            auto K_overlapping_octants_span =
                std::span(K_overlapping_octants.data(), K_overlapping_octants.size());
            auto K_span = std::span(K_view.data(), K_view.size());

            K_window.create(*Comm, K_span.begin(), K_span.end());
            K_window.fence(0);

            size_t start      = 0;
            size_t end        = 0;
            size_t target_idx = 0;
            for (size_t target_rank = 0; target_rank < world_size; ++target_rank) {
                start = end;
                end += K_overlap_offsets(target_rank);
                if (target_rank == world_rank) {
                    continue;
                }
                if(end - start == 0){
                    K_size_window.fence(0);
                    K_window.fence(0);
                    continue;
                }

                auto start_iter = K_span.begin() + K_total_recv_size + start;
                auto end_iter   = K_span.begin() + K_total_recv_size + end;

                if (world_rank != 0) {
                    K_size_window.get(&target_idx, target_rank, world_rank-1);
                }
                K_size_window.fence(0);
                if (start != end) {
                    K_window.put(start_iter, end_iter, target_rank, target_idx);
                }
                K_window.fence(0);
            }
            Kokkos::resize(K_view, K_total_recv_size);
        }

        IpplTimings::stopTimer(algo11_K_view_Timer);

        IpplTimings::TimerRef algo11_H_view_Timer = IpplTimings::getTimer("algo11_H_view");
        IpplTimings::startTimer(algo11_H_view_Timer);

        auto conc_G_T_K = concatenateViews(G_view, concatenateViews(T_view , K_view));

        std::cout << std::to_string(Comm->rank()) + ": Starting algo8\n";
        auto H_view = algo9(conc_G_T_K);

        IpplTimings::stopTimer(algo11_H_view_Timer);

        IpplTimings::TimerRef algo11_R_view_Timer = IpplTimings::getTimer("algo11_R_view");
        IpplTimings::startTimer(algo11_R_view_Timer);

        std::cout << std::to_string(Comm->rank()) + ": Initializing R_view\n";
        auto R_view = initialise_R_view(this->morton_helper, B_view, H_view, F_view);
        R_view = linearise_octants(R_view);

        IpplTimings::stopTimer(algo11_R_view_Timer);

        std::cout << std::to_string(Comm->rank()) + ": Done\n";
        assert(is_balanced(R_view));

        IpplTimings::stopTimer(algo11Timer);

        return R_view;
    }
}  // namespace ippl

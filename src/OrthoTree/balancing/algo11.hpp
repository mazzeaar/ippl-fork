#include <Kokkos_Core.hpp>

#include <cassert>
#include <cstddef>
#include <string>

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
    void sort_if_necessary(Kokkos::View<morton_code*> check_view, const std::string& view_name,
                           const std::string& func_name) {
        const std::string COLOR_RED   = "\e[0;31m";
        const std::string COLOR_RESET = "\e[0;31m";

        if (check_view.size() > 0
            && !std::is_sorted(check_view.data(), check_view.data() + check_view.size())) {
            std::cerr << "{view: " << view_name << "}" << " IS NOT SORTED IN "
                      << "{func: " << func_name << "}"
                      << " THE PROBLEM IS PROBABLY THE PARALLEL FOR TO PUPULATE THE VIEW!";
            std::cerr << std::endl;
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

        // This has to be done in such a way that the concatenated view is sorted
        // TODO
        size_t offset  = 0;
        auto copy_view = [&](auto& view) {
            Kokkos::parallel_for(
                "CopyViewData", view.extent(0),
                KOKKOS_LAMBDA(const size_t i) { result(i + offset) = view(i); });
            offset += view.extent(0);
        };

        (copy_view(views), ...);
        // TODO remove this sort, once the data concatenation keeps the ordering of elements
        std::sort(result.data(), result.data() + result.size());
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

    Kokkos::View<morton_code*> initialise_D_view(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> C_view) {
        auto should_insert = KOKKOS_LAMBDA(morton_code check_octant) {
            const Kokkos::View<morton_code*> neighbour_view =
                get_neighbour_view(morton_helper, check_octant);

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
        sort_if_necessary(G_view, "G_view", __func__);
        return G_view;
    }

    Kokkos::View<morton_code*> initialise_R_view(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> H_view,
                                                 Kokkos::View<morton_code*> F_view) {
        auto should_insert = KOKKOS_LAMBDA(morton_code octant_to_insert) {
            for (int i = 0; i < B_view.size(); i++) {
                auto b_oct = B_view(i);
                if (b_oct == octant_to_insert || morton_helper.is_ancestor(octant_to_insert, b_oct))
                    return true;
            }
            return false;
        };

        Kokkos::View<morton_code*> H_filtered = filter_octants(H_view, should_insert);
        Kokkos::View<morton_code*> F_filtered = filter_octants(F_view, should_insert);
        Kokkos::View<morton_code*> R_view     = concatenateViews(H_filtered, F_filtered);
        sort_if_necessary(R_view, "R_view", __func__);
        return R_view;
    }

    /*
    We do the following: Each rank puts its B_view into a window, then each rank iterates through
    all other ranks and does checks.

    Question: maybe it is more efficient to do the inverse? put B_view into a window and then each
    rank can directly take what it needs. BUT then we also need to communicate what we have received
    to the rank whence we took it...
     */
    auto inter_proc_boundaries(const auto& morton_helper, Kokkos::View<morton_code*> G_view,
                               Kokkos::View<morton_code*> B_view) {
        /*
        the return type should probably be a pair of views.
        one with the octants we send, and one with offsets for each target rank
        */

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
            for (int i = 0; i < B_view.size(); i++) {
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

    auto K_inter_proc_boundaries(const auto& morton_helper, Kokkos::View<morton_code*> G_view,
                               Kokkos::View<morton_code*> T_view, Kokkos::View<morton_code*> old_overlapping_octants, Kokkos::View<size_t*> old_overlap_offsets, Kokkos::View<size_t*> T_recv_sizes) {
        /*
        the return type should probably be a pair of views.
        one with the octants we send, and one with offsets for each target rank
        */

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

                // naive for now, im lazy
                bool new_insert = true;
                for (size_t i = 0; i < G_view.size(); ++i) {
                    new_insert = true;
                    const morton_code G_oct     = G_view(i);

                    size_t T_start = source_rank == 0? 0 : T_recv_sizes(source_rank-1);
                    size_t T_end = T_recv_sizes(source_rank);

                    for (int j = T_start; j < T_end; j++) {
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
        Kokkos::View<morton_code*> B_view = L_view;  // algo4_11(L_view);

        // #define AEIOWFOEWIWELI

#ifndef AEIOWFOEWIWELI
        logger.on(true);
        logger.setOutputLevel(1);
#define LOG              \
    Comm->barrier();     \
    if (world_rank == 0) \
    logger << level1 << "Algo11: "

        auto log_the_view = [&](const auto& view, const auto& view_name) {
            Comm->barrier();
            int ring_buff;
            if (Comm->rank() > 0) {
                mpi::Status status;
                Comm->recv(&ring_buff, 1, Comm->rank() - 1, 0, status);
            } else {
                std::cerr << view_name << ".size():" << std::endl;
            }

            std::cerr << "  " << Comm->rank() << ":  " << view.extent(0) << std::endl;

            if (Comm->rank() + 1 < Comm->size()) {
                Comm->send(ring_buff, 1, Comm->rank() + 1, 0);
            } else {
                std::cerr << std::endl;
            }
            Comm->barrier();
        };

        auto print_the_view = [&](const auto& view, const auto& view_name) {
            Comm->barrier();
            int ring_buff;
            if (Comm->rank() > 0) {
                mpi::Status status;
                Comm->recv(&ring_buff, 1, Comm->rank() - 1, 0, status);
            } else {
                std::cerr << view_name << std::endl;
            }

            std::cerr << "  " << Comm->rank() << ": {";
            for (size_t i = 0; i < view.extent(0); ++i) {
                std::cerr << view(i);
                if (i + 1 < view.extent(0)) {
                    std::cerr << ", ";
                }
            }
            std::cerr << "}" << std::endl;

            if (Comm->rank() + 1 < Comm->size()) {
                Comm->send(ring_buff, 1, Comm->rank() + 1, 0);
            } else {
                std::cerr << std::endl;
            }
            Comm->barrier();
        };

#define LOG_VIEW(logging_view)   log_the_view(logging_view, #logging_view)
#define PRINT_VIEW(logging_view) print_the_view(logging_view, #logging_view)
#else

#define LOG           std::cerr
#define LOG_VIEW(x)   ((void)x)
#define PRINT_VIEW(x) ((void)x)
#endif

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
                              logger << level1 << "why is count == 0?" << endl;
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

                          // PRINT_VIEW(Temp_view);
                          auto algo7_view = algo7(octant_B, Temp_view);
                          // PRINT_VIEW(algo7_view);
                          C_view = concatenateViews(C_view, algo7_view);
                      });
        Kokkos::View<morton_code*> D_view = initialise_D_view(this->morton_helper, B_view, C_view);

        // ripple propagation
        // D_view must be sorted here TODO possible bug
        auto S_view = algo9(D_view);
        auto concatenated_S_C = concatenateViews(S_view, C_view);
        auto F_view = linearise_octants(concatenated_S_C);
        auto G_view = initialise_G_view(this->morton_helper, B_view, F_view);

        /**
         * overlapping_octants is a flat view of the octants we send.
         * overlap_offsets is a prefix sum (?!), where overlap_offsets(0) = send_size rank 0
         */
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

                size_window.fence(0);
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
        for (int i = total_recv_size; i < T_view.size(); i++) {
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
                // TODO
                // if(end - start == 0) continue;

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
        Comm->barrier();

        /**
         * Each rank should now have all the octants it needs
         */

        // =========================================
        // working on K_view
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

                K_size_window.fence(0);
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
        for (int i = K_total_recv_size; i < K_view.size(); i++) {
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
                // TODO
                // if(K_overlap_offsets(target_rank) == 0) continue;

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
        // =========================================

        auto conc_G_T_K = concatenateViews(G_view, T_view , K_view);

        auto H_view = algo9(conc_G_T_K);
        auto R_view = initialise_R_view(this->morton_helper, B_view, H_view, F_view);
        R_view = linearise_octants(R_view);
        return R_view;
    }

}  // namespace ippl

#include <Kokkos_Core.hpp>
#include <cstddef>

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
        std::cerr << Comm->rank() << ": Starting G_view" << std::endl;
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

        // D = inter-proc boundaries
        Kokkos::View<morton_code*> G_view = filter_octants(F_view, should_insert);
        std::cerr << Comm->rank() << ": Now sorting G_view" << std::endl;
        sort_if_necessary(G_view, "G_view", __func__);
        std::cerr << Comm->rank() << ": Survived G_view" << std::endl;
        return G_view;
    }

    Kokkos::View<morton_code*> initialise_R_view(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> H_view,
                                                 Kokkos::View<morton_code*> F_view) {
        auto should_insert = KOKKOS_LAMBDA(morton_code octant_to_insert) {
            for(int i = 0; i < B_view.size(); i++){
                auto b_oct = B_view(i);
                if(b_oct == octant_to_insert || morton_helper.is_ancestor(octant_to_insert, b_oct)) return true;
            }
            return false;
        };

        Kokkos::View<morton_code*> H_filtered = filter_octants(H_view, should_insert);
        Kokkos::View<morton_code*> F_filtered = filter_octants(F_view, should_insert);
        Kokkos::View<morton_code*> R_view = concatenateViews(H_filtered, F_filtered);
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
        Kokkos::View<int*> offsets("offsets", world_size);
        Kokkos::deep_copy(offsets, int(0));

        // scan the window of each rank
        for (size_t source_rank = 0; source_rank < world_size; ++source_rank) {
            size_t offset = 0;
            if (source_rank == world_rank || (window_sizes(source_rank) == 0)) {
                offsets(source_rank) = offset;
                continue;
            }

            // load data
            Kokkos::View<morton_code*> source_data("source_data", std::max(window_sizes(source_rank), B_view.size()));
            // TODO make this better
            for(int i = 0; i < B_view.size(); i++){
                source_data(i) = B_view(i);
            }
            {
                auto source_span =
                    std::span(source_data.data(), source_data.data() + source_data.size());
                Glob_view.fence(0);
                Glob_view.get(source_span.begin(), source_span.begin() + window_sizes(source_rank), source_rank, 0);
                Glob_view.fence(0);
                Kokkos::resize(source_data, window_sizes(source_rank));
            }
            // TODO remove logging
            std::string log_str = "Logging on rank " + std::to_string(Comm->rank()) + " for target rank " + std::to_string(source_rank) + " source_data = {";
            for(int i = 0; i < source_data.size(); i++){
              log_str += std::to_string(source_data(i));
              if(i != source_data.size() - 1) log_str += ", ";
            }
            log_str += "}";
            std::cerr << log_str << std::endl;

            {  // do stuff with data
                auto contains = [&](const auto& i_layer, const morton_code search_code) -> bool {
                    return std::any_of(
                        i_layer.data(), i_layer.data() + i_layer.size(),
                        [&](const morton_code i_oct) {
                            return (i_oct == search_code)
                                       || morton_helper.is_descendant(search_code, i_oct);
                        });
                };

                // naive for now, im lazy
                bool new_insert = true;
                for (size_t i = 0; i < B_view.size(); ++i) {
                    new_insert = true;
                    const morton_code B_oct     = B_view(i);
                    const auto insulation_layer = morton_helper.get_insulation_layer(B_oct);

                    for (size_t j = 0; j < source_data.size(); ++j) {
                        const morton_code Glob_oct = source_data(j);
                        if (contains(insulation_layer, Glob_oct)) {
                            continue;
                        }

                        if (send_idx == data_to_send.size()) {
                            Kokkos::resize(data_to_send,
                                           data_to_send.size() + data_to_send_base_size);
                        }
                        if(!new_insert){
                            if(data_to_send(send_idx-1) != B_oct){
                                data_to_send(send_idx) = B_oct;
                                ++offset;
                                ++send_idx;
                            }
                        } else{
                            new_insert = false;
                            data_to_send(send_idx) = B_oct;
                            ++offset;
                            ++send_idx;
                        }
                    }
                }

                offsets(source_rank) = offset;
            }
            std::cerr << "Done with this iteration\n";
        }

        Glob_view.fence(0);
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
        Kokkos::View<morton_code*> B_view = L_view;//algo4_11(L_view);
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

        LOG_VIEW(B_view);
        PRINT_VIEW(B_view);
        LOG_VIEW(L_view);

        // this has to be sequential, else we have to sort C_view at the end
        Kokkos::View<morton_code*> C_view("C_view", 0);
        std::for_each(B_view.data(), B_view.data() + B_view.size(),
                      [&, this](const morton_code octant_B) {
                          auto morton_helper_copy = this->morton_helper;
                          auto should_copy        = KOKKOS_LAMBDA(const morton_code octant_L) {
                              return octant_L == octant_B || morton_helper_copy.is_descendant(octant_L, octant_B);
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

                          logger << level1 << "Octant_B: " << octant_B << endl;
                          //PRINT_VIEW(Temp_view);
                          auto algo7_view = algo7(octant_B, Temp_view);
                          logger << level1 << "Done with algo7 for this octant" << endl;
                          std::string log_str = "Logging algo7_view = {";
                          for(int i = 0; i < algo7_view.size(); i++){
                              log_str += std::to_string(algo7_view(i));
                              if(i != algo7_view.size() - 1) log_str += ", ";
                          }
                          log_str += "}";
                          logger << level1 << log_str << endl;
                          //PRINT_VIEW(algo7_view);
                          C_view          = concatenateViews(C_view, algo7_view);
                          logger << level1 << "Concatenated the views" << endl;
                      });
        logger << level1 << "Initialized C_view" << endl;
        Comm->barrier();
        //PRINT_VIEW(B_view);
        PRINT_VIEW(C_view);
        // D = intra proc boundaries
        Kokkos::View<morton_code*> D_view = initialise_D_view(this->morton_helper, B_view, C_view);
        PRINT_VIEW(D_view);

        // ripple propagation
        // D_view must be sorted here TODO possible bug
        auto S_view           = algo9(D_view);
        PRINT_VIEW(S_view);
        auto concatenated_S_C = concatenateViews(S_view, C_view);
        // TODO get rid of this sort
        std::sort(concatenated_S_C.data(), concatenated_S_C.data() + concatenated_S_C.size());
        PRINT_VIEW(concatenated_S_C);
        auto F_view           = linearise_octants(concatenated_S_C);
        //LOG_VIEW(F_view);
        PRINT_VIEW(F_view);
        // G = inter proc boundaries
        auto G_view = initialise_G_view(this->morton_helper, B_view, F_view);
        //LOG_VIEW(G_view);
        PRINT_VIEW(G_view);
        logger << level1 << "Got to barrier after initializing G_view" << endl;
        Comm->barrier();
        logger << level1 << "Got after barrier after G_view" << endl;

        /**
         * overlapping_octants is a flat view of the octants we send.
         * overlap_offsets is a prefix sum (?!), where overlap_offsets(0) = send_size rank 0
         */
        auto [overlapping_octants, overlap_offsets] =
            inter_proc_boundaries(morton_helper, G_view, B_view);
        logger << level1 << "overlaps ok" << endl;
        PRINT_VIEW(overlap_offsets);
        PRINT_VIEW(overlapping_octants);

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
         * TODO: this should probably be a prefix sum?
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

                size_buff = overlap_offsets(target_rank);

                logger << level1 << "Putting size_buff: " << size_buff << " in window of rank: " << target_rank << endl;
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

            std::string log_str = "Logging recv_sizes = {";
            for(int i = 0; i < recv_sizes.size(); i++){
              log_str += std::to_string(recv_sizes(i));
              if(i != recv_sizes.size() - 1) log_str += ", ";
            }
            log_str += "}";
            logger << level1 << log_str << endl;
            /*
            Kokkos::parallel_reduce(
                "compute new size", world_size,
                KOKKOS_LAMBDA(const size_t i, size_t& local_new_size) {
                    local_new_size += recv_sizes(i);
                },
                total_recv_size);
                */
            total_recv_size = recv_sizes(recv_sizes.size() - 1);
        }
        Kokkos::View<morton_code*> T_view("T_view", total_recv_size + overlapping_octants.size());
        for (int i = total_recv_size; i < T_view.size(); i++) {
            T_view(i) = overlapping_octants(i-total_recv_size);
        }
        logger << level1 << "Initialized T_view with size " << std::to_string(T_view.size()) << endl;

        /**
         * Put necessary data into each ranks window
         */
        Comm->barrier();
        {
            // TODO this does not work, because you would ovverride the data on the other ranks and the sizes are probably also mismatched
            auto overlapping_octants_span =
                std::span(overlapping_octants.data(), overlapping_octants.size());
            auto T_span = std::span(T_view.data(), T_view.size());

            T_window.create(*Comm, T_span.begin(), T_span.end());
            T_window.fence(0);
            logger << level1 << "Created T_window" << endl;

            size_t start = 0;
            size_t end   = 0;
            size_t target_idx = 0;
            for (size_t target_rank = 0; target_rank < world_size; ++target_rank) {
                start = end;
                end += overlap_offsets(target_rank);
                if (target_rank == world_rank) {
                    size_window.fence(0);
                    T_window.fence(0);
                    continue;
                }
                //if(overlap_offsets(target_rank) == 0) continue;

                // TODO: world_rank is wrong as pos, this must be looked up in size_window of the
                // target_rank :,) -> hence prefix sum
                auto start_iter = T_span.begin() + total_recv_size+ start;
                auto end_iter   = T_span.begin() + total_recv_size+ end;
                size_t dis = 0;
                if(world_rank != 0){
                    dis = world_rank - 1;
                    size_window.get(&target_idx, target_rank, dis);
                }
                size_window.fence(0);                
                if(start != end){

                    T_window.put(start_iter, end_iter, target_rank, target_idx);

                }
                T_window.fence(0);
            }
            Kokkos::resize(T_view, total_recv_size);
        }
        logger << level1 << "Got after the loop" << endl;
        Comm->barrier();

        /**
         * Each rank should now have all the octants it needs
         */
        PRINT_VIEW(T_view);

        /**
         * corresponds to line 15-23 of ps.code
         */
        {
            /**
             * Checks if we already sent the search_oct to the target_rank
             */
            auto already_sent =
                KOKKOS_LAMBDA(const morton_code search_oct, const size_t target_rank) {
                const size_t start   = overlap_offsets(target_rank);
                const size_t end     = (target_rank + 1 == world_size)
                                           ? overlap_offsets(target_rank + 1)
                                           : overlapping_octants.size();
                const auto sent_octs = std::span(overlapping_octants.data() + start, end - start);
                return std::find(sent_octs.begin(), sent_octs.end(), search_oct) != sent_octs.end();
            };
            logger << level1 << "Checked for already sent octants" << endl;

            for (size_t i = 0; i < T_view.size(); ++i) {
                const morton_code T_oct = T_view(i);
                const auto i_layer_view = morton_helper.get_insulation_layer(T_oct);
                const auto i_layer      = std::span(i_layer_view.data(), i_layer_view.size());

                auto contains = KOKKOS_LAMBDA(const morton_code search_oct) {
                    return std::find(i_layer.begin(), i_layer.end(), search_oct) != i_layer.end();
                };

                // i *think* this should initialise rank_t correctly
                size_t rank_t = 0;
                //logger << level1 << "possible infinite loop" << endl;
                while (i <= recv_sizes(rank_t)) {
                    ++rank_t;
                }
                //logger << level1 << "Actually not an infinite loop" << endl;

                for (size_t j = 0; j < G_view.size(); ++j) {
                    const morton_code G_oct = G_view(i);
                    if (!contains(G_oct)) {
                        continue;
                    }

                    if (already_sent(G_oct, rank_t)) {
                        continue;
                    }

                    // TODO: prepare the data to send
                    // Send(G_oct, rank_t)
                }
            }
        }
        logger << level1 << "Got to K_view initialization, some stuff left todo" << endl;

        // TODO: send the data generated in the loop above
        Kokkos::View<morton_code*> K_view;
        // TODO receive octants into K_view
        PRINT_VIEW(K_view);

        auto H_view = algo9(concatenateViews(G_view, T_view /*, K_view */));
        LOG << "H_view ok" << endl;
        PRINT_VIEW(H_view);
        //std::sort(R_view.data(), R_view.data() + R_view.size());
        auto R_view = initialise_R_view(this->morton_helper, B_view, H_view, F_view);
        LOG << "R_view ok" << endl;
        PRINT_VIEW(R_view);
        R_view = linearise_octants(R_view);
        LOG_VIEW(R_view);
        PRINT_VIEW(R_view);
        LOG << "R_view linearise ok" << endl;
        return R_view;
    }

}  // namespace ippl

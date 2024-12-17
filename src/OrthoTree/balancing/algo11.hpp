#include "../OrthoTree.h"

namespace ippl {
    /*
    TODO:
    - IMPLEMENT THIS FUNCTION
    - WRITE TESTS FOR IT
    - THINK HARD IF THE GIVEN SIGNATURE MAKES SENSE
    - UNCOMMENT THE INCLUSION OF THIS FILE IN ORTHO_TREE.HPP (BOTTOM)
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

    Kokkos::View<morton_code*> initialise_C_View(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> L_view) {
        Kokkos::View<size_t> count("count");
        Kokkos::parallel_for(
            "CountValidDescendants", B_view.extent(0), KOKKOS_LAMBDA(const size_t j) {
                morton_code octant_B = B_view(j);
                size_t local_count   = 0;

                for (size_t i = 0; i < L_view.extent(0); ++i) {
                    if (morton_helper.is_descendant(L_view(i), octant_B)) {
                        ++local_count;
                    }
                }
                Kokkos::atomic_add(&count(), local_count);
            });

        Kokkos::View<morton_code*> C_view("C_view", count());

        Kokkos::parallel_for(
            "FillCView", B_view.extent(0), KOKKOS_LAMBDA(const size_t j) {
                size_t index         = 0;
                morton_code octant_B = B_view(j);
                for (size_t i = 0; i < L_view.extent(0); ++i) {
                    if (morton_helper.is_descendant(L_view(i), octant_B)) {
                        C_view(Kokkos::atomic_fetch_add(&index, 1)) = L_view(i);
                    }
                }
            });

        return C_view;
    }

    Kokkos::View<morton_code*> initialise_D_view(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> C_view) {
        auto should_insert = [&](const morton_code octant_X) -> bool {
            const auto neighbours = morton_helper.get_neighbors(octant_X);

            return std::any_of(
                neighbours.data(), neighbours.data() + neighbours.size(),
                [&](const morton_code octant_Z) {
                    return std::any_of(
                        B_view.data(), B_view.data() + B_view.size(),
                        [&](const morton_code octant_B) {
                            // set_A = {z, ancestors of z}
                            // set_B = {ancestors of x}
                            return ((octant_Z == octant_B)
                                    || morton_helper.is_ancestor(octant_Z, octant_B))
                                   && !(morton_helper.is_ancestor(octant_X, octant_B));
                        });
                });
        };

        Kokkos::View<size_t> count("count");
        Kokkos::parallel_for(
            "CountValidOctants", C_view.extent(0), KOKKOS_LAMBDA(const size_t i) {
                if (should_insert(C_view(i))) {
                    Kokkos::atomic_increment(&count());
                }
            });

        // D = intra-proc boundaries
        Kokkos::View<morton_code*> D_view("D_view", count());

        // populate D_view
        Kokkos::View<size_t> index("index");
        Kokkos::parallel_for(
            "PopulateD", C_view.extent(0), KOKKOS_LAMBDA(const size_t i) {
                const auto octant_X = C_view(i);

                if (should_insert(octant_X)) {
                    D_view(Kokkos::atomic_fetch_add(&index(), 1);) = octant_X;
                }
            });

        return D_view;
    }
    those Kokkos::View<morton_code*> initialise_G_view(const auto& morton_helper,
                                                       Kokkos::View<morton_code*> B_view,
                                                       Kokkos::View<morton_code*> F_view) {
        auto should_insert = [&](morton_code octant_to_insert) -> bool {
            const auto neighbours = morton_helper.get_neighbors(octant_to_insert);
            return std::any_of(neighbours.data(), neighbours.data() + neighbours.size(),
                               [&](const morton_code octant_Z) {
                                   return std::any_of(B_view.data(), B_view.data() + B_view.size(),
                                                      [&](const morton_code octant_B) {
                                                          // set_A = {z, ancestors of z}
                                                          // true if: B is in set_A
                                                          return (octant_Z == octant_B)
                                                                 || morton_helper.is_ancestor(
                                                                     octant_Z, octant_B);
                                                      });
                               });
        };

        Kokkos::View<size_t> count("count");
        Kokkos::parallel_for(
            "CountValidOctants", F_view.extent(0), KOKKOS_LAMBDA(const size_t i) {
                if (should_insert(F_view(i))) {
                    Kokkos::atomic_increment(&count());
                }
            });

        Kokkos::View<morton_code*> G_view("G_view", count());

        Kokkos::View<size_t> index("index");
        Kokkos::parallel_for(
            "InsertValidOctants", F_view.extent(0), KOKKOS_LAMBDA(const size_t i) {
                auto octant_X = F(i);
                if (should_insert(octant_X)) {
                    G_view(Kokkos::atomic_fetch_add(&index, 1)) = octant_X;
                }
            });

        return G_view;
    }

    Kokkos::View<morton_code*> initialise_R_view(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> H_view,
                                                 Kokkos::View<morton_code*> F_view) {
        auto should_insert = [&](morton_code octant_to_insert) -> bool {
            const auto neighbours = morton_helper.get_neighbors(octant_to_insert);
            return std::any_of(neighbours.data(), neighbours.data() + neighbours.size(),
                               [&](const morton_code octant_Z) {
                                   return std::any_of(B_view.data(), B_view.data() + B_view.size(),
                                                      [&](const morton_code octant_B) {
                                                          // set_A = {z, ancestors of z}
                                                          // true if: B is in set_A
                                                          return (octant_to_insert == octant_B)
                                                                 || morton_helper.is_ancestor(
                                                                     octant_to_insert, octant_B);
                                                      });
                               });
        };

        Kokkos::View<size_t> count("count");
        Kokkos::parallel_for(
            "CountValidOctants", H_view.extent(0), KOKKOS_LAMBDA(const size_t i) {
                if (should_insert(H_view(i))) {
                    Kokkos::atomic_increment(&count());
                }
            });

        Kokkos::parallel_for(
            "CountValidOctants", F_view.extent(0), KOKKOS_LAMBDA(const size_t i) {
                if (should_insert(F_view(i))) {
                    Kokkos::atomic_increment(&count());
                }
            });

        Kokkos::View<morton_code*> R_view("R_view", count());

        Kokkos::View<size_t> index("index");
        Kokkos::parallel_for(
            "InsertValidOctants", H_view.extent(0), KOKKOS_LAMBDA(const size_t i) {
                auto octant_X = H_view(i);
                if (should_insert(octant_X)) {
                    R_view(Kokkos::atomic_fetch_add(&index, 1)) = octant_X;
                }
            });

        Kokkos::parallel_for(
            "InsertValidOctants", F_view.extent(0), KOKKOS_LAMBDA(const size_t i) {
                auto octant_X = F_view(i);
                if (should_insert(octant_X)) {
                    R_view(Kokkos::atomic_fetch_add(&index, 1)) = octant_X;
                }
            });

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

        const size_t comm_size = Comm->size();
        const size_t rank      = Comm->rank();

        // Step 1: Gather sizes of B_view across ranks
        size_t local_size = B_view.extent(0);
        Kokkos::View<size_t*> sizes("sizes", comm_size);

        // Gather sizes from all ranks
        Comm->allgather(&local_size, sizes.data(), 1);

        // Step 2: Compute rank_offsets using prefix sum
        Kokkos::View<size_t*> rank_offsets("rank_offsets", comm_size + 1);
        Kokkos::parallel_scan(
            "ComputeOffsets", comm_size,
            KOKKOS_LAMBDA(const int i, size_t& partial_sum, const bool final) {
                if (final)
                    rank_offsets(i) = partial_sum;
                partial_sum += sizes(i);
                if (final && i == comm_size - 1)
                    rank_offsets(comm_size) = partial_sum;
            });

        size_t total_size;
        Kokkos::deep_copy(total_size, rank_offsets(comm_size));

        // Step 3: Allocate B_glob
        Kokkos::View<morton_code*> B_glob("B_glob", total_size);

        // Step 4: Copy local data to host space
        Kokkos::View<morton_code*, Kokkos::HostSpace> local_data("local_data", local_size);
        Kokkos::deep_copy(local_data, B_view);

        // Prepare recvcounts and displacements for allgatherv
        std::vector<int> recvcounts(comm_size);
        std::vector<int> displacements(comm_size);
        for (int i = 0; i < comm_size; ++i) {
            recvcounts[i]    = static_cast<int>(sizes(i));
            displacements[i] = static_cast<int>(rank_offsets(i));
        }

        // Step 5: Use Comm::allgatherv to exchange data
        Comm->allgatherv(local_data.data(), local_size, B_glob.data(), recvcounts.data(),
                         displacements.data());

        return {B_glob, rank_offsets};
    }

    Kokkos::View<morton_code*> communicate_T_view(
        Kokkos::View<std::vector<morton_code>*> data_to_send, ippl::mpi::Communicator& Comm) {
        const size_t world_size = Comm.size();
        const size_t world_rank = Comm.rank();

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

    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::algo11(
        Kokkos::View<morton_code*> distributed_complete_tree_L) {
        // B = algo4
        Kokkos::View<morton_code*> B_view =
            block_partition(distributed_complete_tree_L(0),
                            distributed_complete_tree_L(distributed_complete_tree_L.size() - 1));

        // C = algo7(B, L)
        auto C_view = initialise_C_View(this->morton_helper, B_view, distributed_complete_tree_L);

        // D = intra proc boundaries
        Kokkos::View<morton_code*> D_view = initialise_D_view(this->morton_helper, B_view, C_view);

        // ripple propagation
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
            const std::set<morton_code> i_layer(
                insulation_layer_data.data(),
                insulation_layer_data.data() + insulation_layer_data.size());

            for (size_t rank = 0; rank < Comm->size(); ++rank) {
                if (rank == Comm->rank()) {
                    // no need to send own data
                    continue;
                }

                for (size_t i = B_glob_rank_offsets(rank);
                     (i < B_glob_view.size()) && (i < B_glob_rank_offsets(rank)); ++i) {
                    const morton_code ooctant_B_glob = B_glob_view(i);

                    if (i_layer.count(octant_B_glob) == 0) {
                        continue;
                    }

                    data_to_send1(rank).push_back(octant_B_glob);
                }
            }
        }

        Kokkos::View<morton_code*> T_view = communicate_T_view(data_to_send1);

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

                if (data_to_send1.find(octant_G) != data_to_send2.end()) {
                    continue;
                }
                // TODO:
                // if g was not sent to rank(octant_T)
                // i dont get this lol

                // do same thing as above
            }
        }

        // TODO:
        // receive the same way as above
        // K = receive
        Kokkos::View<morton_code*> K_view;

        auto H_view = algo9(concatenateViews(G_view, T_view, K_View));

        auto R_view = initialise_R_view(this->morton_helper, B_view, H_view, F_view);
        R_view = linearise_octants(R_view);

        return R_view;
    }

}  // namespace ippl

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
            "FillMinMaxView", Comm->size(), KOKKOS_LAMBDA(const int rank) {
                min_max_view(rank, 0) = gathered_data_buff[rank * 2];      // min oct
                min_max_view(rank, 1) = gathered_data_buff[rank * 2 + 1];  // max oct
            });

        return min_max_view;
    }

    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::algo11(
        Kokkos::View<morton_code*> distributed_complete_tree_L) {
        // B = algo4
        auto B_view =
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

        for (const morton_code octant_G : std::span(G_view.data(), G_view.data() + G_view.size())) {
            // using a set is probably worth it here,
            // has the inner loop is (probably
            // relatively large)
            const auto insulation_layer_data = morton_helper.get_insulation_layer(octant_G);
            const std::set<morton_code> i_layer(
                insulation_layer_data.data(),
                insulation_layer_data.data() + insulation_layer_data.size());

            // TODO:
            // algo4 octants not on this proc
            // for each b in (B_glob - B)

            // to figure out where to send octants later
            Kokkos::View<morton_code* [2]> rank_boundaries = exchange_min_max_octants(B_view);
            // copy octants that go to ranks in here
            Kokkos::View<std::vector<morton_code>*> data_to_send;

            Kokkos::View<morton_code*> B_glob;
            for (morton_code octant_B_glob : B_glob) {
                if (i_layer.count(octant_B_glob) == 0) {
                    continue;
                }

                // TODO:
                // send g, rank(octant_G) -> step 10
                size_t target_rank;  // figure out from which rank this octant is
                data_to_send(target_rank).push_back(octant_B_glob);
            }
        }

        // send data_to_send to corresponding ranks (gather?)
        // T = receive
        Kokkos::View<morton_code*> T_view;

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

                // TODO:
                // if g was not sent to rank(octant_T)
                // in step 10

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

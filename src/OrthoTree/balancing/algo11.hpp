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

    template <typename... ViewTypes>
    void appendViews(Kokkos::View<morton_code*>& destination, ViewTypes... views) {
        auto result = concatenateViews(destination, views...);

        Kokkos::resize(destination, result.extent(0));
        Kokkos::parallel_for(
            "CopyResultBack", result.extent(0),
            KOKKOS_LAMBDA(const size_t i) { destination(i) = result(i); });
    }

    Kokkos::View<morton_code*> initialise_C_View(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> L_view) {
        auto C_view = Kokkos::View<morton_code*>("C_view", 0);
        std::for_each(
            B_view.data(), B_view.data() + B_view.size(), [&](const morton_code octant_B) {
                // determine size
                size_t count = 0;
                std::for_each(L_view.data(), L_view.data() + L_view.extent(0),
                              [&](const morton_code octant_dist) {
                                  if (morton_helper.is_descendant(octant_dist, octant_B)) {
                                      ++count;
                                  }
                              });

                Kokkos::View<morton_code*> new_view("new_view", count);

                size_t index = 0;
                Kokkos::parallel_for(
                    "FillNewView", L_view.extent(0), KOKKOS_LAMBDA(const size_t i) {
                        if (morton_helper.is_descendant(L_view(i), octant_B)) {
                            new_view(index++) = L_view(i);
                        }
                    });

                auto result_View = algo7(octant_B, new_view);
                appendViews(C, result_View);
            });

        return C_view;
    }

    Kokkos::View<morton_code*> initialise_D_view(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> C_view) {
        // D = intra-proc boundaries
        Kokkos::View<morton_code*> D_view("D_view", 0);

        // Temporary view to store valid candidates
        Kokkos::View<size_t> count("count");

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

        // count valid octants
        std::for_each(C_view.data(), C_view.data() + C_view.size(),
                      [&](const morton_code octant_X) {
                          if (should_insert(octant_X)) {
                              Kokkos::atomic_increment(&count());
                          }
                      });

        Kokkos::resize(D_view, count());

        // populate D_view
        size_t index = 0;
        Kokkos::parallel_for(
            "PopulateD", C_view.extent(0), KOKKOS_LAMBDA(const size_t i) {
                const auto octant_X = C_view(i);

                if (should_insert(octant_X)) {
                    D_view(index++) = octant_X;
                }
            });

        return D_view;
    }

    Kokkos::View<morton_code*> initialise_G_view(const auto& morton_helper,
                                                 Kokkos::View<morton_code*> B_view,
                                                 Kokkos::View<morton_code*> F_view) {
        auto should_insert = [&](morton_code octant_to_insert) {
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

        size_t count = 0;
        std::for_each(F_view.data(), F_view.data() + F_view.size(),
                      [&](const morton_code octant_X) {
                          if (should_insert(octant_X)) {
                              count++;
                          }
                      });

        Kokkos::View<morton_code*> G_view("G_view", count);

        size_t index = 0;
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
        auto should_insert = [&](morton_code octant_to_insert) {
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

        size_t count = 0;
        std::for_each(H_view.data(), H_view.data() + H_view.size(),
                      [&](const morton_code octant_X) {
                          if (should_insert(octant_X)) {
                              count++;
                          }
                      });

        std::for_each(F_view.data(), F_view.data() + F_view.size(),
                      [&](const morton_code octant_X) {
                          if (should_insert(octant_X)) {
                              count++;
                          }
                      });

        Kokkos::View<morton_code*> R_view("R_view", count);

        size_t index = 0;
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
            for (morton_code octant_B : B) {
                if (i_layer.count(octant_B) == 0) {
                    continue;
                }

                // TODO:
                // send g, rank(octant_G) -> step 10
            }
        }

        // TODO:
        // T = receive

        for (const morton_code octant_G : std::span(G_view.data(), G_view.data() + G_view.size())) {
            for (const morton_code octant_T :
                 std::span(T_View.data(), T_View.data() + T_View.size())) {
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
            }
        }

        // TODO:
        // K = receive

        auto H_view = algo9(concatenateViews(G_view, T_View, K_View));

        Kokkos::View<morton_code*> R_view =
            initialise_R_view(this->morton_helper, B_view, H_view, F_view);
        R_view = linearise_octants(R_view);

        return R_view;
    }

}  // namespace ippl

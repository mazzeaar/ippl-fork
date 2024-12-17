#include "../OrthoTree.h"

namespace ippl {
    /*
    TODO:
    - IMPLEMENT THIS FUNCTION
    - WRITE TESTS FOR IT
    - THINK HARD IF THE GIVEN SIGNATURE MAKES SENSE
    - UNCOMMENT THE INCLUSION OF THIS FILE IN ORTHO_TREE.HPP (BOTTOM)
    */

    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::algo11(
        Kokkos::View<morton_code*> distributed_complete_tree_L) {
        const morton_code min_octant = distributed_complete_tree_L(0);
        const morton_code max_octant =
            distributed_complete_tree_L(distributed_complete_tree_L.size() - 1);
        // B = algo4
        auto B = block_partition(min_octant, max_octant);

        // C = algo7
        // B is coarse and (should) not have many blocks
        // -> for each block in B call algo7(block_B, descendants(dist_L))?
        auto C = algo7(B, distributed_complete_tree_L);  // wrong func params?

        // D = intra proc boundaries
        Kokkos::View<morton_code*> D;
        std::for_each(C.data(), C.data() + C.size(), [&, this](const morton_code octant_X) {
            const auto neighbours    = morton_helper.get_neighbors(octant_X);
            const bool should_insert = std::any_of(
                neighbours.data(),
                neighbours.data() + neighbours.size()[&, this](const morton_code octant_Z) {
                    return std::any_of(
                        B.data(), B.data() + B.size(), [&, this](const morton_code octant_B) {
                            // set_A = {z, ancestores of z}
                            // set_B = {ancestores of x}
                            // true if: B is in A, but not in B?
                            return ((octant_Z == octant_B)
                                    || morton_helper.is_ancestor(octant_Z, octant_B))
                                   && !(morton_helper.is_ancestor(octant_X, octant_B));
                        });
                });

            if (should_insert) {
                // TODO:
                // insert octant_X into D
            }
        });

        // ripple propagation
        auto S = algo9(D);

        Kokkos::View<morton_code*> C_u_S("C_u_S", S.size() + C.size());
        Kokkos::parallel_for(
            "Copy_C", C.size(), KOKKOS_LAMBDA(const size_t i) { C_u_S(i) = C(i); });
        Kokkos::parallel_for(
            "Copy_S", S.size(), KOKKOS_LAMBDA(const size_t i) { C_u_S(i + C.size()) = S(i); });

        auto F = linearise_octants(C_u_S);

        // G = inter proc boundaries
        Kokkos::View<morton_code*> G;
        std::for_each(F.data(), F.data() + F.size(), [&, this](const morton_code octant_X) {
            const auto neighbours    = morton_helper.get_neighbors(octant_X);
            const bool should_insert = std::any_of(
                neighbours.data(),
                neighbours.data() + neighbours.size()[&, this](const morton_code octant_Z) {
                    return std::any_of(B.data(), B.data() + B.size(),
                                       [&, this](const morton_code octant_B) {
                                           // set_A = {z, ancestores of z}
                                           // true if: B is in set_A
                                           return (octant_Z == octant_B)
                                                  || morton_helper.is_ancestor(octant_Z, octant_B);
                                       });
                });

            if (should_insert) {
                // TODO:
                // insert octant_X into G
            }
        });

        for (const morton_code octant_G : std::span(G.data(), G.data() + G.size())) {
            // using a set is probably worth it here, has the inner loop is (probably relatively
            // large)
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

        for (const morton_code octant_G : std::span(G.data(), G.data() + G.size())) {
            for (const morton_code octant_T : std::span(T.data(), T.data() + T.size())) {
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
                // if g was not sent to rank(octant_T) in step 10
            }
        }

        // K = receive
        Kokkos::View<morton_code*> G_u_T_u_K("G_u_T_u_K", G.size() + T.size() + K.size());
        Kokkos::parallel_for(
            "Copy_G", C.size(), KOKKOS_LAMBDA(const size_t i) { G_u_T_u_K(i) = G(i); });
        Kokkos::parallel_for(
            "Copy_T", S.size(), KOKKOS_LAMBDA(const size_t i) { G_u_T_u_K(i + G.size()) = T(i); });
        Kokkos::parallel_for(
            "Copy_K", S.size(),
            KOKKOS_LAMBDA(const size_t i) { G_u_T_u_K(i + G.size() + T.size()) = K(i); });

        auto H = algo9(G_u_T_u_K);

        // probably not necessary, just do loop over H, then over F
        Kokkos::View<morton_code*> H_u_F("H_u_F", H.size() + F.size());
        Kokkos::parallel_for(
            "Copy_H", C.size(), KOKKOS_LAMBDA(const size_t i) { H_u_F(i) = H(i); });
        Kokkos::parallel_for(
            "Copy_F", S.size(), KOKKOS_LAMBDA(const size_t i) { H_u_F(i + H.size()) = F(i); });

        Kokkos::View<morton_code*> R;
        std::for_each(
            H_u_F.data(), H_u_F.data() + H_u_F.size(), [&, this](const morton_code octant_X) {
                const auto neighbours    = morton_helper.get_neighbors(octant_X);
                const bool should_insert = std::any_of(
                    neighbours.data(),
                    neighbours.data() + neighbours.size()[&, this](const morton_code octant_Z) {
                        return std::any_of(
                            B.data(), B.data() + B.size(), [&, this](const morton_code octant_B) {
                                // set_A = {x, ancestores of x}
                                // true if: B is in set_A
                                return (octant_X == octant_B)
                                       || morton_helper.is_ancestor(octant_X, octant_B);
                            });
                    });

                if (should_insert) {
                    // TODO:
                    // insert octant_X into R
                }
            });

        R = linearise_octants(R);
        return R;
    }

}  // namespace ippl

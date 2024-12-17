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
        auto B_view = block_partition(min_octant, max_octant);

        // C = algo7
        // B_view is coarse and (should) not have many blocks
        // -> for each block in B call algo7(block_B, descendants(dist_L))?
        auto C = algo7(B_view, distributed_complete_tree_L);  // wrong func params?

        // D = intra proc boundaries

        // ripple propagation
        auto S = algo9(D);

        Kokkos::View<morton_code*> C_u_S("C_u_S", S.size() + C.size());
        Kokkos::parallel_for(
            "Copy_C", C.size(), KOKKOS_LAMBDA(const size_t i) { C_u_S(i) = C(i); });
        Kokkos::parallel_for(
            "Copy_S", S.size(), KOKKOS_LAMBDA(const size_t i) { C_u_S(i + C.size()) = S(i); });

        auto F = linearise(C_u_S);

        // G = inter proc boundaries

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

                // send g, rank(octant_G) -> step 10
            }
        }

        // T = receive

        for (const morton_code octant_G : std::span(G.data(), G.data() + G.size())) {
            for (const morton_code octant_T : std::span(T.data(), T.data() + T.size())) {
                // if octant_G in insulation layer of octant_T
                //   if g was not sent to rank(octant_T) in step 10

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

                // TODO: CONTINUE HERE
                if (is_in_insulation_layer(octant_G, octant_T))
            }
        }

        // K = receive
        // H = Ripple(G u T u K)
        // R = stuff
        // R = algo8(R)
        // return R
    }

}  // namespace ippl

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
        auto C = algo7(B_view, distributed_complete_tree_L);  // wrong func params?

        // D = intra proc boundaries

        // ripple propagation
        auto S = algo9(D);

        Kokkos::View<morton_code*> C_n_S("C_n_S", S.size() + C.size());
        Kokkos::parallel_for(
            "Copy_C", C.size(), KOKKOS_LAMBDA(const size_t i) { C_n_S(i) = C(i); });
        Kokkos::parallel_for(
            "Copy_S", S.size(), KOKKOS_LAMBDA(const size_t i) { C_n_S(i + C.size()) = S(i); });

        // F = linearise(C u S)
        auto F = linearise(C_n_S);

        // G = inter proc boundaries

        for (const morton_code octant_G : std::span(G.data(), G.data() + G.size())) {
            // algo4 octants not on this proc
            // for each b in (B_glob - B)
            //   if b in insulation layer (octant_G)
            //      send g, rank(octant_G) -> step 10
        }

        // T = receive

        for (const morton_code octant_G : std::span(G.data(), G.data() + G.size())) {
            for (const morton_code octant_T : std::span(T.data(), T.data() + T.size())) {
                // if octant_G in insulation layer of octant_T
                //   if g was not sent to rank(octant_T) in step 10
            }
        }

        // K = receive
        // H = Ripple(G u T u K)
        // R = stuff
        // R = algo8(R)
        // return R
    }

}  // namespace ippl

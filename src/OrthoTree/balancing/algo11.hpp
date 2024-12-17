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
        // B = algo4
        // C = algo7
        // D = stuff
        // S = algo9
        // F = linearise(C U S)
        // G = stuff

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

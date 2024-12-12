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
    Kokkos::View<morton_code*> OrthoTree<Dim>::algo9(
        Kokkos::View<morton_code*> sorted_incomplete_tree_L)
    {
        // W <- L
        Kokkos::View<morton_code*> balanced_incomplete_tree = sorted_incomplete_tree_L;

        // for l <- D_max to (L(N) + 1)
        for ()
        {
            // for each w in W
                // if L(w) = l
                    // K <- search_keys(w)
                    // (B, J) <- maximum_lower_bound (K, W)
                        // ( J is the index of B in W )
                    // for each (b, j) in (B, J) | l > (L(b) + 1) & b in A(K)
                        // T[j] <- T[j] + ({N^s(w, (l - 1) ) & {A(K)})
                    // end for
                // end if
            // end for
            // for i <- 1 to len(W)
                // if T[i] != emptySet
                    // R <- R + CompleteSubtree(W[i], T[i]) (Algorithm 10)
                // else
                    // R <- R + W[i]
                // end if
            // end for
            // W <- R, T, R <- emptySet
        // end for
        }
        return balanced_incomplete_tree;
    }

}  // namespace ippl
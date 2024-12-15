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
    Kokkos::View<morton_code*> OrthoTree<Dim>::algo10(
        const morton_code octant_N, Kokkos::View<morton_code*> partial_descendants_L) {
        const size_t depth_N = morton_helper.get_depth(octant_N);
        Kokkos::View<morton_code*> W(partial_descendants_L.data(), partial_descendants_L.size());
        Kokkos::View<morton_code*> R;

        // this is a set because we need unique morton codes
        std::unordered_set<morton_code> P;

        /**
         * Key idea here: we only resize W when we actually need to. To achieve this we store
         * multiple different sizes of W. this way we dont have to resize W all the time (expensive)
         * and we can keep using our allocated memory for as long as possible. We achieve this by
         * swapping unused values we would otherwise remove from W to the (dynamic) back of it,
         * indicated by W_dymanic_size. We mainly use std::vector<> in this loop, because we can
         * easily resize them and std::vector<>::clear() does not deallocate the memory, meaning we
         * (probably) rarely have to resize the vector.
         */
        size_t W_actual_size  = W.size();
        size_t W_dynamic_size = W_actual_size;
        for (size_t depth = max_depth_m; depth <= depth_N + 1; ++depth) {
            // Q = all octants in W at depth depth
            std::vector<morton_code> Q;
            std::for_each(W.data(), W.data() + W.size(),
                          [morton_helper, depth, &Q](const morton_code& oct) {
                              if (morton_helper.get_depth(oct) == depth) {
                                  Q.push_back(oct);
                              }
                              morton_helper.get_parent(oct);
                          });

            // T = all octants in Q s.t. a sibling of Q is not yet contained in T
            std::vector<morton_code> T;
            std::for_each(Q.data(), Q.data() + Q.size(),
                          [morton_helper, &T](const morton_code& oct) {
                              // naive implementation, can be done smarter later
                              const bool contains_sibling =
                                  std::any_of(T.data(), T.data() + T.size(),
                                              [morton_helper, oct](const morton_code& T_oct) {
                                                  return morton_helper.is_sibling(oct, T_oct);
                                              });

                              if (!contains_sibling) {
                                  T.push_back(oct);
                              }
                          });

            const size_t T_size = T.size();
            for (size_t i = 0; i < T_size; ++i) {
                const morton_code octant_t = T[i];
                R.append_range(morton_helper.get_siblings(octant_t));

                // potential issue: is the parent included in here or not?
                // fun fact: this is the collective noun for aunts and uncles:)
                const auto avunculi = morton_helper.get_siblings(morton_helper.get_parent(t));
                for (auto titi : avunculi) {
                    P.insert(titi);
                }
            }

            // basically: we decrease the size of W, then we only have to increase it by the surplus
            // added by P (if P is larger than W)
            const size_t W_old_size = W_dynamic_size;
            size_t W_cur_size       = W_old_size;
            for (size_t i = 0; i < W_cur_size;) {
                const size_t test_depth = morton_helper.get_depth(W[i]);
                if (test_depth == depth - 1) {
                    P.insert(W[i]);
                    std::swap(W[i], W[--W_cur_size]);
                } else {
                    ++i;
                }
            }

            W_dynamic_size += (P.size() - (W_old_size - W_cur_size));
            if (W_dynamic_size > W_actual_size) {
                // resize W
            }

            for (auto P_it = P.begin(); P_it != P.end(); ++P_it, ++W_cur_size) {
                const morton_code P_oct = *P_it;
                W[W_cur_size]           = P_oct;
            }

            P.clear();
        }

        std::sort(R.data(), R.data() + R.size());
        R = linearise_octants(R);
        return R;
    }
}  // namespace ippl

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
        morton_code octant_N, Kokkos::View<morton_code*> partial_descendants_L) {
        std::vector<morton_code> W;
        std::vector<morton_code> R;

        // this is a set because we have to remove duplicates from it
        std::unordered_set<morton_code> P;

        for (size_t depth = max_depth_m; depth <= morton_helper.get_depth(octant_N); ++depth) {
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
                for (auto uncle : avunculi) {
                    P.insert(uncle);
                }
            }

            // basically: we decrease the size of W, then we only have to increase it by the surplus
            // added by P (if P is larger than W)
            const size_t W_old_size = W.size();
            size_t W_cur_size       = W.size();
            for (size_t i = 0; i < W_cur_size;) {
                const size_t test_depth = morton_helper.get_depth(W[i]);
                if (test_depth == depth - 1) {
                    P.insert(W[i]);
                    std::swap(W[i], W[--W_cur_size]);
                } else {
                    ++i;
                }
            }

            // resize W
            size_t W_new_size = W.append_range(P);
            P.clear();
        }

        std::sort(R.data(), R.data() + R.size());
        R = linearise_octants(R);
        return R;
    }
}  // namespace ippl

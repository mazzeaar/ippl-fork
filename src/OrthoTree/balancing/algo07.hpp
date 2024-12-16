#include "../OrthoTree.h"
#include "algo_7_10_base.hpp"

namespace ippl {
    /*
    TODO:
    - IMPLEMENT THIS FUNCTION
    - WRITE TESTS FOR IT
    - THINK HARD IF THE GIVEN SIGNATURE MAKES SENSE
    - UNCOMMENT THE INCLUSION OF THIS FILE IN ORTHO_TREE.HPP (BOTTOM)
    */

    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::algo7(
        morton_code octant_N, Kokkos::View<morton_code*> partial_descendants_L) {
        std::function<void(Kokkos::View<morton_code*>&, size_t&, const morton_code)> R_function =
            [this](auto& R_view, size_t& R_index, const morton_code octant) {
                const auto siblings        = this->morton_helper.get_siblings(octant);
                const size_t siblings_size = siblings.size();

                if (R_index + siblings_size >= R_view.size()) {
                    Kokkos::resize(R_view, 2 * siblings_size);
                }

                for (morton_code sibling : siblings) {
                    R_view[R_index] = sibling;
                    R_index++;
                }
            };

        std::function<void(std::unordered_set<morton_code>&, const morton_code)> P_function =
            [this](auto& P_set, const morton_code octant) {
                // potential issue: should the parent be included in here or not?

                const morton_code parent = this->morton_helper.get_parent(octant);
                const auto neighbors     = this->morton_helper.get_neighbors(parent);

                for (morton_code neighbor : neighbors) {
                    P.insert(neighbor);
                }

                // P ← P + N (P(t), l − 1)
            };

        return algo_7_10_base(R_function, P_function, octant_N, partial_descendants_L);
    }

}  // namespace ippl

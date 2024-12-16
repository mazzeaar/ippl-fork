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
        return algo_7_10_base<7>(octant_N, partial_descendants_L);
    }

}  // namespace ippl

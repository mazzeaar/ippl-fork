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
    Kokkos::View<morton_code*,Kokkos::HostSpace::memory_space> OrthoTree<Dim>::algo9(
        Kokkos::View<morton_code*,Kokkos::HostSpace::memory_space> sorted_incomplete_tree_L);

}  // namespace ippl
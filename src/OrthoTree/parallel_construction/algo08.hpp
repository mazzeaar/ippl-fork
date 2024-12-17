#include "../OrthoTree.h"

/*
TODO:
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::linearise_octants(
        const Kokkos::View<morton_code*>& octants) {
        assert(octants.size() > 0
               && "Octants.size() is zero, dont call this function with an empty list!");

        Kokkos::View<morton_code*> linearised("linearised", octants.size());

        size_t j = 0;
        for (size_t i = 0; i < octants.size() - 1; ++i) {
            if (morton_helper.is_ancestor(octants[i + 1], octants[i])) {
                continue;
            }

            linearised[j] = octants[i];
            ++j;
        }

        linearised[j] = octants[octants.size() - 1];
        Kokkos::resize(linearised, j+1);

        return linearised;
    }
}  // namespace ippl
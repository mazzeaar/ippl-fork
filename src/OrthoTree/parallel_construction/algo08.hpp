#include "../OrthoTree.h"

/*
TODO:
- IMPLEMENT THIS NEW FUNCTION SIGNATURE
- WRITE TESTS FOR THE FUNCTION
- ADJUST THE SIGNATURE IN ORTHOTREE.H

namespace ippl {

    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::linearise_octants(
                                        Kokkos::View<morton_code*> octants);

}  // namespace ippl
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::linearise_octants(
        const Kokkos::View<morton_code*>& octants) {
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

    // wrapper to make it work with old algos, remove this
    template <size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::linearise_octants(
        const Kokkos::vector<morton_code>& octants) {
        Kokkos::View<morton_code*> linearise_view(octants.data(), octants.size());
        auto res = linearise_octants(linearise_view);
        Kokkos::vector<morton_code> vec_res;
        for (size_t i = 0; i < linearise_view.size(); ++i) {
            vec_res.push_back(linearise_view[i]);
        }

        return vec_res;
    }
}  // namespace ippl
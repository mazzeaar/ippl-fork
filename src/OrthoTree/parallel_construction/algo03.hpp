#include <span>

#include "../OrthoTree.h"
/*
TODO:
- IMPLEMENT THIS NEW FUNCTION SIGNATURE
- WRITE TESTS FOR THE FUNCTION
- ADJUST THE SIGNATURE IN ORTHOTREE.H

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::complete_tree(Kokkos::View<morton_code*>& octants);
}  // namespace ippl
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::complete_tree(Kokkos::View<morton_code*> octants) {
        // this removes duplicates, inefficient as of now
        std::map<morton_code, int> m;
        for (auto octant : std::span(octants.data(), octants.size())) {
            ++m[octant];
        }

        Kokkos::resize(octants, m.size());  // shrink
        size_t octants_idx = 0;
        for (const auto [octant, count] : m) {
            octants[octants_idx] = octant;
            octants_idx++;
        }

        octants = linearise_octants(octants);

        Kokkos::View<size_t*> weights_view("weights_view", octants.size());
        for (size_t i = 0; i < octants.size(); ++i) {
            weights_view[i] = 1;
        }

        octants = partition(octants, weights_view);

        Kokkos::resize(octants, octants.size() + 1);

        morton_code first_rank0;
        if (world_rank == 0) {
            const morton_code dfd_root = morton_helper.get_deepest_first_descendant(morton_code(0));
            const morton_code A_finest =
                morton_helper.get_nearest_common_ancestor(dfd_root, octants[0]);

            first_rank0 = morton_helper.get_first_child(A_finest);
        } else if (world_rank == world_size - 1) {
            const morton_code dld_root = morton_helper.get_deepest_last_descendant(morton_code(0));
            const morton_code A_finest =
                morton_helper.get_nearest_common_ancestor(dld_root, octants[octants.size() - 2]);
            const morton_code last_child = morton_helper.get_last_child(A_finest);

            octants[octants.size() - 1] = last_child;
        }

        if (world_rank > 0) {
            Comm->send(*octants.data(), 1, world_rank - 1, 0);
        }

        morton_code buff;
        if (world_rank < world_size - 1) {
            mpi::Status status;
            Comm->recv(&buff, 1, world_rank + 1, 0, status);

            // do we need a status check here or not?
            octants[octants.size() - 1] = buff;
        }

        size_t R_base_size = 100;
        size_t R_index     = 0;
        Kokkos::View<morton_code*> R_view("R_view", R_base_size);

        auto insert_into_R = [&](morton_code octant_a, morton_code octant_b) {
            auto complete_region_view       = complete_region(octant_a, octant_b);
            const size_t additional_octants = complete_region_view.size() + 1;
            size_t remaining_space          = R_view.size() - R_index;

            while (remaining_space <= additional_octants) {
                Kokkos::resize(R_view, R_view.size() + R_base_size);
                remaining_space = R_view.size() - R_index;
            }

            R_view[R_index] = octant_a;
            R_index++;

            for (morton_code elem :
                 std::span(complete_region_view.data(), complete_region_view.size())) {
                R_view[R_index] = elem;
                R_index++;
            }
        };

        if (world_rank == 0) {
            // special case for rank 0, as we push_front'ed earlier
            insert_into_R(first_rank0, octants[0]);
        }

        for (size_t i = 0; i < octants.size() - 1; ++i) {
            insert_into_R(octants[i], octants[i + 1]);
        }

        if (world_rank == world_size - 1) {
            const size_t R_size = R_view.size();
            // we can either be smaller or have a perfect fit, larger (should) not be possible
            if (R_index + 1 < R_size) {
                // shrink
                Kokkos::resize(R_view, R_index + 1);
            } else if (R_index + 1 > R_size) {
                // this is not possible
                assert(false && "how the fuck did we get here?");
            }

            // insert to the back
            R_view[R_index] = octants[octants.size() - 1];
            R_index++;

        } else {
            Kokkos::resize(R_view, R_index);
        }

        return R_view;
    }
}  // namespace ippl
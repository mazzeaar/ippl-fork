#include <algorithm>
#include <span>

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
    inline Kokkos::View<morton_code*> OrthoTree<Dim>::algo9(
        Kokkos::View<morton_code*> sorted_incomplete_tree_L) {

        // Enforce that L is sorted
        if (!std::is_sorted(
            sorted_incomplete_tree_L.data(), 
            sorted_incomplete_tree_L.data() + sorted_incomplete_tree_L.size())) 
        {
            logger << "Input data is not sorted in algo9" << endl;
            throw std::runtime_error("data not sorted in algo9");
        }

        // W <- L balanced_incomplete_tree
        Kokkos::View<morton_code*> W = sorted_incomplete_tree_L;

        size_t R_base_size = 100;
        size_t R_index     = 0;
        Kokkos::View<morton_code*> R_view("R_view", sorted_incomplete_tree_L.size() + R_base_size);
        // for l <- D_max to (L(N) + 1)
        for (size_t depth = max_depth_m; depth >= 3; depth--) {
            Kokkos::View<std::vector<morton_code>*> T("T", W.size());
            // for each w in W
            std::for_each(
                W.data(), W.data() + W.size(), [&, this](const morton_code octant) {
                    if (this->morton_helper.get_depth(octant) != depth) {
                        return;
                    }

                    const auto search_keys = this->morton_helper.get_search_keys(octant);
                    std::for_each(
                        search_keys.begin(), search_keys.end(),
                        [&, this](const morton_code current_key) {
                            const auto neighbor_it =
                                std::lower_bound(W.data(), W.data() + W.size(), current_key);
                            morton_code neighbor = *neighbor_it;
                            size_t neighbor_idx  = neighbor_it - search_keys.data();

                            if (this->morton_helper.get_depth(neighbor) > depth - 1
                                && this->morton_helper.is_ancestor(current_key, neighbor)) {
                                T(neighbor_idx)
                                    .push_back(this->morton_helper.get_parent(current_key));
                            }
                        });
                });

            auto insert_into_R = [&](morton_code octant_a, Kokkos::View<morton_code*> T_view) {
                auto complete_subtree_view = algo10(octant_a, T_view);

                const size_t additional_octants = complete_subtree_view.size() + 1;
                size_t remaining_space          = R_view.size() - R_index;

                while (remaining_space <= additional_octants) {
                    Kokkos::resize(R_view, R_view.size() + R_base_size);
                    remaining_space = R_view.size() - R_index;
                }

                for (morton_code elem :
                     std::span(complete_subtree_view.data(), complete_subtree_view.size())) {
                    R_view[R_index] = elem;
                    R_index++;
                }
            };

            for (size_t i = 0; i < W.size(); i++) {
                if (T(i).size() != 0) {
                    Kokkos::View<morton_code*> T_view(T(i).data(), T(i).size());
                    insert_into_R(W(i), T_view);
                    T(i).clear();
                } else {
                    if (R_view.size() == R_index) {
                        Kokkos::resize(R_view, R_view.size() + R_base_size);
                    }

                    R_view[R_index] = W(i);
                    R_index++;
                }
            }

            Kokkos::resize(R_view, R_index);

            R_index = 0;
            W       = R_view;
        }

        return W;
    }

}  // namespace ippl

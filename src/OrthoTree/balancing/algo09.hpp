#include <algorithm>
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

        Kokkos::View<std::vector<morton_code>> T("T", sorted_incomplete_tree_L.size());

        size_t R_base_size = 100;
        size_t R_index     = 0;
        Kokkos::View<morton_code*> R_view("R_view", sorted_incomplete_tree_L.size() + R_base_size);
        // for l <- D_max to (L(N) + 1)
        for (depth = max_depth_m; depth >= 1; depth--)
        {
          for (auto& octant : balanced_incomplete_tree) {
             if (morton.get_depth(octant) != depth) {
                 continue;
             }

             auto key = search_keys(octant);
             for (size_t i = 0; i < key.size(); i++) {
                 auto it = std::lower_bound(sorted_incomplete_tree_L.begin(), sorted_incomplete_tree_L.end(), key[i]);
                 morton_code neighbor = *it;
                 size_t neighbor_idx(i) = it - key.data();
                 if (morton.get_depth(neighbor) > depth - 1 && morton.is_ancestor(key[i], neighbor)) {
                       T(neighbor_idx).push_back(morton.get_parent(key[i])); 
                 
                 }

             
             }
          }
          auto insert_into_R = [&](morton_code octant_a, Kokkos::View<morton_code*> T_view) {
              // TODO change this line to use algo10
              auto complete_subtree_view       = algo10(octant_a, T_view);

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
          for (size_t i = 0; i < R_base_size; i++) {
              if (T(i).size() != 0) {
                  Kokkos::View<morton_code*> T_view(T(i).data(), T(i).size());
                  insert_into_R(balanced_incomplete_tree(i), T_view);
                  T(i).clear();
              } else {
                  if (R_view.size() == R_index) {
                      Kokkos::resize(R_view, R_view.size() + R_base_size);
                  }
                  R_view[R_index] = balanced_incomplete_tree(i);
                  R_index++;
              }
          }
          R_index = 0;
          balanced_incomplete_tree = R_view;
        }
        return balanced_incomplete_tree;
    }

}  // namespace ippl

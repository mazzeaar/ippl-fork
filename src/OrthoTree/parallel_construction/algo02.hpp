#include "../OrthoTree.h"

/*
TODO:
- WRITE TESTS FOR THE FUNCTION

*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::complete_region(morton_code code_a,
                                                               morton_code code_b) {
        assert(code_a < code_b);

        // special case (not specified in the paper): 
        // one code is an ancestor of the other 
        // -> the bigger code is already the region, don't need to complete anything
        if (morton_helper.is_ancestor(code_a, code_b)
            || morton_helper.is_ancestor(code_b, code_a)) {
            Kokkos::View<morton_code*> min_lin_tree_empty("empty min_lin_tree", 0);
            return min_lin_tree_empty;
        }

        size_t estimated_size = 79;  // should never have to resize with this
        Kokkos::View<morton_code*> min_lin_tree("min_lin_tree", estimated_size);
        size_t idx = 0;

        const morton_code nearest_comm_ancestor =
            morton_helper.get_nearest_common_ancestor(code_a, code_b);

        std::stack<morton_code> stack;
        for (morton_code child : morton_helper.get_children(nearest_comm_ancestor)) {
            stack.push(child);
        }

        while (!stack.empty()) {
            morton_code current_node = stack.top();
            stack.pop();

            bool is_between_a_b   = (code_a < current_node) && (current_node < code_b);
            bool is_ancestor_of_a = morton_helper.is_ancestor(code_a, current_node);
            bool is_ancestor_of_b = morton_helper.is_ancestor(code_b, current_node);

            if (is_between_a_b && !is_ancestor_of_b) {
                if (idx >= min_lin_tree.size()) {
                    // conservative resizing
                    Kokkos::resize(min_lin_tree, min_lin_tree.size() + estimated_size);
                }
                min_lin_tree[idx] = current_node;
                idx++;
            } else if (is_ancestor_of_a || is_ancestor_of_b) {
                for (morton_code child : morton_helper.get_children(current_node)) {
                    stack.push(child);
                }
            }
        }

        // shrink to fit, if necessary
        if (idx != min_lin_tree.size()) {
            Kokkos::resize(min_lin_tree, idx);
        }

        std::sort(min_lin_tree.data(), min_lin_tree.data() + min_lin_tree.size());
        return min_lin_tree;
    }
}  // namespace ippl
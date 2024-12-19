#include <algorithm>
#include <cmath>
#include "MortonHelper.h"

namespace ippl {

    template <size_t Dim>
    morton_code Morton<Dim>::encode(const real_coordinate& coordinate, const real_coordinate& rasterizer, const size_t depth) const
    {
        grid_coordinate grid;

        // also: we use two for loops because encode is called instead of encoding directly.
        for ( size_t i = 0; i < Dim; ++i ) {
            // this will probably break with negative values or if 0 is not min of bounding box
            const double normalised_coords = (coordinate[i] / rasterizer[i]);
            grid[i] = static_cast<morton_code>(normalised_coords);
        }

        return encode(grid, depth);
    }

    template <size_t Dim>
    inline morton_code Morton<Dim>::encode(const grid_coordinate& coordinate, const size_t depth) const
    {
        morton_code code = 0;
        for ( size_t i = 0; i < Dim; ++i ) {
            const morton_code spread_bits = spread_coords(coordinate[i]);
            code |= (spread_bits << i);
        }

        code = (code << depth_mask_shift) | depth;
        return code;
    }

    template <size_t Dim>
    inline Morton<Dim>::grid_coordinate Morton<Dim>::decode(morton_code code) const
    {
        code = code >> depth_mask_shift; // remove depth information
        grid_coordinate grid_pos;
        for ( size_t i = 0; i < Dim; ++i ) {
            // limit amount of traversed bits
            for ( size_t bit = 0; bit < (sizeof(morton_code) * 8) / Dim; ++bit ) {
                const morton_code cur_bit = (code >> ((bit * Dim) + i)) & 1ULL;
                grid_pos[i] |= (cur_bit << bit);
            }
        }

        return grid_pos;
    }

    template <size_t Dim>
    inline size_t Morton<Dim>::get_depth(morton_code code) const
    {
        return code & depth_mask;
    }

    template <size_t Dim>
    inline morton_code Morton<Dim>::get_parent(morton_code code) const
    {
        assert(code != morton_code(0) && "root has not parent");

        const morton_code depth             = get_depth(code);
        const morton_code parent_depth_bits = depth - 1;

        // the first part removes irellevant bits (basically only keeping bits that ALL descendants of a code share with its ancestor)
        // the last part removes the depth bits
        const morton_code cur_shift = (Dim * (max_depth - depth + 1)) + depth_mask_shift;

        // remove all irrelevant bits
        const morton_code parent_code = code >> cur_shift;

        // shift back, resulting in zeros, add parent depth information back in
        return (parent_code << cur_shift) | parent_depth_bits;
    }

    template <size_t Dim>
    inline morton_code Morton<Dim>::get_parent_at_level(morton_code code, morton_code depth) const
    {
        assert(code != morton_code(0) && "root has not parent");

        const morton_code code_depth             = get_depth(code);
        assert(code_depth >= depth && "can't get a parent at a level finer than the current node");

        const morton_code parent_depth_bits = depth;

        // the first part removes irellevant bits (basically only keeping bits that ALL descendants of a code share with its ancestor)
        // the last part removes the depth bits
        const morton_code cur_shift = (Dim * (max_depth - depth)) + depth_mask_shift;

        // remove all irrelevant bits
        const morton_code parent_code = code >> cur_shift;

        // shift back, resulting in zeros, add parent depth information back in
        return (parent_code << cur_shift) | parent_depth_bits;
    }

    template <size_t Dim>
    inline vector_t<morton_code> Morton<Dim>::get_children(morton_code code) const
    {
        /*
        std::string error = std::string("RANK: ") + std::to_string(Comm->rank()).c_str()
                            + std::string(" can't get the first child at the deepest level");
        if (get_depth(code) >= max_depth) {
            std::cerr << "ERROR HERE:    " << error << std::endl;
        }
        assert(get_depth(code) < max_depth && "can't get the first child at the deepest level");
        */

        const morton_code first_child = get_first_child(code);

        // each level has a distinctive step size between siblings, this can maybe be improved upon
        const morton_code step = get_step_size(first_child);

        vector_t<morton_code> vec;
        vec.reserve(n_children);

        for ( size_t i = 0; i < n_children; ++i ) {
            vec.push_back(get_nth_child(code, i));
        }

        return vec;
    }

    template <size_t Dim>
    inline vector_t<morton_code> Morton<Dim>::get_siblings(morton_code code) const
    {
        return get_children(get_parent(code));
    }

    template <size_t Dim>
    inline bool Morton<Dim>::is_descendant(morton_code child, morton_code parent) const
    {

        // child has to be finer than parent
        if ( get_depth(child) <= get_depth(parent) ) return false;

        // descendants are always larger than their parents
        if ( child <= parent ) return false;

        const morton_code step = get_step_size(parent);
        const morton_code next_neighbour = parent + step;

        // if the child is a descendant the parent, is must be smaller than 
        // the parent's next bigger neighbour at the level of the parent
        return child < (next_neighbour);
    }

    template <size_t Dim>
    inline bool Morton<Dim>::is_ancestor(morton_code child, morton_code parent) const
    {
        return is_descendant(child, parent);
    }

    template <size_t Dim>
    inline morton_code Morton<Dim>::get_first_child(morton_code code) const
    {
        /*
        std::string error = std::string("RANK: ") + std::to_string(Comm->rank()).c_str()
                            + std::string(" can't get the first child at the deepest level");
        if (get_depth(code) >= max_depth) {
            std::cerr << "ERROR HERE:    " << error << std::endl;
        }
        assert(get_depth(code) < max_depth && "can't get the first child at the deepest level");
        */
        return code + 1;
    }

    template <size_t Dim>
    inline morton_code Morton<Dim>::get_last_child(morton_code code) const
    {

        return get_nth_child(code, n_children - 1);
    }

    // implementation corrected for absolute level
    template <size_t Dim>
    inline morton_code Morton<Dim>::get_first_descendant(morton_code code, const size_t level) const
    {
        return (code & (~depth_mask)) + level;
    }

    // implementation corrected for absolute level
    template <size_t Dim>
    inline morton_code Morton<Dim>::get_last_descendant(morton_code code, const size_t level) const
    {
        const morton_code current_depth = get_depth(code);
        /*
        assert(level >= current_depth &&
            "can't get descendants at a coarser level than the current node!");
        */
        const morton_code first_descendant = get_first_descendant(code, level);
        const morton_code step = get_step_size(first_descendant);

        // the number of descendants at a given relative level are given by 
        // 2^(Dim * (level difference)) as each level multiplies a factor 2^Dim
        const morton_code num_descendants = (1 << (Dim * (level - current_depth)));

        // the last descendant is num_descendants - 1 morton code steps
        // away from the first descendant
        return first_descendant + step * (num_descendants - 1);
    }

    template<size_t Dim>
    inline morton_code Morton<Dim>::get_nth_descendant(morton_code code, const size_t level, size_t n) const
    {
        assert(level >= get_depth(code) && "can't get descendants at a coarser level than the current node!");
        assert(level <= max_depth && "can't get descendants at a level larger than max_depth");
        assert(n < n_children && "can't get descendant with index larger than n_children");
        const morton_code first_descendant = get_first_descendant(code, level);
        const morton_code step = get_step_size(first_descendant);
        const morton_code num_descendants = (1 << (Dim * (level - get_depth(code))));
        // all the options have to be equally spaced.
        const morton_code num_steps = (num_descendants-1)*n/(n_children-1);

        return first_descendant + step * num_steps;
    }

    template <size_t Dim>
    inline morton_code Morton<Dim>::get_deepest_first_descendant(morton_code code) const
    {
        return get_first_descendant(code, max_depth);
    }

    template <size_t Dim>
    inline morton_code Morton<Dim>::get_deepest_last_descendant(morton_code code) const
    {
        return get_last_descendant(code, max_depth);
    }

    template <size_t Dim>
    inline morton_code Morton<Dim>::get_nearest_common_ancestor(morton_code code_a, morton_code code_b) const
    {
        size_t depth_a = get_depth(code_a);
        size_t depth_b = get_depth(code_b);

        // swap nodes such that b is the coarser nodes
        if ( depth_a < depth_b ) {
            std::swap(code_a, code_b);
            std::swap(depth_a, depth_b);
        }

        morton_code ancestor_b = code_b;
        // climb up until common ancestor is found
        while ( !is_descendant(code_a, ancestor_b) ) {
            ancestor_b = get_parent(ancestor_b);
        }

        return ancestor_b;
    }

    template <size_t Dim>
    inline morton_code Morton<Dim>::get_step_size(morton_code code) const
    {
        // could it be that this can be simplified the following way:
        // the min step size is equal to floor(log2(max_depth)) + 1 == sizeof(depth encoding)
        // each level above min depth increases step size by Dim bits
        // so simplified: 1 << (depth_mask_shift + Dim * (max_depth - get_depth(code)))
        return morton_code(1) << (depth_mask_shift + Dim * (max_depth - get_depth(code)));
    }

    template <size_t Dim>
    inline morton_code Morton<Dim>::spread_coords(grid_t coord) const
    {
        morton_code res = 0;
        for (size_t i = 0; i < max_depth + 1; ++i) {
            // should be right, idk if this is possible without a loop, to guarantee unroll: replace max_depth with sizeof(morton_code)
            const auto current_bit = (coord >> i) & 1ULL;
            const auto shift = (i * Dim);
            res |= (current_bit << shift);
        }

        return res;
    }

    template <size_t Dim>
    inline bool Morton<Dim>::is_sibling(morton_code a, morton_code b) const {
        // i think its faster if we leave this out (no branching)
        // if (get_depth(a) != get_depth(b)) return false;
        return get_parent(a) == get_parent(b);
    }

    template <size_t Dim>
    inline morton_code Morton<Dim>::get_nth_child(morton_code code, size_t n) const {
        assert(n < n_children && "can't get child with index larger than n_children");
        assert(get_depth(code) < max_depth && "can't get children at the deepest level");
        return get_first_child(code) + n * get_step_size(get_first_child(code));
    }

    template <size_t Dim>
    inline int Morton<Dim>::get_child_index(morton_code parent, morton_code child) const {
        if (get_parent(child) != parent) return -1;

        const morton_code step = get_step_size(child);
        return (child - get_first_child(parent)) / step;
    }

    template <size_t Dim>
    inline vector_t<morton_code> Morton<Dim>::get_search_keys(morton_code code) const {
        vector_t<morton_code> keys;
        keys.reserve(n_children - 1);

        int index = get_child_index(get_parent(code), code);
        assert(index != -1 && "code is not a child of its parent");

        morton_code corner_leaf = get_nth_descendant(code, max_depth, index);

        grid_coordinate corner_leaf_coords = decode(corner_leaf);
        grid_coordinate anchor_coords = corner_leaf_coords;
        for (size_t i = 0; i < Dim; ++i) {
           
            // based on which index child we are whether we need to offset in 
            // dimension k depends on whether the k-th bit of index is set
            if ((index & (1 << i)) != 0)
                anchor_coords(i) += 1;
        }
        unsigned int max_coord = 1 << max_depth;
        grid_coordinate offset{};
        for (size_t i = 0; i < n_children; ++i) {
           
            // i == index means we should get the corner leaf itself
            // this is not a useful key for searching
            if (i == index) continue;

            for (size_t j = 0; j < Dim; ++j) {
                // this way the single coordinates of the offset are kind of 
                // counting in binary
                offset(j) = (i / (1 << j)) % 2;
            }
            const grid_coordinate current_coords = anchor_coords - offset;

            auto max = *std::max_element(current_coords.begin(), current_coords.end());
            // if we produce big coordinates due to exiting the domain at the it's maximum
            // or integer underflow, we skip this key
            if (max >= max_coord ) {
                continue;
            } 
            keys.push_back(encode(current_coords, max_depth));
        }

        return keys;
    }

    template<size_t Dim>
    inline vector_t<morton_code> Morton<Dim>::get_neighbors(const morton_code code,
                                                   const size_t neighbor_level) const {
        assert(neighbor_level <= max_depth && "Cant go below max_depth!");

        grid_coordinate coords = decode(code);
        vector_t<morton_code> neighbors;
        size_t level_jump = 1 << (max_depth - neighbor_level);
        grid_coordinate offset{};
        grid_coordinate neighbor_offset(level_jump);

        // we iterate over the 3^Dim hypercube surrounding the node 
        for (size_t i = 0; i < std::pow(3, Dim); ++i) {
            // we skip the center as we are only interested in the neighbors 
            if (i == (int)std::pow(3, Dim) / 2) continue;
            for (size_t j = 0; j < Dim; ++j) {
                int three_pow_j = std::pow(3, j);
                int ternary_digit = (i / three_pow_j) % 3;
                // counting the coordinates in ternary allows us to traverse the 3^Dim hypercube
                offset(j) = ternary_digit*level_jump;
            }
            grid_coordinate current_coords = coords + offset - neighbor_offset;

            auto max = *std::max_element(current_coords.begin(), current_coords.end());
            // if we produce big coordinates due to exiting the domain at the it's maximum
            // or integer underflow, we skip this key 
            if (max >= (1 << max_depth)) {
                continue;
            }
            neighbors.push_back(encode(current_coords, neighbor_level));
        }
        return neighbors;
    }

    template<size_t Dim>
    inline bool Morton<Dim>::are_neighbors(morton_code a, morton_code b) const {
        // if the codes are the same they are not neighbors
        if (a == b) return false;

        // if the codes are siblings they are neighbors
        if (is_sibling(a, b)) return true;

        if (get_depth(a) < get_depth(b)) {
          std::swap(a, b);
        }

        auto neighbors = get_neighbors(a, get_depth(a));

        for (auto& neighbor : neighbors) {
          if (neighbor == b) {
            return true;
          }
          
          if (is_descendant(neighbor, b)) {
            return true;
          }
        }

        return false;
    }

} // namespace ippl

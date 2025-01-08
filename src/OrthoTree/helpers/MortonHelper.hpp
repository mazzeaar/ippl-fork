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
            vec.push_back(first_child + (i * step));
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
        /*
        std::string error = std::string("RANK: ") + std::to_string(Comm->rank()).c_str()
                            + std::string(" can't get the first child at the deepest level");
        if (get_depth(code) >= max_depth) {
            std::cerr << "ERROR HERE:    " << error << std::endl;
            assert(false);
        }
        */

        const morton_code first_child = get_first_child(code);
        const morton_code step = get_step_size(first_child);
        return first_child + (n_children - 1) * step;
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
        const morton_code num_descendants = (1ULL << (Dim * (level - current_depth)));

        // the last descendant is num_descendants - 1 morton code steps
        // away from the first descendant
        return first_descendant + step*(num_descendants - 1);
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

} // namespace ippl
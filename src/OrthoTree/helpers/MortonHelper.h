#ifndef MORTON_ENCODER_H
#define MORTON_ENCODER_H

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>

#include "OrthoTree/OrthoTreeTypes.h"

namespace ippl {

    /**
     * @brief This class manages morton codes for the octree.
     * The functions are implemented based on page 4 of
     * https://padas.oden.utexas.edu/static/papers/OctreeBalance21.pdf
     *
     * The class is a singleton as of now, which is probably something we will have to change.
     * I would suggest that we keep the class design, this way we dont have to add parameters to
     * each function call. Maybe we can also represent a morton code as a class, this would make
     * stuff like code_a.is_parent(potential_parent_code) cleaner. Changing the framework of the
     * functions is easy, so its not important.
     *
     * @tparam Dim how many dimensions our space has.
     */
    template <size_t Dim>
    struct Morton {
        using grid_coordinate = grid_coordinate_template<Dim>;
        using real_coordinate = real_coordinate_template<Dim>;

    public:
        KOKKOS_FUNCTION Morton(size_t max_depth)
            : max_depth(max_depth)
            , depth_mask_shift(std::floor(std::log2(max_depth)) + 1)
            , depth_mask((1 << depth_mask_shift) - 1)
            , n_children((1 << (Dim))) {}

        // deleted to enforce singleton
        KOKKOS_FUNCTION Morton(const Morton&)            = default;
        KOKKOS_FUNCTION Morton& operator=(const Morton&) = default;
        KOKKOS_FUNCTION Morton(Morton&&)                 = default;
        KOKKOS_FUNCTION Morton& operator=(Morton&&)      = default;

        /**
         * @brief Encodes the given coordinate based on the rasterizer. As of now, the rasterizer is
         * used to transform the (real valued) coordinates to a grid based coordinate system, i dont
         * know if this is a good idea, but it works for spaces in the range [0,1]^Dim
         *
         * @param coordinate real values coordinate
         * @param rasterizer real values of one unit length -> should maybe be a boundingbox
         * starting at min, min?
         * @param depth the expected depth of the code, used to scale the rasterizer
         *
         * @warning no checks are done (for anything lol) if the coordinate is encodable for the
         * given grid!
         *
         * @return morton_code
         */
        KOKKOS_INLINE_FUNCTION morton_code encode(const real_coordinate& coordinate,
                                  const real_coordinate& rasterizer, const size_t depth) const;

        /**
         * @brief Encodes the given grid based coordinate to a morton code.
         *
         * @param coordinate grid based coordiante
         * @param depth the depth at which we want to encode the coordinate
         *
         * @warning no checks are done if the coordinate is encodable for the given grid!
         *
         * @return morton_code
         */
        KOKKOS_INLINE_FUNCTION morton_code encode(const grid_coordinate& coordinate, const size_t depth) const;

        /**
         * @brief Decodes the given morton code into an integer based coordiante vector
         *
         * @param code a valid morton code (also works with invalid codes lol)
         * @return grid_coordinate
         */
        KOKKOS_INLINE_FUNCTION grid_coordinate decode(morton_code code) const;

        /**
         * @brief Returns the encoded depth of a code
         *
         * @param code
         * @return size_t
         */
        KOKKOS_INLINE_FUNCTION size_t get_depth(morton_code code) const;

        /**
         * @brief Returns the code of this parent
         *
         * @param code
         * @return morton_code
         */
        KOKKOS_INLINE_FUNCTION morton_code get_parent(morton_code code) const;

        /**
         * @brief Returns a View filled with the (2^Dim) children of a node, in ascending order
         *
         * @param code
         * @return Kokkos::View<morton_code*>
         */
        KOKKOS_INLINE_FUNCTION Kokkos::View<morton_code*> get_children(morton_code code) const;

        /**
         * @brief Returns the siblings of the given code in ascending order.
         * The function works by getting the codes parent and then generating its kids, meaning the
         * siblings contain the given code itself
         *
         * @param code
         * @return Kokkos::View<morton_code*>
         */
        KOKKOS_INLINE_FUNCTION Kokkos::View<morton_code*> get_siblings(morton_code code) const;

        /**
         * @brief Returns the first child of a code
         *
         * @param code
         * @return morton_code
         */
        KOKKOS_INLINE_FUNCTION morton_code get_first_child(morton_code code) const;

        /**
         * @brief Returns the last child of a code
         *
         * @param code
         * @return morton_code
         */
        KOKKOS_INLINE_FUNCTION morton_code get_last_child(morton_code code) const;

        /**
         * @brief Returns the lth first descendand, the walk is done using only first descendants.
         *
         * @param code
         * @param level
         * @return morton_code
         */
        KOKKOS_INLINE_FUNCTION morton_code get_first_descendant(morton_code code, const size_t level) const;

        /**
         * @brief Returns the lth last descendant, the walk is done using only last descendants.
         *
         * @param code
         * @param level
         * @return morton_code
         */
        KOKKOS_INLINE_FUNCTION morton_code get_last_descendant(morton_code code, const size_t level) const;

        /**
         * @brief Returns get_first_descendant(code, max_depth - get_depth(code));
         *
         * @param code
         * @return morton_code
         */
        KOKKOS_INLINE_FUNCTION morton_code get_deepest_first_descendant(morton_code code) const;

        /**
         * @brief Returns get_first_descendant(code, max_depth - get_depth(code));
         *
         * @param code
         * @return morton_code
         */
        KOKKOS_INLINE_FUNCTION morton_code get_deepest_last_descendant(morton_code code) const;

        /**
         * @brief Returns the nearest common ancestor of the two given codes, is implemented very
         * naively and can for sure be done smarter.
         *
         * @param code_a
         * @param code_b
         * @return morton_code
         */
        KOKKOS_INLINE_FUNCTION morton_code get_nearest_common_ancestor(morton_code code_a,
                                                       morton_code code_b) const;

        /**
         * @brief Checks whether the given parent is an ancestor of the given child
         *
         * @param child the child code
         * @param parent the parent code
         * @return true if parent is ancestor of child
         */
        KOKKOS_INLINE_FUNCTION bool is_ancestor(morton_code child, morton_code parent) const;

        /**
         * @brief Checks whether the given child is a descendant of the given parents
         *
         * @param child the child code
         * @param parent the parent code
         * @return true if child is descendant of parent
         */
        KOKKOS_INLINE_FUNCTION bool is_descendant(morton_code child, morton_code parent) const;

        /**
         * @brief Returns the step size with siblings at a given level
         *
         * @param code
         * @return morton_code
         */
        KOKKOS_INLINE_FUNCTION morton_code get_step_size(morton_code code) const;

    private:
        const size_t max_depth;
        const size_t depth_mask_shift;
        const size_t depth_mask;
        const size_t n_children;

        /**
         * @brief Spreads the coordinates of a single axis onto a morton code. This means:
         * For dim = 2
         * coord = 1111, then spread_cord(coord) -> 01010101
         * This will be used to construct the full code in the encode function.
         *
         * @warning NOTE there is no shifting happening, all bits are spread equally, they are only
         * shifted in encode.
         *
         * @param coord
         * @return morton_code
         */
        KOKKOS_INLINE_FUNCTION morton_code spread_coords(grid_t coord) const;
    };

}  // namespace ippl

#include "MortonHelper.hpp"

#endif  // MORTON_ENCODER_H

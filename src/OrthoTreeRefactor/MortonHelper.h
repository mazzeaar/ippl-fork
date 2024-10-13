#ifndef MORTON_HELPER_GUARD
#define MORTON_HELPER_GUARD

#include "common.h"

namespace refactor {

    /**
     * @brief This class encapsulates methods used for morton encoding.
     *
     * Its fully static as of now and accepts dimension and max depth as templates (used to validate encodings)
     * I think this could still be done cleaner, but its ok for now.
     *
     * @tparam DIM
     * @tparam MAX_DEPTH
     */
    template <dim_type DIM, depth_type MAX_DEPTH>
    struct MortonHelper {
        // typedefs
        using position_type = position_type_template<DIM>;
        using grid_position_type = grid_position_type_template<DIM>;
        using particle_type = particle_type_template<DIM>;
        using box_type = BoundingBox<DIM>;

        static constexpr size_t max_nodes_per_edge = get_max_nodes_per_edge(MAX_DEPTH);

        /**
         * @brief encodes a given coordinate or particle to its morton code.
         *
         * @tparam T particle_type or coordinate_type
         * @param coordinate
         * @param root_bounds
         * @return constexpr morton_code_type
         */
        template<typename T>
        static constexpr morton_code_type encode(const T& coordinate, const box_type& root_bounds)
        {
            morton_code_type code = 0;

            if constexpr ( std::is_same_v<T, position_type> ) {
                grid_position_type rasterised_coordinate = fit_to_grid(coordinate, root_bounds);
                for ( size_t i = 0; i < DIM; ++i ) {
                    code += (spread_coords(rasterised_coordinate[i]) << i);
                }
            }
            else if constexpr ( std::is_same_v<T, grid_position_type> ) {
                for ( size_t i = 0; i < DIM; ++i ) {
                    code += (spread_coords(coordinate[i]) << i);
                }
            }

            return code;
        }

        /**
         * @brief Get the morton id pairs object
         * This function creates a vector of <id, morton_code> pairs which are used to associate a particle with its morton code.
         * Works with particles and coordinates
         *
         * @tparam T either particle_type or coordinate_type
         * @param input
         * @param root_bounds
         * @return vector_type<morton_id_pair_type>
         */
        template<typename T>
        static vector_type<morton_id_pair_type> get_morton_id_pairs(const T& input, const box_type& root_bounds)
        {
            vector_type<morton_id_pair_type> pairs;
            size_t n;

            if constexpr ( std::is_same_v<T, vector_type<position_type>> ) {
                n = input.size();
                pairs.resize(n);
                for ( size_t i = 0; i < n; ++i ) {
                    pairs[i] = { i, encode(input[i], root_bounds) };
                }
            }
            else if constexpr ( std::is_same_v<T, particle_type> ) {
                n = input.getLocalNum();
                pairs.resize(n);
                for ( size_t i = 0; i < n; ++i ) {
                    pairs[i] = { i, encode(input.R(i), root_bounds) };
                }
            }

            return pairs;
        }

        /**
         * @brief Normalises the coordinate to be relative to the root_bounds size.
         *
         * @param coordinate
         * @param root_bounds
         * @return constexpr grid_position_type
         */
        static constexpr grid_position_type fit_to_grid(const position_type& coordinate, const box_type& root_bounds)
        {
            grid_position_type normalised;
            for ( size_t i = 0; i < DIM; ++i ) {
                const auto bounds_size = root_bounds.Max[i] - root_bounds.Min[i];
                const auto normalised_coordinate = (coordinate[i] - root_bounds.Min[i]) / bounds_size;

                normalised[i] = normalised_coordinate * max_nodes_per_edge;
            }
            return normalised;
        }

        /**
         * @brief Exctracts the grid coordinate from the morton code.
         *
         * @param morton_code
         * @return constexpr grid_position_type
         */
        static constexpr grid_position_type decode(morton_code_type morton_code)
        {
            grid_position_type grid_pos;
            const depth_type current_depth = get_depth(morton_code);

            morton_code_type bitMask { 1 };
            depth_type bitShift = 0;

            for ( depth_type depth = 0; depth < current_depth; ++depth ) {
                for ( dim_type dimension = 0; dimension < DIM; ++dimension ) {
                    grid_pos[dimension] |= (morton_code & bitMask) >> (bitShift - depth);
                    bitMask <<= 1;
                    ++bitShift;
                }
            }

            return grid_pos;
        }

        /**
         * @brief Get the depth object
         * Works by shifting the key and increasing the depth until its equal to 1 (root node)
         *
         * @param key
         * @return constexpr depth_type
         */
        static constexpr depth_type get_depth(morton_code_type key)
        {
            depth_type depth = 0;

            while ( key != 0 ) {
                if ( key == 1 ) {
                    return depth;
                }

                key >>= DIM;
                ++depth;
            }

            assert(false && "key is invalid!");
            return 0;
        }

        /**
         * @brief Get the parent code. Throws an error if the key is from the root node
         *
         * @param key
         * @return constexpr morton_code_type
         */
        static constexpr morton_code_type get_parent_code(morton_code_type key)
        {
            // idk if this makes sense??
            assert(key != 1 && "key cant be the root node!");
            return key >> DIM;
        }

    private:

        /**
         * @brief This function spreads the bits based on morton encoding.
         * Meaning the bits from the coord get spread by 'DIM' amount.
         * So 0b100111 becomes 0b100 000 000 100 100 1 (for DIM=3)
         *
         * @param coord
         * @return constexpr morton_code_type
         */
        static constexpr morton_code_type spread_coords(grid_id_type coord)
        {
            // this will not compile if we try to generate invalid codes, should help providing nasty bugs for large depths
            // morton_code_type could maybe be uint128 in that case, idk if thats reasonable?
            constexpr auto bitsInType = sizeof(morton_code_type) * 8;
            static_assert(MAX_DEPTH * DIM <= bitsInType, "Morton code type is too small to hold the spread bits.");

            morton_code_type res = 0;
            for ( size_t i = 0; i < MAX_DEPTH; ++i ) {
                const auto current_byte = (coord >> i) & 1ULL;
                const auto shift = (i * DIM);
                res |= (current_byte << shift);
            }

            return res;
        }
    };

} // namespace ippl

#endif // MORTON_HELPER_GUARD
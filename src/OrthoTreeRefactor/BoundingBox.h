#ifndef ORTHOTREE_BOUNDING_BOX_GUARD
#define ORTHOTREE_BOUNDING_BOX_GUARD

#include "common.h"

namespace refactor {

    template <dim_type DIM>
    struct BoundingBox {
        // typedefs
        using position_type = position_type_template<DIM>;
        using particle_type = particle_type_template<DIM>;

        // members
        position_type Min { {} };
        position_type Max { {} };

        /**
         * @brief relationship that two boxes can have with eachother
         */
        enum Relation : int8_t { Overlapping = -1, Adjacent = 0, Separated = 1 };

        /**
         * @brief basicallay numeric_limits<BoundingBox>::max()
         *
         * @return constexpr BoundingBox
         */
        static constexpr BoundingBox max()
        {
            BoundingBox limit_bounds;

            for ( size_t i = 0; i < DIM; ++i ) {
                limit_bounds.min[i] = std::numeric_limits<float_type>::max();
                limit_bounds.max[i] = std::numeric_limits<float_type>::lowest();
            }

            return limit_bounds;
        }

        /**
         * @brief returns a boundingbox that is shrunk to the minimal available size s.t. all particles are still contained.
         * this should maybe include a buffer as well, else we have particles directly on boundaries.
         *
         * @param particles
         * @return constexpr BoundingBox
         */
        static constexpr BoundingBox fit(const vector_type<particle_type>& particles)
        {
            BoundingBox max_bounds = max();

            for ( const auto& particle : particles ) {
                for ( size_t i = 0; i < DIM; ++i ) {
                    max_bounds.min[i] = std::min(particle.position[i], max_bounds.min[i]);
                    max_bounds.max[i] = std::max(particle.position[i], max_bounds.max[i]);
                }
            }

            return max_bounds;
        }

        /**
         * @brief Returns the rasterizer for the current bounding box. Is only called on the root bounding box once afaik.
         *
         * @param n_divisions
         * @return position_type
         */
        position_type get_rasteriser(grid_id_type n_divisions)
        {
            position_type rasterizer;
            for ( dim_type i = 0; i < DIM; ++i ) {
                double boxsize_i = Max[i] - Min[i];
                if ( boxsize_i == 0 ) {
                    rasterizer[i] = 1.0;
                }
                else {
                    rasterizer[i] = boxsize_i / static_cast<float_t>(n_divisions);
                }
            }

            return rasterizer;
        }

        position_type get_center() const
        {
            position_type center;
            for ( size_t i = 0; i < DIM; ++i ) {
                center[i] = ((Min[i] + Max[i]) >> 1);
            }
            return center;
        }

        /**
         * @brief Returns the relationship two nodes have with eachother.
         *
         * @param other
         * @return Relation
         */
        Relation get_relation(const BoundingBox& other) const
        {
            bool any_adjacent = false;

            for ( dim_type i = 0; i < DIM; ++i ) {
                if ( is_separated(other, i) ) {
                    return Relation::Separated;
                }
                if ( is_adjacent(other, i) ) {
                    any_adjacent = true;
                }
            }

            return any_adjacent ? Relation::Adjacent : Relation::Overlapping;
        }

        inline bool is_overlapping(const BoundingBox& other) const { return get_relation(other) == Relation::Overlapping; }
        inline bool is_adjacent(const BoundingBox& other) const { return get_relation(other) == Relation::Adjacent; }
        inline bool is_separated(const BoundingBox& other) const { return get_relation(other) == Relation::Separated; }

        void print_info() const
        {
            static constexpr size_t width = 20;
            std::cout << std::left << std::setw(width) << "Box:" << "[(";
            for ( int i = 0; i < DIM; ++i ) std::cout << Min[i] << ",";
            std::cout << "), (";
            for ( int i = 0; i < DIM; ++i ) std::cout << Max[i] << ",";
            std::cout << ")]" << std::endl;
        }

    private:

        inline bool is_overlapping(const BoundingBox& other, dim_type dim_idx) const { return (Min[dim_idx] < other.Max[dim_idx]) && (Max[dim_idx] > other.Min[dim_idx]); }
        inline bool is_adjacent(const BoundingBox& other, dim_type dim_idx) const { return (Min[dim_idx] == other.Max[dim_idx]) || (Max[dim_idx] == other.Min[dim_idx]); }
        inline bool is_separated(const BoundingBox& other, dim_type dim_idx) const { return (Min[dim_idx] > other.Max[dim_idx]) || (Max[dim_idx] < other.Min[dim_idx]); }
    };
} // namespace ippl

#endif // ORTHOTREE_BOUNDING_BOX_GUARD
#ifndef ORTHOTREE_COMMON_GUARD
#define ORTHOTREE_COMMON_GUARD

#include <cstdint>
#include <iomanip> // for pretty infos

// #define USE_STD_LIB_ONLY

#ifdef USE_STD_LIB_ONLY
#include <vector>
#include <array>
#include <unordered_map>
#else
#include <Kokkos_Vector.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#endif

namespace refactor {

    // this way we (should) be able to compile this stuff only using stl stuff (far way from there)
#ifdef USE_STD_LIB_ONLY
    // will need to define some simple particle type as well here
    template <typename T> using vector_type = std::vector<T>;
    template <typename T, size_t N> using array_type = std::array<T, N>;
    template <typename Key, size_t T> using unordered_map_type = unordered_map_type<Key, T>;
    template <typename T1, T2> using pair_type = std::pair<T1, T2>;
#else
    template <typename T> using vector_type = Kokkos::vector<T>;
    template <typename T, size_t N> using array_type = Kokkos::Array<T, N>;
    template <typename Key, typename T> using unordered_map_type = Kokkos::UnorderedMap<Key, T>;
    template <typename T1, typename T2> using pair_type = Kokkos::pair<T1, T2>;
#endif

    // type aliases
    using float_type = double;

    using dim_type = unsigned short int;
    using child_id_type = unsigned int;
    using morton_code_type = unsigned int;
    using grid_id_type = size_t;
    using particle_id_type = size_t;
    using depth_type = unsigned int;

    using morton_id_pair_type = pair_type<particle_id_type, morton_code_type>;

    // template type aliases
    template <dim_type DIM> using particle_type_template = ippl::OrthoTreeParticle<ippl::ParticleSpatialLayout<float_type, DIM>>;
    template <dim_type DIM> using position_type_template = ippl::Vector<float_t, DIM>;
    template <dim_type DIM> using grid_position_type_template = ippl::Vector<grid_id_type, DIM>;

    // forward declarations
    template <dim_type DIM> struct BoundingBox;
    template <dim_type DIM> class OrthoTreeNode;
    template <dim_type DIM, depth_type max_depth> struct MortonHelper;
    template <dim_type DIM, depth_type max_depth> class OrthoTree;

    // constexpr helpers
    static constexpr size_t get_max_nodes_per_edge(dim_type max_depth) { return (1 << (max_depth)); }

    /**
     * @brief Checks if a grid position is valid for the given range
     * @tparam DIM how many dimensions do we have
     */
    template <dim_type DIM>
    static constexpr bool is_valid_grid_coord(const grid_position_type_template<DIM>& pos, grid_id_type min, grid_id_type max)
    {
        for ( dim_type i = 0; i < DIM; ++i ) {
            if ( pos[i] < min || pos[i] >= max ) return false;
        }
        return true;
    }

} // namespace ippl

#endif // ORTHOTREE_COMMON_GUARD
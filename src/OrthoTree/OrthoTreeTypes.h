#ifndef ORTHO_TREE_TYPES_GUARD
#define ORTHO_TREE_TYPES_GUARD

#include <cstdint>
#include "OrthoTreeParticle.h"
#include <Types/Vector.h>
#include <Kokkos_Core.hpp>

namespace ippl {

    using morton_code = uint64_t;
    using grid_t = uint32_t;

    template <size_t Dim>
    using real_coordinate_template = ippl::Vector<double, Dim>;

    template <size_t Dim>
    using grid_coordinate_template = ippl::Vector<grid_t, Dim>;
    using TreeDefaultExecutionSpace = typename Kokkos::OpenMP;

} // namespace ippl

#endif // ORTHO_TREE_TYPES_GUARD

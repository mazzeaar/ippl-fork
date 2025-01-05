#ifndef ORTHO_TREE_TYPES_GUARD
#define ORTHO_TREE_TYPES_GUARD

#include <cstdint>
#include "OrthoTreeParticle.h"
#include <Types/Vector.h>

namespace ippl {

    using morton_code = uint64_t;
    using grid_t = uint32_t;

    template <size_t Dim>
    using real_coordinate_template = ippl::Vector<double, Dim>;

    template <size_t Dim>
    using grid_coordinate_template = ippl::Vector<grid_t, Dim>;

} // namespace ippl

#endif // ORTHO_TREE_TYPES_GUARD

#include "../OrthoTree.h"

/*
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*,Kokkos::HostSpace::memory_space> OrthoTree<Dim>::block_partition(morton_code min_octant,
                                                               morton_code max_octant) {
        IpplTimings::TimerRef blockPartitionTimer = IpplTimings::getTimer("block_partition");
        IpplTimings::startTimer(blockPartitionTimer);

        Kokkos::View<morton_code*,Kokkos::HostSpace::memory_space> T = complete_region(min_octant, max_octant);

        // find the lowest level (smallest depth)
        size_t lowest_level;
        Kokkos::parallel_reduce("algo4::FindLowestLevel",
            T.size(),
            KOKKOS_LAMBDA(const size_t i, size_t& min_depth) {
                size_t depth = morton_helper.get_depth(T(i));
                if (depth < min_depth) {
                    min_depth = depth;
                }
            },
            Kokkos::Min<size_t>(lowest_level));

        // count the number of elements at the lowest level
        size_t C_size;
        Kokkos::parallel_reduce("algo4::CountAtLowestLevel",
            T.size(),
            KOKKOS_LAMBDA(const size_t i, size_t& count) {
                if (morton_helper.get_depth(T(i)) == lowest_level) {
                    count++;
                }
            },
            C_size);

        Kokkos::View<morton_code*,Kokkos::HostSpace::memory_space> C("algo4::C_view", C_size);

        // populate C_view
        Kokkos::parallel_scan("algo4::PopulateC",
            T.size(), KOKKOS_LAMBDA(const size_t i, size_t& index, bool final) {
                if (morton_helper.get_depth(T(i)) == lowest_level) {
                    if (final) {
                        C(index) = T(i);
                    }
                    index++;
                }
            });

        Kokkos::View<morton_code*,Kokkos::HostSpace::memory_space> G = complete_tree(C);

        Kokkos::View<size_t*,Kokkos::HostSpace::memory_space> weights      = this->aid_list_m.getNumParticlesInOctantsParallel(G);
        Kokkos::View<morton_code*,Kokkos::HostSpace::memory_space> octants = partition(G, weights);

        morton_code min_step   = morton_helper.get_step_size(max_depth_m);
        morton_code max_parent = *(octants.data() + octants.size() - 1);

        morton_code new_min_octant = morton_helper.get_deepest_first_descendant(octants[0]);
        morton_code new_max_octant =
            morton_helper.get_deepest_last_descendant(max_parent) + min_step;

        IpplTimings::TimerRef innitfromoctants = IpplTimings::getTimer("innitfromoctants");
        IpplTimings::startTimer(innitfromoctants);

        this->aid_list_m.innitFromOctants(new_min_octant, new_max_octant);

        IpplTimings::stopTimer(innitfromoctants);

        IpplTimings::stopTimer(blockPartitionTimer);
        return octants;
    }
}  // namespace ippl

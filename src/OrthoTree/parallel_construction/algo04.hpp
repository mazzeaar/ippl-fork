#include "../OrthoTree.h"

/*
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::block_partition(morton_code min_octant,
                                                               morton_code max_octant) {
        IpplTimings::TimerRef blockPartitionTimer = IpplTimings::getTimer("block_partition");
        IpplTimings::startTimer(blockPartitionTimer);


        Kokkos::View<morton_code*> T = complete_region(min_octant, max_octant);

        // find the lowest level (smallest depth)
        size_t lowest_level = max_depth_m;
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
        

        Kokkos::View<morton_code*> C("algo4::C_view", C_size);


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

        if (C_size == 0) {
            Kokkos::resize(C, 2);
            C(0) = min_octant;
            C(1) = max_octant;
        }

        if (aid_list_m.size() == 0) {
            throw std::runtime_error("No particles on rank algo4");
        }
        Kokkos::View<morton_code*> G = complete_tree(C);

        Kokkos::View<size_t*> weights      = this->aid_list_m.getNumParticlesInOctantsParallel(G);
        Kokkos::View<morton_code*> octants = partition(G, weights);

        morton_code min_step   = morton_helper.get_step_size(max_depth_m);
        morton_code max_parent = *(octants.data() + octants.size() - 1);

        morton_code new_min_octant = morton_helper.get_deepest_first_descendant(octants[0]);
        morton_code new_max_octant =
            morton_helper.get_deepest_last_descendant(max_parent) + min_step;

        IpplTimings::TimerRef innitfromoctants = IpplTimings::getTimer("innitfromoctants");
        IpplTimings::startTimer(innitfromoctants);

        this->aid_list_m.innitFromOctants(new_min_octant, new_max_octant);
        n_particles = this->aid_list_m.size();
        IpplTimings::stopTimer(innitfromoctants);

        IpplTimings::stopTimer(blockPartitionTimer);
        return octants;
    }
}  // namespace ippl

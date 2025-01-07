#include "../OrthoTree.h"

/*
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::block_partition(morton_code min_octant,
                                                               morton_code max_octant) {
        // logger.on(true);
        // logger.setOutputLevel(1);
        // logger << level1 << "starting algo4" << endl;

        IpplTimings::TimerRef blockPartitionTimer = IpplTimings::getTimer("block_partition");
        IpplTimings::startTimer(blockPartitionTimer);

        Kokkos::View<morton_code*> T = complete_region(min_octant, max_octant);

        // find the lowest level (smallest depth)
        size_t lowest_level;
        Kokkos::parallel_reduce(
            "algo4::FindLowestLevel", T.size(),
            KOKKOS_LAMBDA(const size_t i, size_t& min_depth) {
                size_t depth = morton_helper.get_depth(T(i));
                if (depth < min_depth) {
                    min_depth = depth;
                }
            },
            Kokkos::Min<size_t>(lowest_level));

        // count the number of elements at the lowest level
        size_t C_size;
        Kokkos::parallel_reduce(
            "algo4::CountAtLowestLevel", T.size(),
            KOKKOS_LAMBDA(const size_t i, size_t& count) {
                if (morton_helper.get_depth(T(i)) == lowest_level) {
                    count++;
                }
            },
            C_size);

        Kokkos::View<morton_code*> C("algo4::C_view", C_size);

        // populate C_view
        Kokkos::parallel_scan(
            "algo4::PopulateC", T.size(), KOKKOS_LAMBDA(const size_t i, size_t& index, bool final) {
                if (morton_helper.get_depth(T(i)) == lowest_level) {
                    if (final) {
                        C(index) = T(i);
                    }
                    index++;
                }
            });

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
        IpplTimings::stopTimer(innitfromoctants);

        IpplTimings::stopTimer(blockPartitionTimer);
        return octants;
    }

    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::algo4_11(Kokkos::View<morton_code*> F_view) {
        assert(F_view.size() > 0 && "Size missmatch");
        logger.setOutputLevel(1);
        logger << "Test in algo4_11" << endl;
        const morton_code min_oct = F_view(0);
        const morton_code max_oct = F_view(F_view.size() - 1);
        Kokkos::View<morton_code*> T = complete_region(min_oct, max_oct);

        // the lowest level is actually the 'highest' (closest to root) node in our tree
        size_t lowest_level;
        Kokkos::parallel_reduce(
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
        Kokkos::parallel_reduce(
            T.size(),
            KOKKOS_LAMBDA(const size_t i, size_t& count) {
                if (morton_helper.get_depth(T(i)) == lowest_level) {
                    count++;
                }
            },
            C_size);

        Kokkos::View<morton_code*> C("C_view", C_size);

        // populate C_view
        Kokkos::parallel_scan(
            T.size(), KOKKOS_LAMBDA(const size_t i, size_t& index, bool final) {
                if (morton_helper.get_depth(T(i)) == lowest_level) {
                    if (final) {
                        C(index) = T(i);
                    }
                    index++;
                }
            });

        Kokkos::View<morton_code*> G = complete_tree(C);

        Kokkos::View<size_t*> weights      = this->aid_list_m.getNumParticlesInOctantsParallel(G);
        
        // TODO is temporary for testing algo11
        Kokkos::View<size_t*> weights_view("weights_view", G.size());
        for (size_t i = 0; i < G.size(); ++i) {
            weights_view[i] = 1;
        }

        logger << level1 << "got weights, now starting partition" << endl;
        Kokkos::View<morton_code*> octants = partition(G, weights_view);

        // update with F_glob
        return octants;
    }
}  // namespace ippl

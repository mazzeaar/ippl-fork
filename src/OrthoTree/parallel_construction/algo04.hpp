#include "../OrthoTree.h"

/*
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::block_partition(morton_code min_octant,
                                                               morton_code max_octant) {
        Kokkos::View<morton_code*> T = complete_region(min_octant, max_octant);

        // the lowest level is actually the 'highest' (closest to root) node in our tree
        size_t lowest_level = morton_helper.get_depth(*std::min_element(
            T.data(), T.data() + T.size(), [this](const morton_code& a, const morton_code& b) {
                return morton_helper.get_depth(a) < morton_helper.get_depth(b);
            }));

        const size_t C_size =
            std::accumulate(T.data(), T.data() + T.size(), 0,
                            [this, lowest_level](auto acc, const morton_code octant) {
                                return acc + (morton_helper.get_depth(octant) == lowest_level);
                            });

        // we only use the 'highest' octants
        Kokkos::View<morton_code*> C("C_view", C_size);
        size_t C_index = 0;
        for (auto it = T.data(); it != T.data() + T.size(); ++it) {
            const morton_code octant = *it;
            if (morton_helper.get_depth(octant) == lowest_level) {
                C[C_index] = octant;
                ++C_index;
            }
        }

        Kokkos::View<morton_code*> G = complete_tree(C);

        Kokkos::View<size_t*> weights      = this->aid_list_m.getNumParticlesInOctantsParallel(G);
        Kokkos::View<morton_code*> octants = partition(G, weights);

        morton_code new_min_octant = octants[0];
        morton_code new_max_octant = *(octants.data() + octants.size() - 1);
        this->aid_list_m.innitFromOctants(new_min_octant, new_max_octant);
        return octants;
    }

    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::algo4_11(Kokkos::View<morton_code*> F_view) {
        assert(F_view.size() > 0 && "Size missmatch");
        const morton_code min_oct = morton_helper.get_deepest_first_descendant(F_view(0));
        const morton_code max_oct =
            morton_helper.get_deepest_first_descendant(F_view(F_view.size() - 1));
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
        Kokkos::View<morton_code*> octants = partition(G, weights);

        // update with F_glob
        return octants;
    }
}  // namespace ippl
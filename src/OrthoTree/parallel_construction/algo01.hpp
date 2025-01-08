#include "../OrthoTree.h"
/*
TODO:
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::build_tree(particle_t const& particles) {
        START_FUNC;

        IpplTimings::TimerRef aidListTimer = IpplTimings::getTimer("aid_list");
        IpplTimings::startTimer(aidListTimer);

        this->aid_list_m.initialize(root_bounds_m, particles);
        if (aid_list_m.size() == 0) {
            END_FUNC;
            throw std::runtime_error("No particles on rank algo1");
        }
        auto [min_octant, max_octant] = this->aid_list_m.getMinReqOctants();

        IpplTimings::stopTimer(aidListTimer);

        IpplTimings::TimerRef buildTreeTimer = IpplTimings::getTimer("build_tree");
        IpplTimings::startTimer(buildTreeTimer);

        auto octants = block_partition(min_octant, max_octant);

        IpplTimings::stopTimer(buildTreeTimer);

        particles_to_file(particles);  // runs much faster if we print here instead of below lol

        // Each proc has now as much of the aid_list as he needs and can start building the
        // tree.

        IpplTimings::startTimer(buildTreeTimer);

        Kokkos::View<morton_code*> tree_view = build_tree_from_octants(octants);

        IpplTimings::stopTimer(buildTreeTimer);

        octants_to_file(tree_view);
        print_stats(tree_view, particles);
        return tree_view;
    }

    template <size_t Dim>
    void OrthoTree<Dim>::build_tree_from_octant(morton_code root_octant,
        Kokkos::View<morton_code*>& tree_view) {
        auto guesstimate_subtree_size = [this](morton_code octant) {
            // we can probably do some really smart guessing here

            const size_t octant_depth = this->morton_helper.get_depth(octant);
            const size_t remaining_depth = this->max_depth_m - octant_depth;

            // empirical guess this could be improved if it was constructed in some 
            // more clever way
            // But a good guess depends a lot on the distribution here.
            // This is more of an upper limit for a good guess probably
            const size_t guess = std::max(2* this->aid_list_m.getNumParticlesInOctant(octant)
                                  / this->max_particles_per_node_m, (size_t)10);
            return guess;
            };

        const size_t old_size = tree_view.size();
        const size_t size_increase = guesstimate_subtree_size(root_octant);

        Kokkos::resize(tree_view, old_size + size_increase);
        size_t new_size = tree_view.size();

        std::stack<morton_code> stack;
        stack.push(root_octant);

        size_t i = old_size;
        while (!stack.empty()) {
            morton_code cur_octant = stack.top();
            stack.pop();

            const size_t octant_depth = morton_helper.get_depth(cur_octant);
            const size_t n_particles_in_octant =
                this->aid_list_m.getNumParticlesInOctant(cur_octant);

            /*
            TODO:
            THIS IS OK FOR DEBUGGING PURPOSES (nicer visualisation), BUT WE SHOULD REENABLE THIS
            LATER, WE DONT CARE ABOUT EMPTY OCTANTS

            if (n_particles_in_octant == 0) { continue;
            }
            */

            if (octant_depth >= this->max_depth_m
                || n_particles_in_octant <= this->max_particles_per_node_m) {
                if (i >= new_size) {
                    Kokkos::resize(tree_view, new_size + (size_increase / 2));
                    new_size = tree_view.size();
                }
                tree_view[i++] = cur_octant;
                continue;
            }

            for (morton_code child_octant : morton_helper.get_children(cur_octant)) {
                stack.push(child_octant);
            }
        }

        const size_t occupied_size = i;
        Kokkos::resize(tree_view, occupied_size);

        // only sort what we need, the rest should already be sorted
        std::sort(tree_view.data() + old_size, tree_view.data() + occupied_size);

        // I WILL LEAVE THIS IN HERE FOR NOW, MIGHT CATCH ERRORS QUICKER IN LATER STAGES

        // these following lines are for testing purposes
        // they check that the resulting tree is sorted and doesn't leave gaps
        // in the domain
        //  The lambda checks that two morton codes are sorted and that there are
        //  no other morton codes between them that overlaps neither
        auto is_sorted_and_contiguous = [this](const morton_code& a, const morton_code& b) {
            if (b <= a || morton_helper.is_descendant(b, a)) {
                return false;
            }
            return b <= morton_helper.get_deepest_first_descendant(
                a + morton_helper.get_step_size(a));
            };

        assert(std::is_sorted(tree_view.data(), tree_view.data() + tree_view.size(),
            is_sorted_and_contiguous)
            && "partitioned_tree is not sorted");
    }

    template <size_t Dim>
    template <typename Container>
    Kokkos::View<morton_code*> OrthoTree<Dim>::build_tree_from_octants(const Container& octants) {
        IpplTimings::TimerRef buildTreeFromOctantsTimer =
            IpplTimings::getTimer("build_tree_from_octants");
        IpplTimings::startTimer(buildTreeFromOctantsTimer);

        Kokkos::View<morton_code*> finished_tree;

        for (auto it = octants.data(); it != (octants.data() + octants.size()); ++it) {
            build_tree_from_octant(*it, finished_tree);
        }

        IpplTimings::stopTimer(buildTreeFromOctantsTimer);

        return finished_tree;
    }
}  // namespace ippl

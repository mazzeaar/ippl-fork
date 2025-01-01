#include <ostream>

#include "OrthoTree/OrthoTreeTypes.h"
#include "OrthoTreeTypes.h"

#include "Communicate/Communicator.h"
#include "Communicate/Operations.h"
#include "Communicate/Request.h"
#include "Kokkos_Vector.hpp"
#include "OrthoTree.h"

namespace ippl {

    template <size_t Dim>
    OrthoTree<Dim>::OrthoTree(size_t max_depth, size_t max_particles_per_node,
        const bounds_t& root_bounds)
        : max_depth_m(max_depth)
        , max_particles_per_node_m(max_particles_per_node)
        , root_bounds_m(root_bounds)
        , morton_helper(max_depth)
        , aid_list_m(AidList<Dim>(max_depth))
        , logger("OrthoTree", std::cout, INFORM_ALL_NODES) {
        world_rank = Comm->rank();
        world_size = Comm->size();

        logger.setOutputLevel(0);
        logger.setPrintNode(INFORM_ALL_NODES);
    }

    inline static int stack_depth = 0;

#define LOG logger << std::string(stack_depth * 2, ' ') << __func__ << ": "

#define END_FUNC               \
    LOG << "FINISHED" << endl; \
    --stack_depth

#define START_FUNC \
    ++stack_depth; \
    LOG << "STARTING" << endl

    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::build_tree_naive(particle_t const& particles) {
        // this needs to be initialized before constructing the tree
        this->aid_list_m.initialize(root_bounds_m, particles);
        IpplTimings::TimerRef timer = IpplTimings::getTimer("build_tree");
        IpplTimings::startTimer(timer);

        // without the step below the parallel/sequential trees can never be identical, as the
        // parallel version never contains the root node
        morton_code root_octant(0);
        auto octants = morton_helper.get_children(root_octant);
        Kokkos::View<morton_code*> finished_tree = build_tree_from_octants(octants);

        particles_to_file(particles);
        octants_to_file(finished_tree);

        IpplTimings::stopTimer(timer);
        return finished_tree;
    }

    template <size_t Dim>
      bool OrthoTree<Dim>::is_balanced(const Kokkos::View<morton_code *>& tree_view) const {
          for (auto& leaf1 : std::span(tree_view.data(), tree_view.data() + tree_view.size())) {
              for (auto& leaf2 : std::span(tree_view.data(), tree_view.data() + tree_view.size())) {
                  if (leaf1 == leaf2) {
                      continue;
                  }

                  int depth1 = morton_helper.get_depth(leaf1);
                  int depth2 = morton_helper.get_depth(leaf2);


                  if (std::abs(depth1 - depth2) <= 1) {
                      continue;
                  }
                 
                  // return false if we find neighbors that are more than one 
                  // level apart
                  if (morton_helper.are_neighbors(leaf1, leaf2)) {
                      return false;
                  }
              }
          }
          
          // If no two leaves are of differing depth and are neighbors, then the tree is balanced
          return true;
      }

}  // namespace ippl

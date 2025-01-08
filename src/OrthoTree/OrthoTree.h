#ifndef ORTHOTREE_GUARD
#define ORTHOTREE_GUARD

#include <Kokkos_Pair.hpp>
#include <Kokkos_Vector.hpp>
#include <fstream>
#include <vector>

#include "OrthoTreeTypes.h"

#include "OrthoTreeParticle.h"
#include "helpers/AidList.h"
#include "helpers/BoundingBox.h"
#include "helpers/Config.h"
#include "helpers/MortonHelper.h"

namespace ippl {

    // this is defined outside of Types.h on purpose, as this is likely to change in the finalised
    // implementation
    template <size_t Dim>
    using particle_type_template = OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>>;

    /**
     * @brief This is the OrthoTree class, nice oder.
     *
     * The idea is:
     *
     * - no complexity
     *      auto tree = OrthoTree<Dimension>(max_depth, max_particles, root_bounds);
     *
     * - all the computational complexity
     *      tree.function_that_builds_the_tree(particles);
     *
     * - maybe some complexity, idc
     *      tree.traverse() or whatever operations your heart desieres.
     *
     * - this way we can later implement update as:
     *      tree.update(updated_particles);
     *
     * @tparam Dim
     */
    template <size_t Dim>
    class OrthoTree {
        using real_coordinate = real_coordinate_template<Dim>;
        using grid_coordinate = grid_coordinate_template<Dim>;
        using particle_t      = particle_type_template<Dim>;
        using bounds_t        = BoundingBox<Dim>;

        const size_t max_depth_m;
        const size_t max_particles_per_node_m;
        const bounds_t root_bounds_m;
        const Morton<Dim> morton_helper;

        size_t n_particles;

        AidList<Dim> aid_list_m;

        size_t world_rank;
        size_t world_size;

        Inform logger;

        bool enable_visualisation = false;
        bool enable_print_stats   = false;

    public:
        OrthoTree(size_t max_depth, size_t max_particles_per_node, const bounds_t& root_bounds);

        void setVisualisation(bool enable) { enable_visualisation = enable; }
        void setPrintStats(bool enable) { enable_print_stats = enable; }
        void setLogOutput(bool enable) {
            logger.on(enable);
            aid_list_m.setLogOutput(enable);
        }
        void setLogLevel(size_t level) {
            logger.setOutputLevel(level);
            aid_list_m.setLogLevel(level);
        }

        size_t getMaxDepth() const { return max_depth_m; }

        /**
         * @brief This is the most basic way to build a tree. Its inefficien, but it (should) be
         * correct. Can be used to compare against parallel implementations later on.
         *
         * @param particles An 'object of arrays' of particles (google it)
         */
        Kokkos::View<morton_code*> build_tree_naive(particle_t const& particles);

#pragma region paralell construction
        /**
         * ALGO 1
         *
         * @brief We take this as an entry function.
         * Each proc will return its own part of the tree?
         *
         * @param particles
         * @return Kokkos::vector<morton_code>
         */
        Kokkos::View<morton_code*> build_tree(particle_t const& particles);

        /**
         * ALGO 2
         *
         * @brief sequential construction of a minimal linear octree between two octants
         *
         * @param morton codes code_a and code_b of the octants
         *
         * @return list of morton codes of minimal linear octree between the two octants
         **/
        Kokkos::View<morton_code*> complete_region(morton_code code_a, morton_code code_b);

        /**
         * ALGO 3
         *
         * @brief Implements the logic part of algorithm 3.
         */
        Kokkos::View<morton_code*> complete_tree(Kokkos::View<morton_code*> tree);

        /**
         * ALGO 4
         *
         * @brief algorithm 4 parallel partitioning of octants into large contiguous blocks
         *
         * @param unpartitioned octree unpartitioned_tree
         *
         * @return block partitioned octree, and unpartitioned_tree is re-distributed
         **/
        Kokkos::View<morton_code*> block_partition(morton_code min_octant, morton_code max_octant);

        /**
         * ALGO 5
         *
         * @brief This function partitions the workload of building the tree across
         * the available mpi ranks.
         */
        Kokkos::View<morton_code*> partition(Kokkos::View<morton_code*> octants,
                                             Kokkos::View<size_t*> weights);

        /**
         * ALGO 5
         *
         * @brief Wrapper for partition for when it is called without weights, will call partition
         * with all weights set to 1
         */
        Kokkos::View<morton_code*> partition(Kokkos::View<morton_code*> octants);

        /**
         * ALGO 8
         *
         * @brief Linearises octants by removing ancestors that would cause overlaps
         * @param list of octants - sorted
         * @return list of linearised octants - sorted
         * @warning THIS FUNCTION ASSUMES THAT THE OCTANTS ARE SORTED
         */
        Kokkos::View<morton_code*> linearise_octants(Kokkos::View<morton_code*> const& octants);

#pragma endregion  // paralell construction

#pragma region balancing

        Kokkos::View<morton_code*> algo6(morton_code octant_N, morton_code descendant_L);

        Kokkos::View<morton_code*> algo7(morton_code octant_N,
                                         Kokkos::View<morton_code*> partial_descendants_L);

        Kokkos::View<morton_code*> algo9(Kokkos::View<morton_code*> sorted_incomplete_tree_L);

        Kokkos::View<morton_code*> algo10(morton_code octant_N,
                                          Kokkos::View<morton_code*> partial_descendants_L);

        Kokkos::View<morton_code*> algo11(Kokkos::View<morton_code*> distributed_complete_tree_L);

#pragma endregion  // balancing

#pragma region helpers

        
        /**
         * @brief Compares the following aspects of the trees:
         * - n_particles
         * - tree_m.size()
         * - contents of tree_m
         *
         * @warning THIS FUNCTION ASSUMES THAT THE TREES ARE SORTED
         *
         * @param other
         * @return true
         * @return false
         */
        bool operator==(const OrthoTree& other);

#pragma endregion  // helpers

        /**
         * @brief Constructs a tree in each octant inside the given container.
         * Templated for now to make it work with Kokkos::View and Kokkos::vectors
         */
        template <typename Container>
        Kokkos::View<morton_code*> build_tree_from_octants(const Container& octants);

        /**
         * @brief Constructs an OrthoTree in the given Octant. It will automatically resize the view
         * to the needed size and apply 'shrink_to_fit' after finishing.
         */
        void build_tree_from_octant(morton_code root_octant, Kokkos::View<morton_code*>& tree_view);

    public:
#pragma region print_helpers

        void print_stats(Kokkos::View<morton_code*>& tree_view, const auto& particles) {
            if (!enable_print_stats) {
                return;
            }

            size_t total_particles = 0;
            for (size_t i = 0; i < tree_view.size(); ++i) {
                auto num_particles = this->aid_list_m.getNumParticlesInOctant(tree_view[i]);
                total_particles += num_particles;
            }

            size_t global_total_particles = 0;
            Comm->allreduce(&total_particles, &global_total_particles, 1, std::plus<size_t>());

            const size_t col_width = 15;

            auto printer = [col_width](const auto& rank, const auto& tree_size,
                                       const auto& num_particles) {
                std::cerr << std::left << std::setw(col_width) << rank << std::left
                          << std::setw(col_width) << tree_size << std::left << std::setw(col_width)
                          << num_particles << std::endl;
            };

            {
                // buffer to print the stats in order
                int ring_buf;
                if (world_rank + 1 < world_size) {
                    mpi::Status status;
                    Comm->recv(&ring_buf, 1, world_rank + 1, 0, status);
                } else {
                    std::cerr << std::string(col_width * 3, '=') << std::endl;
                    printer("rank", "octs_now", "particles");
                    std::cerr << std::string(col_width * 3, '-') << std::endl;
                }

                printer(world_rank, tree_view.size(), total_particles);

                if (world_rank > 0) {
                    Comm->send(ring_buf, 1, world_rank - 1, 0);
                }
            }

            if (world_rank == 0) {
                std::cerr << std::string(col_width * 3, '-') << std::endl
                          << "We now have " << global_total_particles << " particles" << std::endl
                          << "("
                          << (100.0 * (double)global_total_particles
                              / (double)particles.getTotalNum())
                          << "% of starting value lol)" << std::endl
                          << std::string(col_width * 3, '-') << std::endl;
            }
        }

        void print_stats_seq(Kokkos::View<morton_code*>& tree_view, const auto& particles) {
            size_t total_particles = 0;
            for (size_t i = 0; i < tree_view.size(); ++i) {
                auto num_particles = this->aid_list_m.getNumParticlesInOctant(tree_view[i]);
                total_particles += num_particles;
            }
            const size_t col_width = 15;

            auto printer = [col_width](const auto& rank, const auto& tree_size,
                                       const auto& num_particles) {
                std::cerr << std::left << std::setw(col_width) << rank << std::left
                          << std::setw(col_width) << tree_size << std::left << std::setw(col_width)
                          << num_particles << std::endl;
            };
            printer("rank", "octs_now", "particles");
            printer(world_rank, tree_view.size(), total_particles);
        }

        std::ostream& print_octant(std::ostream& os, morton_code octant) {
            const grid_coordinate grid = morton_helper.decode(octant);
            const auto bounds          = bounds_t::bounds_from_grid_coord(
                root_bounds_m, grid, morton_helper.get_depth(octant), max_depth_m);

            return os << octant << " " << bounds.get_min() << " " << bounds.get_max();
        }

        template <typename Iterator>
        std::ostream& print_octant_list(std::ostream& os, Iterator begin, Iterator end) {
            // print_octant(os, 0); os << std::endl;
            for (Iterator it = begin; it != end; ++it) {
                print_octant(os, *it);
                os << std::endl;
            }

            return os;
        }

        std::ostream& print_particles(std::ostream& os, particle_t const& particles) {
            const size_t N =
                (world_rank == 0) ? particles.getLocalNum() / world_size : particles.getLocalNum();
            for (size_t i = 0; i < N; ++i) {
                os << i << " " << particles.R(i) << std::endl;
            }

            return os;
        }

        void particles_to_file(particle_t const& particles) {
            if (!enable_visualisation) {
                return;
            }

            std::string outputPath = std::string(IPPL_SOURCE_DIR)
                                     + "/src/OrthoTree/scripts/output/particles"
                                     + std::to_string(Comm->rank()) + ".txt";

            std::ofstream file(outputPath, std::ofstream::out);
            print_particles(file, particles);
            file.flush();
            file.close();
        }

        /**
         * @brief prints the octants to a file
         *
         * @param octants
         * @template T a container of morton codes that supports data() and size()
         */
        template <typename T>
        void octants_to_file(const T& octants) {
            if (!enable_visualisation) {
                return;
            }

            std::string outputPath = std::string(IPPL_SOURCE_DIR)
                                     + "/src/OrthoTree/scripts/output/octants"
                                     + std::to_string(Comm->rank()) + ".txt";

            std::ofstream file(outputPath, std::ofstream::out);
            print_octant_list(file, octants.data(), octants.data() + octants.size());
            file.flush();
            file.close();
        }
    };
#pragma endregion  // print_helpers

}  // namespace ippl

#include "OrthoTree.hpp"

// implementations of construction algos
#include "parallel_construction/algo01.hpp"
#include "parallel_construction/algo02.hpp"
#include "parallel_construction/algo03.hpp"
#include "parallel_construction/algo04.hpp"
#include "parallel_construction/algo05.hpp"
#include "parallel_construction/algo08.hpp"

// implementations of balancing algos
// #include "balancing/algo06.hpp"
// #include "balancing/algo07.hpp"
// #include "balancing/algo09.hpp"
// #include "balancing/algo10.hpp"
// #include "balancing/algo11.hpp"

#endif  // ORTHOTREE_GUARD

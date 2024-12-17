#ifndef AID_LIST_GUARD
#define AID_LIST_GUARD

#include <Kokkos_Vector.hpp>

#include "OrthoTree/helpers/BoundingBox.h"
#include "OrthoTree/helpers/MortonHelper.h"

namespace ippl {
    template <size_t Dim>
    class AidList {
        const size_t world_rank;
        const size_t world_size;
        const size_t max_depth;
        const Morton<Dim> morton_helper;

        Inform logger;

        /**
         * As of now we have two seperate views to store octants and particle ids.
         * They will only be accessed through the helper functions below, so if we decide to change
         * implementation we (should) only have to change those and everything sould still work.
         */
        Kokkos::View<morton_code*> octants;
        Kokkos::View<size_t*> particle_ids;

    public:
        AidList(size_t max_depth);

        void setLogLevel(int level) { logger.setOutputLevel(level); }
        void setLogOutput(bool enable) { logger.on(enable); }

        template <typename PLayout>
        void initialize(const BoundingBox<Dim>& root_bounds, PLayout const& particles);

        size_t size() const { return octants.extent(0); }

        morton_code getOctant(size_t idx) const { return octants(idx); }
        size_t getID(size_t idx) const { return particle_ids(idx); }

        void setOctant(morton_code octant, size_t idx) { octants(idx) = octant; }
        void setID(size_t particle_id, size_t idx) { particle_ids(idx) = particle_id; }

        auto getOctants() const { return Kokkos::subview(octants, Kokkos::ALL()); }
        auto getParticleIDs() const { return Kokkos::subview(particle_ids, Kokkos::ALL()); }

        auto& getOctants() { return octants; }
        auto& getParticleIDs() { return particle_ids; }

        void resize(size_t num_elements) {
            Kokkos::resize(octants, num_elements);
            Kokkos::resize(particle_ids, num_elements);
        }

        /**
         * @brief Returns the number of particles in the given octant.
         * This function assumes that the AidList is initialized.
         */
        size_t getNumParticlesInOctant(morton_code octant) const;

        /**
         * @brief Returns the lowest index s.t. octants(index-1) < octant <= octants(index)
         * INCLUSIVE
         */
        size_t getLowerBoundIndex(morton_code octant) const;

        /**
         * @brief Returns the highest index s.t. octants(index-1) <= octant < octants(index)
         */
        size_t getUpperBoundIndexExclusive(morton_code octant) const;

        /**
         * @brief Calculates the number of particles for each octant.
         * @warning This function assumes, that the AidList is on rank0, and that it is called from
         * all ranks simultaneously! It will freeze if the later is not fullfilled.
         *
         * @tparam Container Some container with: random_access iterators and functions '.data()'
         * and '.size()'
         */
        template <typename Container>
        Kokkos::View<size_t*> getNumParticlesInOctantsParallel(const Container& container);

        /**
         * @brief Returns the highest index s.t. octants(index-1) <= octant <= octants(index)
         */
        size_t getUpperBoundIndexInclusive(morton_code octant) const;

        /**
         * @brief Checks if the particles have been gathered on the rank it is called on.
         * Will throw if they are not gathered.
         */
        template <typename PLayout>
        bool is_gathered(ippl::ParticleBase<PLayout> const& particles);

        /**
         * @brief Initialises the AidList with {morton_code, particle_id} pairs.
         * This funciton assumes that all particles have been gathered on the given rank, will throw
         * if they are not.
         */
        void initialize_from_rank(
            size_t max_depth, const BoundingBox<Dim>& root_bounds,
            OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles);

        /**
         * @brief Gathers the particles on the rank that calls this function.
         * TODO: this is nice, but not needed at the moment
         */
        void gatherOnRank(
            OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles);

        /**
         * @brief This should send the min octants (so front/back) for each batch for each rank.
         * The distribution is as follows:
         *
         * rank > 0:
         *      rank_N = [(N-1) * batch_size, N * batch_size]
         * rank == 0:
         *      rank_0 = [(#ranks - 1) * batch_size, #octants]
         *
         * @return std::pair<morton_code, morton_code>
         */
        std::pair<morton_code, morton_code> getMinReqOctants();

        /**
         * @brief This function takes all elements between min_octant and max_octant in the AidList
         * and sends it to the requesting rank.
         * Rank 0 keeps the complete AidList to itself.
         */
        void innitFromOctants(morton_code min_octant, morton_code max_octant);
    };

}  // namespace ippl

#include "AidList.hpp"

#endif  // AID_LIST_GUARD
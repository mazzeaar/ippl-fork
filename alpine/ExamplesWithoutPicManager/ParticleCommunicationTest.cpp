// Particle Communication Test
//   Usage:
//     srun ./ParticleCommunicationTest
//                  <nx> [<ny>...] <Np> <Nt> <stype>
//                  <lbthres> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny...    = No. cell-centered points in the y-, z-, ...direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     lbfreq   = Load balancing frequency i.e., Number of time steps after which particle
//                load balancing should happen
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun ./ParticleCommunicationTest 128 128 128 10000 10 --overallocate 1.0 --info 10
//

#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

#include "ChargedParticles.hpp"

constexpr unsigned Dim=3;

const char* TestName = "ParticleCommunicationTest";

template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {
    using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
    // Output View for the random numbers
    view_type vals;

    // The GeneratorPool
    GeneratorPool rand_pool;

    T start, end;

    // Initialize all members
    generate_random(view_type vals_, GeneratorPool rand_pool_, T start_, T end_)
        : vals(vals_)
        , rand_pool(rand_pool_)
        , start(start_)
        , end(end_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        // Draw samples numbers from the pool as double in the range [start, end)
        for (unsigned d = 0; d < Dim; ++d) {
            vals(i)[d] = rand_gen.drand(start[d], end[d]);
        }

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};


int main(int argc, char* argv[]) {
    
    ippl::initialize(argc, argv);
    {
           
        setSignalHandler();
        
        Inform msg("ParticleCommunicationTest");
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        auto start = std::chrono::high_resolution_clock::now();
        int arg = 1;

        // cells per dim
        Vector_t<int, Dim> nr; 
        for(unsigned d = 0; d < Dim; d++){
            nr[d] = std::atoi(argv[arg++]);
        }
        
        // total number of particles and timesteps
        const size_type totalP = std::atoll(argv[arg++]);
        const unsigned int nt = std::atoi(argv[arg++]);

        msg << "Particle Communication Test" << endl << "nt " << nt << " Np= " << totalP << " grid= " << nr << endl;
        
        // pointer to particles
        using bunch_type = ChargedParticles<PLayout_t<double, Dim>, double, Dim>;
        std::unique_ptr<bunch_type> P; 
       
        // nd index for domain decomp 
        ippl::NDIndex<Dim> domain;
        for(unsigned i=0; i<Dim; i++){
            domain[i] = ippl::Index(nr[i]);
        }
        
        // domain bounds 
        Vector_t<double, Dim> rmin(0.0);
        Vector_t<double, Dim> rmax(1.0);

        //cell sizes
        Vector_t<double, Dim> hr = rmax / nr;
        Vector_t<double, Dim> origin = rmin;
        const double dt              = 1.0;
        
        // parallel dims
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        // periodic dims
        const bool isAllPeriodic = true;
        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);
        PLayout_t<double, Dim> PL(FL, mesh);
       
        // total charge 
        double Q           = -1562.5;

        // pointer to bunch of particles
        // choose solver as CG
        std::string solver = "CG";
        P = std::make_unique<bunch_type>(PL, hr, rmin, rmax, isParallel, Q, solver);
       
        // Calculate the nloc particles per rank to be created 
        P->nr_m        = nr;
        size_type nloc = totalP / ippl::Comm->size();
        int rest = (int)(totalP - nloc * ippl::Comm->size());
        if (ippl::Comm->rank() < rest)
            ++nloc;
        
        // create nloc particles per rank
        P->create(nloc);

        // calculate cell local bounds
        const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
        Vector_t<double, Dim> Rmin, Rmax;
        for (unsigned d = 0; d < Dim; ++d) {
            Rmin[d] = origin[d] + lDom[d].first() * hr[d];
            Rmax[d] = origin[d] + (lDom[d].last() + 1) * hr[d];
        }

        Inform m("local sizes");
        m << Rmin << " " << Rmax << endl;




    


    }

    ippl::finalize();
    return 0;

}


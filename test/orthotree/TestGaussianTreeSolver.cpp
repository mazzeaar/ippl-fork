#include "Ippl.h"
#include <Kokkos_Vector.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <random>
#include "Utility/ParameterList.h"
#include "Utility/IpplTimings.h"

KOKKOS_INLINE_FUNCTION double gaussian(double x, double y, double z, double sigma = 0.05,
                                       double mu = 0.5) {
    double pi        = Kokkos::numbers::pi_v<double>;
    double prefactor = (1 / Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma)) ;
    double r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return prefactor * exp(-r2 / (2 * sigma * sigma));
}

int main(int argc, char* argv[]) {
    
    ippl::initialize(argc, argv);
    {
        // Setup
        
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        playout_type PLayout;

        // Targets
        ippl::OrthoTreeParticle targets(PLayout);
        unsigned int nTargets = std::atoi(argv[1]);
        targets.create(nTargets);

        // Sources
        ippl::OrthoTreeParticle sources(PLayout);
        unsigned int nSources = nTargets;
        sources.create(nSources);

        // Random generators for position and charge
        std::mt19937_64 eng(3);
        std::uniform_real_distribution<double> posDis(0.0, 1.0);
        std::uniform_real_distribution<double> chargeDis(-20,20);

        // Generate target points
        
        for(unsigned int idx=0; idx<nTargets; ++idx){
            ippl::Vector<double,3> r = {posDis(eng), posDis(eng), posDis(eng)};
            targets.R(idx) = r;
            targets.rho(idx) = 0.0;
        }

        // Generate source points
        for(unsigned int idx=0; idx<nSources; ++idx){
            ippl::Vector<double,3> r = {posDis(eng), posDis(eng), posDis(eng)};
            sources.R(idx) = r;
            sources.rho(idx) = gaussian(r[0], r[1], r[2]);
        }
        


        // Tree Params
        ippl::ParameterList treeparams;
        treeparams.add("maxdepth",          10);
        treeparams.add("maxleafelements",   static_cast<int>(0.005 * nTargets));
        treeparams.add("boxmin",            0.0);
        treeparams.add("boxmax",            1.0);
        treeparams.add("sourceidx",         nTargets);

        // Solver Params
        ippl::ParameterList solverparams;
        solverparams.add("eps", 1e-6);

        
        ippl::TreeOpenPoissonSolver solver(targets, sources, treeparams, solverparams);

        // Timers 
        static auto explicit_timer = IpplTimings::getTimer("Explicitsol");
        static auto solver_timer = IpplTimings::getTimer("solver");

        IpplTimings::startTimer(explicit_timer);
        auto explicitsol = solver.ExplicitSolution();
        IpplTimings::stopTimer(explicit_timer);
        
        
            
        // timings
        
        IpplTimings::startTimer(solver_timer);
        solver.Solve();
        IpplTimings::stopTimer(solver_timer);
        

        double mse = 0.0;
        double mean = 0.0;
        for(unsigned int i=0; i<nTargets; ++i){
            mse += (Kokkos::abs(explicitsol(i)-targets.rho(i))) / Kokkos::abs(explicitsol(i));
            mean += Kokkos::abs(explicitsol(i)) / nTargets;
        }

        std::cout << mse/nTargets << "  " << mean << "\n";


        Kokkos::parallel_for("Reset target values", nTargets, 
        KOKKOS_LAMBDA(unsigned int i){
            targets.rho(i) = 0.0;
        });

            

        IpplTimings::print(std::string("timings.dat"));


         
    
    }
    ippl::finalize();
}
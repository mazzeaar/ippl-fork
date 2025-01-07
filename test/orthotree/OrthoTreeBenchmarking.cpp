#include "Ippl.h"

#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Vector.hpp>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "Utility/ParameterList.h"

#include "ArgParser.h"
#include "OrthoTree/OrthoTree.h"
#include "OrthoTree/helpers/BoundingBox.h"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using namespace ippl;

struct CustomDistributionFunctions {
    struct CDF {
        KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d,
            const double* params_p) const {
            return x
                + (params_p[d * 2 + 0] / params_p[d * 2 + 1])
                * Kokkos::sin(params_p[d * 2 + 1] * x);
        }
    };

    struct PDF {
        KOKKOS_INLINE_FUNCTION double operator()(double x, unsigned int d,
            double const* params_p) const {
            return 1.0 + params_p[d * 2 + 0] * Kokkos::cos(params_p[d * 2 + 1] * x);
        }
    };

    struct Estimate {
        KOKKOS_INLINE_FUNCTION double operator()(double u, unsigned int d,
            double const* params_p) const {
            return u + params_p[d] * 0.;
        }
    };
};

template <size_t Dim, typename ParticlePositions>
void initializeSpiral(ParticlePositions& particle_positions, const size_t num_particles);

template <size_t Dim, typename ParticlePositions>
void initializeRandom(ParticlePositions& particle_positions, const size_t num_particles);

template <size_t Dim, typename ParticlePositions>
auto initializeGaussian(ParticlePositions& particle_positions, const size_t num_particles);

template <size_t Dim, typename ParticlePositions>
auto initializeShell(ParticlePositions& particle_positions, const size_t num_particles);

template <size_t Dim, typename ParticlePositions> 
auto initializeLogNormal(ParticlePositions& particle_positions, const size_t num_particles);

template <size_t Dim>
auto initializeParticles(const size_t num_particles);

template <size_t Dim>
void run_experiment();

static void define_arguments() {
    // algorithm arguments
    ArgParser::add_argument<size_t>("dim", 2, "Dimension of the simulation");
    ArgParser::add_argument<size_t>("max_particles", 10, "Maximum particles per octant");
    ArgParser::add_argument<size_t>("max_depth", 8, "Maximum depth of the octree");
    ArgParser::add_argument<size_t>("num_particles_tot", 5000, "Number of particles in total");
    ArgParser::add_argument<size_t>("num_particles", 5000,
        "Number of particles per processor");
    ArgParser::add_argument<double>("min_bounds", 0.0, "Min coordinate of the bounding box");
    ArgParser::add_argument<double>("max_bounds", 1.0, "Max coordinate of the bounding box");
    ArgParser::add_argument<size_t>("seed", std::random_device{}(),
        "Seed for the random initialisation, default is random");
    ArgParser::add_argument<std::string>("dist", "spiral",
        "Type of particle distribution, one of: {random, spiral, gauss, shell, lognorm}");
    // output arguments
    ArgParser::add_argument<std::string>("print_stats", "true",
        "Sets the log level for our outputs.");
    ArgParser::add_argument<std::string>("enable_visualisation", "false",
        "Enables or disables the output of visualisation data");
    ArgParser::add_argument<size_t>("log_level", 0, "Sets the log level for our outputs.");

    ArgParser::add_argument<std::string>("parallel", "true",
        "true for parallel, false for sequential run");
    ArgParser::add_argument<std::string>("visualize_helper", "false",
        "Enables the replicate this run message");
    ArgParser::add_argument<size_t>("iterations", 1,
        "Number of iterations to run the benchmark for");
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        define_arguments();
        ArgParser::parse(argc, argv);
        const bool visualize_helper = ArgParser::get<bool>("visualize_helper");
        const size_t dimensions = ArgParser::get<size_t>("dim");
        const size_t iterations = ArgParser::get<size_t>("iterations");

        // Add the for loop here
        for (size_t iteration = 0; iteration < iterations; ++iteration) {
            // Optional: Print which iteration we're on
            if (Comm->rank() == 0) {
                std::cerr << "\nStarting iteration " << iteration + 1 << " of " << iterations
                    << std::endl;
            }

            // logging at the beginning in case the run crashes
            if (Comm->rank() == 0 && visualize_helper) {
                std::cerr << "Replicate this run with: \n"
                    << "./visualise.sh " << Comm->size() << " " << ArgParser::get_args()
                    << std::endl;
            }

            // logging to find where it hangs
            if (Comm->rank() == 0) {
                std::cerr << "Running OrthoTree benchmark in " << dimensions << "D" << std::endl;
            }

            if (dimensions == 2) {
                run_experiment<2>();
            }
            else if (dimensions == 3) {
                run_experiment<3>();
            }
            else {
                std::cerr << "We only specialise for 2D and 3D!";
                exit(1);
            }

            // logging at the end in case the run crashes
            if (Comm->rank() == 0 && visualize_helper) {
                std::cerr << "Finished run with: \n"
                    << "./visualise.sh " << Comm->size() << " " << ArgParser::get_args()
                    << std::endl;
            }

            if (Comm->rank() == 0 && visualize_helper) {
                std::cerr << "Replicate this run with: \n"
                    << "./visualise.sh " << Comm->size() << " " << ArgParser::get_args()
                    << std::endl;
            }
            IpplTimings::print();
            IpplTimings::print("timings.dat");
        }
    }
    ippl::finalize();
    return 0;
}


template <size_t Dim>
void run_experiment() {
    static_assert((Dim == 2 || Dim == 3) && "We only specialise for 2D and 3D!");

    const size_t max_particles = ArgParser::get<size_t>("max_particles");
    const size_t max_depth = ArgParser::get<size_t>("max_depth");

    const double min_bounds = ArgParser::get<double>("min_bounds");
    const double max_bounds = ArgParser::get<double>("max_bounds");

    ippl::OrthoTree<Dim> tree(
        max_depth, max_particles,
        ((Dim == 2) ? ippl::BoundingBox<Dim>({ min_bounds, min_bounds }, { max_bounds, max_bounds })
            : ippl::BoundingBox<Dim>({ min_bounds, min_bounds, min_bounds },
                { max_bounds, max_bounds, max_bounds })));

    const bool enable_visualisation = ArgParser::get<bool>("enable_visualisation");
    const size_t log_level = ArgParser::get<size_t>("log_level");
    const bool enable_stats = ArgParser::get<bool>("print_stats");
    const bool run_parallel = ArgParser::get<bool>("parallel");

    tree.setVisualisation(enable_visualisation);
    tree.setLogLevel(log_level);
    tree.setPrintStats(enable_stats);

    const size_t num_particles = ArgParser::get<size_t>("num_particles_tot");
    size_t num_particles_per_proc;
    if (num_particles != 5000) {
        num_particles_per_proc = num_particles / Comm->size();
    } else {
        num_particles_per_proc = ArgParser::get<size_t>("num_particles");
    }
    auto particles = initializeParticles<Dim>(num_particles_per_proc);

    IpplTimings::TimerRef timer = IpplTimings::getTimer("orthotree_build");
    IpplTimings::clearTimer(timer);
    IpplTimings::startTimer(timer);
    if (run_parallel)
        tree.build_tree(particles);
    else
        tree.build_tree_naive(particles);

    IpplTimings::stopTimer(timer);
    Comm->barrier();
}

template <size_t Dim>
auto initializeParticles(const size_t num_particles) {
    const std::string particle_distribution = ArgParser::get<std::string>("dist");

    typedef ippl::ParticleSpatialLayout<double, Dim> particle_layout_type;
    typedef ippl::OrthoTreeParticle<particle_layout_type> bunch_type;

    particle_layout_type PLayout;
    bunch_type bunch(PLayout);

    bunch.create(num_particles);

    typename bunch_type::particle_position_type::HostMirror positions_host =
        bunch.R.getHostMirror();
    if (particle_distribution == "spiral") {
        initializeSpiral<Dim>(positions_host, num_particles);
    }
    else if (particle_distribution == "random") {
        initializeRandom<Dim>(positions_host, num_particles);
    }
    else if (particle_distribution == "gauss") {
        initializeGaussian<Dim>(positions_host, num_particles);
    }
    else if (particle_distribution == "shell") {
        initializeShell<Dim>(positions_host, num_particles);
    }
    else if (particle_distribution == "lognorm") {
        initializeLogNormal<Dim>(positions_host, num_particles);
    }
    else {
        std::cerr << "Distribution: " << particle_distribution << " is not suppoerted!";
        exit(1);
    }

    Kokkos::deep_copy(bunch.R.getView(), positions_host);

    bunch.update();
    return bunch;
}

template <size_t Dim, typename ParticlePositions>
void initializeRandom(ParticlePositions& particle_positions, const size_t num_particles) {
    static_assert((Dim == 2 || Dim == 3) && "We only specialise for 2D and 3D!");

    const double min_bounds = ArgParser::get<double>("min_bounds");
    const double max_bounds = ArgParser::get<double>("max_bounds");
    const size_t seed = ArgParser::get<size_t>("seed");

    std::mt19937_64 eng(seed);
    std::uniform_real_distribution<double> unif(min_bounds, max_bounds);

    for (unsigned int i = 0; i < num_particles; ++i) {
        if constexpr (Dim == 2) {
            particle_positions(i) = ippl::Vector<double, Dim>{ unif(eng), unif(eng) };
        }
        else if constexpr (Dim == 3) {
            particle_positions(i) = ippl::Vector<double, Dim>{ unif(eng), unif(eng), unif(eng) };
        }
        else {
            std::cerr << "We only specialise for 2D and 3D!" << std::endl;
            exit(1);
        }
    }
}

template <size_t Dim, typename ParticlePositions>
void initializeSpiral(ParticlePositions& particle_positions, const size_t num_particles) {
    static_assert((Dim == 2 || Dim == 3) && "Spirals are only possible in 2D or 3D!");

    const double min_bounds = ArgParser::get<double>("min_bounds");
    const double max_bounds = ArgParser::get<double>("max_bounds");
    const double bounds_size = max_bounds - min_bounds;

    const size_t seed = ArgParser::get<size_t>("seed");
    std::mt19937_64 eng(seed);
    std::uniform_real_distribution<double> unif(min_bounds, max_bounds);

    // center of the bounding box
    const double center_x = bounds_size / 2;
    const double center_y = bounds_size / 2;
    const double center_z = (Dim == 3) ? bounds_size / 2 : 0.0;

    double armCount = 2.0;   // number of spiral arms
    double armTightness = -0.1;  // tightness of the spiral arms

    // max distance from the center
    double max_distance = 0.9 * bounds_size / 2;

    for (unsigned i = 0; i < num_particles; ++i) {
        double angle = unif(eng) * 2.0 * M_PI;
        double distance = unif(eng) * max_distance;

        double totalArmAngle = 5.0;

        for (int j = 0; j < armCount; ++j) {
            double armAngle = armTightness * angle + (j + 1) * distance / max_distance * 2.0 * M_PI;
            double x = center_x + distance * cos(armAngle + totalArmAngle);
            double y = center_y + distance * sin(armAngle + totalArmAngle);
            double z = (Dim == 3) ? center_z + distance * sin(angle) : 0.0;

            if constexpr (Dim == 2) {
                particle_positions(i) = { x, y };
            }
            else if constexpr (Dim == 3) {
                particle_positions(i) = { x, y, z };
            }

            totalArmAngle += armAngle;
        }
    }
}

template <size_t Dim, typename ParticlePositions>
auto initializeGaussian(ParticlePositions& particle_positions, const size_t num_particles) {
    static_assert((Dim == 2 || Dim == 3)
        && "Gaussian distribution is only supported for 2D and 3D!");

    const double mean =
        (ArgParser::get<double>("min_bounds") + ArgParser::get<double>("max_bounds")) / 2;
    const double std_dev =
        (ArgParser::get<double>("max_bounds") - ArgParser::get<double>("min_bounds")) / 6;
    const double min_bounds = ArgParser::get<double>("min_bounds");
    const double max_bounds = ArgParser::get<double>("max_bounds");
    const size_t seed = ArgParser::get<size_t>("seed");

    std::mt19937_64 eng(seed);
    std::normal_distribution<double> norm(mean, std_dev);

    for (unsigned int i = 0; i < num_particles; ++i) {
        if constexpr (Dim == 2) {
            double x = std::clamp(norm(eng), min_bounds, max_bounds);
            double y = std::clamp(norm(eng), min_bounds, max_bounds);
            particle_positions(i) = ippl::Vector<double, Dim>{ x, y };
        }
        else if constexpr (Dim == 3) {
            double x = std::clamp(norm(eng), min_bounds, max_bounds);
            double y = std::clamp(norm(eng), min_bounds, max_bounds);
            double z = std::clamp(norm(eng), min_bounds, max_bounds);
            particle_positions(i) = ippl::Vector<double, Dim>{ x, y, z };
        }
    }

    return particle_positions;
}

template <size_t Dim, typename ParticlePositions>
auto initializeShell(ParticlePositions& particle_positions, const size_t num_particles) {
    static_assert((Dim == 2 || Dim == 3) && "Shell distribution is only supported for 2D and 3D!");

    const double min_bounds = ArgParser::get<double>("min_bounds");
    const double max_bounds = ArgParser::get<double>("max_bounds");
    const double bounds_size = max_bounds - min_bounds;
    const double radius = 0.45 * bounds_size;  // Ensure particles stay inside bounds
    const double center = min_bounds + bounds_size / 2;
    const size_t seed = ArgParser::get<size_t>("seed");

    std::mt19937_64 eng(seed);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

    for (unsigned int i = 0; i < num_particles; ++i) {
        double angle = angle_dist(eng);

        if constexpr (Dim == 2) {
            double x = center + radius * std::cos(angle);
            double y = center + radius * std::sin(angle);
            particle_positions(i) = ippl::Vector<double, Dim>{ x, y };
        }
        else if constexpr (Dim == 3) {
            double z_angle = angle_dist(eng);
            double x = center + radius * std::sin(z_angle) * std::cos(angle);
            double y = center + radius * std::sin(z_angle) * std::sin(angle);
            double z = center + radius * std::cos(z_angle);
            particle_positions(i) = ippl::Vector<double, Dim>{ x, y, z };
        }
    }

    return particle_positions;
}

template <size_t Dim, typename ParticlePositions>
auto initializeLogNormal(ParticlePositions& particle_positions, const size_t num_particles) {
    static_assert((Dim == 2 || Dim == 3)
        && "Log-normal distribution is only supported for 2D and 3D!");

    const double mean =
        (ArgParser::get<double>("min_bounds") + ArgParser::get<double>("max_bounds")) / 2;
    const double std_dev =
        (ArgParser::get<double>("max_bounds") - ArgParser::get<double>("min_bounds")) / 6;
    const double min_bounds = ArgParser::get<double>("min_bounds");
    const double max_bounds = ArgParser::get<double>("max_bounds");
    const size_t seed = ArgParser::get<size_t>("seed");

    std::mt19937_64 eng(seed);
    std::lognormal_distribution<double> lognorm(std::log(mean), std_dev);

    for (unsigned int i = 0; i < num_particles; ++i) {
        if constexpr (Dim == 2) {
            double x = std::clamp(lognorm(eng), min_bounds, max_bounds);
            double y = std::clamp(lognorm(eng), min_bounds, max_bounds);
            particle_positions(i) = ippl::Vector<double, Dim>{ x, y };
        }
        else if constexpr (Dim == 3) {
            double x = std::clamp(lognorm(eng), min_bounds, max_bounds);
            double y = std::clamp(lognorm(eng), min_bounds, max_bounds);
            double z = std::clamp(lognorm(eng), min_bounds, max_bounds);
            particle_positions(i) = ippl::Vector<double, Dim>{ x, y, z };
        }
    }

    return particle_positions;
}
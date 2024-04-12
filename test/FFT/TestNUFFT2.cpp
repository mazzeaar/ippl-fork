#include "Ippl.h"
#include "Utility/ParameterList.h"

#include <iostream>
#include <typeinfo>
#include <array>
#include <Kokkos_Random.hpp>
#include <random>

template<class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout>
{

    Bunch(PLayout& playout)
    : ippl::ParticleBase<PLayout>(playout)
    {
        this->addAttribute(Q);
    }

    ~Bunch(){ }
    
    typedef ippl::ParticleAttrib<double> charge_container_type;
    charge_container_type Q;

};

template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random_particles {

  using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
  using value_type  = typename T::value_type;
  // Output View for the random numbers
  view_type x;

  // The GeneratorPool
  GeneratorPool rand_pool;

  T minU, maxU;

  // Initialize all members
  generate_random_particles(view_type x_, GeneratorPool rand_pool_, 
                  T& minU_, T& maxU_)
      : x(x_), rand_pool(rand_pool_), 
        minU(minU_), maxU(maxU_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    for (unsigned d = 0; d < Dim; ++d) {
        x(i)[d] = rand_gen.drand(minU[d], maxU[d]);
    }

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random_field {

  using view_type = typename ippl::detail::ViewType<T, Dim>::view_type;
  view_type f;

  // The GeneratorPool
  GeneratorPool rand_pool;

  // Initialize all members
  generate_random_field(view_type f_, GeneratorPool rand_pool_)
      : f(f_), rand_pool(rand_pool_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i, const size_t j, const size_t k) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    f(i, j, k).real() = rand_gen.drand(0.0, 1.0);
    f(i, j, k).imag() = rand_gen.drand(0.0, 1.0);

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

int main(int argc, char *argv[]) {
    ippl::initialize(argc, argv);
    {
        // Dimension = 3
        constexpr unsigned int dim = 3;
        const double pi = std::acos(-1.0);

        // Types
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;
        typedef ippl::Vector<double, 3> Vector_t;
        using size_type = ippl::detail::size_type;

        // Particle Layout and Bunch Type
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        typedef Bunch<playout_type> bunch_type;

        // Field Layout
        ippl::Vector<int, dim> pt = {32, 32, 32};
        ippl::Index I(pt[0]);
        ippl::Index J(pt[1]);
        ippl::Index K(pt[2]);
        ippl::NDIndex<dim> owned(I, J, K);

        std::array<bool, dim> isParallel;  
        isParallel.fill(false);

        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        std::array<double, dim> dx = {
            1 * pi / double(pt[0]),
            1 * pi / double(pt[1]),
            1 * pi / double(pt[2]),
        };

        Vector_t hx = {dx[0], dx[1], dx[2]};
        Vector_t origin = {0, 0, 0};
        ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

        playout_type pl;

        bunch_type bunch(pl);
        bunch.setParticleBC(ippl::BC::NO);

        size_type Np = std::pow(32,3);
        
        typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t> field_type;

        field_type field(mesh, layout);

        ippl::ParameterList fftParams;
        fftParams.add("use_finufft_defaults", true); 

        typedef ippl::NUFFT<3,double,Mesh_t, Centering_t> NUFFT_type;
        std::unique_ptr<NUFFT_type> nufft;
        
        int type = 2;
        
        Vector_t minU = origin;
        Vector_t maxU = {dx[0]*pt[0], dx[1]*pt[1], dx[2]*pt[2]};

        size_type nloc = Np/ippl::Comm->size();

        nufft = std::make_unique<NUFFT_type>(layout, nloc, type, fftParams);

        const int nghost = field.getNghost();
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        auto fview = field.getView();
        bunch.create(nloc);
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42));
        Kokkos::parallel_for(nloc,
                            generate_random_particles<Vector_t, Kokkos::Random_XorShift64_Pool<>, dim>(
                            bunch.R.getView(), rand_pool64, minU, maxU));
        
        Kokkos::parallel_for(mdrange_type({nghost, nghost, nghost},
                                        {fview.extent(0) - nghost,
                                        fview.extent(1) - nghost,
                                        fview.extent(2) - nghost}),
                            generate_random_field<Kokkos::complex<double>, Kokkos::Random_XorShift64_Pool<>, dim>(
                            field.getView(), rand_pool64));

        nufft->transform(bunch.R, bunch.Q, field);
        
        auto Q_result = Kokkos::create_mirror_view_and_copy(
                            Kokkos::HostSpace(), bunch.Q.getView());


        //Pick some target point to check. We choose it same as cuFINUFFT testcase cufinufft3d2_test.cu
        
        int idx = nloc/2;

        Kokkos::complex<double> reducedValue(0.0, 0.0);

        auto Rview = bunch.R.getView();

        Kokkos::complex<double> imag = {0.0, 1.0};

        Kokkos::parallel_reduce("NUDFT type2",
                                mdrange_type({0, 0, 0},
                                            {fview.extent(0) - 2 * nghost,
                                            fview.extent(1) - 2 * nghost,
                                            fview.extent(2) - 2 * nghost}),
                                KOKKOS_LAMBDA(const int i,
                                            const int j,
                                            const int k,
                                            Kokkos::complex<double>& valL) 
                                {
                                    ippl::Vector<int, 3> iVec = {i, j, k};
                                    double arg = 0.0;
                                    for(size_t d = 0; d < dim; ++d) {
                                        arg += (iVec[d] - (pt[d]/2)) * Rview(idx)[d];
                                    }

                                    valL += (Kokkos::cos(arg) 
                                    + imag * Kokkos::sin(arg)) * fview(i + nghost, j + nghost, k + nghost);
                                }, Kokkos::Sum<Kokkos::complex<double>>(reducedValue));
        
        double abs_error_real = std::fabs(reducedValue.real() - Q_result(idx));
        double rel_error_real = std::fabs(reducedValue.real() - Q_result(idx)) /std::fabs(reducedValue.real());
        //double abs_error_imag = std::fabs(reducedValue.imag() - Q_result(idx).imag());
        //double rel_error_imag = std::fabs(reducedValue.imag() - Q_result(idx).imag()) /std::fabs(reducedValue.imag());
    
        std::cout << "Abs Error in real part: " << std::setprecision(16) 
                << abs_error_real << " Rel. error in real part: " << std::setprecision(16) << rel_error_real << std::endl;
        //std::cout << "Abs Error in imag part: " << std::setprecision(16) 
        //          << abs_error_imag << " Rel. error in imag part: " << std::setprecision(16) << rel_error_imag << std::endl;


        //Kokkos::complex<double> max_error(0.0, 0.0);
        //MPI_Reduce(&max_error_local, &max_error, 1, 
        //           MPI_C_DOUBLE_COMPLEX, MPI_MAX, 0, Ippl::getComm());

    


    }
    ippl::finalize();
    return 0;
}

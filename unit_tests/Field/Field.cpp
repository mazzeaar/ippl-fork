//
// Unit test FieldTest
//   Test the functionality of the class Field.
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#include "Ippl.h"

#include <cmath>

#include "MultirankUtils.h"
#include "gtest/gtest.h"

class FieldTest : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    template <unsigned Dim>
    using mesh_type = ippl::UniformCartesian<double, Dim>;

    template <unsigned Dim>
    using centering_type = typename mesh_type<Dim>::DefaultCentering;

    template <unsigned Dim>
    using field_type = ippl::Field<double, Dim, mesh_type<Dim>, centering_type<Dim>>;

    template <unsigned Dim>
    using vfield_type =
        ippl::Field<ippl::Vector<double, Dim>, Dim, mesh_type<Dim>, centering_type<Dim>>;

    template <unsigned Dim>
    using layout_type = ippl::FieldLayout<Dim>;

    FieldTest()
        : nPoints(8) {
        setup(this);
    }

    template <unsigned Idx, unsigned Dim>
    void setupDim() {
        ippl::Index I(nPoints);
        std::array<ippl::Index, Dim> args;
        args.fill(I);
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(args);

        double dx = 1.0 / double(nPoints);
        ippl::Vector<double, Dim> hx;
        ippl::Vector<double, Dim> origin;

        ippl::e_dim_tag domDec[Dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned d = 0; d < Dim; d++) {
            domDec[d] = ippl::PARALLEL;
            hx[d]     = dx;
            origin[d] = 0;
        }

        auto layout            = std::make_shared<layout_type<Dim>>(owned, domDec);
        std::get<Idx>(layouts) = layout;

        std::get<Idx>(meshes) = std::make_shared<mesh_type<Dim>>(owned, hx, origin);

        std::get<Idx>(fields) = std::make_shared<field_type<Dim>>(*std::get<Idx>(meshes), *layout);
    }

    PtrCollection<std::shared_ptr, field_type> fields;
    PtrCollection<std::shared_ptr, mesh_type> meshes;
    PtrCollection<std::shared_ptr, layout_type> layouts;
    size_t nPoints;
};

template <unsigned Dim>
struct VFieldVal {
    using vfield_view_type = typename FieldTest::vfield_type<Dim>::view_type;
    const vfield_view_type vview;
    const ippl::NDIndex<Dim> lDom;

    double dx;
    int shift;

    VFieldVal(const vfield_view_type& view, const ippl::NDIndex<Dim>& lDom, double hx = 0,
              int shift = 0)
        : vview(view)
        , lDom(lDom)
        , dx(hx)
        , shift(shift) {}

    template <typename... Idx>
    KOKKOS_INLINE_FUNCTION void operator()(const Idx... args) const {
        ippl::Vector<double, Dim> coords = {(double)args...};
        vview(args...)                   = (0.5 + coords + lDom.first()) * dx;
    }
};

template <unsigned Dim>
struct FieldVal {
    using field_view_type = typename FieldTest::field_type<Dim>::view_type;
    const field_view_type view;

    const ippl::NDIndex<Dim> lDom;

    ippl::Vector<double, Dim> hx = 0;
    double dx                    = 0;
    int shift;

    ~FieldVal() {}

    FieldVal(const field_view_type& view, const ippl::NDIndex<Dim>& lDom, double hx = 0,
             int shift = 0)
        : view(view)
        , lDom(lDom)
        , dx(hx)
        , shift(shift) {}

    FieldVal(const field_view_type& view, const ippl::NDIndex<Dim>& lDom,
             ippl::Vector<double, Dim> hx, int shift = 0)
        : view(view)
        , lDom(lDom)
        , hx(hx)
        , shift(shift) {}

    // range policy tags
    struct Norm {};
    struct Integral {};
    struct Hessian {};

    const double pi = acos(-1.0);

    template <typename... Idx>
    KOKKOS_INLINE_FUNCTION void operator()(const Norm&, const Idx... args) const {
        double tot = (args + ...);
        for (unsigned d = 0; d < Dim; d++)
            tot += lDom[d].first();
        view(args...) = tot - 1;
    }

    template <typename... Idx>
    KOKKOS_INLINE_FUNCTION void operator()(const Integral&, const Idx... args) const {
        ippl::Vector<double, Dim> coords = {(double)args...};
        coords                           = (0.5 + coords + lDom.first() - shift) * dx;
        view(args...)                    = 1;
        for (const auto& x : coords)
            view(args...) *= sin(2 * pi * x);
    }

    template <typename... Idx>
    KOKKOS_INLINE_FUNCTION void operator()(const Hessian&, const Idx... args) const {
        ippl::Vector<double, Dim> coords = {(double)args...};
        coords                           = (0.5 + coords + lDom.first() - shift) * hx;
        view(args...)                    = 1;
        for (const auto& x : coords)
            view(args...) *= x;
    }
};

TEST_F(FieldTest, Sum) {
    double val = 1.0;

    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field = val;

        double sum = field->sum();

        ASSERT_DOUBLE_EQ(val * std::pow(nPoints, Dim), sum);
    };

    apply(check, fields);
}

TEST_F(FieldTest, Norm1) {
    double val = -1.5;

    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field = val;

        double norm1 = ippl::norm(*field, 1);

        ASSERT_DOUBLE_EQ(-val * std::pow(nPoints, Dim), norm1);
    };

    apply(check, fields);
}

TEST_F(FieldTest, Norm2) {
    double val = 1.5;

    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field = val;

        double norm2 = ippl::norm(*field);

        ASSERT_DOUBLE_EQ(std::sqrt(val * val * std::pow(nPoints, Dim)), norm2);
    };

    apply(check, fields);
}

TEST_F(FieldTest, NormInf) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        const ippl::NDIndex<Dim> lDom = field->getLayout().getLocalNDIndex();

        auto view = field->getView();
        FieldVal<Dim> fv(view, lDom);
        Kokkos::parallel_for("Set field",
                             field->template getRangePolicy<typename FieldVal<Dim>::Norm>(), fv);

        double normInf = ippl::norm(*field, 0);

        double val = -1.0 + Dim * nPoints;

        ASSERT_DOUBLE_EQ(val, normInf);
    };

    apply(check, fields);
}

TEST_F(FieldTest, VolumeIntegral) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        const ippl::NDIndex<Dim> lDom = field->getLayout().getLocalNDIndex();
        const int shift               = field->getNghost();

        const double dx = 1. / nPoints;
        auto view       = field->getView();

        FieldVal<Dim> fv(view, lDom, dx, shift);
        Kokkos::parallel_for(
            "Set field", field->template getRangePolicy<typename FieldVal<Dim>::Integral>(), fv);

        ASSERT_NEAR(field->getVolumeIntegral(), 0., 1e-15);
    };

    apply(check, fields);
}

TEST_F(FieldTest, VolumeIntegral2) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field          = 1.;
        double integral = field->getVolumeIntegral();
        double volume   = field->get_mesh().getMeshVolume();
        ASSERT_DOUBLE_EQ(integral, volume);
    };

    apply(check, fields);
}

TEST_F(FieldTest, Grad) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field = 1.;

        vfield_type<Dim> vfield(field->get_mesh(), field->getLayout());
        vfield = grad(*field);

        const int shift = vfield.getNghost();
        auto view       = vfield.getView();
        auto mirror     = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(mirror, view);

        nestedViewLoop<Dim>(mirror, shift, [&]<typename... Idx>(const Idx... args) {
            for (size_t d = 0; d < Dim; d++)
                ASSERT_DOUBLE_EQ(mirror(args...)[d], 0.);
        });
    };

    apply(check, fields);
}

TEST_F(FieldTest, Div) {
    auto dx    = 1. / nPoints;
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        vfield_type<Dim> vfield(field->get_mesh(), field->getLayout());
        auto view        = vfield.getView();
        const int vshift = vfield.getNghost();

        const ippl::NDIndex<Dim> lDom = vfield.getLayout().getLocalNDIndex();

        VFieldVal<Dim> fv(view, lDom, dx, vshift);
        Kokkos::parallel_for("Set field", vfield.getRangePolicy(vshift), fv);

        *field = div(vfield);

        const int shift = field->getNghost();
        auto mirror     = Kokkos::create_mirror_view(field->getView());
        Kokkos::deep_copy(mirror, field->getView());

        nestedViewLoop<Dim>(mirror, shift, [&]<typename... Idx>(const Idx... args) {
            ASSERT_DOUBLE_EQ(mirror(args...), Dim);
        });
    };

    apply(check, fields);
}

TEST_F(FieldTest, Curl) {
    // Restrict to 3D case for now
    auto mesh              = std::get<2>(meshes);
    auto layout            = std::get<2>(layouts);
    constexpr unsigned dim = 3;

    vfield_type<dim> vfield(*mesh, *layout);
    const int nghost = vfield.getNghost();
    auto view_field  = vfield.getView();

    auto lDom                        = layout->getLocalNDIndex();
    ippl::Vector<double, dim> hx     = mesh->getMeshSpacing();
    ippl::Vector<double, dim> origin = mesh->getOrigin();

    auto mirror = Kokkos::create_mirror_view(view_field);
    Kokkos::deep_copy(mirror, view_field);

    for (unsigned int gd = 0; gd < dim; ++gd) {
        bool dim0 = (gd == 0);
        bool dim1 = (gd == 1);
        bool dim2 = (gd == 2);

        for (size_t i = 0; i < view_field.extent(0); ++i) {
            for (size_t j = 0; j < view_field.extent(1); ++j) {
                for (size_t k = 0; k < view_field.extent(2); ++k) {
                    // local to global index conversion
                    const int ig = i + lDom[0].first() - nghost;
                    const int jg = j + lDom[1].first() - nghost;
                    const int kg = k + lDom[2].first() - nghost;

                    double x = (ig + 0.5) * hx[0] + origin[0];
                    double y = (jg + 0.5) * hx[1] + origin[1];
                    double z = (kg + 0.5) * hx[2] + origin[2];

                    mirror(i, j, k)[gd] = dim0 * (y * z) + dim1 * (x * z) + dim2 * (x * y);
                }
            }
        }
    }

    Kokkos::deep_copy(view_field, mirror);

    vfield_type<dim> result(*mesh, *layout);
    result = curl(vfield);

    const int shift = result.getNghost();
    auto view       = result.getView();
    mirror          = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);

    for (size_t i = shift; i < mirror.extent(0) - shift; ++i) {
        for (size_t j = shift; j < mirror.extent(1) - shift; ++j) {
            for (size_t k = shift; k < mirror.extent(2) - shift; ++k) {
                for (size_t d = 0; d < dim; ++d) {
                    ASSERT_DOUBLE_EQ(mirror(i, j, k)[d], 0.);
                }
            }
        }
    }
}

TEST_F(FieldTest, Hessian) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<mesh_type<Dim>>& mesh,
                                   std::shared_ptr<layout_type<Dim>>& layout) {
        typedef ippl::Vector<double, Dim> Vector_t;
        typedef ippl::Field<ippl::Vector<Vector_t, Dim>, Dim, mesh_type<Dim>, centering_type<Dim>>
            MField_t;

        field_type<Dim> field(*mesh, *layout);
        int nghost      = field.getNghost();
        auto view_field = field.getView();

        auto lDom                        = layout->getLocalNDIndex();
        ippl::Vector<double, Dim> hx     = mesh->getMeshSpacing();
        ippl::Vector<double, Dim> origin = mesh->getOrigin();

        FieldVal<Dim> fv(view_field, lDom, hx, nghost);
        Kokkos::parallel_for("Set field",
                             field.template getRangePolicy<typename FieldVal<Dim>::Hessian>(nghost),
                             fv);

        MField_t result(*mesh, *layout);
        result = hess(field);

        nghost             = result.getNghost();
        auto view_result   = result.getView();
        auto mirror_result = Kokkos::create_mirror_view(view_result);
        Kokkos::deep_copy(mirror_result, view_result);

        nestedViewLoop<Dim>(mirror_result, nghost, [&]<typename... Idx>(const Idx... args) {
            double det = 0;
            for (unsigned d = 0; d < Dim; d++)
                det += mirror_result(args...)[d][d];
            ASSERT_DOUBLE_EQ(det, 0.);
        });
    };

    apply(check, meshes, layouts);
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#include "Ippl.h"
#include <Kokkos_Vector.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <random>

#include "Utility/ParameterList.h"
#include "OrthoTree/OrthoTreeParticle.h"
#include "OrthoTree/OrthoTree.h"

#include "Utility/ParameterList.h"

#include "OrthoTreeRefactor/OrthoTree.h"
#include "OrthoTreeRefactor/OrthoTreeParticle.h"

static constexpr size_t dimensions = 3;
static constexpr size_t max_depth = 10;
static constexpr size_t max_elements = 20;

static constexpr bool print_structure = false;

#include <iostream>
#include <cassert>

void print_in_red(const std::string& message)
{
    std::cerr << "\033[1;31m" << message << "\033[0m" << std::endl;
}

template <typename NodeType>
void print_node_details(const NodeType& node, const std::string& node_name)
{
    std::cerr << "----------------------------------------\n";
    std::cerr << node_name << " details:" << std::endl;
    std::cerr << "----------------------------------------\n";
    std::cerr << "parent_m: " << node.parent_m << std::endl;
    std::cerr << "npoints_m: " << node.npoints_m << std::endl;

    std::cerr << "Bounding Box Min: [ ";
    for ( size_t i = 0; i < dimensions; ++i ) {
        std::cerr << node.boundingbox_m.Min[i] << " ";
    }
    std::cerr << "]" << std::endl;

    std::cerr << "Bounding Box Max: [ ";
    for ( size_t i = 0; i < dimensions; ++i ) {
        std::cerr << node.boundingbox_m.Max[i] << " ";
    }
    std::cerr << "]" << std::endl;

    std::cerr << "Children: [ ";
    for ( size_t i = 0; i < node.children_vector.size(); ++i ) {
        std::cerr << node.children_vector[i] << " ";
    }
    std::cerr << "]" << std::endl;

    std::cerr << "Data Vector: [ ";
    for ( size_t i = 0; i < node.data_vector.size(); ++i ) {
        std::cerr << node.data_vector[i] << " ";
    }
    std::cerr << "]" << std::endl;
    std::cerr << "----------------------------------------\n";

}

void compare_nodes(const ippl::OrthoTreeNode& old_node, const refactor::OrthoTreeNode<dimensions>& new_node)
{
    bool has_error = false;

    if ( new_node.parent_m != old_node.parent_m ) {
        print_in_red("Error: parents are not equal!");
        has_error = true;
    }

    if ( new_node.npoints_m != old_node.npoints_m ) {
        print_in_red("Error: npoints_m are not equal!");
        has_error = true;
    }

    for ( size_t i = 0; i < dimensions; ++i ) {
        if ( new_node.boundingbox_m.Min[i] != old_node.boundingbox_m.Min[i] ) {
            print_in_red("Error: bounding box min values are not equal at dimension " + std::to_string(i));
            has_error = true;
        }
        if ( new_node.boundingbox_m.Max[i] != old_node.boundingbox_m.Max[i] ) {
            print_in_red("Error: bounding box max values are not equal at dimension " + std::to_string(i));
            has_error = true;
        }
    }

    if ( new_node.children_vector.size() != old_node.children_vector.size() ) {
        print_in_red("Error: children vector sizes are not equal!");
        has_error = true;
    }
    else {
        for ( size_t i = 0; i < new_node.children_vector.size(); ++i ) {
            if ( new_node.children_vector[i] != old_node.children_vector[i] ) {
                print_in_red("Error: children are not equal at index " + std::to_string(i));
                has_error = true;
            }
        }
    }

    if ( new_node.data_vector.size() != old_node.data_vector.size() ) {
        print_in_red("Error: data vector sizes are not equal!");
        has_error = true;
    }
    else {
        for ( size_t i = 0; i < new_node.data_vector.size(); ++i ) {
            if ( new_node.data_vector[i] != old_node.data_vector[i] ) {
                print_in_red("Error: data vector elements are not equal at index " + std::to_string(i));
                has_error = true;
            }
        }
    }

    // If there were any errors, print the full details of both nodes for comparison
    if ( has_error ) {
        print_node_details(old_node, "Old Node");
        print_node_details(new_node, "New Node");
        assert(false && "Nodes are not equal. See the details above.");
    }
}


int main(int argc, char* argv[])
{
    std::cout << "RUNNING!\n";
    ippl::initialize(argc, argv);
    {
        // OCTREE CONSTRUCTION TEST 
        typedef ippl::ParticleSpatialLayout<double, dimensions> playout_type;
        playout_type PLayout;
        ippl::OrthoTreeParticle particles(PLayout);

        std::mt19937_64 eng(42);
        std::uniform_real_distribution<double> unif(0.25, 0.5);

        unsigned int n = 1000;
        particles.create(n);
        for ( unsigned int i = 0; i<n; ++i ) {
            particles.R(i) = ippl::Vector<double, dimensions> { unif(eng),unif(eng),unif(eng) };
            particles.rho(i) = 0;
        }

        ippl::OrthoTree old_tree(particles, 2, max_depth, max_elements, ippl::BoundingBox<dimensions>{{0, 0, 0}, { 1,1,1 }});
        refactor::OrthoTree<dimensions, max_depth> new_tree(particles, 2, max_elements, refactor::BoundingBox<dimensions>{{0, 0, 0}, { 1,1,1 }});

        auto old_ids = old_tree.CollectIds();

        for ( auto id : old_ids ) {
            compare_nodes(old_tree.nodes_m.value_at(id), new_tree.nodes_m.value_at(id));
        }

        if ( print_structure ) {
            new_tree.PrintStructure();
        }

        std::cout << "\e[0;32mpassed tests!\e[0m\n";
    }

    ippl::finalize();

    return 0;
}

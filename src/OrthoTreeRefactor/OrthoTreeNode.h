#ifndef ORTHOTREENODE_GUARD
#define ORTHOTREENODE_GUARD

#include "common.h"
#include "MortonHelper.h"
#include "BoundingBox.h"

namespace refactor {
    template <dim_type dimension>
    class OrthoTreeNode {
    public:
        // typedefs
        using box_type = BoundingBox<dimension>;
        using position_type = position_type_template<dimension>;

    public:
        vector_type<morton_code_type> children_vector;
        vector_type<morton_code_type> data_vector;
        box_type                            boundingbox_m;
        morton_code_type                 parent_m;
        unsigned int                        npoints_m;

    public:

        OrthoTreeNode() = default;
        OrthoTreeNode(box_type& box) : boundingbox_m(box) { }

        // just vector.push_back
        void AddChildren(morton_code_type child_key)
        {
            children_vector.push_back(child_key);
        }

        // inserts childkey in order (fascinating, isnt it)
        void AddChildInOrder(morton_code_type child_key)
        {
            morton_code_type idx = children_vector.lower_bound(0, children_vector.size(), child_key);
            if ( idx != children_vector.size() && children_vector[idx] == child_key ) {
                return;
            }
            else if ( idx == children_vector.size()-1 ) {
                children_vector.insert(children_vector.begin()+idx+1, child_key);
            }
            else {
                children_vector.insert(children_vector.begin()+idx, child_key);
            }
        }

        inline bool has_child(morton_code_type child_key) { return children_vector.find(child_key) != children_vector.end(); }
        inline bool is_internal_node() const { return !is_leaf_node(); }
        inline bool is_leaf_node() const { return children_vector.empty(); }
        inline Kokkos::vector<morton_code_type>const& GetChildren() const { return children_vector; }

        /**
         * @brief prints info stuff, the template should be removed someday, its only to test against the current implementation
         *
         * @tparam morton
         * @param key
         */
        template <typename morton>
        void print_info(morton_code_type key) const
        {
            static constexpr size_t width = 20;

            auto print_title = [ ] (auto title)
            {
                std::cout << std::left << std::setw(width) << title;
            };

            auto print_info_pair = [&] (auto a, auto b)
            {
                print_title(a);
                std::cout << std::left << std::setw(width) << b
                    << std::endl;
            };

            print_info_pair("ID:", key);
            print_info_pair("Depth:", morton::get_depth(key));
            print_info_pair("Parent:", parent_m);

            print_title("Children IDs:");
            for ( const auto& child : children_vector ) std::cout << child << " ";
            std::cout << std::endl;

            print_title("Grid IDs:");
            std::cout << "[ ";
            for ( const auto& id : morton::decode(key) ) std::cout << id << " ";
            std::cout << "]" << std::endl;

            boundingbox_m.print_info();
        }

    }; // Class node_type

} // namespace ippl
#endif // ORTHOTREENODE_GUARD
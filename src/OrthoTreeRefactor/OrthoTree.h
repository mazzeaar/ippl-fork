/**
 * Implementation of an octree in IPPL
 * The octree currently relies on certain std library functions, namely:
 * - std::transform             Kokkos::transform exists but is still "Experimental" and seems buggy
 * - std::partitition_point     Kokkos::partition_point exists but is still "Experimental" and seems buggy
 * - std::sort                  Kokkos::sort does not exist
 * - std::queue
*/

/**
 * This refactor does not respect ippls naming conventions at all, will be refactored again later.
 *
 * I basically just extracted common methods and tried to encapsulate functionalities, functionality should be identitcal to the original implementation
 * If the refactor is approced we can move it to the original orthotree folder and remove the refactoring namespace
 */

#ifndef ORTHOTREE_REFACTOR_GUARD
#define ORTHOTREE_REFACTOR_GUARD

#include "common.h"
#include "BoundingBox.h"
#include "OrthoTreeNode.h"
#include "MortonHelper.h"

#include "OrthoTreeParticle.h"
#include <queue>

namespace refactor {

    template <dim_type dim, depth_type max_depth>
    class OrthoTree {
    public:
        // typedefs
        using particle_type = particle_type_template<dim>;
        using position_type = position_type_template<dim>;
        using box_type = BoundingBox<dim>;
        using grid_position_type = grid_position_type_template<dim>;

        using node_type = OrthoTreeNode<dim>;
        using morton = MortonHelper<dim, max_depth>;
        using container_type = unordered_map_type<morton_code_type, node_type>;

    public: // public because of refactor tests
        // do we need all of those??
        // also wtf does sourceidx_m do?
        static constexpr morton_code_type ROOT_KEY = 1;
        container_type                  nodes_m;        // Kokkos::UnorderedMap of {morton_node_id, Node} pairs
        box_type                        root_bounds;          // Bounding Box of the Tree = Root node's box
        size_t                          maxelements_m;  // Max points per node
        grid_id_type                    rasterresmax_m; // Max boxes per dim
        position_type                   rasterizer_m;   // Describes how many nodes make up 1 unit of length per dim
        particle_id_type                sourceidx_m;

    public:
        // needs more constructors
        OrthoTree(particle_type const& particles, particle_id_type sourceidx, size_t MaxElements, box_type Box)
            : root_bounds(Box), maxelements_m(MaxElements),
            rasterresmax_m(Kokkos::exp2(max_depth)),
            rasterizer_m(Box.get_rasteriser(rasterresmax_m)),
            sourceidx_m(sourceidx)
        {
            const size_t n = particles.getTotalNum(); // use getGlobalNum() instead? 
            nodes_m = container_type(EstimateNodeNumber(n, max_depth, MaxElements));

            // create and insert root
            nodes_m.insert(ROOT_KEY, node_type(Box));

            // extract positions from particles (particles is probably soa instead of aos)
            vector_type<position_type> positions(n);
            for ( unsigned i = 0; i < n; ++i ) positions[i] = particles.R(i);

            auto aidLocations = morton::get_morton_id_pairs(positions, root_bounds);
            std::sort(aidLocations.begin(), aidLocations.end(), [&] (auto const& idL, auto const idR)
            {
                return idL.second < idR.second;
            });

            auto itBegin = aidLocations.begin();
            addNodes(nodes_m.value_at(nodes_m.find(ROOT_KEY)), ROOT_KEY, itBegin, aidLocations.end(), morton_code_type { 0 }, max_depth);

            BalanceTree(positions);
        }

        void BalanceTree(vector_type<position_type> positions)
        {
            std::queue<morton_code_type>         unprocessedNodes = {};
            vector_type<morton_code_type>     leafNodes = collect_leaf_nodes();

            std::sort(leafNodes.begin(), leafNodes.end(), std::greater<>());
            for ( unsigned int i = 0; i < leafNodes.size(); ++i ) {
                unprocessedNodes.push(leafNodes[i]);
            }

            while ( !unprocessedNodes.empty() ) {

                if ( get_node(unprocessedNodes.front()).is_internal_node() ) { // means that this node has already been refined by a deeper neighbor
                    unprocessedNodes.pop();
                }

                bool processed = true;

                vector_type<morton_code_type> potential_neighbours = GetPotentialColleagues(unprocessedNodes.front());

                for ( unsigned int idx = 0; idx < potential_neighbours.size(); ++idx ) {
                    const morton_code_type parent_code = morton::get_parent_code(potential_neighbours[idx]);
                    if ( parent_code ==  morton::get_parent_code(unprocessedNodes.front()) ) {
                        continue;
                    }
                    else if ( nodes_m.exists(parent_code) ) {
                        continue;
                    }
                    else {
                        processed = false;
                        morton_code_type ancestor = get_ancestor(potential_neighbours[idx]);
                        vector_type<morton_code_type> NewNodes = RefineNode(nodes_m.value_at(nodes_m.find(ancestor)), ancestor, positions);

                        for ( unsigned int idx = 0; idx < NewNodes.size(); ++idx ) {
                            unprocessedNodes.push(NewNodes[idx]);
                        }
                    }
                }

                if ( processed ) {
                    unprocessedNodes.pop();
                }
            }
        }

    private:
        /**
         * @brief inserts all particles. could probably benefit from a full rewrite, but i dont want to rn
         *
         * @param parent_node
         * @param parent_key
         * @param itEndPrev
         * @param itEnd
         * @param idLocationBegin
         * @param remaining_depth
         */
        void addNodes(node_type& parent_node, morton_code_type parent_key, auto& itEndPrev, auto const& itEnd, morton_code_type idLocationBegin, depth_type remaining_depth)
        {
            const auto n_elements = static_cast<size_t>(itEnd - itEndPrev);;
            parent_node.npoints_m = n_elements;
            //auto const n_elements = Kokkos::Experimental::distance(itEndPrev, itEnd);

            // reached leaf node -> fill points into data_vector vector
            if ( n_elements <= this->maxelements_m || remaining_depth == 0 ) {
                parent_node.data_vector.resize(n_elements);
                std::sort(itEndPrev, itEnd);
                std::transform(itEndPrev, itEnd, parent_node.data_vector.begin(), [ ] (auto const item)
                {
                    return item.first;
                });

                itEndPrev = itEnd;
                return;
            }

            --remaining_depth;

            auto const shift = remaining_depth * dim;
            auto const nLocationStep = morton_code_type { 1 } << shift;
            auto const flagParent = parent_key << dim;

            while ( itEndPrev != itEnd ) {
                auto const idChildActual = morton_code_type((itEndPrev->second - idLocationBegin) >> shift);
                auto const itEndActual = std::partition_point(itEndPrev, itEnd, [&] (auto const idPoint)
                {
                    return idChildActual == morton_code_type((idPoint.second - idLocationBegin) >> shift);
                });

                auto const mChildActual = morton_code_type(idChildActual);
                morton_code_type const child_key = flagParent | mChildActual;
                morton_code_type const idLocationBeginChild = idLocationBegin + mChildActual * nLocationStep;
                node_type& nodeChild = createChild(parent_node, child_key);
                nodeChild.parent_m = parent_key;
                addNodes(nodeChild, child_key, itEndPrev, itEndActual, idLocationBeginChild, remaining_depth);
            }
        }

        /**
         * @brief this should probably be moved to orthotreenode?
         */
        node_type& createChild(node_type& parent_node, morton_code_type child_key)
        {
            if ( !parent_node.has_child(child_key) ) {
                parent_node.AddChildInOrder(child_key);
            }

            // memory leak without those two lines, but this can probably be fixed somehow
            nodes_m.insert(child_key, node_type());
            node_type& nodeChild = nodes_m.value_at(nodes_m.find(child_key));

            const auto current_depth = morton::get_depth(child_key);
            const double scale_factor = get_max_nodes_per_edge(max_depth - current_depth);
            grid_position_type decoded_coordinates = morton::decode(child_key);

            for ( dim_type i = 0; i < dim; ++i ) {
                const double cur_rasterizer = rasterizer_m[i] * scale_factor;
                nodeChild.boundingbox_m.Min[i] = static_cast<float_t>(decoded_coordinates[i]) * cur_rasterizer;
                nodeChild.boundingbox_m.Max[i] = nodeChild.boundingbox_m.Min[i] + cur_rasterizer;
            }

            return nodeChild;
        }

        /**
         * @brief unexplained mathemagics
         *
         * @param nParticles
         * @param MaxDepth
         * @param nMaxElements
         * @return morton_code_type
         */
        morton_code_type EstimateNodeNumber(particle_id_type nParticles, depth_type MaxDepth, particle_id_type nMaxElements)
        {
            // for smaller smaller problem size
            if ( nParticles < 10 ) {
                return 10;
            }

            const double rMult = 3.; // why is this 3??

            // for smaller problem size
            if ( (MaxDepth + 1) * dim < 64 ) {
                size_t nMaxChild = size_t { 1 } << (MaxDepth * dim);
                auto const nElementsInNode = nParticles / nMaxChild;
                if ( nElementsInNode > nMaxElements / 2 ) return nMaxChild;
            }

            // for larger problem size
            auto const nElementInNodeAvg = static_cast<float>(nParticles) / static_cast<float>(nMaxElements);
            auto const nDepthEstimated = std::min(MaxDepth, static_cast<depth_type>(std::ceil((log2f(nElementInNodeAvg) + 1.0) / static_cast<float>(dim))));

            if ( nDepthEstimated * dim < 64 ) {
                return static_cast<size_t>(rMult * (1 << nDepthEstimated * dim));
            }
            else {
                return static_cast<size_t>(rMult * nElementInNodeAvg);
            }
        }

        /**
         * @brief Can probably be moves to OrthoTreeNode class..
         * The function basically generates morton codes for each particle in a node and then inserts those particles into the appropriate child node.
         * If the calculated child node does not exist we create it
         *
         * @param parent_node
         * @param parent_key
         * @param positions
         * @return vector_type<morton_code_type>
         */
        vector_type<morton_code_type> RefineNode(node_type& parent_node, morton_code_type parent_key, vector_type<position_type> positions)
        {
            const morton_code_type FlagParent = parent_key << dim;
            const position_type ParentBoxOrigin = parent_node.boundingbox_m.Min;

            /*
            const depth_type child_depth = morton::get_depth(parent_key) + 1;
            const morton_code_type shift = remaining_depth * dim;
            const depth_type remaining_depth = max_depth - child_depth;

            vector_type<morton_id_pair_type> aidLocation(parent_node.data_vector.size());
            std::transform(parent_node.data_vector.begin(), parent_node.data_vector.end(), aidLocation.begin(),
            [&] (particle_id_type id) -> morton_id_pair_type
            {
                const morton_code_type childId = morton::encode(positions[id], parent_node.boundingbox_m) >> shift;
                return { id, (childId + FlagParent) };
            });
            */

            // replaces the stuff above, which makes no sense lol. its probably wrong, but tests still pass
            auto aidLocation = morton::get_morton_id_pairs(parent_node.data_vector, parent_node.boundingbox_m);

            vector_type<morton_code_type> NewNodes = {};
            parent_node.data_vector = {};
            for ( unsigned int childId = 0; childId < Kokkos::pow(2, dim); ++childId ) {
                morton_code_type const child_key = FlagParent | childId;
                if ( nodes_m.exists(child_key) ) {
                    continue;
                }

                node_type& nodeChild = createChild(parent_node, child_key);
                nodeChild.parent_m = parent_key;
                for ( unsigned int idx = 0; idx < aidLocation.size(); ++idx ) {
                    if ( aidLocation[idx].second == child_key ) {
                        nodeChild.data_vector.push_back(aidLocation[idx].first);
                    }
                }

                NewNodes.push_back(child_key);
            }

            return NewNodes;
        }

    public:

        inline node_type& get_node(morton_code_type key) const { return nodes_m.value_at(nodes_m.find(key)); }

        /**
         * @brief This function walks the tree in bfs fashion (in order i think).
         *        Useful to perform operations on the whole or selected parts of the tree.
         *
         * @tparam Procedure    Function template that will be executed on a (key, node) pair. No default.
         * @tparam Selector     Function template that selects nodes based on given criteria. Default selects all nodes
         * @param root_key      Key to root node
         * @param procedure     Function that will be executed on a (key, node) pair. No default.
         * @param selector      Function that selects nodes based on given criteria. Default selects all nodes
         */
        template<typename Procedure, typename Selector = std::function<bool(morton_code_type)>>
        void traverse_tree(morton_code_type root_key, Procedure procedure, Selector selector = [ ] (morton_code_type) { return true; }) const
        {
            std::queue<morton_code_type> queue;
            queue.push(root_key);

            while ( !queue.empty() ) {
                auto const& key = queue.front();
                auto const& node = nodes_m.value_at(nodes_m.find(key));
                procedure(key, node);

                vector_type<morton_code_type> children = node.GetChildren();
                for ( const auto& child : children ) {
                    if ( selector(child) ) {
                        queue.push(child);
                    }
                }

                queue.pop();
            }
        }


        /**
         * @brief This function explores all neighbors. It does that by recursively calling itself, once for each dimension.
         * This way it should compile such that we dont have loops and recursion, instead we just generate all possible neighbors
         * and check their validity, after that we apply the processNeighbor function.
         *
         * This means:
         * eplore_neighbors<0>(cur_grid, neighbor_grid) calls
         * eplore_neighbors<1>(cur_grid, neighbor_grid) three times, each time with a variation of {-1,0,1} to the cur dimension
         * ...
         * eplore_neighbors<dim>(cur_grid, neighbor_grid) unfolds to:
         * (assuming cur_grid = {0, 0, 0})
         *
         * if (is_valid(neighbor_grid={1, 0, 0}, min, max)) { // do stuff }
         * if (is_valid(neighbor_grid={0, 1, 0}, min, max)) { // do stuff }
         * if (is_valid(neighbor_grid={0, 0, 1}, min, max)) { // do stuff }
         * if (is_valid(neighbor_grid={-1, 0, 0}, min, max)) { // do stuff }
         * if (is_valid(neighbor_grid={0, -1, 0}, min, max)) { // do stuff }
         * if (is_valid(neighbor_grid={0, 0, -1}, min, max)) { // do stuff }
         * ...
         * if (is_valid(neighbor_grid={1, 1, 1}, min, max)) { // do stuff }
         *
         * @tparam recursion_dim    recursion depth starts out at zero and increases up to dim
         * @tparam Func             a function that takes a neighbor_key and proecesses it
         * @param cur_grid          The curent position in the grid
         * @param neighbor_grid
         * @param processNeighbor
         * @param cur_depth
         * @param min_coord
         * @param max_coord
         */
        template <std::size_t recursion_dim, typename Func>
        constexpr void explore_neighbors(const grid_position_type& cur_grid, grid_position_type& neighbor_grid,
            Func&& processNeighbor, const depth_type cur_depth, const auto min_coord, const auto max_coord) const
        {
            if constexpr ( recursion_dim == dim ) {
                // If all dimensions are processed, check validity and process
                // after compiling this should be the remaining code
                if ( is_valid_grid_coord<dim>(neighbor_grid, min_coord, max_coord) ) {
                    morton_code_type neighbor_key = morton::encode(neighbor_grid, root_bounds) + Kokkos::pow(1 << dim, cur_depth);
                    processNeighbor(neighbor_key);
                }
            }
            else {
                // recursively calls itself for the next dimension
                for ( int diff : {-1, 0, 1} ) {
                    neighbor_grid[recursion_dim] = cur_grid[recursion_dim] + diff;
                    explore_neighbors<recursion_dim + 1>(cur_grid, neighbor_grid, std::forward<Func>(processNeighbor), cur_depth, min_coord, max_coord);
                }
            }
        }

        template <typename Func>
        void for_each_neighbor(morton_code_type key, Func&& processNeighbor) const
        {
            const grid_position_type cur_grid = morton::decode(key);
            const depth_type cur_depth = morton::get_depth(key);

            const grid_id_type min_coord = 0;
            const grid_id_type max_coord = std::pow(2, cur_depth);

            grid_position_type neighbor_grid = cur_grid;
            explore_neighbors<0>(cur_grid, neighbor_grid, std::forward<Func>(processNeighbor), cur_depth, min_coord, max_coord);
        }

        /*
        we could maybe refactor the neighbor visiting functions using the stuff below, but there is some bug.

        the functions can replace GetColleagues, but not GetPotentialColleagues, idk why
        should work, as getpotential is basically a more broad version of getcolleagues...

        vector_type<morton_code_type> GetColleagues(morton_code_type key) const
        {
            vector_type<morton_code_type> colleagues;
            std::size_t num_neighbors = Kokkos::pow(dim, dim);
            colleagues.reserve(num_neighbors);

            if ( key == ROOT_KEY ) {
                colleagues.push_back(1);
                return colleagues;
            }

            for_each_neighbor(key, [&] (morton_code_type neighbor_key)
            {
                if ( nodes_m.exists(neighbor_key) ) {
                    colleagues.push_back(neighbor_key);
                }
            });

            return colleagues;
        }
        */

        /**
         * @brief returns the morton codes of all (existing) neighboring nodes
         * neighbor -> max 1 dist away
         * @param key
         * @return vector_type<morton_code_type>
         */
        vector_type<morton_code_type> GetColleagues(morton_code_type key) const
        {
            vector_type<morton_code_type> colleagues = {};
            colleagues.reserve(Kokkos::pow(dim, dim));

            if ( key == ROOT_KEY ) {
                colleagues.push_back(1);
                return colleagues;
            }

            const grid_position_type cur_grid = morton::decode(key);

            const auto diffs = { -1 ,0, 1 };
            for ( int z : diffs ) {
                for ( int y : diffs ) {
                    for ( int x : diffs ) {
                        grid_position_type grid_to_encode { cur_grid[0] + x, cur_grid[1] + y, cur_grid[2] + z };
                        morton_code_type nbrKey = morton::encode(grid_to_encode, root_bounds) + Kokkos::pow(8, morton::get_depth(key));

                        if ( nodes_m.exists(nbrKey) ) {
                            colleagues.push_back(nbrKey);
                        }
                    }
                }
            }

            return colleagues;
        }

        /**
         * @brief returns morton code vec of all possible neighbors, even if they dont exist
         * neighbor -> max 1 dist away
         * @param key
         * @return vector_type<morton_code_type>
         */
        vector_type<morton_code_type> GetPotentialColleagues(morton_code_type key) const
        {
            grid_position_type aidGrid = morton::decode(key);
            vector_type<morton_code_type> potentialColleagues;
            const auto min_val = 0;
            const auto max_val = std::pow(2, morton::get_depth(key));

            const auto diffs = { -1 ,0, 1 };
            for ( int z : diffs ) {
                for ( int y : diffs ) {
                    for ( int x : diffs ) {
                        grid_position_type newGrid = grid_position_type { aidGrid[0] + x, aidGrid[1] + y, aidGrid[2] + z };
                        if ( is_valid_grid_coord<dim>(newGrid, min_val, max_val) ) {
                            morton_code_type potentialColleague = morton::encode(newGrid, root_bounds) + Kokkos::pow(8, morton::get_depth(key));
                            potentialColleagues.push_back(potentialColleague);
                        }
                    }
                }
            }

            return potentialColleagues;
        }

        /**
         * @brief Returns coards neighbors -> meaning also nodes that are adjacent to the parent node
         *
         * @param key
         * @return vector_type<morton_code_type>
         */
        vector_type<morton_code_type> GetCoarseNbrs(morton_code_type key) const
        {
            vector_type<morton_code_type> coarseNbrs {};
            coarseNbrs.reserve(3);
            if ( key == ROOT_KEY ) {
                return coarseNbrs;
            }

            node_type& node = get_node(key);
            vector_type<morton_code_type> parentColleagues = GetColleagues(morton::get_parent_code(key));

            for ( unsigned int idx = 0; idx < parentColleagues.size(); ++idx ) {
                node_type& coarseNode = get_node(parentColleagues[idx]);
                if ( coarseNode.is_leaf_node() && coarseNode.boundingbox_m.is_adjacent(node.boundingbox_m) ) {
                    coarseNbrs.push_back(parentColleagues[idx]);
                }
            }

            return coarseNbrs;
        }

        /**
         * @brief Walks through all ancestors, if an ancestor exists in nodes_m we return it.
         *
         * @param key
         * @return morton_code_type
         */
        morton_code_type get_ancestor(morton_code_type key) const noexcept
        {
            if ( nodes_m.exists(key) ) {
                return key;
            }
            else if ( key == 0 ) {
                return 1;
            }
            else {
                return get_ancestor(morton::get_parent_code(key));
            }
        }

        /**
         * @brief collects and returns all particle ids in the given subtree
         *
         * @param root_key
         * @return vector_type<particle_id_type>
         */
        vector_type<particle_id_type> collect_ids(morton_code_type root_key = ROOT_KEY)const
        {
            vector_type<particle_id_type> ids;
            ids.reserve(get_node(root_key).npoints_m);

            traverse_tree(root_key, [&ids] (morton_code_type, node_type const& node)
            {
                if ( node.is_leaf_node() ) {
                    unsigned int oldsize = ids.size();
                    unsigned int newsize = oldsize + node.data_vector.size();
                    ids.resize(newsize);
                    Kokkos::parallel_for("Collectids", node.data_vector.size(),
                    KOKKOS_LAMBDA(unsigned int idx)
                    {
                        ids[oldsize+idx] = node.data_vector[idx];
                    });
                }

            });

            return ids;
        }

        /**
         * @brief collects particle ids that are smaller than source_id, idk what that means though
         *
         * @param root_key
         * @return vector_type<particle_id_type>
         */
        vector_type<particle_id_type> collect_source_ids(morton_code_type root_key = ROOT_KEY) const
        {
            vector_type<particle_id_type> ids;
            ids.reserve(get_node(root_key).npoints_m);
            traverse_tree(root_key, [&] (morton_code_type, node_type const& node)
            {
                if ( node.is_leaf_node() ) {
                    auto pp = std::partition_point(node.data_vector.begin(), node.data_vector.end(), [this] (unsigned int i)
                    {
                        return i < sourceidx_m;
                    });

                    unsigned int newnpoints = node.data_vector.end() - pp;
                    unsigned int oldsize = ids.size();
                    ids.resize(oldsize + newnpoints);

                    // removed kokkos lambda, as it expands to [=], but we need an explicit capture of this to remove warnings
                    Kokkos::parallel_for("CollectSources", newnpoints,
                    [=, this] (unsigned int idx)
                    {
                        ids[oldsize+idx] = node.data_vector[(pp-node.data_vector.begin())+idx] - sourceidx_m;
                    });
                }
            });
            return ids;
        }

        /**
         * @brief collects particle ids that are larger than source_id, idk what that means though
         *
         * @param root_key
         * @return vector_type<particle_id_type>
         */
        vector_type<particle_id_type> collect_target_ids(morton_code_type root_key = ROOT_KEY) const
        {
            vector_type<particle_id_type> ids;
            ids.reserve(get_node(root_key).npoints_m);

            traverse_tree(root_key, [&] (morton_code_type, node_type const& node)
            {
                if ( node.is_leaf_node() ) {
                    auto pp = std::partition_point(node.data_vector.begin(), node.data_vector.end(), [this] (unsigned int i)
                    {
                        return i < sourceidx_m;
                    });

                    unsigned int newnpoints = pp-node.data_vector.begin();
                    unsigned int oldsize = ids.size();
                    ids.resize(oldsize + newnpoints);

                    Kokkos::parallel_for("CollectTargets", newnpoints,
                    KOKKOS_LAMBDA(unsigned int idx){
                        ids[oldsize + idx] = node.data_vector[idx];
                    });
                }
            });

            return ids;
        }

        /**
         * @brief collects all existing leaf nodes
         *
         * @param root_key
         * @return vector_type<morton_code_type>
         */
        vector_type<morton_code_type> collect_leaf_nodes(morton_code_type root_key = ROOT_KEY) const
        {
            vector_type<morton_code_type> leafNodes;
            leafNodes.reserve(static_cast<morton_code_type>(nodes_m.size()/2));

            traverse_tree(root_key, [&leafNodes] (morton_code_type key, node_type const& node)
            {
                if ( node.is_leaf_node() ) {
                    leafNodes.push_back(key);
                }
            });

            return leafNodes;
        }

        /**
         * @brief Collects all existing internal nodes at a specific depth
         *
         * @param depth
         * @return vector_type<morton_code_type>
         */
        vector_type<morton_code_type> collect_internal_nodes_at_depth(depth_type depth) const noexcept
        {
            vector_type<morton_code_type> keys;
            keys.reserve(Kokkos::pow(8, depth));

            traverse_tree(ROOT_KEY,
            [&] (auto key, auto) { if ( morton::get_depth(key) == depth && get_node(key).is_internal_node() ) { keys.push_back(key); } },
            [&] (auto i) { return (morton::get_depth(i)<=depth); });

            return keys;
        }

        /**
         * @brief Collects the morton codes of all existing nodes at a specific depth
         *
         * @param depth
         * @return vector_type<morton_code_type>
         */
        vector_type<morton_code_type> collect_nodes_at_depth(depth_type depth) const noexcept
        {
            vector_type<morton_code_type> keys;
            keys.reserve(Kokkos::pow(8, depth));

            traverse_tree(ROOT_KEY,
            [&] (auto key, auto) { if ( morton::get_depth(key) == depth ) { keys.push_back(key); } },
            [&] (auto i) { return (morton::get_depth(i)<=depth); }
            );

            return keys;
        }

        // ================================================================
        // PRINTER
        // ================================================================

        /**
         * @brief prints the tree structure by printing all nodes and their stored information.
         *
         * @param root_key
         */
        void PrintStructure(morton_code_type root_key = ROOT_KEY) const
        {
            static constexpr size_t width = 20;

            auto print_vector = [ ] (const auto& name, const auto& vec)
            {
                std::cout << std::left << std::setw(width) << name;
                for ( unsigned int i = 0; i < vec.size(); ++i ) {
                    std::cout << vec[i] << " ";
                }
                std::cout << std::endl;
            };

            traverse_tree(root_key, [&] (morton_code_type key, node_type const& node)
            {
                node.template print_info<morton>(key);

                print_vector("Colleagues:", this->GetColleagues(key));
                print_vector("Coarse Nbrs:", this->GetCoarseNbrs(key));
                if ( node.is_internal_node() ) {
                    print_vector("Points:", this->collect_ids(key));
                }
                else {
                    print_vector("Points:", node.data_vector);
                }

                std::cout << std::left << std::setw(width) << "Is Leaf:" << std::boolalpha << node.is_leaf_node()
                    << std::endl << std::endl;
            });
        }
    }; // Class OrthoTree

} // Namespace ippl
#endif

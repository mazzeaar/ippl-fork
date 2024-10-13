/**
 * Implementation of an octree in IPPL
 * The octree currently relies on certain std library functions, namely:
 * - std::transform             Kokkos::transform exists but is still "Experimental" and seems buggy
 * - std::partitition_point     Kokkos::partition_point exists but is still "Experimental" and seems buggy
 * - std::sort                  Kokkos::sort does not exist
 * - std::queue
*/


#ifndef ORTHOTREE_GUARD
#define ORTHOTREE_GUARD

#include <Kokkos_Vector.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <queue>
#include "OrthoTreeParticle.h"


<<<<<<< HEAD
namespace ippl
{

// Types

template <unsigned short int dim>
struct BoundingBox{
    ippl::Vector<double, dim> Min;
    ippl::Vector<double, dim> Max;
}; 

using dim_type                  = unsigned short int;
using child_id_type             = unsigned int;

// Node Types
using morton_node_id_type       = unsigned int;
using grid_id_type              = unsigned int;

// Particle Types
using particle_type             = OrthoTreeParticle<ippl::ParticleSpatialLayout<double,3>>;
using entity_id_type            = size_t;
using position_type             = ippl::Vector<double,3>;

using box_type                  = BoundingBox<3>;
using depth_type                = unsigned int;


/**
 * @class OrthoTreeNode
*/
class OrthoTreeNode
{
public: // Member Variables
    
    Kokkos::vector<morton_node_id_type> children_m;
    Kokkos::vector<morton_node_id_type> vid_m;
    box_type                            boundingbox_m;
    morton_node_id_type                 parent_m;
    unsigned int                        npoints_m;

public: // Member Functions

    void AddChildren(morton_node_id_type kChild){
        children_m.push_back(kChild);
    }

    void AddChildInOrder(morton_node_id_type kChild){
        
        //std::cout << "Adding Child in Order with key " << kChild << "\n";
        morton_node_id_type idx = children_m.lower_bound(0,children_m.size(), kChild);
        if(idx != children_m.size() && children_m[idx] == kChild) return;
        if(idx == children_m.size()-1){
            children_m.insert(children_m.begin()+idx+1, kChild);
            return;
        } 
        children_m.insert(children_m.begin()+idx, kChild);
        
    }

    bool HasChild(morton_node_id_type kChild){
        auto it = children_m.find(kChild);
        if(it == children_m.end()) return false;
        else return true;
    }

    bool IsAnyChildExist() const{
        return !children_m.empty();
    }

    Kokkos::vector<morton_node_id_type>const& GetChildren() const{
        return children_m;
    }

    ippl::Vector<double,3> GetCenter() const{
        ippl::Vector<double,3> center = {
                            boundingbox_m.Min[0] + (boundingbox_m.Max[0]-boundingbox_m.Min[0]) * 0.5,
                            boundingbox_m.Min[1] + (boundingbox_m.Max[1]-boundingbox_m.Min[1]) * 0.5,
                            boundingbox_m.Min[2] + (boundingbox_m.Max[2]-boundingbox_m.Min[2]) * 0.5
                        };
        return center;
    }

}; // Class OrthoTreeNode


// access of the map with key is done as follows : value = map.value_at(map.find(key))
// map.value_at(idx) return the value at idx (!=key) 
// map.find(key) returns the idx of key
using container_type = Kokkos::UnorderedMap<morton_node_id_type, OrthoTreeNode>;


class OrthoTree
{
private:
    
    container_type                  nodes_m;        // Kokkos::UnorderedMap of {morton_node_id, Node} pairs
    box_type                        box_m;          // Bounding Box of the Tree = Root node's box
    depth_type                      maxdepth_m;     // Max Depth of Tree
    size_t                          maxelements_m;  // Max points per node
    grid_id_type                    rasterresmax_m; // Max boxes per dim
    ippl::Vector<double,3>          rasterizer_m;   // Describes how many nodes make up 1 unit of length per dim
    dim_type                        dim_m = 3;      // Dimension (fixed at 3 for now)
    entity_id_type                  sourceidx_m;

public: // Constructors

    OrthoTree () = default;

    OrthoTree (particle_type const& particles, entity_id_type sourceidx, depth_type MaxDepth, size_t MaxElements, box_type Box)
    {

        this->box_m             = Box;
        this->maxdepth_m        = MaxDepth;
        this->maxelements_m     = MaxElements;
        this->sourceidx_m       = sourceidx;
        this->rasterresmax_m    = Kokkos::exp2(MaxDepth);
        this->rasterizer_m      = GetRasterizer(Box, this->rasterresmax_m);
        

        const size_t n = particles.getTotalNum(); // use getGlobalNum() instead? 
        nodes_m = container_type(EstimateNodeNumber(n, MaxDepth, MaxElements));

        // Root (key, node) pair
        morton_node_id_type kRoot = 1;
        OrthoTreeNode NodeRoot;
        NodeRoot.boundingbox_m = Box;
        nodes_m.insert(kRoot, NodeRoot);

        // Vector of point ids
        Kokkos::vector<entity_id_type> vidPoint(n);
        for(unsigned i=0; i<n; ++i) vidPoint[i] = i;

        // Vector of corresponding poisitions
        Kokkos::vector<position_type> positions(n);
        for(unsigned i=0; i<n; ++i) positions[i] = particles.R(i);
 
        // Vector of aid locations
        Kokkos::vector<Kokkos::pair<entity_id_type, morton_node_id_type>> aidLocations(n);

        // transformation of (id, position(id)) -> (id, morton(id))
        std::transform(positions.begin(), positions.end(), vidPoint.begin(), aidLocations.begin(),
        [=](position_type pt, entity_id_type id) -> Kokkos::pair<entity_id_type, morton_node_id_type>
        {
            return {id, this->GetLocationId(pt)};
        });
        std::sort(aidLocations.begin(), aidLocations.end(), [&](auto const& idL, auto const idR) {return idL.second < idR.second; });

        auto itBegin = aidLocations.begin();
        addNodes(nodes_m.value_at(nodes_m.find(kRoot)), kRoot, itBegin, aidLocations.end(), morton_node_id_type{0}, MaxDepth);
  
        BalanceTree(positions);

        //PrintStructure();

    }

    void BalanceTree(Kokkos::vector<position_type> positions){
        
        
        std::queue<morton_node_id_type>         unprocessedNodes    = {};
        Kokkos::vector<morton_node_id_type>     leafNodes           = this->GetLeafNodes();
        
        std::sort(leafNodes.begin(), leafNodes.end(), [](morton_node_id_type l, morton_node_id_type r){return l>r;});
        for(unsigned int idx=0; idx<leafNodes.size(); ++idx) {
            unprocessedNodes.push(leafNodes[idx]);
        }

        while(!unprocessedNodes.empty()){

            if (GetNode(unprocessedNodes.front()).IsAnyChildExist()) { // means that this node has already been refined by a deeper neighbor
                unprocessedNodes.pop();
            }
            
            bool processed = true;
            
            Kokkos::vector<morton_node_id_type> potentialNeighbours = this->GetPotentialColleagues(unprocessedNodes.front());
            
            for(unsigned int idx=0; idx<potentialNeighbours.size();++idx){
                
                if ((potentialNeighbours[idx]>>dim_m) == (unprocessedNodes.front()>>dim_m)) continue;
                
                else if (nodes_m.exists(potentialNeighbours[idx] >> dim_m)) continue;

                else {

                    processed = false;
                    
                    morton_node_id_type ancestor = this->GetNextAncestor(potentialNeighbours[idx]);     
        
                    Kokkos::vector<morton_node_id_type> NewNodes = this->RefineNode(nodes_m.value_at(nodes_m.find(ancestor)), ancestor, positions);

                    for (unsigned int idx=0; idx<NewNodes.size(); ++idx) {
                        unprocessedNodes.push(NewNodes[idx]);
                    }

                }  
            }

            if(processed) unprocessedNodes.pop();
        
        }

    }

private: // Aid Function for Constructor
    
    void addNodes(OrthoTreeNode& nodeParent, morton_node_id_type kParent, auto& itEndPrev, auto const& itEnd, morton_node_id_type idLocationBegin, depth_type nDepthRemain){

        const auto nElement = static_cast<size_t>(itEnd - itEndPrev);;
        nodeParent.npoints_m = nElement;
        //auto const nElement = Kokkos::Experimental::distance(itEndPrev, itEnd);

        // reached leaf node -> fill points into vid_m vector
        if(nElement <= this->maxelements_m || nDepthRemain == 0){
            nodeParent.vid_m.resize(nElement);
            std::sort(itEndPrev, itEnd);
            std::transform(itEndPrev, itEnd, nodeParent.vid_m.begin(), [](auto const item){return item.first;});
            itEndPrev = itEnd;
            return;
        }

        --nDepthRemain;

        auto const shift = nDepthRemain * dim_m;
        auto const nLocationStep = morton_node_id_type{1} << shift;
        auto const flagParent = kParent << dim_m;
        
        while(itEndPrev != itEnd){
            auto const idChildActual = morton_node_id_type((itEndPrev->second - idLocationBegin) >> shift);
            auto const itEndActual = std::partition_point(itEndPrev, itEnd, [&](auto const idPoint)
            {
                return idChildActual == morton_node_id_type((idPoint.second - idLocationBegin) >> shift);
            });

            auto const mChildActual = morton_node_id_type(idChildActual);
            morton_node_id_type const kChild = flagParent | mChildActual;
            morton_node_id_type const idLocationBeginChild = idLocationBegin + mChildActual * nLocationStep;
            OrthoTreeNode& nodeChild = this->createChild(nodeParent,/* idChildActual,*/ kChild);
            nodeChild.parent_m = kParent;
            this->addNodes(nodeChild, kChild, itEndPrev, itEndActual, idLocationBeginChild, nDepthRemain);
        }
    }

    OrthoTreeNode& createChild(OrthoTreeNode& nodeParent,/* child_id_type iChild,*/ morton_node_id_type kChild){
        
        
        
        if(!nodeParent.HasChild(kChild)) nodeParent.AddChildInOrder(kChild);
        
        // inserts {kChild, Node} pair into the unorderd map nodes_m
        nodes_m.insert(kChild, OrthoTreeNode());
        
        OrthoTreeNode& nodeChild = nodes_m.value_at(nodes_m.find(kChild)); // reference to newly created node
        
        position_type ptNodeMin = this->box_m.Min;
        position_type ptNodeMax;

        auto const nDepth   = this->GetDepth(kChild);
        auto mask           = morton_node_id_type{ 1 } << (nDepth * dim_m -1);
        
        double rScale = 1.0;
        for(depth_type iDepth=0; iDepth < nDepth; ++iDepth){
            rScale *= 0.5;
            for(dim_type iDimension = dim_m; iDimension > 0; --iDimension){
                bool const isGreater = (kChild & mask);
                ptNodeMin[iDimension-1] += isGreater * (this->box_m.Max[iDimension - 1] - this->box_m.Min[iDimension - 1]) * rScale;
                mask >>= 1;
            }
        }
        
        for(dim_type iDimension = 0; iDimension < dim_m; ++iDimension){
            ptNodeMax[iDimension] = ptNodeMin[iDimension] + (this->box_m.Max[iDimension] - this->box_m.Min[iDimension]) * rScale;
        }
        //std::cout << "creating child with key = " << kChild << " with box" << ptNodeMax <<" "<<ptNodeMin <<"\n";
        for(dim_type iDimension = 0; iDimension < dim_m; ++iDimension){
            
            nodeChild.boundingbox_m.Min[iDimension] = ptNodeMin[iDimension];
            nodeChild.boundingbox_m.Max[iDimension] = ptNodeMax[iDimension];
        }

        return nodeChild;
    }

    morton_node_id_type EstimateNodeNumber(entity_id_type nParticles, depth_type MaxDepth, entity_id_type nMaxElements){
        
        if (nParticles < 10) return 10;
        
        const double rMult = 3;

        // for smaller problem size
        if ((MaxDepth + 1) * dim_m < 64){
            size_t nMaxChild            = size_t{ 1 } << (MaxDepth * dim_m);
            auto const nElementsInNode  = nParticles / nMaxChild;
            if (nElementsInNode > nMaxElements / 2) return nMaxChild;
        }

        // for larget problem size
        auto const nElementInNodeAvg    = static_cast<float>(nParticles) / static_cast<float>(nMaxElements);
        auto const nDepthEstimated      = std::min(MaxDepth, static_cast<depth_type>(std::ceil((log2f(nElementInNodeAvg) + 1.0) / static_cast<float>(dim_m))));
        
        if (nDepthEstimated * dim_m < 64) return static_cast<size_t>(rMult * (1 << nDepthEstimated * dim_m));

        return static_cast<size_t>(rMult * nElementInNodeAvg);

    }

    ippl::Vector<double,3> GetRasterizer(box_type Box, grid_id_type nDivide){

        const double ndiv = static_cast<double>(nDivide);
        ippl::Vector<double,3> rasterizer;
        for(dim_type i=0; i<dim_m; ++i){
            double boxsize  = Box.Max[i] - Box.Min[i];
            rasterizer[i]   = boxsize == 0 ? 1.0 : (ndiv/boxsize);
        }
        
        return rasterizer;

    }

    Kokkos::vector<morton_node_id_type> RefineNode(OrthoTreeNode& NodeParent, morton_node_id_type kParent, Kokkos::vector<position_type> positions){

        const depth_type            newDepth        = this->GetDepth(kParent)+1;
        const depth_type            nDepthRemain    = maxdepth_m - newDepth;
        const morton_node_id_type   shift           = nDepthRemain * dim_m;

        const position_type ParentBoxOrigin = NodeParent.boundingbox_m.Min;
        Kokkos::vector<Kokkos::pair<entity_id_type,morton_node_id_type>> aidLocation(NodeParent.vid_m.size());
        const morton_node_id_type FlagParent = kParent << dim_m;

        Kokkos::vector<morton_node_id_type> NewNodes={};

        std::transform(NodeParent.vid_m.begin(), NodeParent.vid_m.end(), aidLocation.begin(), [&](entity_id_type id) -> Kokkos::pair<entity_id_type, morton_node_id_type>
        {
            ippl::Vector<grid_id_type,3> gridId = this->GetRelativeGridId(positions[id],ParentBoxOrigin);
            const morton_node_id_type childId = this->MortonEncode(gridId)>>shift;
            return { id, (childId + FlagParent)};
        });
        
        NodeParent.vid_m = {};

        for (unsigned int childId=0; childId<Kokkos::pow(2,dim_m); ++childId){
            
            morton_node_id_type const kChild = FlagParent | childId;
            if(nodes_m.exists(kChild)) continue;
            OrthoTreeNode& nodeChild = this->createChild(NodeParent, kChild);
            nodeChild.parent_m=kParent;
            for (unsigned int idx=0; idx<aidLocation.size(); ++idx){

                if (aidLocation[idx].second == kChild) {
                    nodeChild.vid_m.push_back(aidLocation[idx].first);
                }
            }

                NewNodes.push_back(kChild);
            
        }
        
        return NewNodes;

    }

public: // Morton Encoding Functions

    morton_node_id_type GetLocationId(position_type pt){

        return MortonEncode(GetGridId(pt));

    }

    ippl::Vector<grid_id_type,3> GetGridId(position_type pt){

        ippl::Vector<grid_id_type,3> aid;
        
        for(dim_type i=0; i<dim_m; ++i){
            double r_i          = pt[i] - box_m.Min[i];
            double raster_id    = r_i * rasterizer_m[i];
            aid[i]              = static_cast<grid_id_type>(raster_id);
        }

        return aid;

    }

    ippl::Vector<grid_id_type, 3> GetRelativeGridId(position_type pt, position_type origin){

        ippl::Vector<grid_id_type,3> aid;
        
        for(dim_type i=0; i<dim_m; ++i){
            double r_i          = pt[i] - origin[i];
            double raster_id    = r_i * rasterizer_m[i];
            aid[i]              = static_cast<grid_id_type>(raster_id);
        }

        return aid;
    }

    // Only works for dim = 3 for now
    morton_node_id_type MortonEncode(ippl::Vector<grid_id_type,3> aidGrid) const{

        assert(dim_m == 3);

        return (part1By2(aidGrid[2]) << 2) + (part1By2(aidGrid[1]) << 1) + part1By2(aidGrid[0]);

    }

    ippl::Vector<grid_id_type,3> MortonDecode(morton_node_id_type kNode, depth_type nDepthMax) const{
        
        ippl::Vector<grid_id_type, 3> aidGrid;
        auto const nDepth = GetDepth(kNode);

        auto mask = morton_node_id_type{ 1 };
        for (depth_type iDepth = nDepthMax - nDepth, shift = 0; iDepth < nDepthMax; ++iDepth){
            for (dim_type iDimension = 0; iDimension < dim_m; ++iDimension, ++shift){
                aidGrid[iDimension] |= (kNode & mask) >> (shift - iDepth);
                mask <<= 1;
            }
        }

        return aidGrid;

    }

    static constexpr morton_node_id_type part1By2(grid_id_type n) noexcept{

        // n = ----------------------9876543210 : Bits initially
        // n = ------98----------------76543210 : After (1)
        // n = ------98--------7654--------3210 : After (2)
        // n = ------98----76----54----32----10 : After (3)
        // n = ----9--8--7--6--5--4--3--2--1--0 : After (4)
        n = (n ^ (n << 16)) & 0xff0000ff; // (1)
        n = (n ^ (n << 8)) & 0x0300f00f; // (2)
        n = (n ^ (n << 4)) & 0x030c30c3; // (3)
        n = (n ^ (n << 2)) & 0x09249249; // (4)
        return static_cast<morton_node_id_type>(n);

    }

public: // Node Info Functions


    bool IsValidKey(uint64_t key) const{ 
        
        return key; 
    
    }

    void VisitNodes(morton_node_id_type kRoot, auto procedure) const{
        
        auto q = std::queue<morton_node_id_type>();

        for(q.push(kRoot); !q.empty(); q.pop()){

            auto const& key = q.front();
            auto const& node = nodes_m.value_at(nodes_m.find(key));
            procedure(key, node);

            Kokkos::vector<morton_node_id_type> children = node.GetChildren();
            for(unsigned int i=0; i<children.size(); ++i){
                q.push(children[i]);
=======
namespace ippl {

    // Types

    template <unsigned short int dim>
    struct BoundingBox {
        ippl::Vector<double, dim> Min;
        ippl::Vector<double, dim> Max;
    };

    using dim_type = unsigned short int;
    using child_id_type = unsigned int;

    // Node Types
    using morton_node_id_type = unsigned int;
    using grid_id_type = unsigned int;

    // Particle Types
    using particle_type = OrthoTreeParticle<ippl::ParticleSpatialLayout<double, 3>>;
    using entity_id_type = size_t;
    using position_type = ippl::Vector<double, 3>;

    using box_type = BoundingBox<3>;
    using depth_type = unsigned int;


    /**
     * @class OrthoTreeNode
    */
    class OrthoTreeNode {
    public: // Member Variables

        Kokkos::vector<morton_node_id_type> children_vector;
        Kokkos::vector<morton_node_id_type> data_vector;
        box_type                            boundingbox_m;
        morton_node_id_type                 parent_m;
        unsigned int                        npoints_m;

    public: // Member Functions

        void AddChildren(morton_node_id_type kChild)
        {
            children_vector.push_back(kChild);
        }

        void AddChildInOrder(morton_node_id_type kChild)
        {

            //std::cout << "Adding Child in Order with key " << kChild << "\n";
            morton_node_id_type idx = children_vector.lower_bound(0, children_vector.size(), kChild);
            if ( idx != children_vector.size() && children_vector[idx] == kChild ) return;
            if ( idx == children_vector.size()-1 ) {
                children_vector.insert(children_vector.begin()+idx+1, kChild);
                return;
            }
            children_vector.insert(children_vector.begin()+idx, kChild);

        }

        bool HasChild(morton_node_id_type kChild)
        {
            auto it = children_vector.find(kChild);
            if ( it == children_vector.end() ) return false;
            else return true;
        }

        bool IsAnyChildExist() const
        {
            return !children_vector.empty();
        }

        Kokkos::vector<morton_node_id_type>const& GetChildren() const
        {
            return children_vector;
        }

        ippl::Vector<double, 3> GetCenter() const
        {
            ippl::Vector<double, 3> center = {
                                boundingbox_m.Min[0] + (boundingbox_m.Max[0]-boundingbox_m.Min[0]) * 0.5,
                                boundingbox_m.Min[1] + (boundingbox_m.Max[1]-boundingbox_m.Min[1]) * 0.5,
                                boundingbox_m.Min[2] + (boundingbox_m.Max[2]-boundingbox_m.Min[2]) * 0.5
            };
            return center;
        }

    }; // Class OrthoTreeNode


    // access of the map with key is done as follows : value = map.value_at(map.find(key))
    // map.value_at(idx) return the value at idx (!=key) 
    // map.find(key) returns the idx of key
    using container_type = Kokkos::UnorderedMap<morton_node_id_type, OrthoTreeNode>;


    class OrthoTree {
    public:
        container_type                  nodes_m;        // Kokkos::UnorderedMap of {morton_node_id, Node} pairs
        box_type                        box_m;          // Bounding Box of the Tree = Root node's box
        depth_type                      maxdepth_m;     // Max Depth of Tree
        size_t                          maxelements_m;  // Max points per node
        grid_id_type                    rasterresmax_m; // Max boxes per dim
        ippl::Vector<double, 3>          rasterizer_m;   // Describes how many nodes make up 1 unit of length per dim
        dim_type                        dim_m = 3;      // Dimension (fixed at 3 for now)
        entity_id_type                  sourceidx_m;

    public: // Constructors

        OrthoTree() = default;

        OrthoTree(particle_type const& particles, entity_id_type sourceidx, depth_type MaxDepth, size_t MaxElements, box_type Box)
        {

            this->box_m = Box;
            this->maxdepth_m = MaxDepth;
            this->maxelements_m = MaxElements;
            this->sourceidx_m = sourceidx;
            this->rasterresmax_m = Kokkos::exp2(MaxDepth);
            this->rasterizer_m = GetRasterizer(Box, this->rasterresmax_m);


            const size_t n = particles.getTotalNum(); // use getGlobalNum() instead? 
            nodes_m = container_type(EstimateNodeNumber(n, MaxDepth, MaxElements));

            // Root (key, node) pair
            morton_node_id_type kRoot = 1;
            OrthoTreeNode NodeRoot;
            NodeRoot.boundingbox_m = Box;
            nodes_m.insert(kRoot, NodeRoot);

            // Vector of point ids
            Kokkos::vector<entity_id_type> vidPoint(n);
            for ( unsigned i = 0; i<n; ++i ) vidPoint[i] = i;

            // Vector of corresponding poisitions
            Kokkos::vector<position_type> positions(n);
            for ( unsigned i = 0; i<n; ++i ) positions[i] = particles.R(i);

            // Vector of aid locations
            Kokkos::vector<Kokkos::pair<entity_id_type, morton_node_id_type>> aidLocations(n);

            // transformation of (id, position(id)) -> (id, morton(id))
            std::transform(positions.begin(), positions.end(), vidPoint.begin(), aidLocations.begin(),
            [=, this] (position_type pt, entity_id_type id) -> Kokkos::pair<entity_id_type, morton_node_id_type>
            {
                return { id, this->GetLocationId(pt) };
            });
            std::sort(aidLocations.begin(), aidLocations.end(), [&] (auto const& idL, auto const idR) { return idL.second < idR.second; });

            auto itBegin = aidLocations.begin();
            addNodes(nodes_m.value_at(nodes_m.find(kRoot)), kRoot, itBegin, aidLocations.end(), morton_node_id_type { 0 }, MaxDepth);

            BalanceTree(positions);

            //PrintStructure();

        }

        void BalanceTree(Kokkos::vector<position_type> positions)
        {


            std::queue<morton_node_id_type>         unprocessedNodes = {};
            Kokkos::vector<morton_node_id_type>     leafNodes = this->GetLeafNodes();

            std::sort(leafNodes.begin(), leafNodes.end(), [ ] (morton_node_id_type l, morton_node_id_type r) { return l>r; });
            for ( unsigned int idx = 0; idx<leafNodes.size(); ++idx ) {
                unprocessedNodes.push(leafNodes[idx]);
            }

            while ( !unprocessedNodes.empty() ) {

                if ( GetNode(unprocessedNodes.front()).IsAnyChildExist() ) { // means that this node has already been refined by a deeper neighbor
                    unprocessedNodes.pop();
                }

                bool processed = true;

                Kokkos::vector<morton_node_id_type> potentialNeighbours = this->GetPotentialColleagues(unprocessedNodes.front());

                for ( unsigned int idx = 0; idx<potentialNeighbours.size();++idx ) {

                    if ( (potentialNeighbours[idx]>>dim_m) == (unprocessedNodes.front()>>dim_m) ) continue;

                    else if ( nodes_m.exists(potentialNeighbours[idx] >> dim_m) ) continue;

                    else {

                        processed = false;

                        morton_node_id_type ancestor = this->GetNextAncestor(potentialNeighbours[idx]);

                        Kokkos::vector<morton_node_id_type> NewNodes = this->RefineNode(nodes_m.value_at(nodes_m.find(ancestor)), ancestor, positions);

                        for ( unsigned int idx = 0; idx<NewNodes.size(); ++idx ) {
                            unprocessedNodes.push(NewNodes[idx]);
                        }

                    }
                }

                if ( processed ) unprocessedNodes.pop();

            }

        }

    private: // Aid Function for Constructor

        void addNodes(OrthoTreeNode& nodeParent, morton_node_id_type kParent, auto& itEndPrev, auto const& itEnd, morton_node_id_type idLocationBegin, depth_type nDepthRemain)
        {

            const auto nElement = static_cast<size_t>(itEnd - itEndPrev);;
            nodeParent.npoints_m = nElement;
            //auto const nElement = Kokkos::Experimental::distance(itEndPrev, itEnd);

            // reached leaf node -> fill points into data_vector vector
            if ( nElement <= this->maxelements_m || nDepthRemain == 0 ) {
                nodeParent.data_vector.resize(nElement);
                std::sort(itEndPrev, itEnd);
                std::transform(itEndPrev, itEnd, nodeParent.data_vector.begin(), [ ] (auto const item) { return item.first; });
                itEndPrev = itEnd;
                return;
            }

            --nDepthRemain;

            auto const shift = nDepthRemain * dim_m;
            auto const nLocationStep = morton_node_id_type { 1 } << shift;
            auto const flagParent = kParent << dim_m;

            while ( itEndPrev != itEnd ) {
                auto const idChildActual = morton_node_id_type((itEndPrev->second - idLocationBegin) >> shift);
                auto const itEndActual = std::partition_point(itEndPrev, itEnd, [&] (auto const idPoint)
                {
                    return idChildActual == morton_node_id_type((idPoint.second - idLocationBegin) >> shift);
                });

                auto const mChildActual = morton_node_id_type(idChildActual);
                morton_node_id_type const kChild = flagParent | mChildActual;
                morton_node_id_type const idLocationBeginChild = idLocationBegin + mChildActual * nLocationStep;
                OrthoTreeNode& nodeChild = this->createChild(nodeParent,/* idChildActual,*/ kChild);
                nodeChild.parent_m = kParent;
                this->addNodes(nodeChild, kChild, itEndPrev, itEndActual, idLocationBeginChild, nDepthRemain);
            }
        }

        OrthoTreeNode& createChild(OrthoTreeNode& nodeParent,/* child_id_type iChild,*/ morton_node_id_type kChild)
        {



            if ( !nodeParent.HasChild(kChild) ) nodeParent.AddChildInOrder(kChild);

            // inserts {kChild, Node} pair into the unorderd map nodes_m
            nodes_m.insert(kChild, OrthoTreeNode());

            OrthoTreeNode& nodeChild = nodes_m.value_at(nodes_m.find(kChild)); // reference to newly created node

            position_type ptNodeMin = this->box_m.Min;
            position_type ptNodeMax;

            auto const nDepth = this->GetDepth(kChild);
            auto mask = morton_node_id_type { 1 } << (nDepth * dim_m -1);

            double rScale = 1.0;
            for ( depth_type iDepth = 0; iDepth < nDepth; ++iDepth ) {
                rScale *= 0.5;
                for ( dim_type iDimension = dim_m; iDimension > 0; --iDimension ) {
                    bool const isGreater = (kChild & mask);
                    ptNodeMin[iDimension-1] += isGreater * (this->box_m.Max[iDimension - 1] - this->box_m.Min[iDimension - 1]) * rScale;
                    mask >>= 1;
                }
            }

            for ( dim_type iDimension = 0; iDimension < dim_m; ++iDimension ) {
                ptNodeMax[iDimension] = ptNodeMin[iDimension] + (this->box_m.Max[iDimension] - this->box_m.Min[iDimension]) * rScale;
            }
            //std::cout << "creating child with key = " << kChild << " with box" << ptNodeMax <<" "<<ptNodeMin <<"\n";
            for ( dim_type iDimension = 0; iDimension < dim_m; ++iDimension ) {

                nodeChild.boundingbox_m.Min[iDimension] = ptNodeMin[iDimension];
                nodeChild.boundingbox_m.Max[iDimension] = ptNodeMax[iDimension];
            }

            return nodeChild;
        }

        morton_node_id_type EstimateNodeNumber(entity_id_type nParticles, depth_type MaxDepth, entity_id_type nMaxElements)
        {

            if ( nParticles < 10 ) return 10;

            const double rMult = 3;

            // for smaller problem size
            if ( (MaxDepth + 1) * dim_m < 64 ) {
                size_t nMaxChild = size_t { 1 } << (MaxDepth * dim_m);
                auto const nElementsInNode = nParticles / nMaxChild;
                if ( nElementsInNode > nMaxElements / 2 ) return nMaxChild;
            }

            // for larget problem size
            auto const nElementInNodeAvg = static_cast<float>(nParticles) / static_cast<float>(nMaxElements);
            auto const nDepthEstimated = std::min(MaxDepth, static_cast<depth_type>(std::ceil((log2f(nElementInNodeAvg) + 1.0) / static_cast<float>(dim_m))));

            if ( nDepthEstimated * dim_m < 64 ) return static_cast<size_t>(rMult * (1 << nDepthEstimated * dim_m));

            return static_cast<size_t>(rMult * nElementInNodeAvg);

        }

        ippl::Vector<double, 3> GetRasterizer(box_type Box, grid_id_type nDivide)
        {

            const double ndiv = static_cast<double>(nDivide);
            ippl::Vector<double, 3> rasterizer;
            for ( dim_type i = 0; i<dim_m; ++i ) {
                double boxsize = Box.Max[i] - Box.Min[i];
                rasterizer[i] = boxsize == 0 ? 1.0 : (ndiv/boxsize);
            }

            return rasterizer;

        }

        Kokkos::vector<morton_node_id_type> RefineNode(OrthoTreeNode& NodeParent, morton_node_id_type kParent, Kokkos::vector<position_type> positions)
        {

            const depth_type            newDepth = this->GetDepth(kParent)+1;
            const depth_type            nDepthRemain = maxdepth_m - newDepth;
            const morton_node_id_type   shift = nDepthRemain * dim_m;

            const position_type ParentBoxOrigin = NodeParent.boundingbox_m.Min;
            Kokkos::vector<Kokkos::pair<entity_id_type, morton_node_id_type>> aidLocation(NodeParent.data_vector.size());
            const morton_node_id_type FlagParent = kParent << dim_m;

            Kokkos::vector<morton_node_id_type> NewNodes = {};

            std::transform(NodeParent.data_vector.begin(), NodeParent.data_vector.end(), aidLocation.begin(), [&] (entity_id_type id) -> Kokkos::pair<entity_id_type, morton_node_id_type>
            {
                ippl::Vector<grid_id_type, 3> gridId = this->GetRelativeGridId(positions[id], ParentBoxOrigin);
                const morton_node_id_type childId = this->MortonEncode(gridId)>>shift;
                return { id, (childId + FlagParent) };
            });

            NodeParent.data_vector = {};

            for ( unsigned int childId = 0; childId<Kokkos::pow(2, dim_m); ++childId ) {

                morton_node_id_type const kChild = FlagParent | childId;
                if ( nodes_m.exists(kChild) ) continue;
                OrthoTreeNode& nodeChild = this->createChild(NodeParent, kChild);
                nodeChild.parent_m = kParent;
                for ( unsigned int idx = 0; idx<aidLocation.size(); ++idx ) {

                    if ( aidLocation[idx].second == kChild ) {
                        nodeChild.data_vector.push_back(aidLocation[idx].first);
                    }
                }

                NewNodes.push_back(kChild);

            }

            return NewNodes;

        }

    public: // Morton Encoding Functions

        morton_node_id_type GetLocationId(position_type pt)
        {

            return MortonEncode(GetGridId(pt));

        }

        ippl::Vector<grid_id_type, 3> GetGridId(position_type pt)
        {

            ippl::Vector<grid_id_type, 3> aid;

            for ( dim_type i = 0; i<dim_m; ++i ) {
                double r_i = pt[i] - box_m.Min[i];
                double raster_id = r_i * rasterizer_m[i];
                aid[i] = static_cast<grid_id_type>(raster_id);
            }

            return aid;

        }

        ippl::Vector<grid_id_type, 3> GetRelativeGridId(position_type pt, position_type origin)
        {

            ippl::Vector<grid_id_type, 3> aid;

            for ( dim_type i = 0; i<dim_m; ++i ) {
                double r_i = pt[i] - origin[i];
                double raster_id = r_i * rasterizer_m[i];
                aid[i] = static_cast<grid_id_type>(raster_id);
            }

            return aid;
        }

        // Only works for dim = 3 for now
        morton_node_id_type MortonEncode(ippl::Vector<grid_id_type, 3> aidGrid) const
        {

            assert(dim_m == 3);

            return (part1By2(aidGrid[2]) << 2) + (part1By2(aidGrid[1]) << 1) + part1By2(aidGrid[0]);

        }

        ippl::Vector<grid_id_type, 3> MortonDecode(morton_node_id_type kNode, depth_type nDepthMax) const
        {

            ippl::Vector<grid_id_type, 3> aidGrid;
            auto const nDepth = GetDepth(kNode);

            auto mask = morton_node_id_type { 1 };
            for ( depth_type iDepth = nDepthMax - nDepth, shift = 0; iDepth < nDepthMax; ++iDepth ) {
                for ( dim_type iDimension = 0; iDimension < dim_m; ++iDimension, ++shift ) {
                    aidGrid[iDimension] |= (kNode & mask) >> (shift - iDepth);
                    mask <<= 1;
                }
            }

            return aidGrid;

        }

        static constexpr morton_node_id_type part1By2(grid_id_type n) noexcept
        {

            // n = ----------------------9876543210 : Bits initially
            // n = ------98----------------76543210 : After (1)
            // n = ------98--------7654--------3210 : After (2)
            // n = ------98----76----54----32----10 : After (3)
            // n = ----9--8--7--6--5--4--3--2--1--0 : After (4)
            n = (n ^ (n << 16)) & 0xff0000ff; // (1)
            n = (n ^ (n << 8)) & 0x0300f00f; // (2)
            n = (n ^ (n << 4)) & 0x030c30c3; // (3)
            n = (n ^ (n << 2)) & 0x09249249; // (4)
            return static_cast<morton_node_id_type>(n);

        }

    public: // Node Info Functions


        bool IsValidKey(uint64_t key) const
        {

            return key;

        }

        void VisitNodes(morton_node_id_type kRoot, auto procedure) const
        {

            auto q = std::queue<morton_node_id_type>();

            for ( q.push(kRoot); !q.empty(); q.pop() ) {

                auto const& key = q.front();
                auto const& node = nodes_m.value_at(nodes_m.find(key));
                procedure(key, node);

                Kokkos::vector<morton_node_id_type> children = node.GetChildren();
                for ( unsigned int i = 0; i<children.size(); ++i ) {
                    q.push(children[i]);

                }
>>>>>>> 28e0e755 (refactor of orthotree)

            }

        }

<<<<<<< HEAD
    }

    void VisitSelectedNodes(morton_node_id_type kRoot, auto procedure, auto selector) const{
        
        auto q = std::queue<morton_node_id_type>();

        for(q.push(kRoot); !q.empty(); q.pop()){

            auto const& key = q.front();
            auto const& node = nodes_m.value_at(nodes_m.find(key));
            procedure(key, node);

            Kokkos::vector<morton_node_id_type> children = node.GetChildren();
            for(unsigned int i=0; i<children.size(); ++i){
                
                if(selector(children[i])) q.push(children[i]);
=======
        void VisitSelectedNodes(morton_node_id_type kRoot, auto procedure, auto selector) const
        {

            auto q = std::queue<morton_node_id_type>();

            for ( q.push(kRoot); !q.empty(); q.pop() ) {

                auto const& key = q.front();
                auto const& node = nodes_m.value_at(nodes_m.find(key));
                procedure(key, node);

                Kokkos::vector<morton_node_id_type> children = node.GetChildren();
                for ( unsigned int i = 0; i<children.size(); ++i ) {

                    if ( selector(children[i]) ) q.push(children[i]);

                }
>>>>>>> 28e0e755 (refactor of orthotree)

            }

        }

<<<<<<< HEAD
    }

    void PrintStructure(morton_node_id_type kRoot = 1) const {
        VisitNodes(kRoot, [&](morton_node_id_type key, OrthoTreeNode const& node){
            
            // Node info
            std::cout << "Node ID       = " << key << "\n";
            std::cout << "Depth         = " << int(GetDepth(key)) << "\n";
            std::cout << "Parent ID     = " << node.parent_m << "\n";
            std::cout << "Children IDs  = ";
            Kokkos::vector<morton_node_id_type> children = node.GetChildren();
            for(unsigned int idx=0;idx<children.size();++idx) std::cout << children[idx] << " ";
            std::cout << "\n";
            std::cout << "Grid ID       = " << "[ ";
            for(int i=0;i<3;++i) std::cout << MortonDecode(key,GetDepth(key))[i] << " ";
            std::cout << "] \n";

            // Box extent
            std::cout << "Box           = [(";
            for(int i=0;i<3;++i) std::cout << node.boundingbox_m.Min[i] << ",";
            std::cout << "),(";                 
            for(int i=0;i<3;++i) std::cout << node.boundingbox_m.Max[i] << ",";
            std::cout << ")] \n";

            // Colleagues
            std::cout << "Colleagues    : ";
            Kokkos::vector<morton_node_id_type> colleagues = this->GetColleagues(key);
            for(unsigned int idx=0; idx<colleagues.size(); ++idx){
                std::cout << colleagues[idx] << " ";
            }
            std::cout << "\n";

            // Coarse Nbrs
            if(key != kRoot){
                std::cout << "Coarse Nbrs   : ";
                Kokkos::vector<morton_node_id_type> coarseNbrs = this->GetCoarseNbrs(key);
                for(unsigned int idx=0; idx<coarseNbrs.size(); ++idx){
                    std::cout << coarseNbrs[idx] << " ";
                }
                std::cout << "\n";
            }

            //Points
            std::cout << "Points        : ";
            if(node.IsAnyChildExist()){
                Kokkos::vector<entity_id_type> points = this->CollectIds(key);
                for(unsigned int idx=0; idx<points.size(); ++idx) std::cout << points[idx] << " ";
            }
            else{
                for(unsigned int idx=0; idx<node.vid_m.size(); ++idx) std::cout << node.vid_m[idx] << " ";
            }
            std::cout << "\n";

            // Is Leaf
            std::cout << "Is Leaf? " << (!node.IsAnyChildExist()) << "\n";

            std::cout << "\n\n";
        
        });
    }

    enum BoxRelation : int8_t {Overlapped = -1, Adjacent = 0, Separated = 1};
    BoxRelation boxRelation(box_type& e1, box_type& e2) const{
        
        enum BoxRelationCandidate : uint8_t  { OverlappedC = 0x1, AdjacentC = 0x2, SeparatedC = 0x4 };
        int8_t rel = 0;
        
        for(dim_type iDimension = 0; iDimension < dim_m; ++ iDimension){
            if      ((e1.Min[iDimension] <  e2.Max[iDimension]) && (e1.Max[iDimension] >  e2.Min[iDimension])) rel |= BoxRelationCandidate::OverlappedC;
            else if ((e1.Min[iDimension] == e2.Max[iDimension]) || (e1.Max[iDimension] == e2.Min[iDimension])) rel |= BoxRelationCandidate::AdjacentC;
            else if ((e1.Min[iDimension] >  e2.Max[iDimension]) || (e1.Max[iDimension] <  e2.Min[iDimension])) return BoxRelation::Separated;
        }

        return (rel & BoxRelationCandidate::AdjacentC) == BoxRelationCandidate::AdjacentC ? BoxRelation::Adjacent : BoxRelation::Overlapped;
    
    }

    

    

    

public: // Getters

    dim_type GetDim() const{
        
        return dim_m;
        
    }

    OrthoTreeNode& GetNode(morton_node_id_type key) const{
        
        return nodes_m.value_at(nodes_m.find(key));

    }
    
    depth_type GetDepth(morton_node_id_type key) const{
        
        // Keep shifting off three bits at a time, increasing depth counter
        for (depth_type d = 0; IsValidKey(key); ++d, key >>= dim_m)
            if (key == 1) return d; // If only sentinel bit remains, exit with node depth
            
        assert(false); // Bad key
        
        return 0;

    }

    OrthoTreeNode const& GetParent(morton_node_id_type key){

        assert(key != 1);

        return nodes_m.value_at(nodes_m.find(key >> 3));

    }

    Kokkos::vector<morton_node_id_type> GetColleagues(morton_node_id_type key) const{
        
        Kokkos::vector<morton_node_id_type> colleagues={};
        colleagues.reserve(27);
        
        if(key == 1){
            colleagues.push_back(1);
            return colleagues;
        } 

        ippl::Vector<grid_id_type,3> aidGrid = MortonDecode(key, GetDepth(key));
        

        for(int z=-1; z<=1; ++z){
            for(int y=-1; y<=1; ++y){
                for(int x=-1; x<=1; ++x){

                    morton_node_id_type nbrKey = MortonEncode( ippl::Vector<grid_id_type,3> {aidGrid[0]+x, aidGrid[1]+y, aidGrid[2]+z} ) + Kokkos::pow(8,GetDepth(key));
                    if(nodes_m.exists(nbrKey)) {
                        colleagues.push_back(nbrKey);
                    }
                }
            }
        }

        return colleagues;

    }

    Kokkos::vector<morton_node_id_type> GetCoarseNbrs(morton_node_id_type key) const{

        Kokkos::vector<morton_node_id_type> coarseNbrs{};
        coarseNbrs.reserve(3);
        if(key == 1) return coarseNbrs;
        
        OrthoTreeNode& node = this->GetNode(key);
        
        Kokkos::vector<morton_node_id_type> parentColleagues = this->GetColleagues(key >> dim_m);

        for(unsigned int idx=0; idx<parentColleagues.size(); ++idx){
            
            OrthoTreeNode& coarseNode = this->GetNode(parentColleagues[idx]);
            
            if(!coarseNode.IsAnyChildExist() && boxRelation(coarseNode.boundingbox_m, node.boundingbox_m) == BoxRelation::Adjacent){
                coarseNbrs.push_back(parentColleagues[idx]);
            }

        }

        return coarseNbrs;

    }

    Kokkos::vector<entity_id_type> CollectIds(morton_node_id_type kRoot=1)const{
        
        
        Kokkos::vector<entity_id_type> ids;
        ids.reserve(GetNode(kRoot).npoints_m);

        VisitNodes(kRoot, [&ids](morton_node_id_type, OrthoTreeNode const& node)
        {
            if(!node.IsAnyChildExist()){
                unsigned int oldsize = ids.size();
                unsigned int newsize = oldsize + node.vid_m.size();
                ids.resize(newsize);
                Kokkos::parallel_for("Collectids", node.vid_m.size(),
                KOKKOS_LAMBDA(unsigned int idx){
                    ids[oldsize+idx] = node.vid_m[idx];
            });
            }

        });
        
        return ids;

    }

    Kokkos::vector<entity_id_type> CollectSourceIds(morton_node_id_type kRoot=1)const{
        
        Kokkos::vector<entity_id_type> ids;
        ids.reserve(GetNode(kRoot).npoints_m);
        VisitNodes(kRoot, [&](morton_node_id_type, OrthoTreeNode const& node)
        {
            if(!node.IsAnyChildExist()){
                auto pp = std::partition_point(node.vid_m.begin(), node.vid_m.end(), [this](unsigned int i){return i < sourceidx_m;});
                unsigned int newnpoints = node.vid_m.end() - pp;
                unsigned int oldsize = ids.size();
                ids.resize(oldsize + newnpoints);
                Kokkos::parallel_for("CollectSources", newnpoints,
                KOKKOS_LAMBDA(unsigned int idx){
                    ids[oldsize+idx] = node.vid_m[(pp-node.vid_m.begin())+idx] - sourceidx_m;
            });
            }
            
            
        });
        
        return ids;

    }

    Kokkos::vector<entity_id_type> CollectTargetIds(morton_node_id_type kRoot=1)const{
        
        Kokkos::vector<entity_id_type> ids;
        ids.reserve(GetNode(kRoot).npoints_m);

        VisitNodes(kRoot, [&](morton_node_id_type, OrthoTreeNode const& node)
        {
            if(!node.IsAnyChildExist()){
                auto pp = std::partition_point(node.vid_m.begin(), node.vid_m.end(), [this](unsigned int i){return i < sourceidx_m;});
                unsigned int newnpoints = pp-node.vid_m.begin();
                unsigned int oldsize = ids.size();
                ids.resize(oldsize + newnpoints);
                Kokkos::parallel_for("CollectTargets", newnpoints,
                KOKKOS_LAMBDA(unsigned int idx){
                    ids[oldsize+idx] = node.vid_m[idx];
                });
            }
        });
        
        return ids;

    }

    Kokkos::vector<morton_node_id_type> GetLeafNodes(morton_node_id_type kRoot=1) const{

        Kokkos::vector<morton_node_id_type> leafNodes;
        leafNodes.reserve(static_cast<morton_node_id_type>(nodes_m.size()/2));

        VisitNodes(kRoot, [&leafNodes](morton_node_id_type key, OrthoTreeNode const& node){if(!node.IsAnyChildExist()){leafNodes.push_back(key);} });

        return leafNodes;
    }

    Kokkos::vector<morton_node_id_type> GetPotentialColleagues(morton_node_id_type key) const{

        ippl::Vector<grid_id_type,3> aidGrid = MortonDecode(key, GetDepth(key));
        Kokkos::vector<morton_node_id_type> potentialColleagues;

        for(int z=-1; z<=1; ++z){
            for(int y=-1; y<=1; ++y){
                for(int x=-1; x<=1; ++x){

                    ippl::Vector<grid_id_type,3> newGrid = ippl::Vector<grid_id_type,3>{aidGrid[0]+x, aidGrid[1]+y, aidGrid[2]+z};
                    
                    if (not(newGrid[0]<0 || newGrid[0]>=std::pow(2,GetDepth(key)) || newGrid[1]<0 || newGrid[1]>=std::pow(2,GetDepth(key)) || newGrid[2]<0 || newGrid[2]>=std::pow(2,GetDepth(key)))){
                        
                        morton_node_id_type potentialColleague = MortonEncode(newGrid) + Kokkos::pow(8, GetDepth(key));
                        potentialColleagues.push_back(potentialColleague);
                        
                    }

                }
            }
        }

        return potentialColleagues;

    }

    morton_node_id_type GetNextAncestor(morton_node_id_type key) const noexcept{
            
        if (nodes_m.exists(key)) return key;
        
        else if (key == 0) return 1;

        else return GetNextAncestor(key>>dim_m);
            
    }

    depth_type GetMaxDepth() const noexcept{
        return maxdepth_m;
    }

    
    Kokkos::vector<morton_node_id_type> GetInternalNodeAtDepth(depth_type depth) const noexcept {

        Kokkos::vector<morton_node_id_type> keys;
        keys.reserve(Kokkos::pow(8,depth));
        
        VisitSelectedNodes(1, [&](auto key, auto){
            
            if(GetDepth(key) == depth && GetNode(key).IsAnyChildExist()){

                //std::cout << key << " ";
                keys.push_back(key);

            }
            
        }, [&](auto i){return (GetDepth(i)<=depth);});
        
        return keys;    
    }

    Kokkos::vector<morton_node_id_type> GetNodeAtDepth(depth_type depth) const noexcept {

        Kokkos::vector<morton_node_id_type> keys;
        keys.reserve(Kokkos::pow(8,depth));
        
        VisitSelectedNodes(1, [&](auto key, auto){
            
            if(GetDepth(key) == depth){

                //std::cout << key << " ";
                keys.push_back(key);

            }
            
        }, [&](auto i){return (GetDepth(i)<=depth);});
        
        return keys;    
    }

    unsigned int GetMaxElementsPerNode() const noexcept{
        return maxelements_m;
    }
    
=======
        void PrintStructure(morton_node_id_type kRoot = 1) const
        {
            VisitNodes(kRoot, [&] (morton_node_id_type key, OrthoTreeNode const& node)
            {

                // Node info
                std::cout << "Node ID       = " << key << "\n";
                std::cout << "Depth         = " << int(GetDepth(key)) << "\n";
                std::cout << "Parent ID     = " << node.parent_m << "\n";
                std::cout << "Children IDs  = ";
                Kokkos::vector<morton_node_id_type> children = node.GetChildren();
                for ( unsigned int idx = 0;idx<children.size();++idx ) std::cout << children[idx] << " ";
                std::cout << "\n";
                std::cout << "Grid ID       = " << "[ ";
                for ( int i = 0;i<3;++i ) std::cout << MortonDecode(key, GetDepth(key))[i] << " ";
                std::cout << "] \n";

                // Box extent
                std::cout << "Box           = [(";
                for ( int i = 0;i<3;++i ) std::cout << node.boundingbox_m.Min[i] << ",";
                std::cout << "),(";
                for ( int i = 0;i<3;++i ) std::cout << node.boundingbox_m.Max[i] << ",";
                std::cout << ")] \n";

                // Colleagues
                std::cout << "Colleagues    : ";
                Kokkos::vector<morton_node_id_type> colleagues = this->GetColleagues(key);
                for ( unsigned int idx = 0; idx<colleagues.size(); ++idx ) {
                    std::cout << colleagues[idx] << " ";
                }
                std::cout << "\n";

                // Coarse Nbrs
                if ( key != kRoot ) {
                    std::cout << "Coarse Nbrs   : ";
                    Kokkos::vector<morton_node_id_type> coarseNbrs = this->GetCoarseNbrs(key);
                    for ( unsigned int idx = 0; idx<coarseNbrs.size(); ++idx ) {
                        std::cout << coarseNbrs[idx] << " ";
                    }
                    std::cout << "\n";
                }

                //Points
                std::cout << "Points        : ";
                if ( node.IsAnyChildExist() ) {
                    Kokkos::vector<entity_id_type> points = this->CollectIds(key);
                    for ( unsigned int idx = 0; idx<points.size(); ++idx ) std::cout << points[idx] << " ";
                }
                else {
                    for ( unsigned int idx = 0; idx<node.data_vector.size(); ++idx ) std::cout << node.data_vector[idx] << " ";
                }
                std::cout << "\n";

                // Is Leaf
                std::cout << "Is Leaf? " << (!node.IsAnyChildExist()) << "\n";

                std::cout << "\n\n";

            });
        }

        enum BoxRelation : int8_t { Overlapped = -1, Adjacent = 0, Separated = 1 };
        BoxRelation boxRelation(box_type& e1, box_type& e2) const
        {

            enum BoxRelationCandidate : uint8_t { OverlappedC = 0x1, AdjacentC = 0x2, SeparatedC = 0x4 };
            int8_t rel = 0;

            for ( dim_type iDimension = 0; iDimension < dim_m; ++iDimension ) {
                if ( (e1.Min[iDimension] <  e2.Max[iDimension]) && (e1.Max[iDimension] >  e2.Min[iDimension]) ) rel |= BoxRelationCandidate::OverlappedC;
                else if ( (e1.Min[iDimension] == e2.Max[iDimension]) || (e1.Max[iDimension] == e2.Min[iDimension]) ) rel |= BoxRelationCandidate::AdjacentC;
                else if ( (e1.Min[iDimension] >  e2.Max[iDimension]) || (e1.Max[iDimension] <  e2.Min[iDimension]) ) return BoxRelation::Separated;
            }

            return (rel & BoxRelationCandidate::AdjacentC) == BoxRelationCandidate::AdjacentC ? BoxRelation::Adjacent : BoxRelation::Overlapped;

        }







    public: // Getters

        dim_type GetDim() const
        {

            return dim_m;

        }

        OrthoTreeNode& GetNode(morton_node_id_type key) const
        {

            return nodes_m.value_at(nodes_m.find(key));

        }

        depth_type GetDepth(morton_node_id_type key) const
        {

            // Keep shifting off three bits at a time, increasing depth counter
            for ( depth_type d = 0; IsValidKey(key); ++d, key >>= dim_m )
                if ( key == 1 ) return d; // If only sentinel bit remains, exit with node depth

            assert(false); // Bad key

            return 0;

        }

        OrthoTreeNode const& GetParent(morton_node_id_type key)
        {

            assert(key != 1);

            return nodes_m.value_at(nodes_m.find(key >> 3));

        }

        Kokkos::vector<morton_node_id_type> GetColleagues(morton_node_id_type key) const
        {

            Kokkos::vector<morton_node_id_type> colleagues = {};
            colleagues.reserve(27);

            if ( key == 1 ) {
                colleagues.push_back(1);
                return colleagues;
            }

            ippl::Vector<grid_id_type, 3> aidGrid = MortonDecode(key, GetDepth(key));


            for ( int z = -1; z<=1; ++z ) {
                for ( int y = -1; y<=1; ++y ) {
                    for ( int x = -1; x<=1; ++x ) {

                        morton_node_id_type nbrKey = MortonEncode(ippl::Vector<grid_id_type, 3> {aidGrid[0]+x, aidGrid[1]+y, aidGrid[2]+z}) + Kokkos::pow(8, GetDepth(key));
                        if ( nodes_m.exists(nbrKey) ) {
                            colleagues.push_back(nbrKey);
                        }
                    }
                }
            }

            return colleagues;

        }

        Kokkos::vector<morton_node_id_type> GetCoarseNbrs(morton_node_id_type key) const
        {

            Kokkos::vector<morton_node_id_type> coarseNbrs {};
            coarseNbrs.reserve(3);
            if ( key == 1 ) return coarseNbrs;

            OrthoTreeNode& node = this->GetNode(key);

            Kokkos::vector<morton_node_id_type> parentColleagues = this->GetColleagues(key >> dim_m);

            for ( unsigned int idx = 0; idx<parentColleagues.size(); ++idx ) {

                OrthoTreeNode& coarseNode = this->GetNode(parentColleagues[idx]);

                if ( !coarseNode.IsAnyChildExist() && boxRelation(coarseNode.boundingbox_m, node.boundingbox_m) == BoxRelation::Adjacent ) {
                    coarseNbrs.push_back(parentColleagues[idx]);
                }

            }

            return coarseNbrs;

        }

        Kokkos::vector<entity_id_type> CollectIds(morton_node_id_type kRoot = 1)const
        {


            Kokkos::vector<entity_id_type> ids;
            ids.reserve(GetNode(kRoot).npoints_m);

            VisitNodes(kRoot, [&ids] (morton_node_id_type, OrthoTreeNode const& node)
            {
                if ( !node.IsAnyChildExist() ) {
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

        Kokkos::vector<entity_id_type> CollectSourceIds(morton_node_id_type kRoot = 1)const
        {

            Kokkos::vector<entity_id_type> ids;
            ids.reserve(GetNode(kRoot).npoints_m);
            VisitNodes(kRoot, [&] (morton_node_id_type, OrthoTreeNode const& node)
            {
                if ( !node.IsAnyChildExist() ) {
                    auto pp = std::partition_point(node.data_vector.begin(), node.data_vector.end(), [this] (unsigned int i) { return i < sourceidx_m; });
                    unsigned int newnpoints = node.data_vector.end() - pp;
                    unsigned int oldsize = ids.size();
                    ids.resize(oldsize + newnpoints);
                    Kokkos::parallel_for("CollectSources", newnpoints,
                    [=, this] (unsigned int idx)
                    {
                        ids[oldsize+idx] = node.data_vector[(pp-node.data_vector.begin())+idx] - sourceidx_m;
                    });
                }


            });

            return ids;

        }

        Kokkos::vector<entity_id_type> CollectTargetIds(morton_node_id_type kRoot = 1)const
        {

            Kokkos::vector<entity_id_type> ids;
            ids.reserve(GetNode(kRoot).npoints_m);

            VisitNodes(kRoot, [&] (morton_node_id_type, OrthoTreeNode const& node)
            {
                if ( !node.IsAnyChildExist() ) {
                    auto pp = std::partition_point(node.data_vector.begin(), node.data_vector.end(), [this] (unsigned int i) { return i < sourceidx_m; });
                    unsigned int newnpoints = pp-node.data_vector.begin();
                    unsigned int oldsize = ids.size();
                    ids.resize(oldsize + newnpoints);
                    Kokkos::parallel_for("CollectTargets", newnpoints,
                    KOKKOS_LAMBDA(unsigned int idx){
                        ids[oldsize+idx] = node.data_vector[idx];
                    });
                }
            });

            return ids;

        }

        Kokkos::vector<morton_node_id_type> GetLeafNodes(morton_node_id_type kRoot = 1) const
        {

            Kokkos::vector<morton_node_id_type> leafNodes;
            leafNodes.reserve(static_cast<morton_node_id_type>(nodes_m.size()/2));

            VisitNodes(kRoot, [&leafNodes] (morton_node_id_type key, OrthoTreeNode const& node) { if ( !node.IsAnyChildExist() ) { leafNodes.push_back(key); } });

            return leafNodes;
        }

        Kokkos::vector<morton_node_id_type> GetPotentialColleagues(morton_node_id_type key) const
        {

            ippl::Vector<grid_id_type, 3> aidGrid = MortonDecode(key, GetDepth(key));
            Kokkos::vector<morton_node_id_type> potentialColleagues;

            for ( int z = -1; z<=1; ++z ) {
                for ( int y = -1; y<=1; ++y ) {
                    for ( int x = -1; x<=1; ++x ) {

                        ippl::Vector<grid_id_type, 3> newGrid = ippl::Vector<grid_id_type, 3> { aidGrid[0]+x, aidGrid[1]+y, aidGrid[2]+z };

                        if ( not(newGrid[0]<0 || newGrid[0]>=std::pow(2, GetDepth(key)) || newGrid[1]<0 || newGrid[1]>=std::pow(2, GetDepth(key)) || newGrid[2]<0 || newGrid[2]>=std::pow(2, GetDepth(key))) ) {

                            morton_node_id_type potentialColleague = MortonEncode(newGrid) + Kokkos::pow(8, GetDepth(key));
                            potentialColleagues.push_back(potentialColleague);

                        }

                    }
                }
            }

            return potentialColleagues;

        }

        morton_node_id_type GetNextAncestor(morton_node_id_type key) const noexcept
        {

            if ( nodes_m.exists(key) ) return key;

            else if ( key == 0 ) return 1;

            else return GetNextAncestor(key>>dim_m);

        }

        depth_type GetMaxDepth() const noexcept
        {
            return maxdepth_m;
        }


        Kokkos::vector<morton_node_id_type> GetInternalNodeAtDepth(depth_type depth) const noexcept
        {

            Kokkos::vector<morton_node_id_type> keys;
            keys.reserve(Kokkos::pow(8, depth));

            VisitSelectedNodes(1, [&] (auto key, auto)
            {

                if ( GetDepth(key) == depth && GetNode(key).IsAnyChildExist() ) {

                    //std::cout << key << " ";
                    keys.push_back(key);

                }

            }, [&] (auto i) { return (GetDepth(i)<=depth); });

            return keys;
        }

        Kokkos::vector<morton_node_id_type> GetNodeAtDepth(depth_type depth) const noexcept
        {

            Kokkos::vector<morton_node_id_type> keys;
            keys.reserve(Kokkos::pow(8, depth));

            VisitSelectedNodes(1, [&] (auto key, auto)
            {

                if ( GetDepth(key) == depth ) {

                    //std::cout << key << " ";
                    keys.push_back(key);

                }

            }, [&] (auto i) { return (GetDepth(i)<=depth); });

            return keys;
        }

        unsigned int GetMaxElementsPerNode() const noexcept
        {
            return maxelements_m;
        }
>>>>>>> 28e0e755 (refactor of orthotree)




<<<<<<< HEAD
}; // Class OrthoTree
=======

    }; // Class OrthoTree
>>>>>>> 28e0e755 (refactor of orthotree)






} // Namespace ippl
<<<<<<< HEAD
#endif
=======
#endif
>>>>>>> 28e0e755 (refactor of orthotree)

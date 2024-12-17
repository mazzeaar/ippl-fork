#include <cstddef>
#include "../OrthoTree.h"

/*
TODO:
- IMPLEMENT THIS NEW FUNCTION SIGNATURE
- WRITE TESTS FOR THE FUNCTION
- ADJUST THE SIGNATURE IN ORTHOTREE.H

namespace ippl {

    // entry with weights
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::partition(Kokkos::View<morton_code*> octants,
                                                          Kokkos::View<morton_code*> weights);

    // entry without weights
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::partition(Kokkos::View<morton_code*> octants);
}  // namespace ippl
*/

namespace ippl {

    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::partition(Kokkos::View<morton_code*> octants,
                                                         Kokkos::View<size_t*> weights) {
        START_FUNC;
        Kokkos::View<morton_code*> prefix_sum("prefix_sum", octants.size());

        // the global weight up to right after this rank
        size_t global_total;
        // the local wight total
        size_t local_total;
        // the weight of all octants on this and all smaller ranks combined
        size_t local_prefix;

        // fast computation of the prefix sum
        Kokkos::parallel_scan("prefix_sum", octants.size(), KOKKOS_LAMBDA(const size_t i, size_t& sum,
                                                                         const bool final) {
            sum += weights(i);
            if (final) {
                prefix_sum(i) = sum;
            }
        }, local_total);

        // get the global total weight 
        Comm->scan(&local_total, &local_prefix, 1, std::plus<size_t>());
      
        // broadcast the global total weight
        global_total = local_prefix;
        // last rank will have the global total weight as it's local_prefix
        // so we can just broadcast the global total weight seeding from there
        Comm->broadcast<size_t>(&global_total, 1, world_size - 1);

        // adjust prefix_sum to be the global prefix sum 
        Kokkos::parallel_for("prefix_sum", octants.size(), KOKKOS_LAMBDA(const size_t i) {
            prefix_sum(i) += local_prefix - local_total;
        });

        // initialize the average weight and the remainder
        size_t avg_weight = global_total / world_size;
        size_t k = global_total % world_size;
        
        // these values will be used to mark which octants already on the current
        // rank will stay here
        size_t local_start_idx = 0;
        size_t local_end_idx = 0;


        std::vector<mpi::Request> request;
        request.reserve(2 * world_size);
        Kokkos::View<size_t *> sizes("sizes", world_size);
        for (unsigned rank_iter = 0; rank_iter < world_size; rank_iter++) {
            // this will take care of the remainder
            size_t offset = rank_iter < k ? rank_iter : k;
            // each rank will get a block of octants with a weight as close as 
            // possible to avg_weight. Specifically the n-th of these blocks.
            size_t min_prefix = avg_weight * rank_iter + offset;
            
            // get the iterator to the first element that is greater than min_prefix
            auto start = std::upper_bound(prefix_sum.data(), prefix_sum.data() + octants.size(), min_prefix);
            size_t start_idx = start - prefix_sum.data();
            // if rank < k we allocate a bit more space for weight to adjust
            // for the remainder
            if (rank_iter < k)
                offset++;
            
            size_t max_prefix = avg_weight * (rank_iter + 1) + offset;
            // get the iterator to the first element that is greater than max_prefix
            auto end = std::upper_bound(prefix_sum.data(), prefix_sum.data() + octants.size(), max_prefix);
            size_t end_idx = end - prefix_sum.data();
            
            
            // store how many octants this rank will send to rank_iter
            sizes(rank_iter) = end - start;

            // skip this rank as we can't send to ourselves
            if (rank_iter == world_rank) {
                // mark what octants stay on the rank for later
                local_start_idx = start_idx;
                local_end_idx = end_idx;
                continue;
            }

            // send the number of blocks we'll be sending first
            request.push_back(mpi::Request());
            Comm->isend(sizes(rank_iter), 1, rank_iter, 1, request.back());
            
            // if there are any blocks to be sent do so
            if (end != start && sizes(rank_iter) > 0){
                assert(start_idx + sizes(rank_iter) <= octants.size());
                request.push_back(mpi::Request());
                Comm->isend(octants(start_idx), sizes(rank_iter), rank_iter, 0, request.back());
            }
        }

        Kokkos::View<size_t *> received_sizes("received_sizes", world_size);
        // first we receive all sizes. This let's us prepare a Kokkos::view 
        // of the right size to receive the octants
        for (unsigned rank_iter = 0; rank_iter < world_size; rank_iter++) {
            // can't receive from ourselves
            if (rank_iter == world_rank) {
                // still remember how many of our octants we keep
                received_sizes(world_rank) = sizes(world_rank);
                continue;
            }
            mpi::Status stat;
            Comm->recv(&received_sizes(rank_iter), 1, rank_iter, 1, stat);
        }
        
        // compute the prefix sum of the received sizes 
        // This will allow easy computation of starting indices
        Kokkos::View<size_t *> received_prefix_sum("received_prefix_sum", world_size);
        Kokkos::parallel_scan("received_prefix_sum", world_size, KOKKOS_LAMBDA(const size_t i, size_t& sum,
                                                                         const bool final) {
            sum += received_sizes(i);
            if (final) {
                received_prefix_sum(i) = sum;
            }
        });
        
        // receive buffer that is big enough to fit the sum of all received sizes
        Kokkos::View<morton_code *> partitioned_octants("partitioned_octants", 
            received_prefix_sum(world_size - 1));

        for (unsigned rank_iter = 0; rank_iter < world_size; rank_iter++) {
            // can't receive from ourselves
            if (rank_iter == world_rank) {
                continue;
            }
            // if we don't receive anything we can skip this rank
            if (received_sizes(rank_iter) == 0) {
              continue;
            }
            mpi::Status stat;
            // computes a pointer to the place where rank_iter should insert
            // prefix_sum - received_sizes will give us the sum of all inserted 
            // blocks from ranks smaler than rank_iter
            morton_code *recv_buffer = partitioned_octants.data() + received_prefix_sum(rank_iter) - received_sizes(rank_iter);
            Comm->recv(recv_buffer, received_sizes(rank_iter), rank_iter, 0, stat);
        }
        
        // insert the octants that stay on this rank fast
        Kokkos::parallel_for("partitioned_octants", local_end_idx - local_start_idx, KOKKOS_LAMBDA(const size_t i) {
            unsigned insert_start_idx = received_prefix_sum(world_rank) - received_sizes(world_rank);
            partitioned_octants(insert_start_idx + i) = octants(local_start_idx + i);
        });
        
        // wait on all processes such that we don't free memory before it's done
        for (unsigned req_idx = 0; req_idx < request.size(); req_idx++) {
            request[req_idx].wait();
        }
        Comm->barrier();
        return partitioned_octants;
    }
}  // namespace ippl

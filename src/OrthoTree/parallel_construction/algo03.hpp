#include <span>

#include "../OrthoTree.h"
/*
TODO:
- IMPLEMENT THIS NEW FUNCTION SIGNATURE
- WRITE TESTS FOR THE FUNCTION
- ADJUST THE SIGNATURE IN ORTHOTREE.H

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::complete_tree(Kokkos::View<morton_code*>& octants);
}  // namespace ippl
*/

namespace ippl {

    Kokkos::View<morton_code*> remove_duplicates(Kokkos::View<morton_code*> input_view) {
        const size_t input_size = input_view.extent(0);

        if (input_size == 0) {
            return Kokkos::View<morton_code*>("algo3::deduplicated_view", 0);
        }

        size_t unique_count = 0;
        Kokkos::parallel_reduce(
            "algo3::CountUniqueElements", input_size - 1,
            KOKKOS_LAMBDA(const size_t i, size_t& local_count) {
                local_count += static_cast<size_t>(input_view(i) != input_view(i + 1));
            },
            unique_count);

        const size_t output_size = unique_count + 1;
        Kokkos::View<morton_code*> output_view("algo3::deduplicated_view", output_size);

        /*
        Kokkos::View<size_t> index("algo3::index");
        Kokkos::deep_copy(index, size_t(0));
        Kokkos::parallel_for(
            "algo3::PopulateUniqueElements", input_size - 1, KOKKOS_LAMBDA(const size_t i) {
                if (input_view(i) != input_view(i + 1)) {
                    const size_t current_index = Kokkos::atomic_fetch_add(&index(), size_t(1));
                    output_view(current_index) = input_view(i);
                }
            });
        */

        Kokkos::parallel_scan(
                "algo3::PopulateUniqueElements", input_size - 1, 
                KOKKOS_LAMBDA(const size_t i, size_t& index, bool final) {
                    if (input_view(i) != input_view(i+1)) {
                        if (final) {
                            output_view(index) = input_view(i);
                        }
                        ++index;
                    }
                });
int a;
        Kokkos::parallel_for(
            "algo3::AddLastElement", 1, KOKKOS_LAMBDA(const int) {
                output_view(output_size - 1) = input_view(input_view.extent(0) - 1);
            });

        return output_view;
    }

    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::complete_tree(
        Kokkos::View<morton_code*> input_octants) {
        IpplTimings::TimerRef completeTreeTimer = IpplTimings::getTimer("complete_tree");
        IpplTimings::startTimer(completeTreeTimer);

        if (input_octants.extent(0) == 0) {
            logger << "This looks like trouble" << endl;
            throw std::runtime_error("complete_tree called with empty input");
        }

        auto deduplicated_octants = remove_duplicates(input_octants);
        auto linearised_octants   = linearise_octants(deduplicated_octants);

        auto partitioned_octants      = partition(linearised_octants);
        const size_t partitioned_size = partitioned_octants.extent(0);

        morton_code push_front_buff = 0;
        morton_code push_back_buff  = 0; 
        if (world_rank == 0) {
            const morton_code dfd_root = morton_helper.get_deepest_first_descendant(morton_code(0));
            const morton_code A_finest =
                morton_helper.get_nearest_common_ancestor(dfd_root, partitioned_octants(0));

            push_front_buff = morton_helper.get_first_child(A_finest);
        } else if (world_rank == world_size - 1) {
            const morton_code dld_root = morton_helper.get_deepest_last_descendant(morton_code(0));
            const morton_code A_finest = morton_helper.get_nearest_common_ancestor(
                dld_root, partitioned_octants(partitioned_size - 1));
            const morton_code last_child = morton_helper.get_last_child(A_finest);

            push_back_buff = last_child;
        }

        if (world_rank > 0) {
            Comm->send(*partitioned_octants.data(), 1, world_rank - 1, 0);
        }

        if (world_rank < world_size - 1) {
            mpi::Status status;
            Comm->recv(&push_back_buff, 1, world_rank + 1, 0, status);
        }

        const size_t R_base_size = 100;
        Kokkos::View<morton_code*> R_view("algo3::R_view", R_base_size);

        size_t R_index     = 0;
        auto insert_into_R = KOKKOS_LAMBDA(Kokkos::View<morton_code*> R_view, size_t R_index,
                                           morton_code octant_a, morton_code octant_b)
                                 ->size_t {
            const auto complete_region        = this->complete_region(octant_a, octant_b);
            const size_t complete_region_size = complete_region.extent(0);
            const size_t additional_octants   = complete_region_size + 1;
            const size_t remaining_space      = R_view.extent(0) - R_index;

            if (remaining_space <= additional_octants) {
                const size_t new_size =
                    R_view.extent(0) + (additional_octants - remaining_space) + R_base_size;
                Kokkos::resize(R_view, new_size);
            }

            R_view(R_index) = octant_a;
            R_index += 1;
            //cuda compilation failed using the parallel for - replaced with for loop
            //TODO (optional): look at complete region and see if hierarchical parallelism can be used
            //Kokkos::parallel_for(
            //    complete_region_size,
            //    KOKKOS_LAMBDA(const size_t i) { R_view(R_index + i) = complete_region(i); });
            for (size_t i = 0; i < complete_region_size; ++i) {
                R_view(R_index + i) = complete_region(i);
            }


            return complete_region_size + 1;
        };

        if (world_rank == 0 && push_front_buff != partitioned_octants(0)) {
            R_index += insert_into_R(R_view, R_index, push_front_buff, partitioned_octants(0));
        }

        for (size_t i = 0; i < partitioned_size - 1; ++i) {
            R_index +=
                insert_into_R(R_view, R_index, partitioned_octants(i), partitioned_octants(i + 1));
        }
        if (world_rank < world_size - 1 || push_back_buff != partitioned_octants(partitioned_size - 1)) {
            R_index += insert_into_R(R_view, R_index, partitioned_octants(partitioned_size - 1),
                                     push_back_buff);
        }

        if (world_rank == world_size - 1) {
            Kokkos::resize(R_view, R_index + 1);
            R_view(R_index) = push_back_buff;
        } else {
            Kokkos::resize(R_view, R_index);
        }

        IpplTimings::stopTimer(completeTreeTimer);

        return R_view;
    }
}  // namespace ippl

#include "../OrthoTree.h"

/*
TODO:
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::linearise_octants(
        const Kokkos::View<morton_code*>& input_view) {
        IpplTimings::TimerRef lineariseOctantsTimer = IpplTimings::getTimer("linearise_octants");
        IpplTimings::startTimer(lineariseOctantsTimer);

        /**
         * Idea:
         * 1. we count how many elements we have to copy into our linearised view
         *      -> O(1), so the double calls (probably) dont matter
         * 2. we copy the needed elements into the lienarised_view
         *
         * This way we can initialise our view to spec directly, without having to resize. I dont
         * know if this holds true if input_view contains hunderts of millions of octants, maybe it
         * is more efficient to resize the output then?
         */

        const size_t input_size = input_view.size();

        if (input_view.size() == 0) {
            IpplTimings::stopTimer(lineariseOctantsTimer);
            return Kokkos::View<morton_code*>("algo8::linearised_view", 0);
        }


        // to remove warnings due to KOKKOS_LAMBDA
        const auto local_morton_helper = this->morton_helper;
        size_t count                   = 0;
        Kokkos::parallel_reduce(
            "algo8::CountValidOctants", input_size - 1,
            KOKKOS_LAMBDA(const size_t i, size_t& local_count) {
                // no branching this way
                int should_count = !local_morton_helper.is_ancestor(input_view(i + 1), input_view(i))
                                    && input_view(i + 1) != input_view(i);
                local_count += static_cast<size_t>(should_count);
            },
            count);

        // last element will always be inserted, hence the 'count + 1'
        const size_t output_size = count + 1;
        Kokkos::View<morton_code*> output_view("algo8::linearised_view", output_size);

        Kokkos::parallel_scan(
                "algo8:AddRelevantOctants", input_size - 1, KOKKOS_LAMBDA(const size_t i, size_t& index, bool final) {
                    if (!local_morton_helper.is_ancestor(input_view(i+1), input_view(i)) && input_view(i+1) != input_view(i)) {
                        if (final) {
                            output_view(index) = input_view(i);
                        }
                        index++;
                    }
                });

        // deep copy didnt compile, so we use this georgeous thing lol
        Kokkos::parallel_for(
            "algo8::AddLastElement", 1, KOKKOS_LAMBDA(const int) {
                output_view(output_size - 1) = input_view(input_size - 1);
            });

        IpplTimings::stopTimer(lineariseOctantsTimer);

        return output_view;
    }
}  // namespace ippl

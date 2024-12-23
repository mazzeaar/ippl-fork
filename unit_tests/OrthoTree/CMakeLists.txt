# Get the relative path for better logging
file(RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding unit tests found in ${_relPath}")

# Common include and link directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

# Helper function to add a test with GTest
function(add_test_with_gtest target source_file)
    add_executable(${target} ${source_file})
    target_link_libraries(${target}
        ippl
        pthread
        GTest::gtest_main
        ${MPI_CXX_LIBRARIES}
    )
    gtest_discover_tests(${target} PROPERTIES TEST_DISCOVERY_TIMEOUT 600)
endfunction()

# general tests
add_test_with_gtest(tree_test ortho_tree.cpp)

# helper tests
add_test_with_gtest(bounds_test helpers/bounding_box.cpp)
add_test_with_gtest(morton_test helpers/morton_codes.cpp)
add_test_with_gtest(aid_list_test helpers/aid_list.cpp)

# paralell construction tests
add_test_with_gtest(parallel_tree_test parallel_construction/algo5.cpp)
add_test_with_gtest(linearise_test parallel_construction/algo8.cpp)
add_test_with_gtest(complete_region_test parallel_construction/algo2.cpp)

add_test_with_gtest(parallel_construction_2 parallel_construction/full_test/2_ranks.cpp)
add_test_with_gtest(parallel_construction_4 parallel_construction/full_test/4_ranks.cpp)
add_test_with_gtest(parallel_construction_8 parallel_construction/full_test/8_ranks.cpp)


add_test_with_gtest(balance_subtree balancing/algo07.cpp)
add_test_with_gtest(complete_subtree balancing/algo10.cpp)
add_test_with_gtest(ripple_prop balancing/algo09.cpp)
add_test_with_gtest(balancing balancing/algo11.cpp)

# vi: set et ts=4 sw=4 sts=4:
# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:

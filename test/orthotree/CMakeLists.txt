file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

set (IPPL_LIBS ippl ${MPI_CXX_LIBRARIES})
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (OrthoTreeTest OrthoTreeTest.cpp)
add_executable (OrthoTreeBenchmarking OrthoTreeBenchmarking.cpp) 
target_link_libraries (OrthoTreeTest ${IPPL_LIBS})
target_link_libraries (OrthoTreeBenchmarking ${IPPL_LIBS})

# Relative path for printing info
file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

# Include both the old and refactored tree implementations
include_directories (
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/OrthoTree   # Old OrthoTree
    ${CMAKE_SOURCE_DIR}/src/OrthoTreeRefactor  # Refactored OrthoTree
)

# Link directories
link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

# Set libraries and compile flags
set (IPPL_LIBS ippl ${MPI_CXX_LIBRARIES})
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

# Add executable for testing both OrthoTree implementations
add_executable (TestOrthoTreeRefactor TestOrthoTreeRefactor.cpp)
target_link_libraries (TestOrthoTreeRefactor ${IPPL_LIBS})


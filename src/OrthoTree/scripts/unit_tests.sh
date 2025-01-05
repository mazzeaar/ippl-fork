#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_TEST_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)/build/unit_tests/OrthoTree"



# ================================ 
# ADD YOUR TEST HERE IF IT REQUIRES MORE THAN ONE RANK TO PASS

# test name
MULTI_RANK_TESTS_KEYS=("parallel_tree_test" "aid_list_test" "parallel_construction_2" "parallel_construction_4" "parallel_construction_8" "balancing")

# number of ranks needed to pass
MULTI_RANK_TESTS_VALUES=(4 4 2 4 8 4)

# ================================



# Helper function to get processor count
get_processor_count() {
    local test_name=$1
    for i in "${!MULTI_RANK_TESTS_KEYS[@]}"; do
        if [[ "${MULTI_RANK_TESTS_KEYS[i]}" == "$test_name" ]]; then
            echo "${MULTI_RANK_TESTS_VALUES[i]}"
            return
        fi
    done
    echo 1  # Default processor count
}

# Function to execute test executables
run_tests() {
    local test_dir=$1
    echo "Scanning tests in: $test_dir"

    for test_executable in "$test_dir"/*; do
        if [[ -f "$test_executable" && -x "$test_executable" ]]; then
            local test_name=$(basename "$test_executable")
            local procs=$(get_processor_count "$test_name")  # Resolve processor count dynamically

            echo "Running test '$test_name' with $procs processors."
            mpiexec -n "$procs" --use-hwthread-cpus "$test_executable"

            if [[ $? -ne 0 ]]; then
                echo "Test $test_executable failed."
                exit 1
            fi
        fi
    done
}

# Build unit tests
cd "$UNIT_TEST_DIR" || exit 1
if make -j$(nproc); then
    echo "Unit tests build successful."
else
    echo "Unit tests build failed."
    exit 1
fi

# Iterate through test directories and run tests
for subfolder in "balancing" "helpers" "parallel_construction"; do
    run_tests "$UNIT_TEST_DIR/$subfolder"
done

# Run tests in the root unit_tests/OrthoTree directory
run_tests "$UNIT_TEST_DIR"

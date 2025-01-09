#!/bin/bash

# Ensure the user specifies the number of processors
if [ -z "$1" ]; then
    echo "Error: Please specify the number of processors as the first argument."
    echo "Usage: $0 <num_processors> [additional_args...]"
    exit 1
fi

NUM_PROCESSORS=$1
shift # Remove the first argument, so $@ now contains only additional arguments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)/build_cuda2/test/orthotree"
TEST_EXE="OrthoTreeTest"

# Navigate to the test directory
cd "$TEST_DIR" || exit 1

# Build the tests
if make -j$(nproc); then
    echo "Experimental build successful."
else
    echo "Experimental build failed."
    exit 1
fi

# Run the test executable with additional arguments
echo "Running test with $NUM_PROCESSORS processors: $TEST_EXE $@"
mpiexec -n "$NUM_PROCESSORS" --use-hwthread-cpus "$TEST_EXE" "$@"

# Check the test result
if [[ $? -ne 0 ]]; then
    echo "Test $TEST_EXE failed."
    exit 1
fi

echo "Test $TEST_EXE completed successfully."

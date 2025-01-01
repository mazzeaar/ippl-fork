#!/bin/bash

# Ensure the user specifies the number of processors
if [ -z "$1" ]; then
    echo "Error: Please specify the number of processors as the first argument."
    echo "Usage: $0 <num_processors> <logging_flag> [additional_args...]"
    exit 1
fi

# Read second input argument if loggin is enabled
if [ -z "$2" ]; then
    echo "Error: Please specify 1 for logging or 0 for silent as the second argument."
    echo "Usage: $0 <num_processors> <logging_flag> [additional_args...]"
    exit 1
fi

NUM_PROCESSORS=$1
LOGGING_FLAG=$2
shift 
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Find the build directory for the test executable by navigating up directories and searching for the build directory
TEST_DIR=""
TEST_EXE="OrthoTreeBenchmarking"

for i in {1..5}; do
    if [ -d "$SCRIPT_DIR/build" ]; then
        TEST_DIR="$SCRIPT_DIR/build/test/orthotree"
        if [ "$LOGGING_FLAG" -eq 1 ]; then
            echo "Found build directory at: $TEST_DIR"
        fi
        break
    fi
    SCRIPT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
done

if [ -z "$TEST_DIR" ]; then
    if [ "$LOGGING_FLAG" -eq 1 ]; then
        echo "Error: Could not find build directory within 5 parent directories"
    fi
    exit 1
fi

if [ ! -d "$TEST_DIR" ]; then
    if [ "$LOGGING_FLAG" -eq 1 ]; then
        echo "Error: Test directory does not exist: $TEST_DIR"
    fi
    exit 1
fi
# Navigate to the test directory
cd "$TEST_DIR" || exit 1

# Run the test executable with additional arguments
if [ "$LOGGING_FLAG" -eq 1 ]; then
    echo "Running test with $NUM_PROCESSORS processors: $TEST_EXE $@"
    mpirun -n "$NUM_PROCESSORS" --use-hwthread-cpus "$TEST_EXE" "$@" 
else
    mpirun -n "$NUM_PROCESSORS" --use-hwthread-cpus "$TEST_EXE" "$@" --info 5 -visualize_helper=false -print_stats=false
    echo "Test $TEST_EXE completed successfully."
fi

# Check the test result
if [[ $? -ne 0 ]]; then
    echo "Test $TEST_EXE failed."
    exit 1
fi
if [ "$LOGGING_FLAG" -eq 1 ]; then
    echo "Test $TEST_EXE completed successfully."
fi
exit 0
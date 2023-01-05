#!/bin/bash
# run from git directory
# Examples:
# DRY=echo ./bench/bench.sh
# ./bench/bench.sh

# DRY="echo"
# unset DRY

rocprof_exec="rocprof"
rocprof_option="-i bench/rocprof_input.txt --hip-trace --hsa-trace"

target_exec="./build/bin/test"
target_options="--num-keys=100000000"

$DRY ${rocprof_exec} ${rocprof_option} \
    ${target_exec} ${target_options}

# go to https://ui.perfetto.dev/ and load the json files